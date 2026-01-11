//! Signal, field, and operator parsers.
//!
//! This module handles:
//! - `signal.name { ... }` authoritative state signals
//! - `field.name { ... }` observer data fields
//! - `operator.name { ... }` phase logic operators

use chumsky::prelude::*;

use crate::ast::{
    ConfigEntry, ConstEntry, FieldDef, MeasureBlock, OperatorBody, OperatorDef, OperatorPhase,
    Path, Range, ResolveBlock, SeqConstraint, SignalDef, Spanned, TensorConstraint, Topology,
    TypeExpr,
};

use super::super::ParseError;
use super::super::expr::spanned_expr;
use super::super::primitives::{
    attr_flag, attr_path, attr_string, float, spanned, spanned_path, ws,
};
use super::common::{assert_block, topology};
use super::config::{config_entry, const_entry};
use super::types::type_expr;

// === Signal ===

pub fn signal_def<'src>() -> impl Parser<'src, &'src str, SignalDef, extra::Err<ParseError<'src>>> {
    text::keyword("signal")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            signal_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = SignalDef {
                doc: None,
                path,
                ty: None,
                strata: None,
                title: None,
                symbol: None,
                dt_raw: false,
                local_consts: vec![],
                local_config: vec![],
                warmup: None,
                resolve: None,
                assertions: None,
                tensor_constraints: vec![],
                seq_constraints: vec![],
            };
            for content in contents {
                match content {
                    SignalContent::Type(t) => def.ty = Some(t),
                    SignalContent::Strata(s) => def.strata = Some(s),
                    SignalContent::Title(t) => def.title = Some(t),
                    SignalContent::Symbol(s) => def.symbol = Some(s),
                    SignalContent::DtRaw => def.dt_raw = true,
                    SignalContent::LocalConst(c) => def.local_consts.extend(c),
                    SignalContent::LocalConfig(c) => def.local_config.extend(c),
                    SignalContent::Resolve(r) => def.resolve = Some(r),
                    SignalContent::Assert(a) => def.assertions = Some(a),
                    SignalContent::TensorConstraint(c) => def.tensor_constraints.push(c),
                    SignalContent::SeqConstraint(c) => def.seq_constraints.push(c),
                }
            }
            def
        })
}

#[derive(Clone)]
enum SignalContent {
    Type(Spanned<TypeExpr>),
    Strata(Spanned<Path>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    DtRaw,
    LocalConst(Vec<ConstEntry>),
    LocalConfig(Vec<ConfigEntry>),
    Resolve(ResolveBlock),
    Assert(crate::ast::AssertBlock),
    /// Tensor constraint: `: symmetric` or `: positive_definite`
    TensorConstraint(TensorConstraint),
    /// Sequence constraint: `: each(min..max)` or `: sum(min..max)`
    SeqConstraint(SeqConstraint),
}

fn signal_content<'src>()
-> impl Parser<'src, &'src str, SignalContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_path("strata").map(SignalContent::Strata),
        attr_string("title").map(SignalContent::Title),
        attr_string("symbol").map(SignalContent::Symbol),
        attr_flag("dt_raw").to(SignalContent::DtRaw),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("uses"))
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(just("dt_raw"))
                    .then_ignore(just(')').padded_by(ws())),
            )
            .to(SignalContent::DtRaw),
        // Tensor constraints: `: symmetric`, `: positive_definite`
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("symmetric"))
            .to(SignalContent::TensorConstraint(TensorConstraint::Symmetric)),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("positive_definite"))
            .to(SignalContent::TensorConstraint(
                TensorConstraint::PositiveDefinite,
            )),
        // Sequence constraints: `: each(min..max)`, `: sum(min..max)`
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("each"))
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(constraint_range())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(|r| SignalContent::SeqConstraint(SeqConstraint::Each(r))),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("sum"))
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(constraint_range())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(|r| SignalContent::SeqConstraint(SeqConstraint::Sum(r))),
        // Type expression: `: TypeExpr`
        just(':')
            .padded_by(ws())
            .ignore_then(spanned(type_expr()))
            .map(SignalContent::Type),
        text::keyword("const")
            .padded_by(ws())
            .ignore_then(
                const_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(SignalContent::LocalConst),
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(SignalContent::LocalConfig),
        text::keyword("resolve")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| SignalContent::Resolve(ResolveBlock { body })),
        assert_block().map(SignalContent::Assert),
    ))
}

/// Parses a range for constraint values: `min..max`
fn constraint_range<'src>() -> impl Parser<'src, &'src str, Range, extra::Err<ParseError<'src>>> {
    float()
        .then_ignore(just("..").padded_by(ws()))
        .then(float())
        .map(|(min, max)| Range { min, max })
}

// === Field ===

pub fn field_def<'src>() -> impl Parser<'src, &'src str, FieldDef, extra::Err<ParseError<'src>>> {
    text::keyword("field")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            field_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = FieldDef {
                doc: None,
                path,
                ty: None,
                strata: None,
                topology: None,
                title: None,
                symbol: None,
                measure: None,
            };
            for content in contents {
                match content {
                    FieldContent::Type(t) => def.ty = Some(t),
                    FieldContent::Strata(s) => def.strata = Some(s),
                    FieldContent::Topology(t) => def.topology = Some(t),
                    FieldContent::Title(t) => def.title = Some(t),
                    FieldContent::Symbol(s) => def.symbol = Some(s),
                    FieldContent::Measure(m) => def.measure = Some(m),
                }
            }
            def
        })
}

#[derive(Clone)]
enum FieldContent {
    Type(Spanned<TypeExpr>),
    Strata(Spanned<Path>),
    Topology(Spanned<Topology>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    Measure(MeasureBlock),
}

fn field_content<'src>() -> impl Parser<'src, &'src str, FieldContent, extra::Err<ParseError<'src>>>
{
    choice((
        attr_path("strata").map(FieldContent::Strata),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("topology"))
            .ignore_then(
                spanned(topology())
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(FieldContent::Topology),
        attr_string("title").map(FieldContent::Title),
        attr_string("symbol").map(FieldContent::Symbol),
        just(':')
            .padded_by(ws())
            .ignore_then(spanned(type_expr()))
            .map(FieldContent::Type),
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| FieldContent::Measure(MeasureBlock { body })),
    ))
}

// === Operator ===

pub fn operator_def<'src>()
-> impl Parser<'src, &'src str, OperatorDef, extra::Err<ParseError<'src>>> {
    text::keyword("operator")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            operator_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = OperatorDef {
                doc: None,
                path,
                strata: None,
                phase: None,
                body: None,
                assertions: None,
            };
            for content in contents {
                match content {
                    OperatorContent::Strata(s) => def.strata = Some(s),
                    OperatorContent::Phase(p) => def.phase = Some(p),
                    OperatorContent::Body(b) => def.body = Some(b),
                    OperatorContent::Assert(a) => def.assertions = Some(a),
                }
            }
            def
        })
}

#[derive(Clone)]
enum OperatorContent {
    Strata(Spanned<Path>),
    Phase(Spanned<OperatorPhase>),
    Body(OperatorBody),
    Assert(crate::ast::AssertBlock),
}

fn operator_content<'src>()
-> impl Parser<'src, &'src str, OperatorContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_path("strata").map(OperatorContent::Strata),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("phase"))
            .ignore_then(
                spanned(choice((
                    text::keyword("warmup").to(OperatorPhase::Warmup),
                    text::keyword("collect").to(OperatorPhase::Collect),
                    text::keyword("measure").to(OperatorPhase::Measure),
                )))
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
            )
            .map(OperatorContent::Phase),
        text::keyword("collect")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|e| OperatorContent::Body(OperatorBody::Collect(e))),
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|e| OperatorContent::Body(OperatorBody::Measure(e))),
        assert_block().map(OperatorContent::Assert),
    ))
}
