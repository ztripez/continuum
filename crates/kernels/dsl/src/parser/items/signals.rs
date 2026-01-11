//! Signal, field, and operator parsers.

use chumsky::prelude::*;

use crate::ast::{
    ConfigEntry, ConstEntry, FieldDef, MeasureBlock, OperatorBody, OperatorDef, OperatorPhase,
    Path, Range, ResolveBlock, SeqConstraint, SignalDef, Spanned, TensorConstraint, Topology,
    TypeExpr,
};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{attr_flag, attr_path, attr_string, float, spanned, spanned_path};
use super::super::{ParseError, ParserInput};
use super::common::{assert_block, topology};
use super::config::{config_entry, const_entry};
use super::types::type_expr;

// === Signal ===

pub fn signal_def<'src>()
-> impl Parser<'src, ParserInput<'src>, SignalDef, extra::Err<ParseError<'src>>> {
    just(Token::Signal)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            signal_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
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
-> impl Parser<'src, ParserInput<'src>, SignalContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_path(Token::Strata).map(SignalContent::Strata),
        attr_string(Token::Title).map(SignalContent::Title),
        attr_string(Token::Symbol).map(SignalContent::Symbol),
        attr_flag(Token::DtRaw).to(SignalContent::DtRaw),
        just(Token::Colon)
            .ignore_then(just(Token::Uses))
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(just(Token::DtRaw))
                    .then_ignore(just(Token::RParen)),
            )
            .to(SignalContent::DtRaw),
        // Tensor constraints: `: symmetric`, `: positive_definite`
        just(Token::Colon)
            .ignore_then(just(Token::Symmetric))
            .to(SignalContent::TensorConstraint(TensorConstraint::Symmetric)),
        just(Token::Colon)
            .ignore_then(just(Token::PositiveDefinite))
            .to(SignalContent::TensorConstraint(
                TensorConstraint::PositiveDefinite,
            )),
        // Sequence constraints: `: each(min..max)`, `: sum(min..max)`
        just(Token::Colon)
            .ignore_then(just(Token::Each))
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(constraint_range())
                    .then_ignore(just(Token::RParen)),
            )
            .map(|r| SignalContent::SeqConstraint(SeqConstraint::Each(r))),
        just(Token::Colon)
            .ignore_then(just(Token::Sum))
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(constraint_range())
                    .then_ignore(just(Token::RParen)),
            )
            .map(|r| SignalContent::SeqConstraint(SeqConstraint::Sum(r))),
        // Type expression: `: TypeExpr`
        just(Token::Colon)
            .ignore_then(spanned(type_expr()))
            .map(SignalContent::Type),
        just(Token::Const)
            .ignore_then(
                const_entry()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(SignalContent::LocalConst),
        just(Token::Config)
            .ignore_then(
                config_entry()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(SignalContent::LocalConfig),
        just(Token::Resolve)
            .ignore_then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .map(|body| SignalContent::Resolve(ResolveBlock { body })),
        assert_block().map(SignalContent::Assert),
    ))
}

/// Parses a range for constraint values: `min..max`
fn constraint_range<'src>()
-> impl Parser<'src, ParserInput<'src>, Range, extra::Err<ParseError<'src>>> {
    float()
        .then_ignore(just(Token::DotDot))
        .then(float())
        .map(|(min, max)| Range { min, max })
}

// === Field ===

pub fn field_def<'src>()
-> impl Parser<'src, ParserInput<'src>, FieldDef, extra::Err<ParseError<'src>>> {
    just(Token::Field)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            field_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
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

fn field_content<'src>()
-> impl Parser<'src, ParserInput<'src>, FieldContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_path(Token::Strata).map(FieldContent::Strata),
        just(Token::Colon)
            .ignore_then(just(Token::Topology))
            .ignore_then(spanned(topology()).delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(FieldContent::Topology),
        attr_string(Token::Title).map(FieldContent::Title),
        attr_string(Token::Symbol).map(FieldContent::Symbol),
        just(Token::Colon)
            .ignore_then(spanned(type_expr()))
            .map(FieldContent::Type),
        just(Token::Measure)
            .ignore_then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .map(|body| FieldContent::Measure(MeasureBlock { body })),
    ))
}

// === Operator ===

pub fn operator_def<'src>()
-> impl Parser<'src, ParserInput<'src>, OperatorDef, extra::Err<ParseError<'src>>> {
    just(Token::Operator)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            operator_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
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
-> impl Parser<'src, ParserInput<'src>, OperatorContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_path(Token::Strata).map(OperatorContent::Strata),
        just(Token::Colon)
            .ignore_then(just(Token::Phase))
            .ignore_then(
                spanned(choice((
                    just(Token::Warmup).to(OperatorPhase::Warmup),
                    just(Token::Collect).to(OperatorPhase::Collect),
                    just(Token::Measure).to(OperatorPhase::Measure),
                )))
                .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(OperatorContent::Phase),
        just(Token::Collect)
            .ignore_then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .map(|e| OperatorContent::Body(OperatorBody::Collect(e))),
        just(Token::Measure)
            .ignore_then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .map(|e| OperatorContent::Body(OperatorBody::Measure(e))),
        assert_block().map(OperatorContent::Assert),
    ))
}
