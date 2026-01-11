//! Event-related parsers (impulses, fractures, chronicles).
//!
//! This module handles:
//! - `impulse.name { ... }` external causal inputs
//! - `fracture.name { ... }` tension detectors
//! - `chronicle.name { ... }` observer event handlers

use chumsky::prelude::*;

use crate::ast::{
    ApplyBlock, ChronicleDef, ConfigEntry, Expr, FractureDef, ImpulseDef, ObserveBlock,
    ObserveHandler, Path, Spanned, TypeExpr,
};

use super::super::primitives::attr_string;

use super::super::ParseError;
use super::super::expr::{spanned_effect_expr, spanned_expr};
use super::super::primitives::{attr_path, ident, spanned, spanned_path, ws};
use super::config::config_entry;
use super::types::type_expr;

// === Impulse ===

pub fn impulse_def<'src>() -> impl Parser<'src, &'src str, ImpulseDef, extra::Err<ParseError<'src>>>
{
    text::keyword("impulse")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            impulse_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = ImpulseDef {
                path,
                payload_type: None,
                title: None,
                symbol: None,
                local_config: vec![],
                apply: None,
            };
            for content in contents {
                match content {
                    ImpulseContent::Type(t) => def.payload_type = Some(t),
                    ImpulseContent::Title(t) => def.title = Some(t),
                    ImpulseContent::Symbol(s) => def.symbol = Some(s),
                    ImpulseContent::Config(c) => def.local_config = c,
                    ImpulseContent::Apply(a) => def.apply = Some(a),
                }
            }
            def
        })
}

#[derive(Clone)]
enum ImpulseContent {
    Type(Spanned<TypeExpr>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    Config(Vec<ConfigEntry>),
    Apply(ApplyBlock),
}

fn impulse_content<'src>()
-> impl Parser<'src, &'src str, ImpulseContent, extra::Err<ParseError<'src>>> {
    choice((
        // Attributes with arguments - must come before generic type parser
        attr_string("title").map(ImpulseContent::Title),
        attr_string("symbol").map(ImpulseContent::Symbol),
        // Type expression: `: TypeExpr`
        just(':')
            .padded_by(ws())
            .ignore_then(spanned(type_expr()))
            .map(ImpulseContent::Type),
        // Config block: `config { ... }`
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(ImpulseContent::Config),
        // Apply block: `apply { expr; expr; ... }` - supports semicolon-separated expressions
        text::keyword("apply")
            .padded_by(ws())
            .ignore_then(
                spanned_effect_expr()
                    .padded_by(ws())
                    .separated_by(just(';').padded_by(ws()))
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map_with(|exprs, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                let body = if exprs.len() == 1 {
                    exprs.into_iter().next().unwrap()
                } else {
                    Spanned::new(Expr::Block(exprs), span.start..span.end)
                };
                ImpulseContent::Apply(ApplyBlock { body })
            }),
    ))
}

// === Fracture ===

pub fn fracture_def<'src>()
-> impl Parser<'src, &'src str, FractureDef, extra::Err<ParseError<'src>>> {
    text::keyword("fracture")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            fracture_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut strata = None;
            let mut local_config = vec![];
            let mut conditions = vec![];
            let mut emit = None;
            for content in contents {
                match content {
                    FractureContent::Strata(s) => strata = Some(s),
                    FractureContent::Config(c) => local_config = c,
                    FractureContent::When(w) => conditions = w,
                    FractureContent::Emit(e) => emit = Some(e),
                }
            }
            FractureDef {
                path,
                strata,
                local_config,
                conditions,
                emit,
            }
        })
}

#[derive(Clone)]
enum FractureContent {
    Strata(Spanned<Path>),
    Config(Vec<ConfigEntry>),
    When(Vec<Spanned<Expr>>),
    Emit(Spanned<Expr>),
}

fn fracture_content<'src>()
-> impl Parser<'src, &'src str, FractureContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path) - must come before other choices to avoid matching "strata" elsewhere
        attr_path("strata").map(FractureContent::Strata),
        // config { ... } - local config block
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(FractureContent::Config),
        // when { ... } - trigger conditions
        text::keyword("when")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(FractureContent::When),
        // emit { expr... } - emit expression(s), supports let bindings
        // Multiple expressions are wrapped in a Block
        text::keyword("emit")
            .padded_by(ws())
            .ignore_then(
                spanned_effect_expr()
                    .padded_by(ws())
                    .separated_by(just(';').padded_by(ws()))
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map_with(|exprs, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                if exprs.len() == 1 {
                    FractureContent::Emit(exprs.into_iter().next().unwrap())
                } else {
                    // Multiple expressions -> wrap in a Block
                    FractureContent::Emit(Spanned::new(Expr::Block(exprs), span.start..span.end))
                }
            }),
    ))
}

// === Chronicle ===

pub fn chronicle_def<'src>()
-> impl Parser<'src, &'src str, ChronicleDef, extra::Err<ParseError<'src>>> {
    text::keyword("chronicle")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            observe_block()
                .padded_by(ws())
                .or_not()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, observe)| ChronicleDef { path, observe })
}

fn observe_block<'src>() -> impl Parser<'src, &'src str, ObserveBlock, extra::Err<ParseError<'src>>>
{
    text::keyword("observe")
        .padded_by(ws())
        .ignore_then(
            observe_handler()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|handlers| ObserveBlock { handlers })
}

fn observe_handler<'src>()
-> impl Parser<'src, &'src str, ObserveHandler, extra::Err<ParseError<'src>>> {
    text::keyword("when")
        .padded_by(ws())
        .ignore_then(spanned_expr())
        .then(
            text::keyword("emit")
                .padded_by(ws())
                .ignore_then(text::keyword("event"))
                .padded_by(ws())
                .ignore_then(just('.'))
                .ignore_then(spanned_path())
                .then(
                    event_field()
                        .padded_by(ws())
                        .repeated()
                        .collect()
                        .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
                )
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(condition, (event_name, event_fields))| ObserveHandler {
            condition,
            event_name,
            event_fields,
        })
}

fn event_field<'src>()
-> impl Parser<'src, &'src str, (Spanned<String>, Spanned<Expr>), extra::Err<ParseError<'src>>> {
    spanned(ident())
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned_expr())
}
