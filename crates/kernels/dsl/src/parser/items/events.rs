//! Event-related parsers (impulses, fractures, chronicles).
//!
//! This module handles:
//! - `impulse.name { ... }` external causal inputs
//! - `fracture.name { ... }` tension detectors
//! - `chronicle.name { ... }` observer event handlers

use chumsky::prelude::*;

use crate::ast::{
    ApplyBlock, ChronicleDef, EmitStatement, Expr, FractureDef, ImpulseDef, ObserveBlock,
    ObserveHandler, Spanned, TypeExpr,
};

use super::super::expr::spanned_expr;
use super::super::primitives::{ident, spanned_path, ws};
use super::super::ParseError;
use super::types::type_expr;

// === Impulse ===

pub fn impulse_def<'src>(
) -> impl Parser<'src, &'src str, ImpulseDef, extra::Err<ParseError<'src>>> {
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
                local_config: vec![],
                apply: None,
            };
            for content in contents {
                match content {
                    ImpulseContent::Type(t) => def.payload_type = Some(t),
                    ImpulseContent::Apply(a) => def.apply = Some(a),
                }
            }
            def
        })
}

#[derive(Clone)]
enum ImpulseContent {
    Type(Spanned<TypeExpr>),
    Apply(ApplyBlock),
}

fn impulse_content<'src>(
) -> impl Parser<'src, &'src str, ImpulseContent, extra::Err<ParseError<'src>>> {
    choice((
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(ImpulseContent::Type),
        text::keyword("apply")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| ImpulseContent::Apply(ApplyBlock { body })),
    ))
}

// === Fracture ===

pub fn fracture_def<'src>(
) -> impl Parser<'src, &'src str, FractureDef, extra::Err<ParseError<'src>>> {
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
            let mut conditions = vec![];
            let mut emit = vec![];
            for content in contents {
                match content {
                    FractureContent::When(w) => conditions = w,
                    FractureContent::Emit(e) => emit = e,
                }
            }
            FractureDef {
                path,
                conditions,
                emit,
            }
        })
}

#[derive(Clone)]
enum FractureContent {
    When(Vec<Spanned<Expr>>),
    Emit(Vec<EmitStatement>),
}

fn fracture_content<'src>(
) -> impl Parser<'src, &'src str, FractureContent, extra::Err<ParseError<'src>>> {
    choice((
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
        text::keyword("emit")
            .padded_by(ws())
            .ignore_then(
                emit_statement()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(FractureContent::Emit),
    ))
}

fn emit_statement<'src>(
) -> impl Parser<'src, &'src str, EmitStatement, extra::Err<ParseError<'src>>> + Clone {
    text::keyword("signal")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .then_ignore(just("<-").padded_by(ws()))
        .then(spanned_expr())
        .map(|(target, value)| EmitStatement { target, value })
}

// === Chronicle ===

pub fn chronicle_def<'src>(
) -> impl Parser<'src, &'src str, ChronicleDef, extra::Err<ParseError<'src>>> {
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

fn observe_block<'src>(
) -> impl Parser<'src, &'src str, ObserveBlock, extra::Err<ParseError<'src>>> {
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

fn observe_handler<'src>(
) -> impl Parser<'src, &'src str, ObserveHandler, extra::Err<ParseError<'src>>> {
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

fn event_field<'src>(
) -> impl Parser<'src, &'src str, (Spanned<String>, Spanned<Expr>), extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|i, e| Spanned::new(i, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned_expr())
}
