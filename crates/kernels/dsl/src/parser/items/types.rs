//! Type and function definition parsers.
//!
//! This module handles:
//! - `type.name { ... }` custom type definitions
//! - `fn.name(...) -> Type { ... }` pure functions
//! - Type expressions (Scalar, Vec2/3/4, Named)

use chumsky::prelude::*;

use crate::ast::{FnDef, FnParam, Range, Spanned, TypeDef, TypeExpr, TypeField};

use super::super::expr::spanned_expr;
use super::super::primitives::{float, ident, spanned_path, unit_string, ws};
use super::super::ParseError;

// === Type Definitions ===

pub fn type_def<'src>() -> impl Parser<'src, &'src str, TypeDef, extra::Err<ParseError<'src>>> {
    text::keyword("type")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(ident().map_with(|i, e| Spanned::new(i, e.span().into())))
        .padded_by(ws())
        .then(
            type_field()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(name, fields)| TypeDef { name, fields })
}

fn type_field<'src>() -> impl Parser<'src, &'src str, TypeField, extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|i, e| Spanned::new(i, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
        .map(|(name, ty)| TypeField { name, ty })
}

pub fn type_expr<'src>(
) -> impl Parser<'src, &'src str, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    choice((
        text::keyword("Scalar")
            .ignore_then(
                just('<')
                    .padded_by(ws())
                    .ignore_then(unit_string())
                    .then(just(',').padded_by(ws()).ignore_then(range()).or_not())
                    .then_ignore(just('>').padded_by(ws())),
            )
            .map(|(unit, range)| TypeExpr::Scalar { unit, range }),
        choice((
            text::keyword("Vec2").to(2u8),
            text::keyword("Vec3").to(3u8),
            text::keyword("Vec4").to(4u8),
        ))
        .then(
            just('<')
                .padded_by(ws())
                .ignore_then(unit_string())
                .then_ignore(just('>').padded_by(ws())),
        )
        .map(|(dim, unit)| TypeExpr::Vector {
            dim,
            unit,
            magnitude: None,
        }),
        ident().map(TypeExpr::Named),
    ))
}

fn range<'src>() -> impl Parser<'src, &'src str, Range, extra::Err<ParseError<'src>>> + Clone {
    float()
        .then_ignore(just("..").padded_by(ws()))
        .then(float())
        .map(|(min, max)| Range { min, max })
}

// === Function Definitions ===

pub fn fn_def<'src>() -> impl Parser<'src, &'src str, FnDef, extra::Err<ParseError<'src>>> {
    text::keyword("fn")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .then(
            fn_param()
                .padded_by(ws())
                .separated_by(just(',').padded_by(ws()))
                .allow_trailing()
                .collect()
                .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws())),
        )
        .then(
            just("->")
                .padded_by(ws())
                .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
                .or_not(),
        )
        .then(
            spanned_expr()
                .padded_by(ws())
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(((path, params), return_type), body)| FnDef {
            path,
            params,
            return_type,
            body,
        })
}

fn fn_param<'src>() -> impl Parser<'src, &'src str, FnParam, extra::Err<ParseError<'src>>> + Clone {
    ident()
        .map_with(|i, e| Spanned::new(i, e.span().into()))
        .then(
            just(':')
                .padded_by(ws())
                .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
                .or_not(),
        )
        .map(|(name, ty)| FnParam { name, ty })
}
