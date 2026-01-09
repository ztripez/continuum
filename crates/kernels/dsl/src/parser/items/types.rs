//! Type and function definition parsers.
//!
//! This module handles:
//! - `type.name { ... }` custom type definitions
//! - `fn.name(...) -> Type { ... }` pure functions
//! - Type expressions (Scalar, Vec2/3/4, Named)

use chumsky::prelude::*;

use crate::ast::{FnDef, FnParam, Range, TypeDef, TypeExpr, TypeField};

use super::super::expr::spanned_expr;
use super::super::primitives::{float, ident, spanned, spanned_path, unit_string, ws};
use super::super::ParseError;

// === Type Definitions ===

pub fn type_def<'src>() -> impl Parser<'src, &'src str, TypeDef, extra::Err<ParseError<'src>>> {
    text::keyword("type")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned(ident()))
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
    spanned(ident())
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned(type_expr()))
        .map(|(name, ty)| TypeField { name, ty })
}

pub fn type_expr<'src>(
) -> impl Parser<'src, &'src str, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    recursive(|type_expr_recurse| {
        choice((
            // Scalar<unit, range>
            text::keyword("Scalar")
                .ignore_then(
                    just('<')
                        .padded_by(ws())
                        .ignore_then(unit_string())
                        .then(just(',').padded_by(ws()).ignore_then(range()).or_not())
                        .then_ignore(just('>').padded_by(ws())),
                )
                .map(|(unit, range)| TypeExpr::Scalar { unit, range }),
            // Vec2/Vec3/Vec4<unit>
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
            // Tensor<rows, cols, unit>
            text::keyword("Tensor")
                .ignore_then(
                    just('<')
                        .padded_by(ws())
                        .ignore_then(
                            text::int(10)
                                .map(|s: &str| s.parse::<u8>().unwrap_or(0))
                                .padded_by(ws()),
                        )
                        .then_ignore(just(',').padded_by(ws()))
                        .then(
                            text::int(10)
                                .map(|s: &str| s.parse::<u8>().unwrap_or(0))
                                .padded_by(ws()),
                        )
                        .then_ignore(just(',').padded_by(ws()))
                        .then(unit_string())
                        .then_ignore(just('>').padded_by(ws())),
                )
                .map(|((rows, cols), unit)| TypeExpr::Tensor { rows, cols, unit }),
            // Grid<width, height, element_type>
            text::keyword("Grid")
                .ignore_then(
                    just('<')
                        .padded_by(ws())
                        .ignore_then(
                            text::int(10)
                                .map(|s: &str| s.parse::<u32>().unwrap_or(0))
                                .padded_by(ws()),
                        )
                        .then_ignore(just(',').padded_by(ws()))
                        .then(
                            text::int(10)
                                .map(|s: &str| s.parse::<u32>().unwrap_or(0))
                                .padded_by(ws()),
                        )
                        .then_ignore(just(',').padded_by(ws()))
                        .then(type_expr_recurse.clone())
                        .then_ignore(just('>').padded_by(ws())),
                )
                .map(|((width, height), element_type)| TypeExpr::Grid {
                    width,
                    height,
                    element_type: Box::new(element_type),
                }),
            // Seq<element_type>
            text::keyword("Seq")
                .ignore_then(
                    just('<')
                        .padded_by(ws())
                        .ignore_then(type_expr_recurse)
                        .then_ignore(just('>').padded_by(ws())),
                )
                .map(|element_type| TypeExpr::Seq {
                    element_type: Box::new(element_type),
                }),
            // Named type
            ident().map(TypeExpr::Named),
        ))
    })
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
            spanned(ident())
                .separated_by(just(',').padded_by(ws()))
                .collect()
                .delimited_by(just('<').padded_by(ws()), just('>').padded_by(ws()))
                .or_not()
                .map(|o| o.unwrap_or_default()),
        )
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
                .ignore_then(spanned(type_expr()))
                .or_not(),
        )
        .then(
            spanned_expr()
                .padded_by(ws())
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|((((path, generics), params), return_type), body)| FnDef {
            path,
            generics,
            params,
            return_type,
            body,
        })
}

fn fn_param<'src>() -> impl Parser<'src, &'src str, FnParam, extra::Err<ParseError<'src>>> + Clone {
    spanned(ident())
        .then(
            just(':')
                .padded_by(ws())
                .ignore_then(spanned(type_expr()))
                .or_not(),
        )
        .map(|(name, ty)| FnParam { name, ty })
}
