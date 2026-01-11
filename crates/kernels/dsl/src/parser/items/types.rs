//! Type and function definition parsers.
//!
//! This module handles:
//! - `type.name { ... }` custom type definitions
//! - `fn.name(...) -> Type { ... }` pure functions
//! - Type expressions (Scalar, Vec2/3/4, Named)

use chumsky::prelude::*;

use crate::ast::{FnDef, FnParam, Range, TypeDef, TypeExpr, TypeField};
use crate::math_consts;

use super::super::ParseError;
use super::super::expr::spanned_expr;
use super::super::primitives::{float, ident, spanned, spanned_path, unit_string, ws};

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
        .map(|(name, fields)| TypeDef { doc: None, name, fields })
}

fn type_field<'src>() -> impl Parser<'src, &'src str, TypeField, extra::Err<ParseError<'src>>> {
    spanned(ident())
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned(type_expr()))
        .map(|(name, ty)| TypeField { name, ty })
}

pub fn type_expr<'src>()
-> impl Parser<'src, &'src str, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
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
            // Vec2/Vec3/Vec4<unit> or Vec2/Vec3/Vec4<unit, magnitude: range>
            choice((
                text::keyword("Vec2").to(2u8),
                text::keyword("Vec3").to(3u8),
                text::keyword("Vec4").to(4u8),
            ))
            .then(
                just('<')
                    .padded_by(ws())
                    .ignore_then(unit_string())
                    .then(
                        just(',')
                            .padded_by(ws())
                            .ignore_then(text::keyword("magnitude"))
                            .ignore_then(just(':').padded_by(ws()))
                            .ignore_then(magnitude_value())
                            .or_not(),
                    )
                    .then_ignore(just('>').padded_by(ws())),
            )
            .map(|(dim, (unit, magnitude))| TypeExpr::Vector {
                dim,
                unit,
                magnitude,
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
                .map(|((rows, cols), unit)| TypeExpr::Tensor {
                    rows,
                    cols,
                    unit,
                    constraints: Vec::new(),
                }),
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
                    constraints: Vec::new(),
                }),
            // Named type
            ident().map(TypeExpr::Named),
        ))
    })
}

/// Parses a numeric value that can be either a float literal or a math constant.
/// Math constants like PI, TAU, E, PHI are looked up from the registry.
fn numeric_value<'src>() -> impl Parser<'src, &'src str, f64, extra::Err<ParseError<'src>>> + Clone
{
    choice((
        // Float literal
        float(),
        // Math constant (uppercase identifiers like PI, TAU, E, PHI, SQRT2, etc.)
        // Also supports Unicode variants like π, τ, φ, ℯ
        // Must start with uppercase letter/underscore/unicode, but can contain digits afterwards
        any()
            .filter(|c: &char| c.is_ascii_uppercase() || *c == '_' || !c.is_ascii())
            .then(
                any()
                    .filter(|c: &char| {
                        c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_' || !c.is_ascii()
                    })
                    .repeated(),
            )
            .to_slice()
            .try_map(|name: &str, span| {
                math_consts::lookup(name)
                    .ok_or_else(|| Rich::custom(span, format!("unknown math constant '{name}'")))
            }),
    ))
}

fn range<'src>() -> impl Parser<'src, &'src str, Range, extra::Err<ParseError<'src>>> + Clone {
    numeric_value()
        .then_ignore(just("..").padded_by(ws()))
        .then(numeric_value())
        .map(|(min, max)| Range { min, max })
}

/// Parses a magnitude value, which can be either a range (min..max) or a single value.
/// A single value is converted to an exact range (value..value).
/// Supports both float literals and math constants.
fn magnitude_value<'src>()
-> impl Parser<'src, &'src str, Range, extra::Err<ParseError<'src>>> + Clone {
    choice((
        // Range: 1e10..1e12 or 0..PI
        numeric_value()
            .then_ignore(just("..").padded_by(ws()))
            .then(numeric_value())
            .map(|(min, max)| Range { min, max }),
        // Single value: 1 -> Range { min: 1, max: 1 }
        numeric_value().map(|v| Range { min: v, max: v }),
    ))
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
            doc: None,
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
