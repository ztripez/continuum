//! Type and function definition parsers.

use chumsky::prelude::*;

use crate::ast::{FnDef, FnParam, Range, TypeDef, TypeExpr, TypeField};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{float, ident, spanned, spanned_path, unit_string};
use super::super::{ParseError, ParserInput};

// === Type Definitions ===

pub fn type_def<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeDef, extra::Err<ParseError<'src>>> {
    just(Token::Type)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned(ident()))
        .then(
            type_field()
                .repeated()
                .collect()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(name, fields)| TypeDef {
            doc: None,
            name,
            fields,
        })
}

fn type_field<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeField, extra::Err<ParseError<'src>>> {
    spanned(ident())
        .then_ignore(just(Token::Colon))
        .then(spanned(type_expr()))
        .map(|(name, ty)| TypeField { name, ty })
}

pub fn type_expr<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    recursive(|type_expr_recurse| {
        choice((
            // Scalar or Scalar<unit, range>
            just(Token::Scalar)
                .then(
                    just(Token::LAngle)
                        .ignore_then(unit_string())
                        .then(just(Token::Comma).ignore_then(range()).or_not())
                        .then_ignore(just(Token::RAngle))
                        .or_not(),
                )
                .map(|(_, maybe_params)| match maybe_params {
                    Some((unit, range)) => TypeExpr::Scalar { unit, range },
                    None => TypeExpr::Scalar {
                        unit: "".to_string(),
                        range: None,
                    },
                }),
            // Vec2/Vec3/Vec4 or Vec2/Vec3/Vec4<unit> or Vec2/Vec3/Vec4<unit, magnitude: range>
            choice((
                just(Token::Vec2).to(2u8),
                just(Token::Vec3).to(3u8),
                just(Token::Vec4).to(4u8),
            ))
            .then(
                just(Token::LAngle)
                    .ignore_then(unit_string())
                    .then(
                        just(Token::Comma)
                            .ignore_then(just(Token::Magnitude))
                            .ignore_then(just(Token::Colon))
                            .ignore_then(magnitude_value())
                            .or_not(),
                    )
                    .then_ignore(just(Token::RAngle))
                    .or_not(),
            )
            .map(|(dim, maybe_params)| match maybe_params {
                Some((unit, magnitude)) => TypeExpr::Vector {
                    dim,
                    unit,
                    magnitude,
                },
                None => TypeExpr::Vector {
                    dim,
                    unit: "".to_string(),
                    magnitude: None,
                },
            }),
            // Tensor<rows, cols, unit>
            just(Token::Tensor)
                .ignore_then(
                    just(Token::LAngle)
                        .ignore_then(select! { Token::Integer(i) => i as u8 })
                        .then_ignore(just(Token::Comma))
                        .then(select! { Token::Integer(i) => i as u8 })
                        .then_ignore(just(Token::Comma))
                        .then(unit_string())
                        .then_ignore(just(Token::RAngle)),
                )
                .map(|((rows, cols), unit)| TypeExpr::Tensor {
                    rows,
                    cols,
                    unit,
                    constraints: Vec::new(),
                }),
            // Grid<width, height, element_type>
            just(Token::Grid)
                .ignore_then(
                    just(Token::LAngle)
                        .ignore_then(select! { Token::Integer(i) => i as u32 })
                        .then_ignore(just(Token::Comma))
                        .then(select! { Token::Integer(i) => i as u32 })
                        .then_ignore(just(Token::Comma))
                        .then(type_expr_recurse.clone())
                        .then_ignore(just(Token::RAngle)),
                )
                .map(|((width, height), element_type)| TypeExpr::Grid {
                    width,
                    height,
                    element_type: Box::new(element_type),
                }),
            // Seq<element_type>
            just(Token::Seq)
                .ignore_then(
                    just(Token::LAngle)
                        .ignore_then(type_expr_recurse)
                        .then_ignore(just(Token::RAngle)),
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

fn range<'src>() -> impl Parser<'src, ParserInput<'src>, Range, extra::Err<ParseError<'src>>> + Clone
{
    float()
        .then_ignore(just(Token::DotDot))
        .then(float())
        .map(|(min, max)| Range { min, max })
}

/// Parses a magnitude value, which can be either a range (min..max) or a single value.
/// A single value is converted to an exact range (value..value).
fn magnitude_value<'src>()
-> impl Parser<'src, ParserInput<'src>, Range, extra::Err<ParseError<'src>>> + Clone {
    choice((
        // Range: 1e10..1e12
        float()
            .then_ignore(just(Token::DotDot))
            .then(float())
            .map(|(min, max)| Range { min, max }),
        // Single value: 1 -> Range { min: 1, max: 1 }
        float().map(|v| Range { min: v, max: v }),
    ))
}

// === Function Definitions ===

pub fn fn_def<'src>() -> impl Parser<'src, ParserInput<'src>, FnDef, extra::Err<ParseError<'src>>> {
    just(Token::Fn)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            spanned(ident())
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LAngle), just(Token::RAngle))
                .or_not()
                .map(|o: Option<Vec<_>>| o.unwrap_or_default()),
        )
        .then(
            fn_param()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(
            just(Token::Arrow)
                .ignore_then(spanned(type_expr()))
                .or_not(),
        )
        .then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
        .map(|((((path, generics), params), return_type), body)| FnDef {
            doc: None,
            path,
            generics,
            params,
            return_type,
            body,
        })
}

fn fn_param<'src>()
-> impl Parser<'src, ParserInput<'src>, FnParam, extra::Err<ParseError<'src>>> + Clone {
    spanned(ident())
        .then(
            just(Token::Colon)
                .ignore_then(spanned(type_expr()))
                .or_not(),
        )
        .map(|(name, ty)| FnParam { name, ty })
}
