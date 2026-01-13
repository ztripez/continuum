use chumsky::{error::Rich, prelude::*, span::SimpleSpan};

use crate::ast::{Range, TypeExpr};

use super::super::super::lexer::Token;
use super::super::super::primitives::{float, ident, tok, unit_string};
use super::super::super::{ParseError, ParserInput};

/// Parser for all primitive types registered in the DSL.
pub fn primitive_type_parser<'src>(
    type_expr_recurse: impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>>
    + Clone,
) -> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    choice((
        scalar_parser(),
        vector_parser(),
        quat_parser(),
        tensor_parser(),
        grid_parser(type_expr_recurse.clone()),
        seq_parser(type_expr_recurse),
    ))
}

fn scalar_parser<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
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
        })
}

fn vector_parser<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
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
    })
}

fn quat_parser<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Quat)
        .then(
            just(Token::LAngle)
                .ignore_then(just(Token::Magnitude))
                .ignore_then(just(Token::Colon))
                .ignore_then(magnitude_value())
                .then_ignore(just(Token::RAngle))
                .or_not(),
        )
        .map(|(_, magnitude)| TypeExpr::Quat { magnitude })
}

fn tensor_parser<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
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
        })
}

fn grid_parser<'src>(
    type_expr_recurse: impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>>
    + Clone,
) -> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Grid)
        .ignore_then(
            just(Token::LAngle)
                .ignore_then(select! { Token::Integer(i) => i as u32 })
                .then_ignore(just(Token::Comma))
                .then(select! { Token::Integer(i) => i as u32 })
                .then_ignore(just(Token::Comma))
                .then(type_expr_recurse)
                .then_ignore(just(Token::RAngle)),
        )
        .map(|((width, height), element_type)| TypeExpr::Grid {
            width,
            height,
            element_type: Box::new(element_type),
        })
}

fn seq_parser<'src>(
    type_expr_recurse: impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>>
    + Clone,
) -> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Seq)
        .ignore_then(
            just(Token::LAngle)
                .ignore_then(type_expr_recurse)
                .then_ignore(just(Token::RAngle)),
        )
        .map(|element_type| TypeExpr::Seq {
            element_type: Box::new(element_type),
            constraints: Vec::new(),
        })
}

fn range<'src>() -> impl Parser<'src, ParserInput<'src>, Range, extra::Err<ParseError<'src>>> + Clone
{
    numeric_value()
        .then_ignore(just(Token::DotDot))
        .then(numeric_value())
        .map(|(min, max)| Range { min, max })
}

fn magnitude_value<'src>()
-> impl Parser<'src, ParserInput<'src>, Range, extra::Err<ParseError<'src>>> + Clone {
    choice((
        numeric_value()
            .then_ignore(just(Token::DotDot))
            .then(numeric_value())
            .map(|(min, max)| Range { min, max }),
        numeric_value().map(|v| Range { min: v, max: v }),
    ))
}

fn numeric_value<'src>()
-> impl Parser<'src, ParserInput<'src>, f64, extra::Err<ParseError<'src>>> + Clone {
    let value = choice((
        float(),
        ident().try_map(|name: String, span: SimpleSpan| {
            crate::math_consts::lookup(&name)
                .ok_or_else(|| Rich::custom(span.into(), format!("unknown math constant '{name}'")))
        }),
    ));

    tok(Token::Minus)
        .or_not()
        .then(value)
        .map(|(minus, val): (Option<Token>, f64)| if minus.is_some() { -val } else { val })
}
