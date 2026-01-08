//! Primitive parser combinators

use chumsky::prelude::*;

use crate::ast::{Literal, Path, Spanned};

use super::ParseError;

/// Whitespace and comments
pub fn ws<'src>() -> impl Parser<'src, &'src str, (), extra::Err<ParseError<'src>>> + Clone {
    let line_comment = just("//")
        .then(any().and_is(just('\n').not()).repeated())
        .padded();
    let hash_comment = just("#")
        .then(any().and_is(just('\n').not()).repeated())
        .padded();
    let block_comment = just("/*")
        .then(any().and_is(just("*/").not()).repeated())
        .then(just("*/"))
        .padded();

    choice((
        line_comment.ignored(),
        hash_comment.ignored(),
        block_comment.ignored(),
        text::whitespace().at_least(1).ignored(),
    ))
    .repeated()
    .ignored()
}

/// Identifier
pub fn ident<'src>() -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    text::ascii::ident().map(|s: &str| s.to_string())
}

/// Dot-separated path
pub fn path<'src>() -> impl Parser<'src, &'src str, Path, extra::Err<ParseError<'src>>> + Clone {
    ident()
        .separated_by(just('.'))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(Path::new)
}

/// Spanned path
pub fn spanned_path<'src>(
) -> impl Parser<'src, &'src str, Spanned<Path>, extra::Err<ParseError<'src>>> + Clone {
    path().map_with(|p, e| Spanned::new(p, e.span().into()))
}

/// String literal
pub fn string_lit<'src>(
) -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    none_of("\"\\")
        .or(just('\\').ignore_then(any()))
        .repeated()
        .collect::<String>()
        .delimited_by(just('"'), just('"'))
}

/// Float number
pub fn float<'src>() -> impl Parser<'src, &'src str, f64, extra::Err<ParseError<'src>>> + Clone {
    just('-')
        .or_not()
        .then(text::int(10))
        .then(just('.').then(text::digits(10)).or_not())
        .then(
            one_of("eE")
                .then(one_of("+-").or_not())
                .then(text::digits(10))
                .or_not(),
        )
        .to_slice()
        .map(|s: &str| s.parse().unwrap_or(0.0))
}

/// Number literal
pub fn number<'src>() -> impl Parser<'src, &'src str, Literal, extra::Err<ParseError<'src>>> + Clone
{
    float().map(Literal::Float)
}

/// Literal value
pub fn literal<'src>(
) -> impl Parser<'src, &'src str, Literal, extra::Err<ParseError<'src>>> + Clone {
    choice((number(), string_lit().map(Literal::String)))
}

/// Unit in angle brackets: <K>, <W/m²>
pub fn unit<'src>() -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    none_of(">")
        .repeated()
        .at_least(1)
        .collect::<String>()
        .delimited_by(just('<'), just('>'))
}

/// Optional spanned unit
pub fn optional_unit<'src>(
) -> impl Parser<'src, &'src str, Option<Spanned<String>>, extra::Err<ParseError<'src>>> + Clone {
    unit()
        .map_with(|u, e| Spanned::new(u, e.span().into()))
        .or_not()
}
