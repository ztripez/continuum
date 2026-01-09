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

/// Unit string content (without angle brackets): K, W/m², kg/m³
/// Used in type expressions like Scalar<kg/m³, 0..1000>
/// Accepts Unicode superscripts and common unit characters
pub fn unit_string<'src>(
) -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    // Unit strings can contain:
    // - Letters (a-z, A-Z)
    // - Digits (0-9)
    // - Unicode superscripts (⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺)
    // - Division and multiplication (/, *, ·)
    // - Degree symbol (°)
    // - Common unit prefixes are just letters
    // Stop at: comma, closing bracket, whitespace
    any()
        .filter(|c: &char| {
            c.is_alphanumeric()
                || *c == '/'
                || *c == '*'
                || *c == '·'
                || *c == '°'
                || *c == '-'
                || *c == '_'
                // Unicode superscripts
                || *c == '⁰'
                || *c == '¹'
                || *c == '²'
                || *c == '³'
                || *c == '⁴'
                || *c == '⁵'
                || *c == '⁶'
                || *c == '⁷'
                || *c == '⁸'
                || *c == '⁹'
                || *c == '⁻'
                || *c == '⁺'
                // Unicode subscripts (less common but might be useful)
                || *c == '₀'
                || *c == '₁'
                || *c == '₂'
                || *c == '₃'
                || *c == '₄'
                || *c == '₅'
                || *c == '₆'
                || *c == '₇'
                || *c == '₈'
                || *c == '₉'
        })
        .repeated()
        .at_least(1)
        .collect::<String>()
}

/// Optional spanned unit
pub fn optional_unit<'src>(
) -> impl Parser<'src, &'src str, Option<Spanned<String>>, extra::Err<ParseError<'src>>> + Clone {
    unit()
        .map_with(|u, e| Spanned::new(u, e.span().into()))
        .or_not()
}

// === Common attribute parsers (DRY helpers) ===

/// Parse `: keyword(string_lit)` pattern used for title/symbol attributes
/// Example: `: title("My Title")`
pub fn attr_string<'src>(
    keyword: &'static str,
) -> impl Parser<'src, &'src str, Spanned<String>, extra::Err<ParseError<'src>>> + Clone {
    just(':')
        .padded_by(ws())
        .ignore_then(text::keyword(keyword))
        .ignore_then(
            string_lit()
                .map_with(|s, e| Spanned::new(s, e.span().into()))
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
        )
}

/// Parse `: keyword(path)` pattern used for strata attributes
/// Example: `: strata(terra.crust)`
pub fn attr_path<'src>(
    keyword: &'static str,
) -> impl Parser<'src, &'src str, Spanned<Path>, extra::Err<ParseError<'src>>> + Clone {
    just(':')
        .padded_by(ws())
        .ignore_then(text::keyword(keyword))
        .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
}

/// Parse `: keyword` pattern (no value) used for flag attributes
/// Example: `: initial`, `: terminal`
pub fn attr_flag<'src>(
    keyword: &'static str,
) -> impl Parser<'src, &'src str, (), extra::Err<ParseError<'src>>> + Clone {
    just(':')
        .padded_by(ws())
        .ignore_then(text::keyword(keyword))
        .ignored()
}

/// Parse `: keyword(int)` pattern returning a spanned integer
/// Example: `: stride(4)`
pub fn attr_int<'src>(
    keyword: &'static str,
) -> impl Parser<'src, &'src str, Spanned<u32>, extra::Err<ParseError<'src>>> + Clone {
    just(':')
        .padded_by(ws())
        .ignore_then(text::keyword(keyword))
        .ignore_then(
            text::int(10)
                .map(|s: &str| s.parse::<u32>().unwrap_or(0))
                .map_with(|n, e| {
                    let span: chumsky::span::SimpleSpan = e.span();
                    Spanned::new(n, span.start..span.end)
                })
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
        )
}
