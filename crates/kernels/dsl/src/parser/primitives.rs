//! Primitive parser combinators for the Continuum DSL.
//!
//! This module provides low-level parsers for the fundamental tokens and
//! patterns used throughout the DSL grammar. These primitives are composed
//! by higher-level parsers in [`super::items`] and [`super::expr`].
//!
//! # Token Types
//!
//! - **Whitespace**: Spaces, newlines, and comments (`//`, `#`, `/* */`)
//! - **Identifiers**: ASCII alphanumeric names (e.g., `terra`, `temperature`)
//! - **Paths**: Dot-separated identifier sequences (e.g., `terra.surface.temp`)
//! - **Literals**: Numbers and strings
//! - **Units**: Physical unit annotations (e.g., `<K>`, `<W/m²>`)
//!
//! # Span Tracking
//!
//! The [`spanned`] function wraps any parser output in our [`Spanned`] wrapper
//! with source location tracking. This is essential for error reporting and
//! IDE features.
//!
//! ```ignore
//! use crate::parser::primitives::spanned;
//!
//! // Instead of:
//! parser().map_with(|v, e| Spanned::new(v, e.span().into()))
//!
//! // Use:
//! spanned(parser())
//! ```

use chumsky::prelude::*;

use crate::ast::{Literal, Path, Spanned};

use super::ParseError;

/// Wraps a parser's output in our [`Spanned`] with source location.
///
/// This helper function captures the source span during parsing and wraps
/// the parsed value in a `Spanned` for error reporting and IDE features.
///
/// # Example
///
/// ```ignore
/// // Parse an identifier and capture its span
/// spanned(ident()) // Returns Parser<..., Spanned<String>, ...>
/// ```
pub fn spanned<'src, O>(
    parser: impl Parser<'src, &'src str, O, extra::Err<ParseError<'src>>> + Clone,
) -> impl Parser<'src, &'src str, Spanned<O>, extra::Err<ParseError<'src>>> + Clone {
    parser.map_with(|value, extra| Spanned::new(value, extra.span().into()))
}

/// Parses whitespace and comments, consuming all of them.
///
/// Recognized comment styles:
/// - Line comments: `// text` and `# text` (but NOT `///` or `//!` doc comments)
/// - Block comments: `/* text */`
///
/// Returns `()` since whitespace is typically ignored.
pub fn ws<'src>() -> impl Parser<'src, &'src str, (), extra::Err<ParseError<'src>>> + Clone {
    // Regular line comment: // but NOT /// or //!
    // We match // then capture the rest of the line, filtering out doc comments
    let line_comment = just("//")
        .ignore_then(any().and_is(just('\n').not()).repeated().to_slice())
        .filter(|rest: &&str| {
            // Keep only if rest is empty (//\n) or doesn't start with / or !
            rest.is_empty() || (!rest.starts_with('/') && !rest.starts_with('!'))
        })
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

/// Parses item documentation comments (`///`).
///
/// Captures consecutive `///` lines and returns them as a single string
/// with the `///` prefix stripped. Returns `None` if no doc comments found.
pub fn doc_comment<'src>(
) -> impl Parser<'src, &'src str, Option<String>, extra::Err<ParseError<'src>>> + Clone {
    just("///")
        .ignore_then(any().and_is(just('\n').not()).repeated().to_slice())
        .then_ignore(just('\n').or(end().to('\n')))
        .map(|s: &str| s.trim().to_string())
        .separated_by(text::whitespace().at_most(1000))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(|lines| Some(lines.join("\n")))
        .or(empty().to(None))
}

/// Parses module-level documentation comments (`//!`).
///
/// Captures consecutive `//!` lines at the start of a file and returns them
/// as a single string with the `//!` prefix stripped.
pub fn module_doc<'src>(
) -> impl Parser<'src, &'src str, Option<String>, extra::Err<ParseError<'src>>> + Clone {
    // Allow leading whitespace before the first //!
    text::whitespace()
        .ignore_then(
            just("//!")
                .ignore_then(any().and_is(just('\n').not()).repeated().to_slice())
                .then_ignore(just('\n').or(end().to('\n')))
                .map(|s: &str| s.trim().to_string())
                .separated_by(text::whitespace().at_most(1000))
                .at_least(1)
                .collect::<Vec<_>>()
                .map(|lines| Some(lines.join("\n"))),
        )
        .or(empty().to(None))
}

/// Parses an ASCII identifier.
///
/// Identifiers start with a letter or underscore and contain alphanumeric
/// characters or underscores. Examples: `terra`, `surface_temp`, `_internal`.
pub fn ident<'src>() -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    text::ascii::ident().map(|s: &str| s.to_string())
}

/// Parses a dot-separated path of identifiers.
///
/// Paths are used throughout the DSL to reference signals, strata, config
/// values, and other named entities. Examples: `terra.surface.temperature`,
/// `config.dt`, `const.physics.gravity`.
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
    spanned(path())
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
        .map(|s: &str| {
            s.parse().unwrap_or_else(|e| {
                panic!("Internal parser error: matched float pattern '{}' but parse failed: {}", s, e)
            })
        })
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

/// Unit in angle brackets: `<K>`, `<W/m²>`
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
    spanned(unit()).or_not()
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
            spanned(string_lit())
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
            spanned(text::int(10).map(|s: &str| {
                s.parse::<u32>().unwrap_or_else(|e| {
                    panic!("Internal parser error: matched integer pattern '{}' but parse failed: {}", s, e)
                })
            }))
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
        )
}
