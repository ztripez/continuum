//! Primitive parser combinators for the Continuum DSL.

use chumsky::input::{self, MapExtra};
use chumsky::prelude::*;

use super::lexer::Token;
use super::{ParseError, ParserInput};
use crate::ast::{Literal, Path, Spanned};

/// Wraps a parser's output in our [`Spanned`] with source location.
pub fn spanned<'src, I, O>(
    parser: impl Parser<'src, I, O, extra::Err<ParseError<'src>>> + Clone,
) -> impl Parser<'src, I, Spanned<O>, extra::Err<ParseError<'src>>> + Clone
where
    I: input::Input<'src, Span = SimpleSpan, Token = Token>,
{
    parser.map_with(|value, extra: &mut MapExtra<'src, '_, I, _>| {
        Spanned::new(value, extra.span().into())
    })
}

/// Parses an identifier or a keyword that can act as an identifier.
pub fn ident<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    let base_ident = select! { Token::Ident(name) => name };

    let kw_group1 = choice((
        just(Token::World).to("world".to_string()),
        just(Token::Strata).to("strata".to_string()),
        just(Token::Era).to("era".to_string()),
        just(Token::Signal).to("signal".to_string()),
        just(Token::Field).to("field".to_string()),
        just(Token::Operator).to("operator".to_string()),
        just(Token::Fn).to("fn".to_string()),
        just(Token::Type).to("type".to_string()),
        just(Token::Impulse).to("impulse".to_string()),
        just(Token::Fracture).to("fracture".to_string()),
        just(Token::Chronicle).to("chronicle".to_string()),
        just(Token::Entity).to("entity".to_string()),
        just(Token::Member).to("member".to_string()),
        just(Token::Const).to("const".to_string()),
        just(Token::Config).to("config".to_string()),
        just(Token::Policy).to("policy".to_string()),
        just(Token::Version).to("version".to_string()),
        just(Token::Initial).to("initial".to_string()),
        just(Token::Terminal).to("terminal".to_string()),
        just(Token::Stride).to("stride".to_string()),
        just(Token::Title).to("title".to_string()),
        just(Token::Symbol).to("symbol".to_string()),
        just(Token::Active).to("active".to_string()),
        just(Token::Converge).to("converge".to_string()),
        just(Token::Warmup).to("warmup".to_string()),
    ));

    let kw_group2 = choice((
        just(Token::Iterate).to("iterate".to_string()),
        just(Token::Phase).to("phase".to_string()),
        just(Token::Magnitude).to("magnitude".to_string()),
        just(Token::Symmetric).to("symmetric".to_string()),
        just(Token::PositiveDefinite).to("positive_definite".to_string()),
        just(Token::Topology).to("topology".to_string()),
        just(Token::Min).to("min".to_string()),
        just(Token::Max).to("max".to_string()),
        just(Token::Mean).to("mean".to_string()),
        just(Token::Sum).to("sum".to_string()),
        just(Token::Product).to("product".to_string()),
        just(Token::Any).to("any".to_string()),
        just(Token::All).to("all".to_string()),
        just(Token::None).to("none".to_string()),
        just(Token::First).to("first".to_string()),
        just(Token::Nearest).to("nearest".to_string()),
        just(Token::Within).to("within".to_string()),
        just(Token::Other).to("other".to_string()),
        just(Token::Pairs).to("pairs".to_string()),
        just(Token::Filter).to("filter".to_string()),
        just(Token::Event).to("event".to_string()),
        just(Token::Observe).to("observe".to_string()),
        just(Token::Apply).to("apply".to_string()),
        just(Token::When).to("when".to_string()),
        just(Token::Emit).to("emit".to_string()),
    ));

    let kw_group3 = choice((
        just(Token::Assert).to("assert".to_string()),
        just(Token::Resolve).to("resolve".to_string()),
        just(Token::Measure).to("measure".to_string()),
        just(Token::Collect).to("collect".to_string()),
        just(Token::Transition).to("transition".to_string()),
        just(Token::Gated).to("gated".to_string()),
        just(Token::Dt).to("dt".to_string()),
        just(Token::To).to("to".to_string()),
        just(Token::Warn).to("warn".to_string()),
        just(Token::Error).to("error".to_string()),
        just(Token::Fatal).to("fatal".to_string()),
        just(Token::Pi).to("PI".to_string()),
        just(Token::Tau).to("TAU".to_string()),
        just(Token::Phi).to("PHI".to_string()),
        just(Token::E).to("E".to_string()),
        just(Token::I).to("I".to_string()),
        just(Token::Scalar).to("Scalar".to_string()),
        just(Token::Vec2).to("Vec2".to_string()),
        just(Token::Vec3).to("Vec3".to_string()),
        just(Token::Vec4).to("Vec4".to_string()),
        just(Token::Vector).to("Vector".to_string()),
        just(Token::Tensor).to("Tensor".to_string()),
        just(Token::Grid).to("Grid".to_string()),
        just(Token::Seq).to("Seq".to_string()),
        just(Token::Uses).to("uses".to_string()),
    ));

    choice((base_ident, kw_group1, kw_group2, kw_group3))
}

/// Parses a dot-separated path of identifiers.
pub fn path<'src>()
-> impl Parser<'src, ParserInput<'src>, Path, extra::Err<ParseError<'src>>> + Clone {
    ident()
        .separated_by(just(Token::Dot))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(Path::new)
}

/// Spanned path
pub fn spanned_path<'src>()
-> impl Parser<'src, ParserInput<'src>, Spanned<Path>, extra::Err<ParseError<'src>>> + Clone {
    spanned(path())
}

/// String literal
pub fn string_lit<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    select! {
        Token::String(s) => s,
    }
}

/// Float number
pub fn float<'src>()
-> impl Parser<'src, ParserInput<'src>, f64, extra::Err<ParseError<'src>>> + Clone {
    select! {
        Token::Integer(i) => i as f64,
        Token::Float(f) => f,
    }
}

/// Number literal (with optional sign)
pub fn number<'src>()
-> impl Parser<'src, ParserInput<'src>, Literal, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Minus)
        .or_not()
        .then(float())
        .map(|(minus, val)| {
            if minus.is_some() {
                Literal::Float(-val)
            } else {
                Literal::Float(val)
            }
        })
}

/// Literal value
pub fn literal<'src>()
-> impl Parser<'src, ParserInput<'src>, Literal, extra::Err<ParseError<'src>>> + Clone {
    choice((
        number(),
        string_lit().map(Literal::String),
        select! { Token::Bool(b) => Literal::Bool(b) },
    ))
}

/// Unit in angle brackets: `<K>`, `<W/m²>`
pub fn unit<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    just(Token::LAngle)
        .ignore_then(unit_content())
        .then_ignore(just(Token::RAngle))
}

/// Internal helper to match unit content tokens
fn unit_content<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    // Reconstruct unit from tokens
    choice((
        ident(),
        just(Token::Slash).to("/".to_string()),
        just(Token::Star).to("*".to_string()),
        select! {
            Token::Integer(i) => i.to_string(),
            Token::UnitPart(s) => s,
        },
    ))
    .repeated()
    .at_least(1)
    .collect::<Vec<_>>()
    .map(|parts| parts.join(""))
}

/// Unit string content (without angle brackets): K, W/m², kg/m³
pub fn unit_string<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    unit_content()
}

/// Optional spanned unit
pub fn optional_unit<'src>()
-> impl Parser<'src, ParserInput<'src>, Option<Spanned<String>>, extra::Err<ParseError<'src>>> + Clone
{
    spanned(unit()).or_not()
}

// === Common attribute parsers (DRY helpers) ===

/// Parse `: keyword(string_lit)` pattern used for title/symbol attributes
pub fn attr_string<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, Spanned<String>, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Colon)
        .ignore_then(just(token))
        .ignore_then(spanned(string_lit()).delimited_by(just(Token::LParen), just(Token::RParen)))
}

/// Parse `: keyword(path)` pattern used for strata attributes
pub fn attr_path<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, Spanned<Path>, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Colon)
        .ignore_then(just(token))
        .ignore_then(spanned_path().delimited_by(just(Token::LParen), just(Token::RParen)))
}

/// Parse `: keyword` pattern (no value) used for flag attributes
pub fn attr_flag<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, (), extra::Err<ParseError<'src>>> + Clone {
    just(Token::Colon).ignore_then(just(token)).ignored()
}

/// Parse `: keyword(int)` pattern returning a spanned integer
pub fn attr_int<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, Spanned<u32>, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Colon).ignore_then(just(token)).ignore_then(
        spanned(select! {
            Token::Integer(i) => i as u32,
        })
        .delimited_by(just(Token::LParen), just(Token::RParen)),
    )
}

pub fn doc_comment<'src>()
-> impl Parser<'src, ParserInput<'src>, Option<String>, extra::Err<ParseError<'src>>> + Clone {
    select! {
        Token::DocComment(s) => s,
    }
    .repeated()
    .collect::<Vec<_>>()
    .map(|lines| {
        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    })
}

pub fn module_doc<'src>()
-> impl Parser<'src, ParserInput<'src>, Option<String>, extra::Err<ParseError<'src>>> + Clone {
    select! {
        Token::ModuleDoc(s) => s,
    }
    .repeated()
    .collect::<Vec<_>>()
    .map(|lines| {
        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    })
}
