//! Primitive parser combinators for the Continuum DSL.

use chumsky::input;
use chumsky::prelude::*;
use chumsky::span::SimpleSpan;

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
    parser.map_with(|value, extra| Spanned::new(value, extra.span().into()))
}

/// Helper to match a token, ignoring its span in the input
pub fn tok<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, Token, extra::Err<ParseError<'src>>> + Clone {
    just(token)
}

/// Parses an identifier or a keyword that can act as an identifier.
pub fn ident<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    let base_ident = select! { Token::Ident(name) => name };

    let kw_group1 = choice((
        tok(Token::World).to("world".to_string()),
        tok(Token::Strata).to("strata".to_string()),
        tok(Token::Era).to("era".to_string()),
        tok(Token::Signal).to("signal".to_string()),
        tok(Token::Field).to("field".to_string()),
        tok(Token::Operator).to("operator".to_string()),
        tok(Token::Fn).to("fn".to_string()),
        tok(Token::Type).to("type".to_string()),
        tok(Token::Impulse).to("impulse".to_string()),
        tok(Token::Fracture).to("fracture".to_string()),
        tok(Token::Chronicle).to("chronicle".to_string()),
        tok(Token::Entity).to("entity".to_string()),
        tok(Token::Count).to("count".to_string()),
        tok(Token::Member).to("member".to_string()),
        tok(Token::Const).to("const".to_string()),
        tok(Token::Config).to("config".to_string()),
        tok(Token::Policy).to("policy".to_string()),
        tok(Token::Version).to("version".to_string()),
        tok(Token::Initial).to("initial".to_string()),
        tok(Token::Terminal).to("terminal".to_string()),
        tok(Token::Stride).to("stride".to_string()),
        tok(Token::Title).to("title".to_string()),
        tok(Token::Symbol).to("symbol".to_string()),
        tok(Token::Active).to("active".to_string()),
        tok(Token::Converge).to("converge".to_string()),
        tok(Token::Warmup).to("warmup".to_string()),
    ));

    let kw_group2 = choice((
        tok(Token::Iterate).to("iterate".to_string()),
        tok(Token::Phase).to("phase".to_string()),
        tok(Token::Magnitude).to("magnitude".to_string()),
        tok(Token::Symmetric).to("symmetric".to_string()),
        tok(Token::PositiveDefinite).to("positive_definite".to_string()),
        tok(Token::Topology).to("topology".to_string()),
        tok(Token::Min).to("min".to_string()),
        tok(Token::Max).to("max".to_string()),
        tok(Token::Mean).to("mean".to_string()),
        tok(Token::Sum).to("sum".to_string()),
        tok(Token::Product).to("product".to_string()),
        tok(Token::Any).to("any".to_string()),
        tok(Token::All).to("all".to_string()),
        tok(Token::None).to("none".to_string()),
        tok(Token::First).to("first".to_string()),
        tok(Token::Nearest).to("nearest".to_string()),
        tok(Token::Within).to("within".to_string()),
        tok(Token::Other).to("other".to_string()),
        tok(Token::Pairs).to("pairs".to_string()),
        tok(Token::Filter).to("filter".to_string()),
        tok(Token::Event).to("event".to_string()),
        tok(Token::Observe).to("observe".to_string()),
        tok(Token::Apply).to("apply".to_string()),
        tok(Token::When).to("when".to_string()),
        tok(Token::Emit).to("emit".to_string()),
    ));

    let kw_group3 = choice((
        tok(Token::Assert).to("assert".to_string()),
        tok(Token::Resolve).to("resolve".to_string()),
        tok(Token::Measure).to("measure".to_string()),
        tok(Token::Collect).to("collect".to_string()),
        tok(Token::Transition).to("transition".to_string()),
        tok(Token::Gated).to("gated".to_string()),
        tok(Token::Dt).to("dt".to_string()),
        tok(Token::To).to("to".to_string()),
        tok(Token::Warn).to("warn".to_string()),
        tok(Token::Error).to("error".to_string()),
        tok(Token::Fatal).to("fatal".to_string()),
        tok(Token::Pi).to("PI".to_string()),
        tok(Token::Tau).to("TAU".to_string()),
        tok(Token::Phi).to("PHI".to_string()),
        tok(Token::E).to("E".to_string()),
        tok(Token::I).to("I".to_string()),
        tok(Token::Scalar).to("Scalar".to_string()),
        tok(Token::Vec2).to("Vec2".to_string()),
        tok(Token::Vec3).to("Vec3".to_string()),
        tok(Token::Vec4).to("Vec4".to_string()),
        tok(Token::Vector).to("Vector".to_string()),
        tok(Token::Tensor).to("Tensor".to_string()),
        tok(Token::Grid).to("Grid".to_string()),
        tok(Token::Seq).to("Seq".to_string()),
        tok(Token::Uses).to("uses".to_string()),
    ));

    choice((base_ident, kw_group1, kw_group2, kw_group3))
}

/// Parses a dot-separated path of identifiers.
pub fn path<'src>()
-> impl Parser<'src, ParserInput<'src>, Path, extra::Err<ParseError<'src>>> + Clone {
    ident()
        .separated_by(tok(Token::Dot))
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
    tok(Token::Minus)
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
    tok(Token::LAngle)
        .ignore_then(unit_content())
        .then_ignore(tok(Token::RAngle))
}

/// Internal helper to match unit content tokens
fn unit_content<'src>()
-> impl Parser<'src, ParserInput<'src>, String, extra::Err<ParseError<'src>>> + Clone {
    // Reconstruct unit from tokens
    choice((
        ident(),
        tok(Token::Slash).to("/".to_string()),
        tok(Token::Star).to("*".to_string()),
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
    tok(Token::Colon)
        .ignore_then(tok(token))
        .ignore_then(spanned(string_lit()).delimited_by(tok(Token::LParen), tok(Token::RParen)))
}

/// Parse `: keyword(path)` pattern used for strata attributes
pub fn attr_path<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, Spanned<Path>, extra::Err<ParseError<'src>>> + Clone {
    tok(Token::Colon)
        .ignore_then(tok(token))
        .ignore_then(spanned_path().delimited_by(tok(Token::LParen), tok(Token::RParen)))
}

/// Parse `: keyword` pattern (no value) used for flag attributes
pub fn attr_flag<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, (), extra::Err<ParseError<'src>>> + Clone {
    tok(Token::Colon).ignore_then(tok(token)).ignored()
}

/// Parse `: keyword(int)` pattern returning a spanned integer
pub fn attr_int<'src>(
    token: Token,
) -> impl Parser<'src, ParserInput<'src>, Spanned<u32>, extra::Err<ParseError<'src>>> + Clone {
    tok(Token::Colon).ignore_then(tok(token)).ignore_then(
        spanned(select! {
            Token::Integer(i) => i as u32,
        })
        .delimited_by(tok(Token::LParen), tok(Token::RParen)),
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
