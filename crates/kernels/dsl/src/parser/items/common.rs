//! Common shared parsers used across multiple item types.

use chumsky::prelude::*;

use crate::ast::{AssertBlock, AssertSeverity, Assertion, Topology};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{spanned, string_lit};
use super::super::{ParseError, ParserInput};

// === Assertions ===

pub fn assert_block<'src>()
-> impl Parser<'src, ParserInput<'src>, AssertBlock, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Assert)
        .ignore_then(
            assertion()
                .repeated()
                .at_least(1)
                .collect()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|assertions| AssertBlock { assertions })
}

fn assertion<'src>()
-> impl Parser<'src, ParserInput<'src>, Assertion, extra::Err<ParseError<'src>>> + Clone {
    // Parse: condition [: severity] [, "message"]
    spanned_expr()
        .then(just(Token::Colon).ignore_then(assert_severity()).or_not())
        .then(
            just(Token::Comma)
                .ignore_then(spanned(string_lit()))
                .or_not(),
        )
        .map(
            |((condition, severity), message): ((_, Option<AssertSeverity>), _)| Assertion {
                condition,
                severity: severity.unwrap_or_default(),
                message,
            },
        )
}

fn assert_severity<'src>()
-> impl Parser<'src, ParserInput<'src>, AssertSeverity, extra::Err<ParseError<'src>>> + Clone {
    choice((
        just(Token::Warn).to(AssertSeverity::Warn),
        just(Token::Error).to(AssertSeverity::Error),
        just(Token::Fatal).to(AssertSeverity::Fatal),
    ))
}

// === Topology ===

pub fn topology<'src>()
-> impl Parser<'src, ParserInput<'src>, Topology, extra::Err<ParseError<'src>>> + Clone {
    choice((
        just(Token::SphereSurface).to(Topology::SphereSurface),
        just(Token::PointCloud).to(Topology::PointCloud),
        just(Token::Volume).to(Topology::Volume),
    ))
}
