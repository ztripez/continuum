//! Common shared parsers used across multiple item types.
//!
//! This module provides utility parsers that are shared between signals,
//! operators, fields, and entities.

use chumsky::prelude::*;

use crate::ast::{AssertBlock, AssertSeverity, Assertion, Topology};

use super::super::expr::spanned_expr;
use super::super::primitives::{spanned, string_lit, ws};
use super::super::ParseError;

// === Assertions ===

pub fn assert_block<'src>(
) -> impl Parser<'src, &'src str, AssertBlock, extra::Err<ParseError<'src>>> + Clone {
    text::keyword("assert")
        .padded_by(ws())
        .ignore_then(
            assertion()
                .padded_by(ws())
                .repeated()
                .at_least(1)
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|assertions| AssertBlock { assertions })
}

fn assertion<'src>() -> impl Parser<'src, &'src str, Assertion, extra::Err<ParseError<'src>>> + Clone
{
    // Parse: condition [: severity] [, "message"]
    spanned_expr()
        .then(
            just(':')
                .padded_by(ws())
                .ignore_then(assert_severity())
                .or_not(),
        )
        .then(
            just(',')
                .padded_by(ws())
                .ignore_then(spanned(string_lit()))
                .or_not(),
        )
        .map(|((condition, severity), message)| Assertion {
            condition,
            severity: severity.unwrap_or_default(),
            message,
        })
}

fn assert_severity<'src>(
) -> impl Parser<'src, &'src str, AssertSeverity, extra::Err<ParseError<'src>>> + Clone {
    choice((
        text::keyword("warn").to(AssertSeverity::Warn),
        text::keyword("error").to(AssertSeverity::Error),
        text::keyword("fatal").to(AssertSeverity::Fatal),
    ))
}

// === Topology ===

pub fn topology<'src>(
) -> impl Parser<'src, &'src str, Topology, extra::Err<ParseError<'src>>> + Clone {
    choice((
        text::keyword("sphere_surface").to(Topology::SphereSurface),
        text::keyword("point_cloud").to(Topology::PointCloud),
        text::keyword("volume").to(Topology::Volume),
    ))
}
