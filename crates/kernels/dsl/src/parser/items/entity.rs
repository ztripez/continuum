//! Entity definition parsers.
//!
//! This module handles `entity.name { ... }` index space declarations.
//! Entities define index spaces for per-entity state (member signals).
//!
//! Entities are pure index spaces - they don't have strata. Only member
//! signals have strata because they are the ones being scheduled.

use chumsky::prelude::*;

use crate::ast::{CountBounds, EntityDef, Path, Spanned};

use super::super::primitives::{spanned_path, ws};
use super::super::ParseError;

// === Entity ===

/// Parses an entity definition.
///
/// # DSL Syntax
///
/// ```cdsl
/// entity.human.person {
///     : count(config.human.pop_size)
/// }
///
/// entity.stellar.moon {
///     : count(1..100)
/// }
/// ```
///
/// Entities are pure index spaces. Per-entity state is defined via member signals:
///
/// ```cdsl
/// member.human.person.age {
///     : Scalar
///     : strata(human.physiology)
///     resolve { prev + 1 }
/// }
/// ```
pub fn entity_def<'src>() -> impl Parser<'src, &'src str, EntityDef, extra::Err<ParseError<'src>>> {
    text::keyword("entity")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            entity_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = EntityDef {
                doc: None,
                path,
                count_source: None,
                count_bounds: None,
            };
            for content in contents {
                match content {
                    EntityContent::CountSource(c) => def.count_source = Some(c),
                    EntityContent::CountBounds(b) => def.count_bounds = Some(b),
                }
            }
            def
        })
}

#[derive(Clone)]
enum EntityContent {
    CountSource(Spanned<Path>),
    CountBounds(CountBounds),
}

fn entity_content<'src>(
) -> impl Parser<'src, &'src str, EntityContent, extra::Err<ParseError<'src>>> {
    choice((
        // : count(config.path) - count from config
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("count"))
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("config"))
                    .ignore_then(just('.'))
                    .ignore_then(spanned_path())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(EntityContent::CountSource),
        // : count(min..max) - count bounds
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("count"))
            .ignore_then(
                count_bounds()
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EntityContent::CountBounds),
    ))
}

fn count_bounds<'src>(
) -> impl Parser<'src, &'src str, CountBounds, extra::Err<ParseError<'src>>> + Clone {
    text::int(10)
        .map(|s: &str| s.parse::<u32>().unwrap_or(0))
        .then_ignore(just("..").padded_by(ws()))
        .then(text::int(10).map(|s: &str| s.parse::<u32>().unwrap_or(u32::MAX)))
        .map(|(min, max)| CountBounds { min, max })
}
