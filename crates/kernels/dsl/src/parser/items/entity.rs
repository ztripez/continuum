//! Entity definition parsers.
//!
//! This module handles `entity.name { ... }` indexed state collections
//! with schemas, fields, and resolution logic.

use chumsky::prelude::*;

use crate::ast::{
    ConfigEntry, CountBounds, EntityDef, EntityFieldDef, EntitySchemaField, MeasureBlock, Path,
    ResolveBlock, Spanned, Topology, TypeExpr,
};

use super::super::expr::spanned_expr;
use super::super::primitives::{attr_path, ident, spanned, spanned_path, ws};
use super::super::ParseError;
use super::common::{assert_block, topology};
use super::config::config_entry;
use super::types::type_expr;

// === Entity ===

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
                strata: None,
                count_source: None,
                count_bounds: None,
                schema: vec![],
                config_defaults: vec![],
                resolve: None,
                assertions: None,
                fields: vec![],
            };
            for content in contents {
                match content {
                    EntityContent::Strata(s) => def.strata = Some(s),
                    EntityContent::CountSource(c) => def.count_source = Some(c),
                    EntityContent::CountBounds(b) => def.count_bounds = Some(b),
                    EntityContent::Schema(s) => def.schema = s,
                    EntityContent::Config(c) => def.config_defaults = c,
                    EntityContent::Resolve(r) => def.resolve = Some(r),
                    EntityContent::Assert(a) => def.assertions = Some(a),
                    EntityContent::Field(f) => def.fields.push(f),
                }
            }
            def
        })
}

#[derive(Clone)]
enum EntityContent {
    Strata(Spanned<Path>),
    CountSource(Spanned<Path>),
    CountBounds(CountBounds),
    Schema(Vec<EntitySchemaField>),
    Config(Vec<ConfigEntry>),
    Resolve(ResolveBlock),
    Assert(crate::ast::AssertBlock),
    Field(EntityFieldDef),
}

fn entity_content<'src>(
) -> impl Parser<'src, &'src str, EntityContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path)
        attr_path("strata").map(EntityContent::Strata),
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
        // schema { fields }
        text::keyword("schema")
            .padded_by(ws())
            .ignore_then(
                entity_schema_field()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EntityContent::Schema),
        // config { defaults }
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EntityContent::Config),
        // resolve { expr }
        text::keyword("resolve")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| EntityContent::Resolve(ResolveBlock { body })),
        // assert { assertions }
        assert_block().map(EntityContent::Assert),
        // field.name { ... } - nested field definition
        entity_field_def().map(EntityContent::Field),
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

fn entity_schema_field<'src>(
) -> impl Parser<'src, &'src str, EntitySchemaField, extra::Err<ParseError<'src>>> + Clone {
    spanned(ident())
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned(type_expr()))
        .map(|(name, ty)| EntitySchemaField { name, ty })
}

fn entity_field_def<'src>(
) -> impl Parser<'src, &'src str, EntityFieldDef, extra::Err<ParseError<'src>>> + Clone {
    text::keyword("field")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned(ident()))
        .padded_by(ws())
        .then(
            entity_field_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(name, contents)| {
            let mut def = EntityFieldDef {
                name,
                ty: None,
                topology: None,
                measure: None,
            };
            for content in contents {
                match content {
                    EntityFieldContent::Type(t) => def.ty = Some(t),
                    EntityFieldContent::Topology(t) => def.topology = Some(t),
                    EntityFieldContent::Measure(m) => def.measure = Some(m),
                }
            }
            def
        })
}

#[derive(Clone)]
enum EntityFieldContent {
    Type(Spanned<TypeExpr>),
    Topology(Spanned<Topology>),
    Measure(MeasureBlock),
}

fn entity_field_content<'src>(
) -> impl Parser<'src, &'src str, EntityFieldContent, extra::Err<ParseError<'src>>> + Clone {
    choice((
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("topology"))
            .ignore_then(
                spanned(topology())
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EntityFieldContent::Topology),
        just(':')
            .padded_by(ws())
            .ignore_then(spanned(type_expr()))
            .map(EntityFieldContent::Type),
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| EntityFieldContent::Measure(MeasureBlock { body })),
    ))
}
