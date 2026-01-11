use chumsky::prelude::*;

use crate::ast::{PolicyBlock, Spanned, WorldDef};

use super::super::ParseError;
use super::super::primitives::{attr_string, spanned_path, ws};
use super::config::config_entry;

pub fn world_def<'src>() -> impl Parser<'src, &'src str, WorldDef, extra::Err<ParseError<'src>>> {
    text::keyword("world")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            world_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = WorldDef {
                path,
                title: None,
                version: None,
                policy: None,
            };
            for content in contents {
                match content {
                    WorldContent::Title(t) => def.title = Some(t),
                    WorldContent::Version(v) => def.version = Some(v),
                    WorldContent::Policy(p) => def.policy = Some(p),
                }
            }
            def
        })
}

#[derive(Clone)]
enum WorldContent {
    Title(Spanned<String>),
    Version(Spanned<String>),
    Policy(PolicyBlock),
}

fn world_content<'src>() -> impl Parser<'src, &'src str, WorldContent, extra::Err<ParseError<'src>>>
{
    choice((
        attr_string("title").map(WorldContent::Title),
        attr_string("version").map(WorldContent::Version),
        text::keyword("policy")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|entries| WorldContent::Policy(PolicyBlock { entries })),
    ))
}
