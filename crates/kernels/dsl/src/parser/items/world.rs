use chumsky::prelude::*;

use crate::ast::{PolicyBlock, Spanned, WorldDef};

use super::super::lexer::Token;
use super::super::primitives::{attr_string, spanned_path};
use super::super::{ParseError, ParserInput};
use super::config::config_entry;

pub fn world_def<'src>()
-> impl Parser<'src, ParserInput<'src>, WorldDef, extra::Err<ParseError<'src>>> {
    just(Token::World)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            world_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
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

fn world_content<'src>()
-> impl Parser<'src, ParserInput<'src>, WorldContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_string(Token::Title).map(WorldContent::Title),
        attr_string(Token::Version).map(WorldContent::Version),
        just(Token::Policy)
            .ignore_then(
                config_entry()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(|entries| WorldContent::Policy(PolicyBlock { entries })),
    ))
}
