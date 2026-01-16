//! Entity definition parsers.

use chumsky::prelude::*;

use crate::ast::{CountBounds, EntityDef, Path, Spanned};

use super::super::lexer::Token;
use super::super::primitives::spanned_path;
use super::super::{ParseError, ParserInput};

// === Entity ===

pub fn entity_def<'src>()
-> impl Parser<'src, ParserInput<'src>, EntityDef, extra::Err<ParseError<'src>>> {
    just(Token::Entity)
        .ignore_then(spanned_path())
        .then(
            entity_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
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

fn entity_content<'src>()
-> impl Parser<'src, ParserInput<'src>, EntityContent, extra::Err<ParseError<'src>>> {
    choice((
        // : count(config.path) - count from config
        just(Token::Colon)
            .ignore_then(just(Token::Count))
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(just(Token::Config))
                    .ignore_then(just(Token::Dot))
                    .ignore_then(spanned_path())
                    .then_ignore(just(Token::RParen)),
            )
            .map(EntityContent::CountSource),
        // : count(min..max) - count bounds
        just(Token::Colon)
            .ignore_then(just(Token::Count))
            .ignore_then(count_bounds().delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(EntityContent::CountBounds),
    ))
}

fn count_bounds<'src>()
-> impl Parser<'src, ParserInput<'src>, CountBounds, extra::Err<ParseError<'src>>> + Clone {
    select! { Token::Integer(i) => i as u32 }
        .then_ignore(just(Token::DotDot))
        .then(select! { Token::Integer(i) => i as u32 })
        .map(|(min, max)| CountBounds { min, max })
}
