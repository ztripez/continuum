//! Configuration block parsers.

use chumsky::prelude::*;

use crate::ast::{ConfigBlock, ConfigEntry, ConstBlock, ConstEntry};

use super::super::lexer::Token;
use super::super::primitives::{doc_comment, literal, optional_unit, spanned, spanned_path};
use super::super::{ParseError, ParserInput};

// === Const Block ===

pub fn const_block<'src>()
-> impl Parser<'src, ParserInput<'src>, ConstBlock, extra::Err<ParseError<'src>>> {
    just(Token::Const)
        .ignore_then(
            const_entry()
                .repeated()
                .collect()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|entries| ConstBlock { entries })
}

pub fn const_entry<'src>()
-> impl Parser<'src, ParserInput<'src>, ConstEntry, extra::Err<ParseError<'src>>> + Clone {
    doc_comment()
        .then(spanned_path())
        .then_ignore(just(Token::Colon))
        .then(spanned(literal()))
        .then(optional_unit())
        .map(|(((doc, path), value), unit)| ConstEntry {
            doc,
            path,
            value,
            unit,
        })
}

// === Config Block ===

pub fn config_block<'src>()
-> impl Parser<'src, ParserInput<'src>, ConfigBlock, extra::Err<ParseError<'src>>> {
    just(Token::Config)
        .ignore_then(
            config_entry()
                .repeated()
                .collect()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|entries| ConfigBlock { entries })
}

pub fn config_entry<'src>()
-> impl Parser<'src, ParserInput<'src>, ConfigEntry, extra::Err<ParseError<'src>>> + Clone {
    doc_comment()
        .then(spanned_path())
        .then_ignore(just(Token::Colon))
        .then(spanned(literal()))
        .then(optional_unit())
        .map(|(((doc, path), value), unit)| ConfigEntry {
            doc,
            path,
            value,
            unit,
        })
}
