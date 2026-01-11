//! Configuration block parsers.
//!
//! This module handles `const { ... }` and `config { ... }` blocks
//! for compile-time constants and runtime parameters.

use chumsky::prelude::*;

use crate::ast::{ConfigBlock, ConfigEntry, ConstBlock, ConstEntry};

use super::super::primitives::{doc_comment, literal, optional_unit, spanned, spanned_path, ws};
use super::super::ParseError;

// === Const Block ===

pub fn const_block<'src>(
) -> impl Parser<'src, &'src str, ConstBlock, extra::Err<ParseError<'src>>> {
    text::keyword("const")
        .padded_by(ws())
        .ignore_then(
            const_entry()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|entries| ConstBlock { entries })
}

pub fn const_entry<'src>(
) -> impl Parser<'src, &'src str, ConstEntry, extra::Err<ParseError<'src>>> + Clone {
    doc_comment()
        .then_ignore(ws()) // Consume whitespace after doc comment (or leading ws if no doc)
        .then(spanned_path())
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned(literal()))
        .then(optional_unit().padded_by(ws()))
        .map(|(((doc, path), value), unit)| ConstEntry {
            doc,
            path,
            value,
            unit,
        })
}

// === Config Block ===

pub fn config_block<'src>(
) -> impl Parser<'src, &'src str, ConfigBlock, extra::Err<ParseError<'src>>> {
    text::keyword("config")
        .padded_by(ws())
        .ignore_then(
            config_entry()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|entries| ConfigBlock { entries })
}

pub fn config_entry<'src>(
) -> impl Parser<'src, &'src str, ConfigEntry, extra::Err<ParseError<'src>>> + Clone {
    doc_comment()
        .then_ignore(ws()) // Consume whitespace after doc comment (or leading ws if no doc)
        .then(spanned_path())
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned(literal()))
        .then(optional_unit().padded_by(ws()))
        .map(|(((doc, path), value), unit)| ConfigEntry {
            doc,
            path,
            value,
            unit,
        })
}
