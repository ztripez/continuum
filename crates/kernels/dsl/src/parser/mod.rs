//! Parser for Continuum DSL
//!
//! Uses Chumsky for direct string parsing with error recovery.

mod expr;
mod items;
mod primitives;

use chumsky::prelude::*;

use crate::ast::{CompilationUnit, Spanned};
use primitives::ws;

pub type ParseError<'src> = Rich<'src, char>;

/// Parse source code into a compilation unit
pub fn parse(source: &str) -> (Option<CompilationUnit>, Vec<ParseError<'_>>) {
    compilation_unit().parse(source).into_output_errors()
}

fn compilation_unit<'src>(
) -> impl Parser<'src, &'src str, CompilationUnit, extra::Err<ParseError<'src>>> {
    ws().ignore_then(
        items::item()
            .map_with(|i, e| Spanned::new(i, e.span().into()))
            .padded_by(ws())
            .repeated()
            .collect()
            .map(|items| CompilationUnit { items }),
    )
}

#[cfg(test)]
mod tests;
