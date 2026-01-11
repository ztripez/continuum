//! Parser for Continuum DSL.

mod expr;
mod items;
pub mod lexer;
mod primitives;

use chumsky::input::{MapExtra, Stream};
use chumsky::prelude::*;
use logos::Logos;

use crate::ast::CompilationUnit;
use crate::ast::Spanned;
pub use lexer::Token;
use primitives::module_doc;

/// A rich parse error with span and context information.
pub type ParseError<'src> = Rich<'src, Token>;

/// Input type for the parser.
pub type ParserInput<'src> = Stream<
    std::iter::Map<
        logos::SpannedIter<'src, Token>,
        fn(
            (
                Result<Token, <Token as logos::Logos<'src>>::Error>,
                std::ops::Range<usize>,
            ),
        ) -> Token,
    >,
>;

/// Parses DSL source code into a compilation unit.
pub fn parse(source: &str) -> (Option<CompilationUnit>, Vec<ParseError<'_>>) {
    fn lex_map(
        tok_span: (
            Result<Token, <Token as logos::Logos>::Error>,
            std::ops::Range<usize>,
        ),
    ) -> Token {
        tok_span.0.unwrap_or(Token::Error)
    }

    let lexer = Token::lexer(source).spanned().map(lex_map as fn(_) -> _);

    let stream = Stream::from_iter(lexer);

    compilation_unit().parse(stream).into_output_errors()
}

fn compilation_unit<'src>()
-> impl Parser<'src, ParserInput<'src>, CompilationUnit, extra::Err<ParseError<'src>>> {
    // Parse optional module doc comment (//!) at the start of the file
    module_doc()
        .then(
            items::item()
                .map_with(
                    |item, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                        Spanned::new(item, extra.span().into())
                    },
                )
                .repeated()
                .collect(),
        )
        .map(|(module_doc, items)| CompilationUnit { module_doc, items })
}

#[cfg(test)]
mod tests;
