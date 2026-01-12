//! Parser for Continuum DSL.

mod expr;
mod items;
pub mod lexer;
mod primitives;

use chumsky::input::{MapExtra, MappedInput, Stream};
use chumsky::prelude::*;
use chumsky::span::SimpleSpan;
use logos::Logos;

use crate::ast::{CompilationUnit, Spanned};
pub use lexer::Token;
use primitives::module_doc;

/// A rich parse error with span and context information.
pub type ParseError<'src> = Rich<'src, Token>;

/// Iterator that yields tokens from `logos` with raw spans.
type LexerIter<'src> = std::iter::Map<
    logos::SpannedIter<'src, Token>,
    fn(
        (
            Result<Token, <Token as Logos<'src>>::Error>,
            std::ops::Range<usize>,
        ),
    ) -> (Token, std::ops::Range<usize>),
>;

/// Mapper that attaches `SimpleSpan` to parsed tokens.
type TokenSpanMapper<'src> = fn((Token, std::ops::Range<usize>)) -> (Token, SimpleSpan);

/// Input type for the parser, yielding tokens with `SimpleSpan` metadata.
pub type ParserInput<'src> =
    MappedInput<Token, SimpleSpan, Stream<LexerIter<'src>>, TokenSpanMapper<'src>>;

fn lex_map<'src>(
    tok_span: (
        Result<Token, <Token as Logos<'src>>::Error>,
        std::ops::Range<usize>,
    ),
) -> (Token, std::ops::Range<usize>) {
    let token = tok_span.0.unwrap_or(Token::Error);
    (token, tok_span.1)
}

fn attach_span(token_span: (Token, std::ops::Range<usize>)) -> (Token, SimpleSpan) {
    let (token, range) = token_span;
    (token, SimpleSpan::from(range))
}

/// Parses DSL source code into a compilation unit.
pub fn parse(source: &str) -> (Option<CompilationUnit>, Vec<ParseError<'_>>) {
    let lexer = Token::lexer(source).spanned().map(lex_map as fn(_) -> _);

    let stream = Stream::from_iter(lexer).map(
        SimpleSpan::from(source.len()..source.len()),
        attach_span as TokenSpanMapper,
    );

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
