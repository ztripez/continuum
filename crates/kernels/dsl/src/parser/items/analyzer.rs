//! Parser for analyzer definitions.
//!
//! Analyzers are post-hoc analysis queries on field snapshots.

use chumsky::prelude::*;

use crate::ast::{AnalyzerDef, Spanned};
use crate::parser::lexer::Token;
use crate::parser::primitives::path;
use crate::parser::{ParseError, ParserInput};

/// Parse an analyzer definition.
///
/// # Grammar
///
/// ```text
/// analyzer terra.name {
///     : doc "description"
///     : requires(fields: [field1, field2])
///     : compute { <expr> }
///     : validate { check <expr> ... }
/// }
/// ```
pub fn analyzer_def<'src>()
-> impl Parser<'src, ParserInput<'src>, AnalyzerDef, extra::Err<ParseError<'src>>> {
    just(Token::Analyzer)
        .ignore_then(path().map_with(|p, e| Spanned::new(p, e.span().into())))
        .then_ignore(just(Token::LBrace))
        .then(
            // For now, collect any attributes
            any().repeated().collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map(|(path, _attrs)| AnalyzerDef {
            doc: None,
            path,
            required_fields: Vec::new(),
            compute: None,
            validations: Vec::new(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be added after parser integration
}
