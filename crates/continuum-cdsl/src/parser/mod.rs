//! Hand-written recursive descent parser for Continuum DSL.
//!
//! This module implements a recursive descent parser that replaces the
//! chumsky combinator-based parser to eliminate compile-time monomorphization
//! explosion.
//!
//! ## Architecture
//!
//! - `stream`: TokenStream wrapper with lookahead
//! - `error`: ParseError and recovery mechanisms
//! - `expr`: Expression parser using Pratt parsing
//! - `decl`: Declaration parsers (keyword-dispatched)
//! - `types`: Type and unit expression parsers
//! - `blocks`: Block body, warmup, when, observe parsers
//!
//! ## Public API
//!
//! The public API matches the chumsky-based parser exactly:
//!
//! ```rust,ignore
//! pub fn parse_expr(tokens: &[Token], file_id: u16) -> Result<Expr, Vec<ParseError>>
//! pub fn parse_declarations(tokens: &[Token], file_id: u16) -> Result<Vec<Declaration>, Vec<ParseError>>
//! ```

mod error;
mod stream;

pub use error::ParseError;
use stream::TokenStream;

use crate::ast::{Declaration, Expr};
use crate::lexer::Token;

/// Parse an expression from a token stream.
///
/// # Parameters
/// - `tokens`: Token stream to parse.
/// - `file_id`: SourceMap file identifier for spans.
///
/// # Returns
/// Parsed [`Expr`] with spans referencing `file_id`.
///
/// # Errors
/// Returns parse errors if the tokens are not a valid expression.
pub fn parse_expr(tokens: &[Token], file_id: u16) -> Result<Expr, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    match expr::parse_expr(&mut stream) {
        Ok(expr) => {
            if !stream.at_end() {
                Err(vec![ParseError::unexpected_token(
                    stream.peek(),
                    "end of expression",
                    stream.current_span(),
                )])
            } else {
                Ok(expr)
            }
        }
        Err(e) => Err(vec![e]),
    }
}

/// Parse declarations from a token stream.
///
/// # Parameters
/// - `tokens`: Token stream for a full CDSL file.
/// - `file_id`: SourceMap file identifier for spans.
///
/// # Returns
/// Parsed declarations in source order.
///
/// # Errors
/// Returns parse errors if the token stream is invalid.
pub fn parse_declarations(
    tokens: &[Token],
    file_id: u16,
) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    decl::parse_declarations(&mut stream)
}

mod blocks;
mod decl;
mod expr;
mod types;
