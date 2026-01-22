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

mod blocks;
mod decl;
mod expr;

/// Token utility functions for canonical keyword-to-string mappings.
///
/// Provides the single source of truth for converting keyword tokens to their
/// string representations. Used throughout the parser when keywords can appear
/// as identifiers or when dispatching on keyword types.
pub mod token_utils;
mod types;

// Public API wrappers matching old chumsky-based interface
use continuum_cdsl_ast::{Declaration, Expr};
use continuum_cdsl_lexer::Token;
use std::ops::Range;

/// Create fake byte spans for backward compatibility with old API.
///
/// Converts a slice of tokens to (token, span) pairs using enumeration indices
/// as fake byte offsets. This maintains API compatibility while the new span-aware
/// API (`parse_declarations_with_spans`, `parse_expr_with_spans`) accepts real spans.
///
/// # Parameters
///
/// * `tokens` - Slice of tokens without span information
///
/// # Returns
///
/// Vector of (token, fake_span) pairs where each span is `i..i+1`
///
/// # Note
///
/// Fake spans use token index as byte offset, which is incorrect but preserves
/// backward compatibility. New code should use span-aware APIs with real byte offsets.
fn tokens_with_fake_spans(tokens: &[Token]) -> Vec<(Token, Range<usize>)> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, tok)| (tok.clone(), i..i + 1))
        .collect()
}

/// Parse a sequence of tokens into declarations.
///
/// # Parameters
/// - `tokens`: Slice of tokens to parse
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Vec<Declaration>)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails
///
/// # Note
/// This function creates fake byte spans for backward compatibility.
/// Use `parse_declarations_with_spans` for accurate error locations.
pub fn parse_declarations(
    tokens: &[Token],
    file_id: u16,
) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let tokens_with_spans = tokens_with_fake_spans(tokens);
    let mut stream = TokenStream::new(&tokens_with_spans, file_id);
    decl::parse_declarations(&mut stream)
}

/// Parse a sequence of tokens with byte spans into declarations.
///
/// # Parameters
/// - `tokens`: Slice of (token, byte_span) pairs
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Vec<Declaration>)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails
pub fn parse_declarations_with_spans(
    tokens: &[(Token, Range<usize>)],
    file_id: u16,
) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    decl::parse_declarations(&mut stream)
}

/// Parse a sequence of tokens into an expression.
///
/// # Parameters
/// - `tokens`: Slice of tokens to parse
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Expr)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails (single error wrapped in Vec for API compatibility)
///
/// # Note
/// This function creates fake byte spans for backward compatibility.
/// Use `parse_expr_with_spans` for accurate error locations.
pub fn parse_expr(tokens: &[Token], file_id: u16) -> Result<Expr, Vec<ParseError>> {
    let tokens_with_spans = tokens_with_fake_spans(tokens);
    let mut stream = TokenStream::new(&tokens_with_spans, file_id);
    expr::parse_expr(&mut stream).map_err(|e| vec![e])
}

/// Parse a sequence of tokens with byte spans into an expression.
///
/// # Parameters
/// - `tokens`: Slice of (token, byte_span) pairs
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Expr)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails
pub fn parse_expr_with_spans(
    tokens: &[(Token, Range<usize>)],
    file_id: u16,
) -> Result<Expr, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    expr::parse_expr(&mut stream).map_err(|e| vec![e])
}
