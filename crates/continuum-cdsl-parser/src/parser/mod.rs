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
//! ```rust,ignore
//! pub fn parse_expr(tokens: &[(Token, Range<usize>)], file_id: u16) -> Result<Expr, Vec<ParseError>>
//! pub fn parse_declarations(tokens: &[(Token, Range<usize>)], file_id: u16) -> Result<Vec<Declaration>, Vec<ParseError>>
//! ```

mod error;
mod stream;

pub use error::ParseError;
use stream::TokenStream;

mod blocks;
mod decl;
mod expr;
mod helpers;

/// Token utility functions for canonical keyword-to-string mappings.
///
/// Provides the single source of truth for converting keyword tokens to their
/// string representations. Used throughout the parser when keywords can appear
/// as identifiers or when dispatching on keyword types.
pub mod token_utils;
mod types;

// Public API — all functions require real byte spans from the lexer.
use continuum_cdsl_ast::{Declaration, Expr};
use continuum_cdsl_lexer::Token;
use std::ops::Range;

/// Parse a sequence of tokens with byte spans into declarations.
///
/// # Parameters
/// - `tokens`: Slice of (token, byte_span) pairs from [`logos::Lexer::spanned()`]
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Vec<Declaration>)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails
pub fn parse_declarations(
    tokens: &[(Token, Range<usize>)],
    file_id: u16,
) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    decl::parse_declarations(&mut stream)
}

/// Parse a sequence of tokens with byte spans into an expression.
///
/// # Parameters
/// - `tokens`: Slice of (token, byte_span) pairs from [`logos::Lexer::spanned()`]
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Expr)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails
pub fn parse_expr(tokens: &[(Token, Range<usize>)], file_id: u16) -> Result<Expr, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    expr::parse_expr(&mut stream).map_err(|e| vec![e])
}
