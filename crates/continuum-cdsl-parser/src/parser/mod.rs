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

/// Parse a sequence of tokens into declarations.
///
/// # Parameters
/// - `tokens`: Slice of tokens to parse
/// - `file_id`: File identifier for span tracking
///
/// # Returns
/// - `Ok(Vec<Declaration>)` if parsing succeeds
/// - `Err(Vec<ParseError>)` if parsing fails
pub fn parse_declarations(
    tokens: &[Token],
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
pub fn parse_expr(tokens: &[Token], file_id: u16) -> Result<Expr, Vec<ParseError>> {
    let mut stream = TokenStream::new(tokens, file_id);
    expr::parse_expr(&mut stream).map_err(|e| vec![e])
}
