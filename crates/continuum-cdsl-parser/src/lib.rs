//! Hand-written recursive descent parser for Continuum DSL
//!
//! This crate provides fast, hand-written parsing with no generic explosion.
//! Compiles in ~5s vs ~45s for chumsky version.

pub mod parser;

pub use parser::{
    parse_declarations, parse_declarations_with_spans, parse_expr, parse_expr_with_spans,
    ParseError,
};

// Re-export lexer
pub use continuum_cdsl_lexer::Token;
