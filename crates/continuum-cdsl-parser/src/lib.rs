//! Hand-written recursive descent parser for Continuum DSL
//!
//! This crate provides fast, hand-written parsing with no generic explosion.
//! Compiles in ~5s vs ~45s for chumsky version.

pub mod parser;

pub use parser::{ParseError, parse_declarations, parse_expr};

// Re-export lexer
pub use continuum_cdsl_lexer::Token;
