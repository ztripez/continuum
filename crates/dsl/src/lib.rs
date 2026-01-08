//! Continuum DSL
//!
//! Compiler for the Continuum Domain-Specific Language.
//! Parses .cdsl files into typed IR for DAG construction.

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::{lex, LexError, Token};
pub use parser::parse;
