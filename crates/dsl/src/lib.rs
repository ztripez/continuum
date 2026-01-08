//! Continuum DSL
//!
//! Compiler for the Continuum Domain-Specific Language.
//! Parses .cdsl files into typed IR for DAG construction.

pub mod ast;
mod parser;
pub mod validate;

pub use ast::*;
pub use parser::parse;
pub use validate::{validate, ValidationError};
