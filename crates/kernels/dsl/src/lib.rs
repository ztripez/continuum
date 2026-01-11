//! Continuum DSL
//!
//! Compiler for the Continuum Domain-Specific Language.
//! Parses .cdsl files into typed IR for DAG construction.

pub mod ast;
mod loader;
mod parser;
pub mod validate;

pub use ast::*;
pub use loader::{LoadError, LoadResult, collect_cdsl_files, load_file, load_world};
pub use parser::parse;
pub use validate::{ValidationError, validate};
