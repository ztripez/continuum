//! # Continuum DSL Compiler
//!
//! Complete compiler pipeline for Continuum DSL (CDSL).
//!
//! This crate is a facade that re-exports functionality from:
//! - `continuum-cdsl-ast` - AST types and foundation
//! - `continuum-cdsl-lexer` - Tokenization
//! - `continuum-cdsl-parser` - Parsing to AST
//! - `continuum-cdsl-resolve` - Name resolution, type checking, validation
//!
//! ## Architecture
//!
//! The compiler is split into modular sub-crates for faster incremental builds:
//!
//! ```text
//! continuum-cdsl-ast      (~10k LOC) - AST + foundation types
//!     ↓
//! continuum-cdsl-lexer    (~1k LOC)  - Tokenization
//!     ↓
//! continuum-cdsl-parser   (~2k LOC)  - Hand-written recursive descent parser
//!     ↓
//! continuum-cdsl-resolve  (~13k LOC) - Resolution + validation
//!     ↓
//! continuum-cdsl (facade) - Re-exports + compile API
//! ```
//!
//! This split provides:
//! - **70-85% faster** incremental builds
//! - **Parallel compilation** of independent modules
//! - **No chumsky dependency** = no generic explosion
//!
//! ## Usage
//!
//! ```rust,ignore
//! use continuum_cdsl::compile;
//! use std::collections::HashMap;
//! use std::path::PathBuf;
//!
//! let mut sources = HashMap::new();
//! sources.insert(PathBuf::from("world.cdsl"), source_code);
//!
//! let result = compile(&sources);
//! ```

// Force-link: populates KERNEL_SIGNATURES distributed slice
#[allow(unused_imports)]
use continuum_functions::tensor_ops as _;

// Re-export AST and foundation types
pub use continuum_cdsl_ast::{self as ast, *};

// Re-export lexer
pub use continuum_cdsl_lexer as lexer;
pub use continuum_cdsl_lexer::Token;

// Re-export parser
pub use continuum_cdsl_parser as parser;
pub use continuum_cdsl_parser::{parse_declarations, parse_expr, ParseError};

// Re-export resolve
pub use continuum_cdsl_resolve as resolve;
pub use continuum_cdsl_resolve::*;

// Keep compile module (high-level API)
pub mod compile;

pub use compile::{
    compile, compile_with_sources, deserialize_world, format_errors, serialize_world,
    CompileResultWithSources,
};

// Version info
/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
