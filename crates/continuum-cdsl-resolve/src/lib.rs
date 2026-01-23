// Allow unwrap in tests
#![cfg_attr(test, allow(clippy::unwrap_used))]

//! Resolution and validation for Continuum DSL
//!
//! This crate performs name resolution, type resolution, and validation
//! of parsed AST.

pub mod desugar;
pub mod error;
pub mod resolve;

pub use error::CompileError;
pub use resolve::*;
