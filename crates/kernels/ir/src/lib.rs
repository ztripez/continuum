//! Continuum IR - Intermediate Representation
//!
//! Lowers DSL AST into typed IR that can be compiled to runtime DAGs.
//!
//! Pipeline: DSL AST -> IR -> Runtime DAGs

mod compile;
mod lower;
mod types;

pub use compile::{compile, CompilationResult, CompileError};
pub use lower::{lower, LowerError};
pub use types::*;
