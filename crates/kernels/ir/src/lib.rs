//! Continuum IR - Intermediate Representation
//!
//! Lowers DSL AST into typed IR that can be compiled to runtime DAGs.
//!
//! Pipeline: DSL AST -> IR -> Runtime DAGs

mod lower;
mod types;

pub use lower::{lower, LowerError};
pub use types::*;
