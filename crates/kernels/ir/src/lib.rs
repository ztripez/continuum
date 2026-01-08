//! Continuum IR - Intermediate Representation
//!
//! Lowers DSL AST into typed IR that can be compiled to runtime DAGs.
//!
//! Pipeline: DSL AST -> IR -> Runtime DAGs

mod compile;
mod interpret;
mod lower;
mod types;
mod validate;

pub use compile::{compile, CompilationResult, CompileError};
pub use interpret::{
    build_assertion, build_era_configs, build_resolver, convert_assertion_severity,
    get_initial_signal_value, get_initial_value,
};
pub use lower::{lower, LowerError};
pub use types::*;
pub use validate::{validate, CompileWarning, WarningCode};
