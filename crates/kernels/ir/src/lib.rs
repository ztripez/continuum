//! Continuum IR - Intermediate Representation
//!
//! Lowers DSL AST into typed IR that can be compiled to runtime DAGs.
//!
//! Pipeline: DSL AST -> IR -> SSA -> Bytecode -> Runtime DAGs

mod codegen;
mod compile;
mod interpret;
mod lower;
pub mod ssa;
mod types;
pub mod units;
mod validate;
pub mod vectorized;

pub use codegen::compile as compile_to_bytecode;
pub use compile::{compile, CompilationResult, CompileError};
pub use interpret::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_resolver,
    convert_assertion_severity, get_initial_signal_value, get_initial_value,
};
pub use lower::{lower, LowerError};
pub use types::*;
pub use validate::{validate, CompileWarning, WarningCode};
pub use vectorized::{L2ExecutionError, L2VectorizedExecutor, ScalarL2Kernel, VRegBuffer};
