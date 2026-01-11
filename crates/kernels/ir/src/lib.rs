//! Continuum IR - Intermediate Representation
//!
//! Lowers DSL AST into typed IR that can be compiled to runtime DAGs.
//!
//! Pipeline: DSL AST -> IR -> SSA -> Bytecode -> Runtime DAGs

mod codegen;
mod compile;
pub mod fusion;
mod interpret;
mod lower;
pub mod patterns;
pub mod ssa;
mod types;
pub mod units;
mod validate;
pub mod vectorized;

pub use codegen::compile as compile_to_bytecode;
pub use compile::{CompilationResult, CompileError, compile};
pub use interpret::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_resolver,
    build_signal_resolver, convert_assertion_severity, get_initial_signal_value, get_initial_value,
};
pub use lower::{LowerError, lower};
pub use types::*;
pub use validate::{CompileWarning, WarningCode, validate};
pub use vectorized::{L2ExecutionError, L2VectorizedExecutor, ScalarL2Kernel, VRegBuffer};
