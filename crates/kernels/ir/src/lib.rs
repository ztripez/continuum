//! Continuum IR - Intermediate Representation
//!
//! Lowers DSL AST into typed IR that can be compiled to runtime DAGs.
//!
//! Pipeline: DSL AST -> IR -> SSA -> Bytecode -> Runtime DAGs

pub mod analysis;
mod codegen;
mod compile;
pub mod expressions;
pub mod fusion;
mod interpret;
mod lower;
pub mod patterns;
pub mod ssa;
mod types;
pub mod unified_nodes;
#[cfg(test)]
mod unified_nodes_test;
pub mod units;
mod validate;
pub mod vectorized;

pub use continuum_foundation::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, MemberId,
    OperatorId, Path, SignalId, StratumId, TypeId,
};

pub use codegen::compile as compile_to_bytecode;
pub use compile::{CompilationResult, CompileError, compile};
pub use expressions::*;
pub use interpret::{
    InterpValue, MemberInterpContext, MemberResolverFn, Vec3MemberResolverFn,
    build_aggregate_resolver, build_assertion, build_era_configs, build_field_measure,
    build_fracture, build_member_resolver, build_resolver, build_signal_resolver,
    build_vec3_member_resolver, convert_assertion_severity, eval_initial_expr,
    get_initial_signal_value, get_initial_value,
};
pub use lower::{LowerError, lower};
pub use types::*;
pub use unified_nodes::*;
pub use validate::{CompileWarning, WarningCode, validate};
pub use vectorized::{L2ExecutionError, L2VectorizedExecutor, ScalarL2Kernel, VRegBuffer};
