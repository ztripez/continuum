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
pub mod scenario;
pub mod ssa;
pub mod types;
pub mod unified_nodes;
#[cfg(test)]
mod unified_nodes_test;
pub mod units;
mod validate;
pub mod vectorized;

pub use codegen::compile as compile_to_bytecode;
pub use compile::{CompilationResult, CompileError, compile};
pub use continuum_foundation::{PrimitiveParamKind, PrimitiveParamSpec};
pub use expressions::*;
pub use interpret::{
    InterpValue, MemberInterpContext, MemberResolverFn, MemberResolverStats, RuntimeBuildError,
    RuntimeBuildOptions, RuntimeBuildReport, Vec3MemberResolverFn, build_aggregate_resolver,
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_member_resolver,
    build_resolver, build_runtime, build_signal_resolver, build_vec3_member_resolver,
    build_warmup_fn, convert_assertion_severity, eval_initial_expr, get_initial_signal_value,
    get_initial_value, register_member_resolvers,
};

pub use lower::{LowerError, lower, lower_multi, lower_with_file};
pub use scenario::{
    Scenario, ScenarioError, ScenarioMetadata, ScenarioResult, ScenarioValue, find_scenarios,
    load_scenarios,
};
pub use types::*;
pub use unified_nodes::*;
pub use validate::{CompileWarning, WarningCode, validate};
pub use vectorized::{L2ExecutionError, L2VectorizedExecutor, ScalarL2Kernel, VRegBuffer};
