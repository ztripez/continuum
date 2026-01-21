//! Continuum Runtime.
//!
//! This crate provides the execution engine for Continuum simulations.
//! It takes compiled IR from `continuum_ir` and executes simulations
//! tick by tick according to the phase model.
//!
//! # Architecture
//!
//! The runtime is organized into several modules:
//!
//! - [`types`] - Core types: [`Phase`], [`Value`], [`StratumState`], [`Dt`]
//! - [`storage`] - Signal and entity storage with tick management
//! - [`soa_storage`] - SoA (Struct-of-Arrays) storage for vectorized execution
//! - [`reductions`] - Deterministic reduction operations for entity aggregates
//! - [`vectorized`] - Unified vectorized primitive abstraction
//! - [`executor`] - Phase executors and the main [`Runtime`] type
//! - [`dag`] - Execution graph construction and scheduling
//! - [`error`] - Error types for runtime failures
//!
//! # Execution Model
//!
//! Each simulation tick proceeds through five phases in order:
//!
//! 1. **Configure** - Freeze execution context for the tick
//! 2. **Collect** - Accumulate inputs and impulse payloads
//! 3. **Resolve** - Compute new signal values from expressions
//! 4. **Fracture** - Detect tension conditions and emit responses
//! 5. **Measure** - Emit field values for observation
//!
//! # Example
//!
//! ```rust,no_run
//! use continuum_cdsl::compile;
//! use continuum_runtime::build_runtime;
//!
//! let root = std::env::temp_dir().join("cdsl-demo");
//! std::fs::create_dir_all(&root).unwrap();
//! std::fs::write(
//!     root.join("demo.cdsl"),
//!     r#"
//! world demo { }
//! strata sim { : stride(1) }
//! era main {
//!     : initial
//!     : dt(1.0 <s>)
//!     strata { sim: active }
//! }
//! signal counter : type Scalar : stratum(sim) { resolve { prev } }
//! "#,
//! )
//! .unwrap();
//!
//! let compiled = compile(&root).unwrap();
//! let mut runtime = build_runtime(compiled);
//! runtime.tick();
//! ```

pub mod checkpoint;
pub mod bytecode;
pub mod dag;
pub mod error;
pub mod executor;
pub mod lens_sink;
pub mod reductions;
pub mod soa_storage;
pub mod storage;
pub mod types;
pub mod vectorized;

pub use error::{Error, Result};
pub use executor::cost_model::{ComplexityScore, ComplexityThresholds, CostModel, CostWeights};
pub use bytecode::{
    BytecodeExecutor, CompiledBlock, Compiler, ExecutionContext, ExecutionError, ExecutionRuntime,
    Instruction, OpcodeKind, OpcodeMetadata,
};
pub use executor::{
    AssertContext, AssertionChecker, AssertionFn, AssertionSeverity, ChunkConfig, CollectContext,
    CollectFn, EraConfig, FractureContext, FractureFn, ImpulseContext, ImpulseFn, LaneKernel,
    LaneKernelError, LaneKernelRegistry, LaneKernelResult, LoweringHeuristics, LoweringStrategy,
    MeasureContext, MeasureFn, MemberResolveContext, MemberSignalResolver, PhaseExecutor,
    ResolveContext, ResolverFn, Runtime, ScalarKernelFn, ScalarL1Kernel, ScalarL1Resolver,
    ScalarResolveContext, ScalarResolverFn, TransitionFn, Vec3KernelFn, Vec3L1Kernel,
    Vec3L1Resolver, Vec3ResolveContext, Vec3ResolverFn, WarmupContext, WarmupExecutor, WarmupFn,
};
pub use soa_storage::{
    AlignedBuffer, MemberSignalBuffer, MemberSignalMeta, MemberSignalRegistry, PopulationStorage,
    SIMD_ALIGNMENT, TypedBuffer, ValueType,
};
pub use types::*;
pub use vectorized::{
    Cardinality, EntityIndex, FieldPrimitive, FieldSampleIdentity, FractureIdentity,
    FracturePrimitive, GlobalSignal, IndexSpace, MemberSignal, MemberSignalId,
    MemberSignalIdentity, SampleIndex, VectorizedPrimitive,
};

use crate::dag::NodeKind;
use continuum_cdsl::ast::{CompiledWorld, ExprKind, RoleId, TypedExpr};
use indexmap::IndexMap;

/// Build a runtime from a compiled world.
///
/// Initializes era configuration, compiles bytecode blocks, and seeds global
/// signals from literal resolve expressions.
///
/// # Parameters
/// - `compiled`: Compiled CDSL world with resolved declarations and DAGs.
///
/// # Returns
/// A [`Runtime`] ready to execute ticks for the initial era.
///
/// # Panics
/// Panics if any era `dt` is not a literal scalar, if transitions are declared
/// without a runtime compiler, or if the initial era is missing or unknown.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::compile;
/// use continuum_runtime::build_runtime;
/// use std::time::{SystemTime, UNIX_EPOCH};
///
/// let mut root = std::env::temp_dir();
/// let stamp = SystemTime::now()
///     .duration_since(UNIX_EPOCH)
///     .unwrap()
///     .as_nanos();
/// root.push(format!("cdsl-demo-{}", stamp));
/// std::fs::create_dir_all(&root).unwrap();
/// std::fs::write(
///     root.join("demo.cdsl"),
///     r#"
/// world demo { }
/// strata sim { : stride(1) }
/// era main {
///     : initial
///     : dt(1.0 <s>)
///     strata { sim: active }
/// }
/// signal counter : type Scalar : stratum(sim) { resolve { prev } }
/// "#,
/// )
/// .unwrap();
///
/// let compiled = compile(&root).unwrap();
/// let _runtime = build_runtime(compiled);
/// ```
pub fn build_runtime(compiled: CompiledWorld) -> Runtime {
    let mut era_configs = IndexMap::new();

    for era in compiled.world.eras.values() {
        if !era.transitions.is_empty() {
            panic!(
                "era '{}' declares transitions, but runtime transition compilation is not implemented",
                era.path
            );
        }
        let dt = literal_scalar(&era.dt).map(Dt).unwrap_or_else(|| {
            panic!("Era {} has non-literal dt expression", era.path)
        });
        let mut strata = IndexMap::new();
        for policy in &era.strata_policy {
            let state = if policy.active {
                if let Some(stride) = policy.cadence_override {
                    StratumState::ActiveWithStride(stride)
                } else {
                    StratumState::Active
                }
            } else {
                StratumState::Gated
            };
            strata.insert(policy.stratum.clone(), state);
        }

        era_configs.insert(
            EraId::new(era.path.to_string()),
            EraConfig {
                dt,
                strata,
                transition: None,
            },
        );
    }

    let initial_era = compiled
        .world
        .initial_era
        .clone()
        .unwrap_or_else(|| {
            panic!("world missing initial era; mark exactly one era with :initial")
        });

    if !era_configs.contains_key(&initial_era) {
        panic!("initial era '{}' not found in era configs", initial_era);
    }

    let (dag_set, bytecode_blocks, impulse_map) = compile_bytecode_and_dags(&compiled);

    let mut runtime = Runtime::new(initial_era, era_configs, dag_set, bytecode_blocks);
    for (id, idx) in impulse_map {
        runtime.add_impulse_mapping(id, idx);
    }

    // Initialize signals from world defaults/metadata
    for (path, node) in &compiled.world.globals {
        if let Some(literal) = node
            .executions
            .iter()
            .find(|execution| execution.phase == Phase::Resolve)
            .and_then(|execution| match &execution.body {
                continuum_cdsl::ast::ExecutionBody::Expr(expr) => literal_scalar(expr),
                _ => None,
            })
        {
            runtime.init_signal(SignalId::from(path.to_string()), Value::Scalar(literal));
        }
    }

    runtime
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl::ast::{EraTransition, TypedExpr, World, WorldDecl};
    use continuum_cdsl::foundation::{Path, Shape, Span, Type, Unit};
    use continuum_kernel_types::KernelId;

    fn empty_world() -> World {
        World::new(WorldDecl {
            path: Path::from_path_str("demo"),
            title: None,
            version: None,
            warmup: None,
            attributes: Vec::new(),
            span: Span::new(0, 0, 0, 0),
            doc: None,
            debug: false,
        })
    }

    #[test]
    #[should_panic(expected = "world missing initial era")]
    fn test_runtime_panics_without_initial_era() {
        let world = empty_world();
        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled);
    }

    #[test]
    #[should_panic(expected = "initial era 'missing' not found in era configs")]
    fn test_runtime_panics_when_initial_missing_in_configs() {
        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("missing"));
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        world
            .eras
            .insert(Path::from_path_str("main"), continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ));

        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled);
    }

    #[test]
    #[should_panic(expected = "non-literal dt expression")]
    fn test_runtime_panics_on_non_literal_dt() {
        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        let left = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        let right = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 2.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![left, right],
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );

        world
            .eras
            .insert(Path::from_path_str("main"), continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ));

        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled);
    }

    #[test]
    #[should_panic(expected = "declares transitions")]
    fn test_runtime_panics_on_transitions() {
        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        let condition = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let mut era = continuum_cdsl::ast::Era::new(
            EraId::new("main"),
            Path::from_path_str("main"),
            dt,
            span,
        );
        era.transitions.push(EraTransition::new(
            EraId::new("next"),
            condition,
            span,
        ));
        world.eras.insert(Path::from_path_str("main"), era);

        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled);
    }
}

fn compile_bytecode_and_dags(compiled: &CompiledWorld) -> (crate::dag::DagSet, Vec<CompiledBlock>, std::collections::HashMap<ImpulseId, usize>) {
    let mut compiler = crate::bytecode::Compiler::new();
    let mut bytecode_blocks = Vec::new();
    let mut block_indices: IndexMap<(continuum_foundation::Path, Phase), usize> = IndexMap::new();
    let mut impulse_map = std::collections::HashMap::new();

    let mut runtime_dags: IndexMap<(Phase, StratumId), crate::dag::ExecutableDag> = IndexMap::new();

    for ((phase, stratum), dag) in &compiled.dag_set.dags {
        let mut levels = Vec::with_capacity(dag.levels.len());
        for level in &dag.levels {
            let mut nodes = Vec::with_capacity(level.nodes.len());
            for path in &level.nodes {
                let (role_id, exec, node_path) = if let Some(node) = compiled.world.globals.get(path)
                {
                    let exec = node
                        .executions
                        .iter()
                        .find(|execution| execution.phase == *phase)
                        .unwrap_or_else(|| panic!("Missing execution for {} in {:?}", path, phase));
                    (node.role_id(), exec, node.path.clone())
                } else if let Some(node) = compiled.world.members.get(path) {
                    let exec = node
                        .executions
                        .iter()
                        .find(|execution| execution.phase == *phase)
                        .unwrap_or_else(|| panic!("Missing execution for {} in {:?}", path, phase));
                    (node.role_id(), exec, node.path.clone())
                } else {
                    panic!("DAG references unknown node {}", path);
                };

                let key = (path.clone(), *phase);
                let block_idx = if let Some(idx) = block_indices.get(&key) {
                    *idx
                } else {
                    let compiled_block = compiler
                        .compile_execution(exec)
                        .unwrap_or_else(|err| panic!("Failed to compile {}: {err:?}", path));
                    let idx = bytecode_blocks.len();
                    bytecode_blocks.push(compiled_block);
                    block_indices.insert(key, idx);
                    idx
                };

                let node_kind = match (role_id, *phase) {
                    (RoleId::Signal, Phase::Resolve) => NodeKind::SignalResolve {
                        signal: SignalId::from(path.to_string()),
                        resolver_idx: block_idx,
                    },
                    (RoleId::Operator, Phase::Collect) | (RoleId::Impulse, Phase::Collect) => {
                        if role_id == RoleId::Impulse {
                            impulse_map.insert(ImpulseId::new(path.to_string()), block_idx);
                        }
                        NodeKind::OperatorCollect {
                            operator_idx: block_idx,
                        }
                    }
                    (RoleId::Operator, Phase::Measure) => NodeKind::OperatorMeasure {
                        operator_idx: block_idx,
                    },
                    (RoleId::Field, Phase::Measure) => NodeKind::FieldEmit {
                        field_idx: block_idx,
                    },
                    (RoleId::Fracture, Phase::Fracture) => NodeKind::Fracture {
                        fracture_idx: block_idx,
                    },
                    (RoleId::Chronicle, Phase::Measure) => NodeKind::ChronicleObserve {
                        chronicle_idx: block_idx,
                    },
                    _ => panic!(
                        "Unsupported execution for {} in phase {:?} ({:?})",
                        path,
                        phase,
                        role_id
                    ),
                };

                let reads = exec
                    .reads
                    .iter()
                    .map(|read| {
                        if let Some(read_node) = compiled.world.globals.get(read) {
                            if read_node.role_id() != RoleId::Signal {
                                panic!("read '{}' is not a signal", read);
                            }
                            SignalId::from(read.to_string())
                        } else if let Some(read_node) = compiled.world.members.get(read) {
                            if read_node.role_id() != RoleId::Signal {
                                panic!("read '{}' is not a signal", read);
                            }
                            SignalId::from(read.to_string())
                        } else {
                            panic!("read '{}' references unknown node", read);
                        }
                    })
                    .collect();

                let writes = match role_id {
                    RoleId::Signal => Some(SignalId::from(node_path.to_string())),
                    _ => None,
                };

                nodes.push(crate::dag::DagNode {
                    id: crate::dag::NodeId(node_path.to_string()),
                    reads,
                    writes,
                    kind: node_kind,
                });
            }
            levels.push(crate::dag::Level { nodes });
        }

        runtime_dags.insert(
            (*phase, stratum.clone()),
            crate::dag::ExecutableDag {
                phase: *phase,
                stratum: stratum.clone(),
                levels,
            },
        );
    }

    let mut era_dags = crate::dag::EraDags::default();
    for dag in runtime_dags.values() {
        era_dags.insert(dag.clone());
    }

    let mut dag_set = crate::dag::DagSet::default();
    for era in compiled.world.eras.values() {
        dag_set.insert_era(EraId::new(era.path.to_string()), era_dags.clone());
    }

    (dag_set, bytecode_blocks, impulse_map)
}

fn literal_scalar(expr: &TypedExpr) -> Option<f64> {
    match &expr.expr {
        ExprKind::Literal { value, .. } => Some(*value),
        _ => None,
    }
}
