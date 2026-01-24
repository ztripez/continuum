// Allow unwrap in tests
#![cfg_attr(test, allow(clippy::unwrap_used))]

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

pub mod bytecode;
pub mod checkpoint;
pub mod dag;
pub mod error;
pub mod executor;
pub mod lens_sink;
pub mod reductions;
pub mod soa_storage;
pub mod storage;
pub mod types;
pub mod vectorized;

pub use bytecode::{
    BytecodeExecutor, CompiledBlock, Compiler, ExecutionContext, ExecutionError, ExecutionRuntime,
    Instruction, OpcodeKind, OpcodeMetadata,
};
pub use continuum_foundation::WorldPolicy;
pub use error::{Error, Result};
pub use executor::cost_model::{ComplexityScore, ComplexityThresholds, CostModel, CostWeights};
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
use continuum_cdsl::ast::{CompiledWorld, ExprKind, RoleId, TypedExpr, TypeExpr};
use continuum_cdsl::Type;
use continuum_foundation::Path;
use indexmap::IndexMap;
use tracing::debug;

/// Scenario configuration for a Continuum world.
///
/// A Scenario defines how a World is instantiated for execution. It provides
/// initial conditions and parameter overrides without changing causal structure.
///
/// # Config Overrides
///
/// Scenarios may override config values declared in `config {}` blocks. Config
/// overrides take precedence over world defaults but must target declared config
/// paths.
///
/// **Const values cannot be overridden** - they are immutable world-level constants.
///
/// # Example
///
/// ```ignore
/// // World defines:
/// config {
///     thermal.decay_halflife: 1.42e17 <s>
///     thermal.initial_temp: 5500.0 <K>
/// }
///
/// // Scenario overrides:
/// let scenario = Scenario {
///     config_overrides: [
///         ("thermal.initial_temp".into(), Value::Scalar(6000.0))
///     ].into_iter().collect()
/// };
/// ```
///
/// # Lifecycle
///
/// Scenario config overrides are applied during `build_runtime()` (lifecycle stage 4:
/// Scenario Application). Overrides are merged with world defaults, with scenario
/// values taking precedence.
#[derive(Debug, Clone, Default)]
pub struct Scenario {
    /// Config value overrides keyed by Path.
    ///
    /// These override world defaults from `config {}` blocks. Only config values
    /// may be overridden - const values are immutable.
    pub config_overrides: IndexMap<Path, Value>,
}

impl Scenario {
    /// Creates an empty scenario with no overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a scenario with the given config overrides.
    pub fn with_config_overrides(config_overrides: IndexMap<Path, Value>) -> Self {
        Self { config_overrides }
    }
}

/// Validates that a runtime Value matches the expected TypeExpr.
///
/// Returns `true` if the value is compatible with the declared type.
/// This performs basic shape matching without unit checking (units are
/// tracked separately in the type system).
fn value_matches_type_expr(value: &Value, type_expr: &TypeExpr) -> bool {
    match (value, type_expr) {
        // Scalar matches Scalar type
        (Value::Scalar(_), TypeExpr::Scalar { .. }) => true,
        
        // Integer can match Scalar (implicit conversion)
        (Value::Integer(_), TypeExpr::Scalar { .. }) => true,
        
        // Vec2 matches Vector<2, _>
        (Value::Vec2(_), TypeExpr::Vector { dim: 2, .. }) => true,
        
        // Vec3 matches Vector<3, _>
        (Value::Vec3(_), TypeExpr::Vector { dim: 3, .. }) => true,
        
        // Vec4 matches Vector<4, _>
        (Value::Vec4(_), TypeExpr::Vector { dim: 4, .. }) => true,
        
        // Boolean matches Bool type
        (Value::Boolean(_), TypeExpr::Bool) => true,
        
        // Infer accepts anything (type will be inferred)
        (_, TypeExpr::Infer) => true,
        
        // Matrix types (Mat2, Mat3, Mat4)
        (Value::Mat2(_), TypeExpr::Matrix { rows: 2, cols: 2, .. }) => true,
        (Value::Mat3(_), TypeExpr::Matrix { rows: 3, cols: 3, .. }) => true,
        (Value::Mat4(_), TypeExpr::Matrix { rows: 4, cols: 4, .. }) => true,
        
        // Quaternion could match Vector<4, _> or a custom type
        (Value::Quat(_), TypeExpr::Vector { dim: 4, .. }) => true,
        
        // All other combinations are mismatches
        _ => false,
    }
}

/// Build a runtime from a compiled world with optional scenario overrides.
///
/// Initializes era configuration, compiles bytecode blocks, loads config/const values
/// (with scenario overrides applied), and seeds global signals from literal resolve
/// expressions.
///
/// # Parameters
/// - `compiled`: Compiled CDSL world with resolved declarations and DAGs.
/// - `scenario`: Optional scenario with config value overrides. If `None`, uses world
///   defaults for all config values.
///
/// # Returns
/// A [`Runtime`] ready to execute ticks for the initial era.
///
/// # Panics
/// Panics if any era `dt` is not a literal scalar, if transitions are declared
/// without a runtime compiler, if the initial era is missing or unknown, or if
/// config/const declarations contain non-literal expressions.
///
/// # Scenario Config Overrides
///
/// If `scenario` is provided with `config_overrides`, those values will override
/// world defaults for matching config paths. Const values cannot be overridden.
///
/// # Examples
///
/// Basic usage without scenario:
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
/// let _runtime = build_runtime(compiled, None);
/// ```
///
/// With scenario config overrides:
/// ```ignore
/// use continuum_runtime::{build_runtime, Scenario, Value};
///
/// let scenario = Scenario::with_config_overrides(
///     [("thermal.initial_temp".into(), Value::Scalar(6000.0))]
///         .into_iter()
///         .collect()
/// );
/// let runtime = build_runtime(compiled, Some(scenario));
/// ```
pub fn build_runtime(compiled: CompiledWorld, scenario: Option<Scenario>) -> Runtime {
    let mut era_configs = IndexMap::new();

    for era in compiled.world.eras.values() {
        if !era.transitions.is_empty() {
            panic!(
                "era '{}' declares transitions, but runtime transition compilation is not implemented",
                era.path
            );
        }
        let dt = literal_scalar(&era.dt)
            .map(Dt)
            .unwrap_or_else(|| panic!("Era {} has non-literal dt expression", era.path));
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

    let initial_era =
        compiled.world.initial_era.clone().unwrap_or_else(|| {
            panic!("world missing initial era; mark exactly one era with :initial")
        });

    if !era_configs.contains_key(&initial_era) {
        panic!("initial era '{}' not found in era configs", initial_era);
    }

    let (dag_set, bytecode_blocks, impulse_map) = compile_bytecode_and_dags(&compiled);

    let mut runtime = Runtime::new(
        initial_era,
        era_configs,
        dag_set,
        bytecode_blocks,
        compiled.world.metadata.policy,
    );
    for (id, idx) in impulse_map {
        runtime.add_impulse_mapping(id, idx);
    }

    // Extract and populate config/const values
    use continuum_cdsl::ast::Declaration;
    let mut config_values = IndexMap::new();
    let mut const_values = IndexMap::new();
    let mut config_types = IndexMap::new();

    // Load world defaults for config and const
    for decl in &compiled.world.declarations {
        match decl {
            Declaration::Config(entries) => {
                for entry in entries {
                    // Store type information for validation
                    config_types.insert(entry.path.clone(), entry.type_expr.clone());
                    
                    if let Some(default) = &entry.default {
                        let value = evaluate_literal(default).unwrap_or_else(|| {
                            panic!(
                                "Config '{}' has non-literal default expression. \
                                 Config defaults must be compile-time literals (Scalar or Vec3 with literal components).",
                                entry.path
                            )
                        });
                        config_values.insert(entry.path.clone(), value);
                    }
                }
            }
            Declaration::Const(entries) => {
                for entry in entries {
                    let value = evaluate_literal(&entry.value).unwrap_or_else(|| {
                        panic!(
                            "Const '{}' has non-literal expression. \
                             Const values must be compile-time literals (Scalar or Vec3 with literal components).",
                            entry.path
                        )
                    });
                    const_values.insert(entry.path.clone(), value);
                }
            }
            _ => {}
        }
    }

    // Load nested config/const blocks from global nodes
    use continuum_cdsl::ast::NestedBlock;
    for (_path, node) in &compiled.world.globals {
        for block in &node.nested_blocks {
            match block {
                NestedBlock::Config(entries) => {
                    for entry in entries {
                        // Construct qualified path: node.path + entry.path
                        let qualified_path = node.path.append(entry.path.to_string());
                        config_types.insert(qualified_path.clone(), entry.type_expr.clone());
                        
                        if let Some(default) = &entry.default {
                            let value = evaluate_literal(default).unwrap_or_else(|| {
                                panic!(
                                    "Config '{}' has non-literal default expression. \
                                     Config defaults must be compile-time literals (Scalar or Vec3 with literal components).",
                                    qualified_path
                                )
                            });
                            config_values.insert(qualified_path, value);
                        }
                    }
                }
                NestedBlock::Const(entries) => {
                    for entry in entries {
                        let qualified_path = node.path.append(entry.path.to_string());
                        let value = evaluate_literal(&entry.value).unwrap_or_else(|| {
                            panic!(
                                "Const '{}' has non-literal expression. \
                                 Const values must be compile-time literals (Scalar or Vec3 with literal components).",
                                qualified_path
                            )
                        });
                        const_values.insert(qualified_path, value);
                    }
                }
            }
        }
    }

    // Load nested config/const blocks from member nodes
    for (_path, node) in &compiled.world.members {
        for block in &node.nested_blocks {
            match block {
                NestedBlock::Config(entries) => {
                    for entry in entries {
                        // Construct qualified path: node.path + entry.path
                        let qualified_path = node.path.append(entry.path.to_string());
                        config_types.insert(qualified_path.clone(), entry.type_expr.clone());
                        
                        if let Some(default) = &entry.default {
                            let value = evaluate_literal(default).unwrap_or_else(|| {
                                panic!(
                                    "Config '{}' has non-literal default expression. \
                                     Config defaults must be compile-time literals (Scalar or Vec3 with literal components).",
                                    qualified_path
                                )
                            });
                            config_values.insert(qualified_path, value);
                        }
                    }
                }
                NestedBlock::Const(entries) => {
                    for entry in entries {
                        let qualified_path = node.path.append(entry.path.to_string());
                        let value = evaluate_literal(&entry.value).unwrap_or_else(|| {
                            panic!(
                                "Const '{}' has non-literal expression. \
                                 Const values must be compile-time literals (Scalar or Vec3 with literal components).",
                                qualified_path
                            )
                        });
                        const_values.insert(qualified_path, value);
                    }
                }
            }
        }
    }

    // Apply scenario config overrides (const values cannot be overridden)
    if let Some(scenario) = scenario {
        for (path, value) in scenario.config_overrides {
            // Validation 1: Ensure path is NOT a const value (check this FIRST)
            if const_values.contains_key(&path) {
                panic!(
                    "Scenario attempts to override immutable const path '{}'. \
                     Const values cannot be overridden by scenarios.",
                    path
                );
            }
            
            // Validation 2: Check if path exists in config
            if !config_types.contains_key(&path) {
                let valid_paths: Vec<_> = config_types.keys().collect();
                panic!(
                    "Scenario attempts to override non-existent config path '{}'. \
                     This may be a typo. Valid config paths: {:?}",
                    path, valid_paths
                );
            }
            
            // Validation 3: Type validation
            let type_expr = config_types.get(&path).unwrap();
            if !value_matches_type_expr(&value, type_expr) {
                panic!(
                    "Scenario override for '{}' has incompatible type. \
                     Expected type matching {:?}, got value {:?}",
                    path, type_expr, value
                );
            }
            
            config_values.insert(path, value);
        }
    }

    // Initialize entities - spawn instances based on :count(...) attributes
    // (Must happen before set_config_values which moves config_values)
    use continuum_cdsl::ast::UntypedKind;
    let mut max_entity_count = 0;
    for (entity_path, entity) in &compiled.world.entities {
        // Look for :count(...) attribute
        let count_attr = entity.attributes.iter().find(|attr| attr.name == "count");
        
        if let Some(attr) = count_attr {
            if attr.args.len() == 1 {
                // Evaluate the count expression
                let count_expr = &attr.args[0];
                
                // Try to evaluate as literal or config reference
                let count = match &count_expr.kind {
                    UntypedKind::Config(config_path) => {
                        // It's a config reference like config.plates.count
                        if let Some(Value::Scalar(val)) = config_values.get(config_path) {
                            *val as usize
                        } else if let Some(Value::Integer(val)) = config_values.get(config_path) {
                            *val as usize
                        } else {
                            panic!("Entity '{}' count references unknown config: {}", entity_path, config_path);
                        }
                    }
                    _ => {
                        // Try literal
                        if let Some(Value::Scalar(literal)) = evaluate_literal(count_expr) {
                            literal as usize
                        } else if let Some(Value::Integer(literal)) = evaluate_literal(count_expr) {
                            literal as usize
                        } else {
                            panic!(
                                "Entity '{}' has non-literal, non-config count expression. \
                                 Count must be a compile-time literal or config reference.",
                                entity_path
                            );
                        }
                    }
                };
                
                runtime.register_entity_count(&entity.id.to_string(), count);
                max_entity_count = max_entity_count.max(count);
            } else {
                panic!(
                    "Entity '{}' has :count attribute with {} arguments, expected 1",
                    entity_path,
                    attr.args.len()
                );
            }
        } else {
            // No count attribute - default to 0 instances
            runtime.register_entity_count(&entity.id.to_string(), 0);
        }
    }
    
    // Register all member signals with the member signal buffer
    // This must happen BEFORE init_member_instances
    for (path, node) in &compiled.world.members {
        // Use output type (set by type resolution) instead of type_expr (cleared after resolution)
        let Some(output_type) = node.output.as_ref() else {
            // Not a signal (probably entity field metadata)
            continue;
        };
        
        use continuum_kernel_types::Shape;
        let value_type = match output_type {
            Type::Kernel(kt) => match &kt.shape {
                Shape::Scalar => ValueType::scalar(),
                Shape::Vector { dim: 2 } => ValueType::vec2(),
                Shape::Vector { dim: 3 } => ValueType::vec3(),
                Shape::Vector { dim: 4 } => ValueType::vec4(),
                Shape::Vector { dim } => {
                    panic!("Unsupported vector dimension {} for member signal {}", dim, path)
                }
                _ => panic!("Unsupported shape {:?} for member signal {}", kt.shape, path),
            },
            Type::Bool => ValueType::boolean(),
            Type::String => panic!("String member signals not yet supported: {}", path),
            _ => panic!("Unsupported type {:?} for member signal {}", output_type, path),
        };
        
        runtime.register_member_signal(&path.to_string(), value_type);
    }
    
    // Initialize member signal storage with the maximum entity count
    // This must happen AFTER registering all signals
    if max_entity_count > 0 {
        runtime.init_member_instances(max_entity_count);
    }

    // Populate config/const in bytecode executor  
    runtime.set_config_values(config_values);
    runtime.set_const_values(const_values);

    // Extract signal types for zero value initialization
    let mut signal_types = IndexMap::new();
    
    // Extract types from global signals
    for (path, node) in &compiled.world.globals {
        if let Some(output_type) = &node.output {
            signal_types.insert(SignalId::from(path.to_string()), output_type.clone());
        }
    }
    
    // Extract types from member signals
    for (path, node) in &compiled.world.members {
        if let Some(output_type) = &node.output {
            signal_types.insert(SignalId::from(path.to_string()), output_type.clone());
        }
    }
    
    runtime.set_signal_types(signal_types);

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
            policy: WorldPolicy::default(),
        })
    }

    #[test]
    #[should_panic(expected = "world missing initial era")]
    fn test_runtime_panics_without_initial_era() {
        let world = empty_world();
        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled, None);
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
        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled, None);
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

        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled, None);
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
        era.transitions
            .push(EraTransition::new(EraId::new("next"), condition, span));
        world.eras.insert(Path::from_path_str("main"), era);

        let compiled = CompiledWorld::new(world, Default::default());
        let _runtime = build_runtime(compiled, None);
    }

    // ========== Scenario Config Override Validation Tests ==========

    #[test]
    #[should_panic(expected = "non-existent config path")]
    fn test_scenario_rejects_non_existent_config_path() {
        use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        // Add a valid config entry
        let config_entry = ConfigEntry {
            path: Path::from_path_str("thermal.decay_halflife"),
            default: Some(Expr {
                kind: UntypedKind::Literal {
                    value: 1.42e17,
                    unit: None,
                },
                span,
            }),
            type_expr: TypeExpr::Scalar {
                unit: None,
                bounds: None,
            },
            span,
            doc: None,
        };
        world
            .declarations
            .push(Declaration::Config(vec![config_entry]));

        // Add era
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());

        // Try to override a NON-EXISTENT path (typo: "inital" instead of "initial")
        let scenario = Scenario::with_config_overrides(
            [(
                Path::from_path_str("thermal.inital_temp"),
                Value::Scalar(6000.0),
            )]
            .into_iter()
            .collect(),
        );

        let _runtime = build_runtime(compiled, Some(scenario));
    }

    #[test]
    #[should_panic(expected = "override immutable const path")]
    fn test_scenario_rejects_const_override() {
        use continuum_cdsl::ast::{ConstEntry, Declaration, Expr, UntypedKind};

        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        // Add a const entry
        let const_entry = ConstEntry {
            path: Path::from_path_str("physics.stefan_boltzmann"),
            value: Expr {
                kind: UntypedKind::Literal {
                    value: 5.67e-8,
                    unit: None,
                },
                span,
            },
            type_expr: TypeExpr::Scalar {
                unit: None,
                bounds: None,
            },
            span,
            doc: None,
        };
        world
            .declarations
            .push(Declaration::Const(vec![const_entry]));

        // Add era
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());

        // Try to override a CONST value (should fail)
        let scenario = Scenario::with_config_overrides(
            [(
                Path::from_path_str("physics.stefan_boltzmann"),
                Value::Scalar(6.0e-8),
            )]
            .into_iter()
            .collect(),
        );

        let _runtime = build_runtime(compiled, Some(scenario));
    }

    #[test]
    #[should_panic(expected = "incompatible type")]
    fn test_scenario_rejects_wrong_value_type() {
        use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        // Add a config entry expecting Scalar
        let config_entry = ConfigEntry {
            path: Path::from_path_str("thermal.initial_temp"),
            default: Some(Expr {
                kind: UntypedKind::Literal {
                    value: 5500.0,
                    unit: None,
                },
                span,
            }),
            type_expr: TypeExpr::Scalar {
                unit: None,
                bounds: None,
            },
            span,
            doc: None,
        };
        world
            .declarations
            .push(Declaration::Config(vec![config_entry]));

        // Add era
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());

        // Try to override with WRONG TYPE (Vec3 instead of Scalar)
        let scenario = Scenario::with_config_overrides(
            [(
                Path::from_path_str("thermal.initial_temp"),
                Value::Vec3([1.0, 2.0, 3.0]),
            )]
            .into_iter()
            .collect(),
        );

        let _runtime = build_runtime(compiled, Some(scenario));
    }

    #[test]
    fn test_scenario_valid_override_succeeds() {
        use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        // Add a config entry
        let config_entry = ConfigEntry {
            path: Path::from_path_str("thermal.initial_temp"),
            default: Some(Expr {
                kind: UntypedKind::Literal {
                    value: 5500.0,
                    unit: None,
                },
                span,
            }),
            type_expr: TypeExpr::Scalar {
                unit: None,
                bounds: None,
            },
            span,
            doc: None,
        };
        world
            .declarations
            .push(Declaration::Config(vec![config_entry]));

        // Add era
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());

        // Valid override with correct type
        let scenario = Scenario::with_config_overrides(
            [(
                Path::from_path_str("thermal.initial_temp"),
                Value::Scalar(6000.0),
            )]
            .into_iter()
            .collect(),
        );

        // Should succeed without panic
        let _runtime = build_runtime(compiled, Some(scenario));
    }

    #[test]
    #[should_panic(expected = "incompatible type")]
    fn test_scenario_rejects_vector_dimension_mismatch() {
        use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

        let span = Span::new(0, 0, 0, 0);
        let mut world = empty_world();
        world.initial_era = Some(EraId::new("main"));

        // Add a config entry expecting Vector<3>
        let config_entry = ConfigEntry {
            path: Path::from_path_str("physics.position"),
            default: Some(Expr {
                kind: UntypedKind::Vector(vec![
                    Expr {
                        kind: UntypedKind::Literal {
                            value: 0.0,
                            unit: None,
                        },
                        span,
                    },
                    Expr {
                        kind: UntypedKind::Literal {
                            value: 0.0,
                            unit: None,
                        },
                        span,
                    },
                    Expr {
                        kind: UntypedKind::Literal {
                            value: 0.0,
                            unit: None,
                        },
                        span,
                    },
                ]),
                span,
            }),
            type_expr: TypeExpr::Vector {
                dim: 3,
                unit: None,
            },
            span,
            doc: None,
        };
        world
            .declarations
            .push(Declaration::Config(vec![config_entry]));

        // Add era
        let dt = TypedExpr::new(
            continuum_cdsl::ast::ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            span,
        );
        world.eras.insert(
            Path::from_path_str("main"),
            continuum_cdsl::ast::Era::new(
                EraId::new("main"),
                Path::from_path_str("main"),
                dt,
                span,
            ),
        );

        let compiled = CompiledWorld::new(world, Default::default());

        // Try to override with WRONG VECTOR DIMENSION (Vec2 instead of Vec3)
        let scenario = Scenario::with_config_overrides(
            [(
                Path::from_path_str("physics.position"),
                Value::Vec2([1.0, 2.0]),
            )]
            .into_iter()
            .collect(),
        );

        let _runtime = build_runtime(compiled, Some(scenario));
    }

}

fn compile_bytecode_and_dags(
    compiled: &CompiledWorld,
) -> (
    crate::dag::DagSet,
    Vec<CompiledBlock>,
    std::collections::HashMap<ImpulseId, usize>,
) {
    let mut compiler = crate::bytecode::Compiler::new();
    let mut bytecode_blocks = Vec::new();
    let mut block_indices: IndexMap<(continuum_foundation::Path, Phase), usize> = IndexMap::new();
    let mut impulse_map = std::collections::HashMap::new();

    let mut runtime_dags: IndexMap<(EraId, Phase, StratumId), crate::dag::ExecutableDag> =
        IndexMap::new();

    for ((era, phase, stratum), dag) in &compiled.dag_set.dags {
        let mut levels = Vec::with_capacity(dag.levels.len());
        for level in &dag.levels {
            let mut nodes = Vec::with_capacity(level.nodes.len());
            for path in &level.nodes {
                let (role_id, exec, node_path) = if let Some(node) =
                    compiled.world.globals.get(path)
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
                    (RoleId::Signal, Phase::Configure) | (RoleId::Signal, Phase::Resolve) => {
                        // Check if this is a member signal (domain.entity.member)
                        if compiled.world.members.contains_key(path) {
                            // Parse domain.entity.member path
                            let path_str = path.to_string();
                            let path_parts: Vec<&str> = path_str.split('.').collect();
                            if path_parts.len() < 2 {
                                panic!("Invalid member signal path format: {}", path_str);
                            }
                            // Last part is the member signal name
                            let signal_name = path_parts[path_parts.len() - 1].to_string();
                            // Everything before the last part is the entity ID
                            let entity_id = EntityId::from(
                                path_parts[..path_parts.len() - 1].join(".")
                            );

                            NodeKind::MemberSignalResolve {
                                member_signal: MemberSignalId {
                                    entity_id,
                                    signal_name,
                                },
                                kernel_idx: block_idx,
                            }
                        } else {
                            // Regular global signal
                            NodeKind::SignalResolve {
                                signal: SignalId::from(path.to_string()),
                                resolver_idx: block_idx,
                            }
                        }
                    }
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
                        path, phase, role_id
                    ),
                };

                // Filter reads to only include runtime dependencies (signals).
                // Config/const values are compile-time constants and don't need DAG tracking.
                // Entity reads are also filtered out - they're structural dependencies, not signal reads.
                let reads = exec
                    .reads
                    .iter()
                    .filter_map(|read| {
                        if let Some(read_node) = compiled.world.globals.get(read) {
                            if read_node.role_id() != RoleId::Signal {
                                panic!("read '{}' is not a signal", read);
                            }
                            Some(SignalId::from(read.to_string()))
                        } else if let Some(read_node) = compiled.world.members.get(read) {
                            if read_node.role_id() != RoleId::Signal {
                                panic!("read '{}' is not a signal", read);
                            }
                            Some(SignalId::from(read.to_string()))
                        } else {
                            // Not a node - must be config, const, or entity dependency
                            // These don't need runtime DAG tracking
                            None
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
            (era.clone(), *phase, stratum.clone()),
            crate::dag::ExecutableDag {
                phase: *phase,
                stratum: stratum.clone(),
                levels,
            },
        );
    }

    // Build runtime DagSet with proper era separation
    // Group DAGs by era
    let mut dags_by_era: IndexMap<EraId, Vec<crate::dag::ExecutableDag>> = IndexMap::new();
    for ((era_id, _, _), dag) in &runtime_dags {
        dags_by_era
            .entry(era_id.clone())
            .or_default()
            .push(dag.clone());
    }

    let mut dag_set = crate::dag::DagSet::default();
    for (era_id, dags) in dags_by_era {
        let mut era_dags = crate::dag::EraDags::default();
        for dag in dags {
            era_dags.insert(dag);
        }
        dag_set.insert_era(era_id, era_dags);
    }

    (dag_set, bytecode_blocks, impulse_map)
}

fn literal_scalar(expr: &TypedExpr) -> Option<f64> {
    match &expr.expr {
        ExprKind::Literal { value, .. } => Some(*value),
        _ => None,
    }
}

/// Evaluates a literal DSL expression to a runtime Value.
///
/// This function extracts compile-time constant values from config and const
/// declarations during world loading. Only simple literal expressions are supported:
///
/// - **Scalar literals**: `42.0`, `1.5e-8`, `290.0 <K>`
/// - **Vec3 literals**: `[1.0, 2.0, 3.0]` (all components must be literals)
///
/// # Returns
///
/// - `Some(Value::Scalar(f64))` for scalar literals
/// - `Some(Value::Vec3([f64; 3]))` for 3-element vector literals
/// - `None` for any non-literal expression (kernel calls, references, operators, etc.)
///
/// # Usage Context
///
/// Called during `build_runtime()` (lifecycle stage 4: Scenario Application) to extract
/// default values from `config{}` and `const{}` blocks. Non-literal expressions cause
/// a panic with a clear error message (enforcing the Fail Loudly principle).
///
/// # Examples
///
/// ```ignore
/// // Valid expressions (return Some):
/// config { physics.gravity: 9.81 }           // Scalar
/// const { physics.origin: [0.0, 0.0, 0.0] }  // Vec3
///
/// // Invalid expressions (return None, causing panic):
/// config { foo: 1.0 + 2.0 }                  // Kernel call
/// config { bar: const.base_value }           // Reference
/// ```
fn evaluate_literal(expr: &continuum_cdsl::ast::Expr) -> Option<Value> {
    use continuum_cdsl::ast::UntypedKind;
    use continuum_cdsl::foundation::UnaryOp;
    
    match &expr.kind {
        UntypedKind::Literal { value, .. } => Some(Value::Scalar(*value)),
        // Handle negative literals (parsed as UnaryOp::Neg)
        UntypedKind::Unary { op, operand } if matches!(op, UnaryOp::Neg) => {
            if let UntypedKind::Literal { value, .. } = &operand.kind {
                Some(Value::Scalar(-value))
            } else {
                None
            }
        }
        UntypedKind::Vector(elements) if elements.len() == 3 => {
            // Helper to extract scalar from element (handles negative literals)
            let extract_scalar = |elem: &continuum_cdsl::ast::Expr| -> Option<f64> {
                match &elem.kind {
                    UntypedKind::Literal { value, .. } => Some(*value),
                    UntypedKind::Unary { op, operand } if matches!(op, UnaryOp::Neg) => {
                        if let UntypedKind::Literal { value, .. } = &operand.kind {
                            Some(-value)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            };
            
            let x = extract_scalar(&elements[0])?;
            let y = extract_scalar(&elements[1])?;
            let z = extract_scalar(&elements[2])?;
            Some(Value::Vec3([x, y, z]))
        }
        _ => None,
    }
}
