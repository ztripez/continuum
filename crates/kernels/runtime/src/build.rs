//! Runtime construction from compiled worlds.
//!
//! This module converts a compiled CDSL world into an executable [`Runtime`].
//! It handles era configuration, bytecode compilation, DAG construction,
//! config/const loading, scenario overrides, entity initialization, and
//! signal seeding.

use crate::bytecode::CompiledBlock;
use crate::dag::{self, NodeKind};
use crate::executor::{EraConfig, Runtime};
use crate::soa_storage::ValueType;
use crate::types::*;
use crate::vectorized::MemberSignalId;
use continuum_cdsl::ast::{CompiledWorld, ExprKind, RoleId, TypeExpr, TypedExpr};
use continuum_cdsl::foundation::Type;
use continuum_foundation::Path;
use indexmap::IndexMap;

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
        (
            Value::Mat2(_),
            TypeExpr::Matrix {
                rows: 2, cols: 2, ..
            },
        ) => true,
        (
            Value::Mat3(_),
            TypeExpr::Matrix {
                rows: 3, cols: 3, ..
            },
        ) => true,
        (
            Value::Mat4(_),
            TypeExpr::Matrix {
                rows: 4, cols: 4, ..
            },
        ) => true,

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
/// use continuum_cdsl::compile_with_sources;
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

        // Compile era transition conditions into a TransitionFn
        let transition: Option<crate::executor::TransitionFn> = if era.transitions.is_empty() {
            None
        } else {
            Some(compile_era_transitions(&era.transitions, dt))
        };

        era_configs.insert(
            EraId::new(era.path.to_string()),
            EraConfig {
                dt,
                strata,
                transition,
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
            // Other declaration types are handled by their respective registration phases
            Declaration::Node(_)
            | Declaration::Member(_)
            | Declaration::Entity(_)
            | Declaration::Stratum(_)
            | Declaration::Era(_)
            | Declaration::Type(_)
            | Declaration::World(_)
            | Declaration::Function(_) => {}
        }
    }

    // Load nested config/const blocks from global and member nodes
    use continuum_cdsl::ast::NestedBlock;

    /// Extract config/const values from a node's nested blocks into the accumulator maps.
    fn load_nested_config_const(
        node_path: &Path,
        nested_blocks: &[NestedBlock],
        config_values: &mut IndexMap<Path, Value>,
        const_values: &mut IndexMap<Path, Value>,
        config_types: &mut IndexMap<Path, continuum_cdsl::ast::TypeExpr>,
    ) {
        for block in nested_blocks {
            match block {
                NestedBlock::Config(entries) => {
                    for entry in entries {
                        let qualified_path = node_path.append(entry.path.to_string());
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
                        let qualified_path = node_path.append(entry.path.to_string());
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

    for (_path, node) in &compiled.world.nodes {
        load_nested_config_const(
            &node.path,
            &node.nested_blocks,
            &mut config_values,
            &mut const_values,
            &mut config_types,
        );
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
            let type_expr = config_types
                .get(&path)
                .expect("path must exist in config_types after validation 2");
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
                            panic!(
                                "Entity '{}' count references unknown config: {}",
                                entity_path, config_path
                            );
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
    for (path, node) in compiled
        .world
        .nodes
        .iter()
        .filter(|(_, n)| n.entity.is_some())
    {
        // Use output type (set by type resolution) instead of type_expr (cleared after resolution)
        let Some(output_type) = node.output.as_ref() else {
            // Not a signal (probably entity field metadata)
            continue;
        };

        let value_type = output_type_to_value_type(output_type, &path.to_string());
        runtime.register_member_signal(&path.to_string(), value_type);
    }

    // Register the root entity and global signals in the member signal buffer.
    // Global signals are stored as instance-0 of the root entity.
    runtime.register_root_entity();
    for (path, node) in compiled
        .world
        .nodes
        .iter()
        .filter(|(_, n)| n.entity.is_none())
    {
        let Some(output_type) = node.output.as_ref() else {
            continue;
        };

        let value_type = output_type_to_value_type(output_type, &path.to_string());
        runtime.register_global_signal(&path.to_string(), value_type);
    }

    // Initialize member signal storage with the maximum entity count.
    // Always allocate at least 1 instance for the root entity (global signals).
    runtime.init_member_instances(max_entity_count.max(1));

    // Populate config/const in bytecode executor
    runtime.set_config_values(config_values);
    runtime.set_const_values(const_values);

    // Extract signal types for zero value initialization
    let mut signal_types = IndexMap::new();
    for (path, node) in &compiled.world.nodes {
        if let Some(output_type) = &node.output {
            signal_types.insert(SignalId::from(path.to_string()), output_type.clone());
        }
    }

    runtime.set_signal_types(signal_types);

    // Initialize spatial topologies from entity declarations
    runtime.initialize_topologies(&compiled.world.entities);

    // Initialize global signals from world defaults/metadata
    for (path, node) in compiled
        .world
        .nodes
        .iter()
        .filter(|(_, n)| n.entity.is_none())
    {
        // Try to initialize from literal resolve block first
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
            continue; // Already initialized, skip config check
        }

        // If no literal resolve, check for :initial() attribute
        if let Some(initial_value) = node.initial {
            runtime.init_signal(
                SignalId::from(path.to_string()),
                Value::Scalar(initial_value),
            );
        }
    }

    runtime
}

/// Convert an output `Type` to a `ValueType` for signal storage registration.
///
/// Panics if the type is not supported for signal storage (e.g., String, unsupported
/// matrix dimensions).
fn output_type_to_value_type(output_type: &Type, path: &str) -> ValueType {
    use continuum_kernel_types::Shape;
    match output_type {
        Type::Kernel(kt) => match &kt.shape {
            Shape::Scalar => ValueType::scalar(),
            Shape::Vector { dim: 2 } => ValueType::vec2(),
            Shape::Vector { dim: 3 } => ValueType::vec3(),
            Shape::Vector { dim: 4 } => ValueType::vec4(),
            Shape::Vector { dim } => {
                panic!("Unsupported vector dimension {} for signal {}", dim, path)
            }
            _ => panic!("Unsupported shape {:?} for signal {}", kt.shape, path),
        },
        Type::Bool => ValueType::boolean(),
        Type::String => panic!("String signals not yet supported: {}", path),
        _ => panic!("Unsupported type {:?} for signal {}", output_type, path),
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
                let node = compiled
                    .world
                    .nodes
                    .get(path)
                    .unwrap_or_else(|| panic!("DAG references unknown node {}", path));
                let exec = node
                    .executions
                    .iter()
                    .find(|execution| execution.phase == *phase)
                    .unwrap_or_else(|| panic!("Missing execution for {} in {:?}", path, phase));
                let (role_id, node_path) = (node.role_id(), node.path.clone());

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
                        // Check if this is a member signal (has entity association)
                        let entity = if node.entity.is_some() {
                            // Parse domain.entity.member path
                            let path_str = path.to_string();
                            let path_parts: Vec<&str> = path_str.split('.').collect();
                            if path_parts.len() < 2 {
                                panic!("Invalid member signal path format: {}", path_str);
                            }
                            // Last part is the member signal name
                            let signal_name = path_parts[path_parts.len() - 1].to_string();
                            // Everything before the last part is the entity ID
                            let entity_id =
                                EntityId::from(path_parts[..path_parts.len() - 1].join("."));

                            Some(dag::SignalEntityContext {
                                member_signal: MemberSignalId {
                                    entity_id,
                                    signal_name,
                                },
                            })
                        } else {
                            None
                        };

                        NodeKind::SignalResolve {
                            signal: SignalId::from(path.to_string()),
                            resolver_idx: block_idx,
                            entity,
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
                        if let Some(read_node) = compiled.world.nodes.get(read) {
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
/// Compile era transition conditions into a runtime `TransitionFn`.
///
/// Each transition has a target era and a boolean condition (already typed and
/// compiled by the pipeline). This function compiles each condition into bytecode
/// and builds a closure that evaluates them in declaration order, returning the
/// first matching target.
///
/// # Parameters
/// - `transitions`: Typed transition declarations from the resolved era
/// - `dt`: The era's time step (needed for kernel evaluation context)
///
/// # Returns
/// A `TransitionFn` closure that evaluates all transition conditions and returns
/// `Some(target_era)` for the first condition that evaluates to `true`.
///
/// # Panics
/// Panics if a transition condition fails to compile to bytecode (indicates a
/// compiler bug — the type checker should have already validated the expression).
fn compile_era_transitions(
    transitions: &[continuum_cdsl::ast::EraTransition],
    dt: Dt,
) -> crate::executor::TransitionFn {
    use crate::bytecode::Compiler;
    use continuum_cdsl::ast::{Execution, ExecutionBody};

    let mut compiler = Compiler::new();
    let mut compiled_transitions: Vec<(EraId, CompiledBlock)> = Vec::new();

    for transition in transitions {
        // Wrap the condition TypedExpr in a synthetic Execution block
        let synthetic_execution = Execution {
            name: format!("transition_to_{}", transition.target.as_str()),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(transition.condition.clone()),
            reads: Vec::new(),
            temporal_reads: Vec::new(),
            emits: Vec::new(),
            span: transition.span,
        };

        let compiled_block = compiler
            .compile_execution(&synthetic_execution)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to compile transition condition to '{}': {:?}. \
                 This is a compiler bug — the type checker should have validated the expression.",
                    transition.target.as_str(),
                    e
                )
            });

        let target = EraId::new(transition.target.as_str().to_string());
        compiled_transitions.push((target, compiled_block));
    }

    // Move compiled blocks into an Arc so the closure can be Send + Sync
    let compiled_transitions = std::sync::Arc::new(compiled_transitions);

    Box::new(move |signals, _entities, _sim_time| {
        use crate::bytecode::BytecodeExecutor;
        use crate::executor::transition_context::TransitionEvalContext;

        // Transition evaluation uses empty config/const maps since the conditions
        // only reference signals (config/const are inlined during compilation).
        // If a transition condition references config/const at runtime, the
        // TransitionEvalContext will return an error.
        let config_values = IndexMap::new();
        let const_values = IndexMap::new();

        for (target, block) in compiled_transitions.iter() {
            let mut ctx = TransitionEvalContext::new(dt, signals, &config_values, &const_values);

            let mut executor = BytecodeExecutor::new();
            match executor.execute_block(block.root, &block.program, &mut ctx) {
                Ok(Some(Value::Boolean(true))) => return Some(target.clone()),
                Ok(Some(Value::Boolean(false))) | Ok(None) => continue,
                Ok(Some(other)) => {
                    tracing::error!(
                        target_era = %target,
                        value = ?other,
                        "transition condition evaluated to non-boolean value"
                    );
                    continue;
                }
                Err(e) => {
                    tracing::error!(
                        target_era = %target,
                        error = %e,
                        "transition condition evaluation failed"
                    );
                    continue;
                }
            }
        }

        None
    })
}

fn evaluate_literal(expr: &continuum_cdsl::ast::Expr) -> Option<Value> {
    use continuum_cdsl::ast::UntypedKind;
    use continuum_cdsl::foundation::UnaryOp;

    match &expr.kind {
        UntypedKind::Literal { value, .. } => Some(Value::Scalar(*value)),
        // Handle negative literals (parsed as UnaryOp::Neg)
        UntypedKind::Unary {
            op: UnaryOp::Neg,
            operand,
        } => {
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
                    UntypedKind::Unary {
                        op: UnaryOp::Neg,
                        operand,
                    } => {
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
