//! Intermediate Representation (IR) Types
//!
//! This module defines the typed intermediate representation produced by lowering
//! DSL AST nodes. The IR is a simplified, executable form of the DSL that serves
//! as input to DAG construction and runtime execution.
//!
//! # Overview
//!
//! The IR sits between the AST (parsing output) and the execution graph (runtime input).
//! It performs several transformations:
//!
//! - **Type resolution**: All types are fully resolved with constraints
//! - **Dependency analysis**: Signal dependencies are explicitly tracked via `reads` fields
//! - **Function inlining**: User-defined functions are inlined at call sites
//! - **Expression simplification**: Complex AST nodes are reduced to simpler IR forms
//!
//! # Key Types
//!
//! - [`CompiledWorld`]: The top-level container holding all compiled definitions
//! - [`CompiledSignal`]: A signal with resolved expression and dependencies
//! - [`CompiledExpr`]: Expression tree suitable for bytecode compilation
//! - [`ValueType`]: Scalar or vector type with optional range constraints
//!
//! # Usage
//!
//! The IR is typically produced by the [`crate::lower()`] function and consumed by:
//! - `crate::codegen` for bytecode compilation
//! - `crate::interpret` for building runtime closures
//! - `crate::validate` for warning generation

use indexmap::IndexMap;

use continuum_foundation::{ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, MemberId, OperatorId, SignalId, StratumId, TypeId};

// Re-export StratumState from foundation for backwards compatibility
pub use continuum_foundation::StratumState;

/// The complete compiled simulation world, ready for DAG construction.
///
/// `CompiledWorld` is the top-level IR container produced by [`crate::lower()`].
/// It holds all definitions from all parsed `.cdsl` files in a single, unified
/// representation with resolved dependencies and types.
///
/// # Structure
///
/// The world is organized into several categories:
///
/// - **Configuration**: Constants (compile-time) and config values (scenario-time)
/// - **Time structure**: Strata (simulation layers) and eras (time phases)
/// - **Causal entities**: Signals, operators, and fractures that affect simulation state
/// - **Observation**: Fields for data extraction (non-causal)
/// - **External input**: Impulses for external causal events
/// - **Structured data**: Entities for collections of similar instances
///
/// # Ordering
///
/// All maps use [`IndexMap`] to preserve insertion order, which ensures
/// deterministic iteration for graph construction and execution scheduling.
///
/// # Example
///
/// ```ignore
/// let world = lower(&compilation_unit)?;
/// println!("Signals defined: {}", world.signals.len());
/// for (id, signal) in &world.signals {
///     println!("  {} reads {:?}", id.0, signal.reads);
/// }
/// ```
#[derive(Debug)]
pub struct CompiledWorld {
    /// Global constants (evaluated at compile time)
    pub constants: IndexMap<String, f64>,
    /// Runtime configuration values
    pub config: IndexMap<String, f64>,
    /// User-defined functions
    pub functions: IndexMap<FnId, CompiledFn>,
    /// Strata definitions
    pub strata: IndexMap<StratumId, CompiledStratum>,
    /// Era definitions
    pub eras: IndexMap<EraId, CompiledEra>,
    /// Signal definitions
    pub signals: IndexMap<SignalId, CompiledSignal>,
    /// Field definitions
    pub fields: IndexMap<FieldId, CompiledField>,
    /// Operator definitions
    pub operators: IndexMap<OperatorId, CompiledOperator>,
    /// Impulse definitions
    pub impulses: IndexMap<ImpulseId, CompiledImpulse>,
    /// Fracture definitions
    pub fractures: IndexMap<FractureId, CompiledFracture>,
    /// Entity definitions
    pub entities: IndexMap<EntityId, CompiledEntity>,
    /// Member signal definitions (per-entity authoritative state)
    pub members: IndexMap<MemberId, CompiledMember>,
    /// Chronicle definitions (observer-only event recording)
    pub chronicles: IndexMap<ChronicleId, CompiledChronicle>,
    /// Custom type definitions
    pub types: IndexMap<TypeId, CompiledType>,
}

/// A compiled stratum definition representing a simulation layer.
///
/// Strata partition the simulation into independent layers that can execute
/// at different cadences. Each signal, field, and operator belongs to exactly
/// one stratum.
///
/// # Stride
///
/// The `default_stride` controls how often this stratum executes relative to
/// the base tick rate. A stride of 10 means the stratum executes every 10th
/// tick of the containing era.
#[derive(Debug, Clone)]
pub struct CompiledStratum {
    /// Unique identifier for the stratum.
    pub id: StratumId,
    /// Human-readable title for display.
    pub title: Option<String>,
    /// Unicode symbol for visualization.
    pub symbol: Option<String>,
    /// Default stride (ticks between updates).
    pub default_stride: u32,
}

/// A compiled user-defined function declaration.
///
/// Functions in the DSL are pure and are inlined at their call sites during
/// lowering. This struct stores the function definition for reference during
/// inlining.
///
/// # Restrictions
///
/// User-defined functions have strict purity requirements:
///
/// - **No `prev` access**: Cannot reference the previous value of any signal
/// - **No `dt` access**: Cannot depend on the current time step
/// - **No signal writes**: Cannot emit or modify signal values
/// - **Allowed**: `const.*` and `config.*` references, calling other functions
///
/// These restrictions ensure functions can be safely inlined without changing
/// semantics or introducing hidden dependencies.
///
/// # Inlining
///
/// When a function call is lowered, it becomes a nested sequence of `Let`
/// expressions binding arguments to parameters, followed by the function body.
#[derive(Debug, Clone)]
pub struct CompiledFn {
    /// Unique identifier for the function.
    pub id: FnId,
    /// Parameter names in order
    pub params: Vec<String>,
    /// Function body expression
    pub body: CompiledExpr,
}

/// A compiled era definition representing a distinct time phase.
///
/// Eras partition simulation time into phases with different characteristics.
/// Each era specifies its own time step (`dt`), stratum activation states,
/// and transition conditions to other eras.
///
/// # Era Lifecycle
///
/// 1. Simulation starts in the era marked `is_initial = true`
/// 2. Each tick, transition conditions are evaluated in order
/// 3. First matching transition moves to that era
/// 4. Simulation ends when reaching an era with `is_terminal = true`
///
/// # Strata States
///
/// The `strata_states` map controls which strata are active in this era:
/// - `Active`: Executes every tick
/// - `ActiveWithStride(n)`: Executes every nth tick
/// - `Gated`: Disabled (does not execute)
#[derive(Debug, Clone)]
pub struct CompiledEra {
    /// Unique identifier for the era.
    pub id: EraId,
    /// Whether this is the starting era.
    pub is_initial: bool,
    /// Whether this era ends simulation.
    pub is_terminal: bool,
    /// Human-readable title.
    pub title: Option<String>,
    /// Time step in seconds
    pub dt_seconds: f64,
    /// Stratum states for this era
    pub strata_states: IndexMap<StratumId, StratumState>,
    /// Transitions to other eras
    pub transitions: Vec<CompiledTransition>,
}

/// A compiled transition between eras.
///
/// Transitions define conditions under which the simulation moves from the
/// current era to a target era. The condition expression is evaluated each
/// tick; if it returns a non-zero value, the transition fires.
///
/// When multiple transitions are defined, they are evaluated in order and
/// the first matching transition is taken.
#[derive(Debug, Clone)]
pub struct CompiledTransition {
    /// Target era to transition to.
    pub target_era: EraId,
    /// Condition that must be true (non-zero) to transition.
    pub condition: CompiledExpr,
}

/// A compiled signal definition representing authoritative simulation state.
///
/// Signals are the primary state-carrying entities in Continuum. Each signal
/// has a resolve expression that computes its next value based on its previous
/// value, other signals, and constants/config.
///
/// # Dependencies
///
/// The `reads` field lists all signals this signal depends on, enabling the
/// DAG builder to determine execution order. Circular dependencies are not
/// allowed and should be detected during validation.
///
/// # dt-robustness
///
/// Signals that use `dt_raw` must declare this explicitly via `uses_dt_raw = true`.
/// This enables dt-robustness auditing: signals using raw dt values may produce
/// different results with different time steps, which is sometimes intentional
/// (e.g., physical integration) but should be tracked.
///
/// # Vector/Tensor Signals
///
/// For vector types (Vec2, Vec3, Vec4), expressions are expanded at lowering time
/// into per-component scalar expressions stored in `resolve_components`. This
/// enables the bytecode VM to remain scalar-only while supporting vector signals.
///
/// - Scalar signals: Use `resolve` field
/// - Vector signals: Use `resolve_components` field (x, y, z, w in order)
///
/// # Warmup
///
/// Signals may define a warmup phase that runs before normal simulation to
/// establish initial equilibrium through iterative convergence.
#[derive(Debug, Clone)]
pub struct CompiledSignal {
    /// Unique identifier for the signal.
    pub id: SignalId,
    /// Stratum binding for scheduling.
    pub stratum: StratumId,
    /// Human-readable title.
    pub title: Option<String>,
    /// Unicode symbol for display.
    pub symbol: Option<String>,
    /// Value type with optional bounds.
    pub value_type: ValueType,
    /// Whether `dt_raw` is explicitly used.
    pub uses_dt_raw: bool,
    /// Signals this signal reads
    pub reads: Vec<SignalId>,
    /// The resolve expression (for scalar signals).
    pub resolve: Option<CompiledExpr>,
    /// Component-wise resolve expressions for vector signals.
    ///
    /// For Vec2: [x, y]
    /// For Vec3: [x, y, z]
    /// For Vec4: [x, y, z, w]
    ///
    /// When present, this takes precedence over `resolve` for vector signals.
    /// Each expression is a scalar expression that computes one component.
    pub resolve_components: Option<Vec<CompiledExpr>>,
    /// Warmup configuration
    pub warmup: Option<CompiledWarmup>,
    /// Assertions to validate after resolution
    pub assertions: Vec<CompiledAssertion>,
}

/// A compiled field definition for observable (non-causal) data.
///
/// Fields are derived measurements computed from signal values. Unlike signals,
/// fields do not affect the causal simulation - they exist purely for observation
/// and data extraction. Removing all fields from a world must not change the
/// simulation outcome.
///
/// # Topology
///
/// Fields may have spatial topology (e.g., `SphereSurface`, `PointCloud`) that
/// describes how their data is distributed in space, which affects visualization
/// and analysis.
///
/// # Phase
///
/// Fields are computed during the Measure phase, after all signals have resolved.
/// They may read any signal value but cannot affect signal resolution.
#[derive(Debug, Clone)]
pub struct CompiledField {
    /// Unique identifier for the field.
    pub id: FieldId,
    /// Stratum binding.
    pub stratum: StratumId,
    /// Human-readable title.
    pub title: Option<String>,
    /// Spatial topology for reconstruction.
    pub topology: TopologyIr,
    /// Value type at each sample point.
    pub value_type: ValueType,
    /// Signals this field reads
    pub reads: Vec<SignalId>,
    /// The measure expression
    pub measure: Option<CompiledExpr>,
}

/// A compiled operator definition for phase-specific computation.
///
/// Operators are execution units that run during specific simulation phases.
/// Unlike signals (which resolve state), operators perform side-effect actions
/// like collecting inputs, running warmup iterations, or measuring outputs.
///
/// # Phases
///
/// - `Warmup`: Runs during the warmup phase before main simulation
/// - `Collect`: Gathers and accumulates inputs for signal resolution
/// - `Measure`: Computes derived values during the observation phase
#[derive(Debug, Clone)]
pub struct CompiledOperator {
    /// Unique identifier for the operator.
    pub id: OperatorId,
    /// Stratum binding.
    pub stratum: StratumId,
    /// Execution phase.
    pub phase: OperatorPhaseIr,
    /// Signals this operator reads
    pub reads: Vec<SignalId>,
    /// The operator body
    pub body: Option<CompiledExpr>,
    /// Assertions to validate after execution
    pub assertions: Vec<CompiledAssertion>,
}

/// A compiled impulse definition for external causal input.
///
/// Impulses provide a mechanism for external systems to inject causal events
/// into the simulation. Each impulse has a typed payload and an apply
/// expression that determines how the payload affects signal state.
///
/// # Usage
///
/// Impulses are typically triggered by:-
/// User interaction events
/// - External system inputs
/// - Scenario-driven event scripts
#[derive(Debug, Clone)]
pub struct CompiledImpulse {
    /// Unique identifier for the impulse.
    pub id: ImpulseId,
    /// Type of data carried by the impulse.
    pub payload_type: ValueType,
    /// The apply expression
    pub apply: Option<CompiledExpr>,
}

/// A compiled fracture definition for emergent tension detection.
///
/// Fractures detect when simulation state reaches critical thresholds and
/// trigger corrective emissions to other signals. They represent emergent
/// phenomena that occur when certain conditions are met.
///
/// # Evaluation
///
/// All conditions must evaluate to non-zero for the fracture to trigger.
/// When triggered, all emit expressions are evaluated and their values
/// are applied to the target signals.
///
/// # Ordering
///
/// Fractures execute during the Fracture phase, after signal resolution
/// but before measurement. This allows corrective actions before observation.
#[derive(Debug, Clone)]
pub struct CompiledFracture {
    /// Unique identifier for the fracture.
    pub id: FractureId,
    /// Signals this fracture reads
    pub reads: Vec<SignalId>,
    /// Condition expressions (all must be true)
    pub conditions: Vec<CompiledExpr>,
    /// Emit statements
    pub emits: Vec<CompiledEmit>,
}

/// A signal emission from a fracture.
///
/// When a fracture triggers, it emits values to one or more signals. Each
/// emit specifies a target signal and a value expression to compute.
#[derive(Debug, Clone)]
pub struct CompiledEmit {
    /// Target signal path.
    pub target: SignalId,
    /// Expression for the emitted value.
    pub value: CompiledExpr,
}

/// A compiled entity definition representing a pure index space.
///
/// Entities define collections of similar instances (e.g., moons, tectonic plates)
/// that can be indexed and iterated. Entities are pure identity providers - they
/// define *what exists* but not *what state it has*.
///
/// Per-entity state is defined via member signals (`member.entity.field { ... }`),
/// which are top-level primitives with their own resolve expressions and strata.
/// This separation allows multi-rate scheduling on the same entity.
///
/// # Instance Count
///
/// The number of instances can be:
/// - Fixed: Determined at compile time via `count_bounds`
/// - Config-driven: Read from scenario configuration via `count_source`
/// - Bounded: Validated against `count_bounds` at runtime
///
/// # Example DSL
///
/// ```cdsl
/// entity.stellar.moon {
///     : strata(stellar)
///     : count(1..100)
/// }
///
/// // Per-entity state is defined separately as member signals:
/// member.stellar.moon.mass {
///     : Scalar<kg>
///     : strata(stellar.orbital)
///     resolve { prev }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CompiledEntity {
    /// Unique identifier for the entity type.
    pub id: EntityId,
    /// Count source from config (e.g., "stellar.moon_count")
    pub count_source: Option<String>,
    /// Count validation bounds
    pub count_bounds: Option<(u32, u32)>,
}

/// A compiled member signal definition representing per-entity authoritative state.
///
/// Member signals are top-level primitives that define state attached to entity
/// instances. Unlike entity schema fields (which are resolved together), each
/// member signal has its own resolve expression and can belong to a different
/// stratum, enabling multi-rate scheduling on the same entity.
///
/// # Path Structure
///
/// The member's path consists of:
/// - Entity path: identifies which entity type this belongs to
/// - Signal name: the specific signal within that entity
///
/// For example, `human.person.age` belongs to entity `human.person` with signal `age`.
///
/// # Snapshot Semantics
///
/// All `self.*` reads within a member's resolve block see the previous tick's values,
/// maintaining snapshot/next-state semantics for determinism.
///
/// # Example DSL
///
/// ```cdsl
/// member.human.person.age {
///     : Scalar
///     : strata(human.physiology)
///     resolve { integrate(prev, 1) }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CompiledMember {
    /// Unique identifier for this member signal.
    pub id: MemberId,
    /// The entity this member belongs to.
    pub entity_id: EntityId,
    /// The signal name within the entity (e.g., "age" from "human.person.age").
    pub signal_name: String,
    /// Stratum binding for scheduling.
    pub stratum: StratumId,
    /// Human-readable title for display.
    pub title: Option<String>,
    /// Unicode symbol for visualization.
    pub symbol: Option<String>,
    /// Value type with optional bounds.
    pub value_type: ValueType,
    /// Whether `dt_raw` is explicitly used.
    pub uses_dt_raw: bool,
    /// Signals this member reads (global signals).
    pub reads: Vec<SignalId>,
    /// Other member signals this member reads.
    pub member_reads: Vec<MemberId>,
    /// The resolve expression.
    pub resolve: Option<CompiledExpr>,
    /// Assertions to validate after resolution.
    pub assertions: Vec<CompiledAssertion>,
}

/// A compiled chronicle definition for observer-only event recording.
///
/// Chronicles observe simulation state and emit events for logging, analytics,
/// or user notification. They are strictly non-causal - removing all chronicles
/// must not change simulation results.
///
/// # Observer Boundary
///
/// Chronicles execute during the Measure phase and can only read signal values.
/// They cannot:
/// - Write to signals
/// - Emit impulses
/// - Affect any causal state
///
/// # Event Emission
///
/// When an observation handler's condition evaluates to true (non-zero), the
/// chronicle emits a structured event with the specified name and payload fields.
#[derive(Debug, Clone)]
pub struct CompiledChronicle {
    /// Unique identifier for the chronicle.
    pub id: ChronicleId,
    /// Signals this chronicle reads for its observation handlers.
    pub reads: Vec<SignalId>,
    /// Observation handlers that emit events when conditions are met.
    pub handlers: Vec<CompiledObserveHandler>,
}

/// An observation handler within a chronicle.
///
/// Each handler watches for a specific condition and emits a named event
/// with structured payload when the condition is met.
#[derive(Debug, Clone)]
pub struct CompiledObserveHandler {
    /// Condition that must be true (non-zero) to emit the event.
    pub condition: CompiledExpr,
    /// Name of the event to emit (e.g., "supercontinent_formed").
    pub event_name: String,
    /// Payload fields for the emitted event.
    pub event_fields: Vec<CompiledEventField>,
}

/// A field within a chronicle event payload.
///
/// Each field has a name and an expression that computes its value
/// when the event is emitted.
#[derive(Debug, Clone)]
pub struct CompiledEventField {
    /// Name of the field in the event payload.
    pub name: String,
    /// Expression to compute the field value.
    pub value: CompiledExpr,
}

/// Warmup configuration for signal initialization.
///
/// Warmup runs a signal's iterate expression repeatedly until convergence
/// or the maximum iteration count is reached. This establishes initial
/// equilibrium before the main simulation begins.
///
/// # Convergence
///
/// If `convergence` is specified, iteration stops early when the absolute
/// change between iterations falls below this threshold.
#[derive(Debug, Clone)]
pub struct CompiledWarmup {
    /// Maximum iterations to run.
    pub iterations: u32,
    /// Optional convergence threshold.
    pub convergence: Option<f64>,
    /// Expression evaluated each warmup iteration.
    pub iterate: CompiledExpr,
}

/// A compiled assertion for runtime validation.
///
/// Assertions validate signal invariants after resolution. They do not modify
/// values - they only check conditions and emit faults when violated.
///
/// # Severity Levels
///
/// - `Warn`: Logs a warning but continues execution
/// - `Error`: May halt execution based on policy configuration
/// - `Fatal`: Always halts execution immediately
#[derive(Debug, Clone)]
pub struct CompiledAssertion {
    /// The condition that must be true
    pub condition: CompiledExpr,
    /// Severity of the assertion failure
    pub severity: AssertionSeverity,
    /// Optional message to emit on failure
    pub message: Option<String>,
}

/// The severity level of an assertion failure.
///
/// Determines how the runtime responds when an assertion condition evaluates
/// to false (zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssertionSeverity {
    /// Warning only, execution continues
    Warn,
    /// Error, may halt based on policy
    #[default]
    Error,
    /// Fatal, always halts
    Fatal,
}

/// The type of a signal or field value.
///
/// Values can be scalars or fixed-size vectors. Types may optionally carry
/// unit information (e.g., "K" for Kelvin, "m/s" for velocity) and range
/// constraints.
///
/// # Unit Representation
///
/// Units are stored in two forms:
/// - `unit: Option<String>`: The original unit string for display and serialization
/// - `dimension: Option<Unit>`: Parsed dimensional representation for analysis
///
/// The string form is used for:
/// - Error message clarity (e.g., "expected K, got m/s")
/// - Observer output formatting
/// - Documentation generation
///
/// The structured form (`Unit`) enables:
/// - Dimensional analysis at compile time
/// - Unit algebra for expression type checking
/// - Physics safety validation
///
/// # Examples
///
/// - `Scalar { unit: None, dimension: None, range: None }`: Unbounded dimensionless scalar
/// - `Scalar { unit: Some("K"), dimension: Some(Unit::temperature()), ... }`: Temperature
/// - `Vec3 { unit: Some("m/s"), dimension: Some(Unit::velocity()) }`: Velocity vector
#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    /// Single scalar value.
    Scalar {
        /// Unit string (e.g., "K", "Pa", "W/m²").
        unit: Option<String>,
        /// Parsed dimensional representation for analysis.
        dimension: Option<crate::units::Unit>,
        /// Optional value bounds.
        range: Option<ValueRange>,
    },
    /// 2D vector.
    Vec2 {
        /// Component unit (e.g., "m", "m/s").
        unit: Option<String>,
        /// Parsed dimensional representation for analysis.
        dimension: Option<crate::units::Unit>,
        /// Optional magnitude constraint.
        magnitude: Option<ValueRange>,
    },
    /// 3D vector.
    Vec3 {
        /// Component unit (e.g., "m", "m/s").
        unit: Option<String>,
        /// Parsed dimensional representation for analysis.
        dimension: Option<crate::units::Unit>,
        /// Optional magnitude constraint (e.g., for position bounds).
        magnitude: Option<ValueRange>,
    },
    /// 4D vector (quaternions, homogeneous coordinates).
    Vec4 {
        /// Component unit (typically "1" for quaternions).
        unit: Option<String>,
        /// Parsed dimensional representation for analysis.
        dimension: Option<crate::units::Unit>,
        /// Optional magnitude constraint (e.g., magnitude: 1 for unit quaternions).
        magnitude: Option<ValueRange>,
    },
    /// NxM tensor (matrices, stress/strain tensors).
    Tensor {
        /// Number of rows.
        rows: u8,
        /// Number of columns.
        cols: u8,
        /// Element unit (e.g., "Pa" for stress tensors).
        unit: Option<String>,
        /// Parsed dimensional representation for analysis.
        dimension: Option<crate::units::Unit>,
        /// Mathematical constraints (symmetric, positive_definite).
        constraints: Vec<TensorConstraintIr>,
    },
    /// 2D grid of values (e.g., temperature maps, heightmaps).
    Grid {
        /// Grid width.
        width: u32,
        /// Grid height.
        height: u32,
        /// Element type.
        element_type: Box<ValueType>,
    },
    /// Ordered sequence of values.
    Seq {
        /// Element type.
        element_type: Box<ValueType>,
        /// Aggregate constraints (each, sum).
        constraints: Vec<SeqConstraintIr>,
    },
}

/// A numeric range constraint for scalar values.
///
/// Defines the valid bounds for a scalar value. Values outside this range
/// may trigger assertions or validation warnings depending on configuration.
///
/// Both bounds are inclusive: a value `v` is valid if `min <= v <= max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueRange {
    /// Minimum allowed value.
    pub min: f64,
    /// Maximum allowed value.
    pub max: f64,
}

/// Tensor mathematical constraint.
///
/// Tensors can have constraints that enforce mathematical properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorConstraintIr {
    /// Matrix must be symmetric (A = A^T).
    Symmetric,
    /// Matrix must be positive definite (all eigenvalues > 0).
    PositiveDefinite,
}

/// Sequence aggregate constraint.
///
/// Sequences can have constraints that enforce aggregate properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeqConstraintIr {
    /// Each element must be within the given range.
    Each(ValueRange),
    /// The sum of all elements must be within the given range.
    Sum(ValueRange),
}

/// Spatial topology for field data distribution.
///
/// Describes how field data is organized in space, affecting how it can be
/// visualized, sampled, and analyzed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyIr {
    /// Data distributed on a spherical surface (e.g., planetary data)
    SphereSurface,
    /// Discrete point samples in 3D space
    PointCloud,
    /// Volumetric data in 3D space
    Volume,
}

/// The execution phase for an operator.
///
/// Determines when in the tick lifecycle the operator runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorPhaseIr {
    /// Runs during warmup initialization
    Warmup,
    /// Runs during input collection before signal resolution
    Collect,
    /// Runs during the measurement/observation phase
    Measure,
}

/// A compiled expression tree ready for bytecode generation or interpretation.
///
/// `CompiledExpr` is a simplified representation of DSL expressions with all
/// syntactic sugar removed and user-defined functions inlined. It serves as
/// the input to both the bytecode compiler and the closure-based interpreter.
///
/// # Expression Categories
///
/// - **Literals and references**: `Literal`, `Prev`, `DtRaw`, `Signal`, `Const`, `Config`
/// - **Operators**: `Binary`, `Unary`
/// - **Control flow**: `If`, `Let`, `Local`
/// - **Function calls**: `Call` (for kernel functions)
/// - **Entity operations**: `SelfField`, `Aggregate`, `Filter`, etc.
///
/// # Entity Expressions
///
/// Entity-related variants (`SelfField`, `EntityAccess`, `Aggregate`, etc.) are
/// not compiled to bytecode directly. They are handled by the entity executor
/// at runtime, which has access to entity storage for iteration and aggregation.
///
/// # Example
///
/// The DSL expression `prev + signal.heat * 0.5` becomes:
///
/// ```ignore
/// CompiledExpr::Binary {
///     op: BinaryOpIr::Add,
///     left: Box::new(CompiledExpr::Prev),
///     right: Box::new(CompiledExpr::Binary {
///         op: BinaryOpIr::Mul,
///         left: Box::new(CompiledExpr::Signal(SignalId::from("heat"))),
///         right: Box::new(CompiledExpr::Literal(0.5)),
///     }),
/// }
/// ```
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    /// Literal value
    Literal(f64),
    /// Previous value of current signal
    Prev,
    /// Raw dt value
    DtRaw,
    /// Accumulated simulation time in seconds
    SimTime,
    /// Collected/accumulated inputs from Collect phase
    Collected,
    /// Reference to a signal
    Signal(SignalId),
    /// Reference to a constant
    Const(String),
    /// Reference to a config value
    Config(String),
    /// Binary operation
    Binary {
        /// The operator.
        op: BinaryOpIr,
        /// Left operand.
        left: Box<CompiledExpr>,
        /// Right operand.
        right: Box<CompiledExpr>,
    },
    /// Unary operation
    Unary {
        /// The operator.
        op: UnaryOpIr,
        /// Operand expression.
        operand: Box<CompiledExpr>,
    },
    /// User-defined function call
    ///
    /// This is kept for user-defined functions that couldn't be inlined.
    /// Most user functions get inlined during lowering.
    Call {
        /// Name of the function.
        function: String,
        /// Call arguments.
        args: Vec<CompiledExpr>,
    },
    /// Kernel function call (engine-provided)
    ///
    /// These are engine-provided primitives called via `kernel.*` syntax.
    /// They may be GPU-accelerated and have guaranteed determinism properties.
    /// Examples: `kernel.sqrt`, `kernel.sin`, `kernel.gravity_acceleration`.
    KernelCall {
        /// Name of the kernel function (without the `kernel.` prefix).
        function: String,
        /// Call arguments.
        args: Vec<CompiledExpr>,
    },
    /// dt-robust operator call
    ///
    /// These operators provide numerically stable time integration with
    /// implicit dt handling. They are distinguished from regular function
    /// calls to enable special code generation and validation.
    DtRobustCall {
        /// The dt-robust operator being called.
        operator: DtRobustOperator,
        /// Call arguments (varies by operator).
        args: Vec<CompiledExpr>,
        /// Integration method (when applicable).
        method: IntegrationMethod,
    },
    /// Field access
    FieldAccess {
        /// Object to access.
        object: Box<CompiledExpr>,
        /// Field name.
        field: String,
    },
    /// Conditional
    If {
        /// Condition to test.
        condition: Box<CompiledExpr>,
        /// Expression if true.
        then_branch: Box<CompiledExpr>,
        /// Expression if false.
        else_branch: Box<CompiledExpr>,
    },
    /// Let binding
    Let {
        /// Variable name.
        name: String,
        /// Value to bind.
        value: Box<CompiledExpr>,
        /// Body where binding is visible.
        body: Box<CompiledExpr>,
    },
    /// Local variable reference
    Local(String),

    // === Entity expressions ===

    /// Access current entity instance field: self.mass
    SelfField(String),

    /// Access entity instance by ID: entity.moon["luna"].mass
    EntityAccess {
        /// Entity type ID.
        entity: EntityId,
        /// Instance identifier.
        instance: InstanceId,
        /// Field name.
        field: String,
    },

    /// Aggregate operation over entity instances: sum(entity.moon, self.mass)
    Aggregate {
        /// Aggregation operator.
        op: AggregateOpIr,
        /// Entity type to aggregate over.
        entity: EntityId,
        /// Expression evaluated per instance.
        body: Box<CompiledExpr>,
    },

    /// Other instances (self-exclusion): sum(other(entity.moon), ...)
    /// Used within entity resolve for N-body interactions
    Other {
        /// Entity type.
        entity: EntityId,
        /// Body expression.
        body: Box<CompiledExpr>,
    },

    /// Pairwise iteration: for (a, b) in pairs(entity.moon)
    Pairs {
        /// Entity type.
        entity: EntityId,
        /// Body expression.
        body: Box<CompiledExpr>,
    },

    /// Filter entity instances: filter(entity.moon, self.mass > 1e20)
    Filter {
        /// Entity type.
        entity: EntityId,
        /// Filter predicate.
        predicate: Box<CompiledExpr>,
        /// Body expression.
        body: Box<CompiledExpr>,
    },

    /// First matching instance: first(entity.plate, self.type == Continental)
    First {
        /// Entity type.
        entity: EntityId,
        /// Filter predicate.
        predicate: Box<CompiledExpr>,
    },

    /// Nearest instance to position: nearest(entity.plate, position)
    Nearest {
        /// Entity type.
        entity: EntityId,
        /// Center position.
        position: Box<CompiledExpr>,
    },

    /// All instances within radius: within(entity.moon, pos, 1e9)
    Within {
        /// Entity type.
        entity: EntityId,
        /// Center position.
        position: Box<CompiledExpr>,
        /// Search radius.
        radius: Box<CompiledExpr>,
        /// Body expression.
        body: Box<CompiledExpr>,
    },
}

/// Aggregate operations over collections of entity instances.
///
/// These operations reduce a collection of values to a single value.
/// They are used with entity iteration constructs like `sum(entity.moon, ...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateOpIr {
    /// Sum of all values
    Sum,
    /// Product of all values
    Product,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Arithmetic mean
    Mean,
    /// Number of instances
    Count,
    /// True (1.0) if any value is non-zero
    Any,
    /// True (1.0) if all values are non-zero
    All,
    /// True (1.0) if no values are non-zero
    None,
}

/// Binary operators for two-operand expressions.
///
/// All comparison operators return 1.0 for true and 0.0 for false.
/// Logical operators treat any non-zero value as true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOpIr {
    /// Addition: `a + b`
    Add,
    /// Subtraction: `a - b`
    Sub,
    /// Multiplication: `a * b`
    Mul,
    /// Division: `a / b`
    Div,
    /// Exponentiation: `a ^ b`
    Pow,
    /// Equality: `a == b`
    Eq,
    /// Inequality: `a != b`
    Ne,
    /// Less than: `a < b`
    Lt,
    /// Less than or equal: `a <= b`
    Le,
    /// Greater than: `a > b`
    Gt,
    /// Greater than or equal: `a >= b`
    Ge,
    /// Logical AND: `a && b`
    And,
    /// Logical OR: `a || b`
    Or,
}

/// Unary operators for single-operand expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpIr {
    /// Numeric negation: `-x`
    Neg,
    /// Logical NOT: `!x` (returns 1.0 if x is 0.0, otherwise 0.0)
    Not,
}

/// dt-robust operators that provide numerically stable time integration.
///
/// These operators implement time-step invariant calculations that:
/// - Are deterministic (same inputs → same outputs)
/// - Are stable (bounded output at any reasonable dt)
/// - Are convergent (approach correct solution as dt → 0)
/// - Are symmetric (dt=0.1 twice equals dt=0.2 once)
///
/// Unlike raw expressions like `prev + rate * dt`, these operators handle
/// the implicit `dt` parameter correctly and can use higher-order methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtRobustOperator {
    /// Integrate a rate over time: `integrate(prev, rate)`
    /// Raw equivalent: `prev + rate * dt`
    Integrate,
    /// Exponential decay toward zero: `decay(value, halflife)`
    /// Raw equivalent: `prev * (1 - k * dt)` (but stable)
    Decay,
    /// Relax toward a target value: `relax(current, target, tau)`
    /// Raw equivalent: `prev + (target - prev) * k * dt`
    Relax,
    /// Bounded accumulation: `accumulate(prev, delta, min, max)`
    /// Raw equivalent: `clamp(prev + delta * dt, min, max)`
    Accumulate,
    /// Phase advancement with wrapping: `advance_phase(phase, omega)`
    /// Raw equivalent: `wrap(prev + omega * dt, 0, TAU)`
    AdvancePhase,
    /// Exponential smoothing: `smooth(prev, input, tau)`
    Smooth,
    /// Spring-damper system: `damp(pos, vel, target, stiffness, damping)`
    Damp,
}

/// Integration method for dt-robust operators.
///
/// Higher-order methods provide better accuracy at larger time steps
/// but require more computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IntegrationMethod {
    /// First-order Euler integration (default)
    #[default]
    Euler,
    /// Fourth-order Runge-Kutta
    Rk4,
    /// Velocity Verlet (for position-velocity systems)
    Verlet,
}

/// A compiled custom type definition.
///
/// Custom types allow users to define composite types for signals,
/// impulse payloads, and entity schemas. Each type consists of named
/// fields with specific value types.
///
/// # Example DSL
///
/// ```cdsl
/// type.PlateState {
///   position: Vec3<m>
///   velocity: Vec3<m/s>
///   strain: Tensor<3,3,Pa>
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CompiledType {
    /// Unique identifier for this type.
    pub id: TypeId,
    /// Named fields with their value types.
    pub fields: Vec<CompiledTypeField>,
}

/// A field within a compiled custom type.
#[derive(Debug, Clone)]
pub struct CompiledTypeField {
    /// Field name (e.g., "position", "velocity").
    pub name: String,
    /// The resolved value type for this field.
    pub value_type: ValueType,
}
