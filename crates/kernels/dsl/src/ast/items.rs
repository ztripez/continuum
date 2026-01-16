//! Item definition types for the Continuum DSL AST.
//!
//! This module defines all top-level declaration types that can appear
//! in DSL source files: configuration blocks, signals, fields, operators,
//! entities, and other simulation structure definitions.

use super::expr::{Expr, Literal};
use super::{Path, Spanned, TypeExpr};

// === Const/Config ===

/// Block of compile-time constant definitions.
///
/// Constants are immutable values resolved at compile time. They cannot
/// be changed by scenarios or at runtime.
///
/// # DSL Syntax
///
/// ```cdsl
/// const {
///     physics.gravitational: 6.674e-11
///     physics.stefan_boltzmann: 5.670e-8 <W/m²/K⁴>
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ConstBlock {
    /// Individual constant definitions.
    pub entries: Vec<ConstEntry>,
}

/// A single constant definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstEntry {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Namespaced path (e.g., `physics.gravitational`).
    pub path: Spanned<Path>,
    /// Constant value (must be a literal).
    pub value: Spanned<Literal>,
    /// Optional unit annotation (e.g., `W/m²/K⁴`).
    pub unit: Option<Spanned<String>>,
}

/// Block of runtime configuration values.
///
/// Config values can be overridden by scenarios and provide tunable
/// parameters for simulation behavior.
///
/// # DSL Syntax
///
/// ```cdsl
/// config {
///     terra.thermal.tau: 1000.0 <s>
///     terra.initial_temp: 288.0 <K>
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigBlock {
    /// Individual config definitions.
    pub entries: Vec<ConfigEntry>,
}

/// A single config definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigEntry {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Namespaced path (e.g., `terra.thermal.tau`).
    pub path: Spanned<Path>,
    /// Default value (can be overridden by scenario).
    pub value: Spanned<Literal>,
    /// Optional unit annotation.
    pub unit: Option<Spanned<String>>,
}

// === World ===

/// World definition replacing the world.yaml manifest.
///
/// # DSL Syntax
///
/// ```cdsl
/// world.terra {
///     : title("Earth Planetary Simulation")
///     : version("1.0.0")
///
///     policy {
///         determinism: strict
///         faults: fatal
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WorldDef {
    /// World path (e.g., `world.terra`).
    pub path: Spanned<Path>,
    /// Human-readable title.
    pub title: Option<Spanned<String>>,
    /// Version string.
    pub version: Option<Spanned<String>>,
    /// Policy configuration.
    pub policy: Option<PolicyBlock>,
}

/// Block of execution policy configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyBlock {
    /// Individual policy definitions.
    pub entries: Vec<ConfigEntry>,
}

// === Types ===

/// Custom type definition for structured values.
///
/// # DSL Syntax
///
/// ```cdsl
/// type OrbitalElements {
///     semi_major: Scalar<m>
///     eccentricity: Scalar<1, 0..1>
///     inclination: Scalar<rad>
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Type name (e.g., `OrbitalElements`).
    pub name: Spanned<String>,
    /// Struct fields.
    pub fields: Vec<TypeField>,
}

/// A field within a custom type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeField {
    /// Field name.
    pub name: Spanned<String>,
    /// Field type expression.
    pub ty: Spanned<TypeExpr>,
}

// === Functions ===

/// User-defined function declaration
///
/// Example: `fn.physics.stefan_boltzmann_loss(temp: Scalar<K>) -> Scalar<W/m²> { ... }`
#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Function path (e.g., `physics.stefan_boltzmann_loss`)
    pub path: Spanned<Path>,
    /// Generic type parameters (e.g. `<T>`)
    pub generics: Vec<Spanned<String>>,
    /// Function parameters
    pub params: Vec<FnParam>,
    /// Return type (optional, can be inferred)
    pub return_type: Option<Spanned<TypeExpr>>,
    /// Function body expression
    pub body: Spanned<Expr>,
}

/// A function parameter
#[derive(Debug, Clone, PartialEq)]
pub struct FnParam {
    /// Parameter name
    pub name: Spanned<String>,
    /// Parameter type (optional)
    pub ty: Option<Spanned<TypeExpr>>,
}

// === Strata ===

/// Stratum definition for organizing simulation time.
///
/// Strata group signals by their temporal resolution and update frequency.
/// Each stratum can have a different stride (how many base ticks between updates).
///
/// # DSL Syntax
///
/// ```cdsl
/// strata.terra.thermal {
///     : title("Thermal Dynamics")
///     : symbol("🌡")
///     : stride(10)
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StrataDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Stratum path (e.g., `terra.thermal`).
    pub path: Spanned<Path>,
    /// Human-readable title for display.
    pub title: Option<Spanned<String>>,
    /// Unicode symbol for visualization.
    pub symbol: Option<Spanned<String>>,
    /// Default stride (ticks between updates).
    pub stride: Option<Spanned<u32>>,
}

// === Era ===

/// Era definition controlling simulation phases.
///
/// Eras define distinct simulation phases with different time steps,
/// active strata, and transition conditions. One era must be marked
/// as initial; terminal eras end the simulation.
///
/// # DSL Syntax
///
/// ```cdsl
/// era.formation {
///     : initial
///     : title("Planet Formation")
///     : dt(1000 <yr>)
///     : strata(terra.thermal, active)
///     : strata(terra.orbital, gated)
///
///     transition(stable) {
///         when { signal.terra.age > 4.5e9 <yr> }
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EraDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Era identifier.
    pub name: Spanned<String>,
    /// Whether this is the starting era.
    pub is_initial: bool,
    /// Whether this era ends simulation.
    pub is_terminal: bool,
    /// Human-readable title.
    pub title: Option<Spanned<String>>,
    /// Time step for this era.
    pub dt: Option<Spanned<ValueWithUnit>>,
    /// Config overrides active during this era.
    pub config_overrides: Vec<ConfigEntry>,
    /// Strata activation states.
    pub strata_states: Vec<StrataState>,
    /// Transitions to other eras.
    pub transitions: Vec<Transition>,
}

/// A literal value with its unit annotation.
///
/// Used for typed numeric literals like `1000 <yr>` or `288.15 <K>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueWithUnit {
    /// The numeric value.
    pub value: Literal,
    /// Unit string (e.g., "yr", "K", "m/s").
    pub unit: String,
}

/// Stratum activation state within an era.
#[derive(Debug, Clone, PartialEq)]
pub struct StrataState {
    /// Reference to the stratum.
    pub strata: Spanned<Path>,
    /// Activation state.
    pub state: StrataStateKind,
}

/// How a stratum behaves within an era.
#[derive(Debug, Clone, PartialEq)]
pub enum StrataStateKind {
    /// Stratum executes every tick (respecting its stride).
    Active,
    /// Stratum executes with overridden stride.
    ActiveWithStride(u32),
    /// Stratum is suspended (signals frozen).
    Gated,
}

/// Era transition definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// Target era to transition to.
    pub target: Spanned<Path>,
    /// Conditions that trigger the transition (all must be true).
    pub conditions: Vec<Spanned<Expr>>,
}

// === Signal ===

/// Signal definition for authoritative simulation state.
///
/// Signals are the core building blocks of simulation. Each signal has
/// a value that evolves over time according to its resolve expression.
/// Signals can reference other signals, constants, and config values.
///
/// # DSL Syntax
///
/// ```cdsl
/// signal.terra.surface.temperature {
///     : Scalar<K, 50..1000>
///     : strata(terra.thermal)
///     : title("Surface Temperature")
///     : symbol("T")
///
///     config {
///         initial: 288.0 <K>
///     }
///
///     resolve {
///         relax(prev, equilibrium_temp, config.terra.surface.temperature.tau)
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SignalDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Signal path (e.g., `terra.surface.temperature`).
    pub path: Spanned<Path>,
    /// Value type with optional bounds.
    pub ty: Option<Spanned<TypeExpr>>,
    /// Stratum binding for scheduling.
    pub strata: Option<Spanned<Path>>,
    /// Human-readable title.
    pub title: Option<Spanned<String>>,
    /// Unicode symbol for display.
    pub symbol: Option<Spanned<String>>,
    /// Whether `dt_raw` is explicitly declared via `: uses(dt_raw)`.
    pub dt_raw: bool,
    /// Uses declarations (e.g., ["dt.raw", "maths.clamping"]).
    /// Generic uses() declarations beyond the legacy dt_raw flag.
    pub uses: Vec<String>,
    /// Signal-local constants.
    pub local_consts: Vec<ConstEntry>,
    /// Signal-local config with defaults.
    pub local_config: Vec<ConfigEntry>,
    /// Optional warmup block for initial convergence.
    pub warmup: Option<WarmupBlock>,
    /// Resolution expression evaluated each tick.
    pub resolve: Option<ResolveBlock>,
    /// Assertions validated after resolution.
    pub assertions: Option<AssertBlock>,
    /// Tensor constraints declared as child clauses (e.g., `: symmetric`).
    /// These are validated against the type during lowering.
    pub tensor_constraints: Vec<super::TensorConstraint>,
    /// Sequence constraints declared as child clauses (e.g., `: each(0..1)`).
    /// These are validated against the type during lowering.
    pub seq_constraints: Vec<super::SeqConstraint>,
}

/// Warmup block for iterative signal initialization.
///
/// Runs multiple iterations before simulation starts to reach
/// a stable initial state.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmupBlock {
    /// Maximum iterations to run.
    pub iterations: Spanned<u32>,
    /// Optional convergence threshold to stop early.
    pub convergence: Option<Spanned<f64>>,
    /// Expression evaluated each warmup iteration.
    pub iterate: Spanned<Expr>,
}

/// Resolution block containing the expression to evaluate each tick.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveBlock {
    /// Expression that produces the new signal value.
    pub body: Spanned<Expr>,
}

// === Assertions ===

/// An assertion block containing one or more assertions
#[derive(Debug, Clone, PartialEq)]
pub struct AssertBlock {
    /// The individual assertions within the block.
    pub assertions: Vec<Assertion>,
}

/// A single assertion
#[derive(Debug, Clone, PartialEq)]
pub struct Assertion {
    /// The condition that must be true
    pub condition: Spanned<Expr>,
    /// Optional severity level (defaults to Error)
    pub severity: AssertSeverity,
    /// Optional message to emit on failure
    pub message: Option<Spanned<String>>,
}

/// Severity of an assertion failure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssertSeverity {
    /// Warning only, execution continues
    Warn,
    /// Error, may halt based on policy
    #[default]
    Error,
    /// Fatal, always halts
    Fatal,
}

// === Field ===

/// Field definition for observable derived measurements.
///
/// Fields are computed during the Measure phase and emitted for observation.
/// They do not affect simulation causality and exist purely for visualization
/// and analysis.
///
/// # DSL Syntax
///
/// ```cdsl
/// field.terra.surface.heat_flux {
///     : Scalar<W/m²>
///     : strata(terra.thermal)
///     : topology(SphereSurface)
///
///     measure {
///         emit(position, signal.terra.surface.flux_out)
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Field path.
    pub path: Spanned<Path>,
    /// Value type at each sample point.
    pub ty: Option<Spanned<TypeExpr>>,
    /// Stratum binding.
    pub strata: Option<Spanned<Path>>,
    /// Spatial topology for reconstruction.
    pub topology: Option<Spanned<Topology>>,
    /// Human-readable title.
    pub title: Option<Spanned<String>>,
    /// Symbol for display.
    pub symbol: Option<Spanned<String>>,
    /// Measurement expression emitting samples.
    pub measure: Option<MeasureBlock>,
}

/// Spatial topology for field reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Topology {
    /// Samples on a sphere surface (e.g., planetary surface).
    SphereSurface,
    /// Discrete point samples (e.g., entity positions).
    PointCloud,
    /// Volumetric samples (e.g., atmospheric layers).
    Volume,
}

/// Measure block containing the field sampling expression.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureBlock {
    /// Expression that emits field samples.
    pub body: Spanned<Expr>,
}

// === Operator ===

/// Operator definition for phase-specific logic.
///
/// Operators execute during specific phases and can aggregate inputs,
/// produce side effects, or coordinate multi-signal behaviors.
///
/// # DSL Syntax
///
/// ```cdsl
/// operator.terra.heat_diffusion {
///     : strata(terra.thermal)
///     : phase(collect)
///
///     collect {
///         // Gather heat flux from neighbors
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Operator path.
    pub path: Spanned<Path>,
    /// Stratum binding.
    pub strata: Option<Spanned<Path>>,
    /// Execution phase.
    pub phase: Option<Spanned<OperatorPhase>>,
    /// Phase-specific body.
    pub body: Option<OperatorBody>,
    /// Operator assertions.
    pub assertions: Option<AssertBlock>,
    /// Dangerous function uses declarations (e.g., "dt.raw", "maths.clamping").
    pub uses: Vec<String>,
}

/// Execution phase for operators.
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorPhase {
    /// Runs during initialization warmup.
    Warmup,
    /// Runs during signal input collection.
    Collect,
    /// Runs during field measurement emission.
    Measure,
}

/// Phase-specific operator body.
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorBody {
    /// Warmup phase expression.
    Warmup(Spanned<Expr>),
    /// Collect phase expression.
    Collect(Spanned<Expr>),
    /// Measure phase expression.
    Measure(Spanned<Expr>),
}

// === Impulse ===

/// Impulse definition for external causal events.
///
/// Impulses represent external inputs that can modify signal state,
/// such as user actions, API calls, or scenario-triggered events.
///
/// # DSL Syntax
///
/// ```cdsl
/// impulse.stellar.flare {
///     : payload(Scalar<W/m²>)
///
///     apply {
///         signal.terra.surface.flux_in += payload
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ImpulseDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Impulse path.
    pub path: Spanned<Path>,
    /// Type of data carried by the impulse.
    pub payload_type: Option<Spanned<TypeExpr>>,
    /// Human-readable title.
    pub title: Option<Spanned<String>>,
    /// Symbol for display.
    pub symbol: Option<Spanned<String>>,
    /// Impulse-local config.
    pub local_config: Vec<ConfigEntry>,
    /// Application logic when impulse fires.
    pub apply: Option<ApplyBlock>,
    /// Dangerous function uses declarations (e.g., "dt.raw", "maths.clamping").
    pub uses: Vec<String>,
}

/// Apply block containing impulse application logic.
#[derive(Debug, Clone, PartialEq)]
pub struct ApplyBlock {
    /// Expression executed when impulse fires.
    pub body: Spanned<Expr>,
}

// === Fracture ===

/// Fracture definition for tension detection and response.
///
/// Fractures detect when physical constraints are violated and emit
/// events or impulses in response (e.g., earthquakes from plate stress).
///
/// # DSL Syntax
///
/// ```cdsl
/// fracture.terra.quake {
///     when {
///         signal.terra.plate.stress > config.terra.quake.threshold
///     }
///     emit {
///         impulse.terra.seismic(magnitude_from_stress(signal.terra.plate.stress))
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FractureDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Fracture path.
    pub path: Spanned<Path>,
    /// Stratum this fracture belongs to.
    pub strata: Option<Spanned<Path>>,
    /// Fracture-local config with defaults.
    pub local_config: Vec<ConfigEntry>,
    /// Trigger conditions.
    pub conditions: Vec<Spanned<Expr>>,
    /// Emit expression when triggered. Contains signal emit expressions.
    pub emit: Option<Spanned<Expr>>,
    /// Dangerous function uses declarations (e.g., "dt.raw", "maths.clamping").
    pub uses: Vec<String>,
}

// === Chronicle ===

/// Chronicle definition for observer-only event recording.
///
/// Chronicles watch simulation state and emit events for logging,
/// analytics, or user notification. They cannot affect causality.
///
/// # DSL Syntax
///
/// ```cdsl
/// chronicle.stellar.events {
///     observe {
///         when signal.terra.temp < 273.0 {
///             emit event.ice_age {
///                 temp: signal.terra.temp
///                 tick: tick
///             }
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChronicleDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Chronicle path.
    pub path: Spanned<Path>,
    /// Observation handlers.
    pub observe: Option<ObserveBlock>,
}

/// Block of observation handlers.
#[derive(Debug, Clone, PartialEq)]
pub struct ObserveBlock {
    /// Individual event handlers.
    pub handlers: Vec<ObserveHandler>,
}

/// Handler that emits an event when a condition is met.
#[derive(Debug, Clone, PartialEq)]
pub struct ObserveHandler {
    /// Trigger condition.
    pub condition: Spanned<Expr>,
    /// Event name to emit.
    pub event_name: Spanned<Path>,
    /// Event payload fields.
    pub event_fields: Vec<(Spanned<String>, Spanned<Expr>)>,
}

// === Member Signal ===

/// Member signal definition - per-entity authoritative state as top-level primitive.
///
/// Member signals are signals indexed over an entity's index space, with each
/// instance having its own value. Unlike embedded entity schema fields, member
/// signals are declared separately as top-level primitives with their own
/// resolve blocks. This enables:
///
/// - **Multi-rate scheduling**: Different member signals on the same entity
///   can be in different strata
/// - **Independent dependencies**: Each member signal has its own reads
/// - **Snapshot semantics**: All `self.*` reads see previous tick values
///
/// # DSL Syntax
///
/// ```cdsl
/// member.human.person.age {
///     : Scalar
///     : strata(human.physiology)
///     initial { 0.0 }
///     resolve { integrate(prev, 1) }
/// }
///
/// member.stellar.star.rotation_period {
///     : Scalar<day, 0.1..100>
///     : strata(stellar.activity)
///     initial { config.stellar.default_rotation_period_days }
///     resolve { prev }
/// }
/// ```
///
/// # Path Structure
///
/// The member path follows the pattern `entity_path.signal_name`:
/// - `human.person.age` → entity `human.person`, signal `age`
/// - `stellar.moon.velocity` → entity `stellar.moon`, signal `velocity`
#[derive(Debug, Clone, PartialEq)]
pub struct MemberDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Full member path (e.g., `human.person.age`).
    /// The last segment is the signal name, preceding segments form the entity path.
    pub path: Spanned<Path>,
    /// Value type with optional bounds.
    pub ty: Option<Spanned<TypeExpr>>,
    /// Stratum binding for scheduling.
    pub strata: Option<Spanned<Path>>,
    /// Human-readable title.
    pub title: Option<Spanned<String>>,
    /// Unicode symbol for display.
    pub symbol: Option<Spanned<String>>,
    /// Whether member uses raw dt (`: uses(dt.raw)`).
    pub dt_raw: bool,
    /// Uses declarations (e.g., ["dt.raw", "maths.clamping"]).
    /// Generic uses() declarations beyond the legacy dt_raw flag.
    pub uses: Vec<String>,
    /// Member-local config with defaults.
    pub local_config: Vec<ConfigEntry>,
    /// Initial value expression (evaluated once at entity creation).
    pub initial: Option<ResolveBlock>,
    /// Resolution expression evaluated each tick (per instance).
    pub resolve: Option<ResolveBlock>,
    /// Assertions validated after resolution.
    pub assertions: Option<AssertBlock>,
}

// === Entity ===

/// Entity definition - defines an index space over which member signals are indexed.
///
/// Entities are pure index spaces that provide:
/// - **Identity**: How instances are identified (key type)
/// - **Ordering**: Deterministic instance ordering
/// - **Lifecycle**: Instance count from config
///
/// Entities do NOT define dynamics - that's the role of member signals.
///
/// # DSL Syntax
///
/// ```cdsl
/// entity.human.person {
///     : key(String)
///     : count(config.human.pop_size)
///     : count(1..1000)
/// }
/// ```
///
/// Member signals are then declared separately:
///
/// ```cdsl
/// member.human.person.age {
///     : Scalar
///     : strata(human.physiology)
///     resolve { prev + 1 }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EntityDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Entity path (e.g., `stellar.moon`)
    pub path: Spanned<Path>,
    /// Count source from config (e.g., `config.stellar.moon_count`)
    pub count_source: Option<Spanned<Path>>,
    /// Count validation bounds (e.g., `1..20`)
    pub count_bounds: Option<CountBounds>,
}

/// Count validation bounds for entity instances.
///
/// Ensures the entity instance count from config falls within valid bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct CountBounds {
    /// Minimum required instances.
    pub min: u32,
    /// Maximum allowed instances.
    pub max: u32,
}

// === Analyzer ===

/// Analyzer definition for post-hoc field analysis.
///
/// Analyzers are pure observers that compute derived analysis results from field
/// snapshots captured during simulation. They run post-hoc (after simulation ends)
/// on field data and produce structured JSON results with optional validation checks.
///
/// # DSL Syntax
///
/// ```cdsl
/// analyzer terra.hypsometric_integral {
///     : doc "Land/ocean ratio and elevation distribution"
///     : requires(fields: [geophysics.elevation])
///     
///     : compute {
///         let samples = field.samples(geophysics.elevation)
///         let above_sea = samples.filter(|s| s.value > 0.0).count()
///         let total = samples.count()
///         let integral = above_sea / total
///         
///         emit {
///             integral: integral,
///             land_fraction: integral,
///             ocean_fraction: 1.0 - integral,
///             sample_count: total
///         }
///     }
///     
///     : validate {
///         check integral in 0.2..0.4
///             : severity(warning)
///             : message("Land fraction {integral*100:.1}%")
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzerDef {
    /// Documentation comment from `///` lines.
    pub doc: Option<String>,
    /// Analyzer path (e.g., `terra.hypsometric_integral`).
    pub path: Spanned<Path>,
    /// Field dependencies (fields required for this analyzer).
    pub required_fields: Vec<Spanned<Path>>,
    /// Compute expression that produces the result.
    pub compute: Option<Spanned<Expr>>,
    /// Validation checks (optional).
    pub validations: Vec<ValidationCheck>,
}

/// A single validation check in an analyzer.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationCheck {
    /// The condition to check.
    pub condition: Spanned<Expr>,
    /// Severity level (error, warning, or info).
    pub severity: Severity,
    /// Optional message template for the check.
    pub message: Option<Spanned<String>>,
}

/// Severity level for validation checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Critical failure.
    Error,
    /// Potential issue worth investigating.
    Warning,
    /// Informational note.
    Info,
}
