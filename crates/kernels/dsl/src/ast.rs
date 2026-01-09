//! Abstract Syntax Tree (AST) for the Continuum DSL.
//!
//! This module defines the typed representation of parsed DSL source code.
//! The AST preserves source spans for all nodes, enabling precise error
//! reporting and IDE hover text.
//!
//! # Structure
//!
//! A [`CompilationUnit`] contains a list of top-level [`Item`]s, which include:
//! - **Configuration**: [`ConstBlock`], [`ConfigBlock`] for compile-time values
//! - **Types**: [`TypeDef`] for custom type declarations
//! - **Functions**: [`FnDef`] for pure inlined expressions
//! - **Simulation Structure**: [`StrataDef`], [`EraDef`] for time organization
//! - **Signals**: [`SignalDef`] for authoritative state
//! - **Observation**: [`FieldDef`] for derived measurements
//! - **Operators**: [`OperatorDef`] for phase-specific logic
//! - **Events**: [`ImpulseDef`], [`FractureDef`], [`ChronicleDef`]
//! - **Collections**: [`EntityDef`] for indexed state
//!
//! # Span Tracking
//!
//! All nodes are wrapped in [`Spanned<T>`] which associates the AST node
//! with its byte range in the source file. This enables:
//! - Precise error messages pointing to exact source locations
//! - IDE features like go-to-definition and hover documentation
//! - Source mapping for compiled IR
//!
//! # Example
//!
//! ```ignore
//! use continuum_dsl::{parse, ast::*};
//!
//! let src = r#"
//!     signal.terra.temperature {
//!         : Scalar<K, 50..1000>
//!         : strata(terra.thermal)
//!         resolve { prev + 0.1 }
//!     }
//! "#;
//!
//! let (unit, errors) = parse(src);
//! let unit = unit.unwrap();
//! for item in &unit.items {
//!     if let Item::SignalDef(sig) = &item.node {
//!         println!("Signal: {}", sig.path.node);
//!     }
//! }
//! ```

use std::ops::Range as StdRange;

/// Source span representing a byte range in the source file.
///
/// Used for error reporting and source mapping.
pub type Span = StdRange<usize>;

/// A spanned AST node that associates a value with its source location.
///
/// All significant AST nodes are wrapped in `Spanned` to preserve their
/// position in the source file for error reporting and IDE features.
///
/// # Example
///
/// ```ignore
/// let path = Spanned::new(
///     Path::new(vec!["terra".into(), "temperature".into()]),
///     0..18
/// );
/// assert_eq!(path.node.to_string(), "terra.temperature");
/// assert_eq!(path.span, 0..18);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    /// The wrapped AST node.
    pub node: T,
    /// Byte range in source (start..end).
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Creates a new spanned node.
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// A complete DSL compilation unit representing a parsed source file.
///
/// Contains all top-level items defined in the source. Multiple compilation
/// units from different files are merged during world loading.
#[derive(Debug, Clone, Default)]
pub struct CompilationUnit {
    /// All top-level items in declaration order.
    pub items: Vec<Spanned<Item>>,
}

/// Top-level items that can appear in DSL source files.
///
/// Each variant corresponds to a distinct declaration type in the DSL.
/// Items are processed in a specific order during compilation regardless
/// of their declaration order in source.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// Compile-time constant definitions: `const { physics.g: 9.81 }`.
    ConstBlock(ConstBlock),
    /// Runtime configuration values: `config { thermal.tau: 1000.0 }`.
    ConfigBlock(ConfigBlock),
    /// Custom type declaration: `type Vec2 { x: Scalar<m>, y: Scalar<m> }`.
    TypeDef(TypeDef),
    /// User-defined function: `fn.math.lerp(a, b, t) { a + (b - a) * t }`.
    FnDef(FnDef),
    /// Time stratum definition: `strata.terra { : stride(10) }`.
    StrataDef(StrataDef),
    /// Era definition: `era.main { : initial : dt(1 <yr>) }`.
    EraDef(EraDef),
    /// Signal (authoritative state): `signal.terra.temp { resolve { prev } }`.
    SignalDef(SignalDef),
    /// Field (derived measurement): `field.terra.surface { measure { ... } }`.
    FieldDef(FieldDef),
    /// Operator (phase logic): `operator.terra.diffuse { collect { ... } }`.
    OperatorDef(OperatorDef),
    /// Impulse (external event): `impulse.stellar.flare { apply { ... } }`.
    ImpulseDef(ImpulseDef),
    /// Fracture (tension detector): `fracture.terra.quake { when { ... } }`.
    FractureDef(FractureDef),
    /// Chronicle (observer): `chronicle.stellar.events { observe { ... } }`.
    ChronicleDef(ChronicleDef),
    /// Entity (indexed collection): `entity.stellar.moon { schema { ... } }`.
    EntityDef(EntityDef),
}

/// Dot-separated path identifying a named entity in the DSL.
///
/// Paths are used for signal references (`signal.terra.temperature`),
/// function names (`fn.math.lerp`), strata (`strata.terra`), and other
/// namespaced identifiers.
///
/// # Example
///
/// ```ignore
/// let path = Path::new(vec!["terra".into(), "surface".into(), "temp".into()]);
/// assert_eq!(path.to_string(), "terra.surface.temp");
/// assert_eq!(path.join("/"), "terra/surface/temp");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    /// Individual path segments (e.g., `["terra", "surface", "temp"]`).
    pub segments: Vec<String>,
}

impl Path {
    /// Creates a new path from segments.
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    /// Joins segments with a custom separator.
    pub fn join(&self, sep: &str) -> String {
        self.segments.join(sep)
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.join("."))
    }
}

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
    /// Namespaced path (e.g., `terra.thermal.tau`).
    pub path: Spanned<Path>,
    /// Default value (can be overridden by scenario).
    pub value: Spanned<Literal>,
    /// Optional unit annotation.
    pub unit: Option<Spanned<String>>,
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

/// Type expression representing a value's shape and constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// Scalar value with unit and optional bounds: `Scalar<K, 0..1000>`.
    Scalar {
        /// Unit string (e.g., "K", "m/s", "W/m²").
        unit: String,
        /// Optional value bounds.
        range: Option<Range>,
    },
    /// Vector value: `Vec3<m>` (dimension, unit, optional magnitude bounds).
    Vector {
        /// Dimension (2 or 3).
        dim: u8,
        /// Component unit.
        unit: String,
        /// Optional magnitude bounds.
        magnitude: Option<Range>,
    },
    /// Reference to a named type: `OrbitalElements`.
    Named(String),
}

/// Numeric range for value bounds validation.
///
/// Used in type expressions to constrain valid values.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    /// Minimum allowed value (inclusive).
    pub min: f64,
    /// Maximum allowed value (inclusive).
    pub max: f64,
}

// === Functions ===

/// User-defined function declaration
///
/// Example: `fn.physics.stefan_boltzmann_loss(temp: Scalar<K>) -> Scalar<W/m²> { ... }`
#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    /// Function path (e.g., `physics.stefan_boltzmann_loss`)
    pub path: Spanned<Path>,
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
    /// Impulse path.
    pub path: Spanned<Path>,
    /// Type of data carried by the impulse.
    pub payload_type: Option<Spanned<TypeExpr>>,
    /// Impulse-local config.
    pub local_config: Vec<ConfigEntry>,
    /// Application logic when impulse fires.
    pub apply: Option<ApplyBlock>,
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
    /// Fracture path.
    pub path: Spanned<Path>,
    /// Trigger conditions.
    pub conditions: Vec<Spanned<Expr>>,
    /// Emissions when triggered.
    pub emit: Vec<EmitStatement>,
}

/// Statement emitting an impulse or event when fracture triggers.
#[derive(Debug, Clone, PartialEq)]
pub struct EmitStatement {
    /// Target impulse or event path.
    pub target: Spanned<Path>,
    /// Value to emit.
    pub value: Spanned<Expr>,
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
///         when { signal.terra.temp < 273.0 } {
///             event.ice_age {
///                 temp: signal.terra.temp
///                 tick: tick
///             }
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChronicleDef {
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

// === Entity ===

/// Entity definition - a named, indexed collection of structured state
///
/// Example:
/// ```cdsl
/// entity.stellar.moon {
///   : strata(stellar.orbital)
///   : count(config.stellar.moon_count)
///   : count(1..20)
///
///   schema {
///     mass: Scalar<kg, 1e18..1e24>
///     radius: Scalar<m, 1e5..1e7>
///   }
///
///   resolve {
///     self.velocity = integrate(self.velocity, acceleration)
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EntityDef {
    /// Entity path (e.g., `stellar.moon`)
    pub path: Spanned<Path>,
    /// Stratum binding
    pub strata: Option<Spanned<Path>>,
    /// Count source from config (e.g., `config.stellar.moon_count`)
    pub count_source: Option<Spanned<Path>>,
    /// Count validation bounds (e.g., `1..20`)
    pub count_bounds: Option<CountBounds>,
    /// Schema fields for each instance
    pub schema: Vec<EntitySchemaField>,
    /// Default config values for schema fields
    pub config_defaults: Vec<ConfigEntry>,
    /// Resolution logic (executed per instance)
    pub resolve: Option<ResolveBlock>,
    /// Entity-level assertions
    pub assertions: Option<AssertBlock>,
    /// Nested field definitions for observation
    pub fields: Vec<EntityFieldDef>,
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

/// A field in an entity schema defining per-instance state.
#[derive(Debug, Clone, PartialEq)]
pub struct EntitySchemaField {
    /// Field name (e.g., `mass`, `position`).
    pub name: Spanned<String>,
    /// Field type with constraints.
    pub ty: Spanned<TypeExpr>,
}

/// A field definition nested within an entity for observation.
///
/// These fields exist purely for measurement and visualization,
/// not for simulation state.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityFieldDef {
    /// Field name.
    pub name: Spanned<String>,
    /// Optional type annotation.
    pub ty: Option<Spanned<TypeExpr>>,
    /// Spatial topology for reconstruction.
    pub topology: Option<Spanned<Topology>>,
    /// Measurement expression.
    pub measure: Option<MeasureBlock>,
}

// === Expressions ===

/// Expression node representing computations and data flow.
///
/// Expressions form the core of DSL computation. They appear in resolve
/// blocks, assertions, function bodies, and anywhere values are computed.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value: `42`, `3.14`, `"hello"`, `true`.
    Literal(Literal),

    /// Literal with unit annotation: `1000 <yr>`, `288.15 <K>`.
    LiteralWithUnit {
        /// The numeric or string value.
        value: Literal,
        /// Unit string (e.g., "yr", "K", "m/s").
        unit: String,
    },

    /// Unqualified path reference: `foo.bar.baz`.
    Path(Path),

    /// Previous tick's value of the current signal: `prev`.
    Prev,

    /// Previous tick's value of a specific field: `prev.temperature`.
    PrevField(String),

    /// Raw (unscaled) time step for dt-robust expressions: `dt_raw`.
    DtRaw,

    /// Impulse payload in apply blocks: `payload`.
    Payload,

    /// Field access on impulse payload: `payload.magnitude`.
    PayloadField(String),

    /// Explicit signal reference: `signal.terra.temperature`.
    SignalRef(Path),

    /// Constant reference: `const.physics.G`.
    ConstRef(Path),

    /// Config value reference: `config.terra.thermal.tau`.
    ConfigRef(Path),

    /// Field reference (observation): `field.terra.surface.temp`.
    FieldRef(Path),

    /// Binary operation: `a + b`, `x * y`, `p && q`.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// Left operand.
        left: Box<Spanned<Expr>>,
        /// Right operand.
        right: Box<Spanned<Expr>>,
    },

    /// Unary operation: `-x`, `!flag`.
    Unary {
        /// The operator.
        op: UnaryOp,
        /// Operand expression.
        operand: Box<Spanned<Expr>>,
    },

    /// Function call: `lerp(a, b, t)`, `sin(angle)`.
    Call {
        /// Function being called.
        function: Box<Spanned<Expr>>,
        /// Call arguments.
        args: Vec<Spanned<Expr>>,
    },

    /// Method call on an object: `vec.normalize()`, `signal.clamp(0, 1)`.
    MethodCall {
        /// Object receiving the method call.
        object: Box<Spanned<Expr>>,
        /// Method name.
        method: String,
        /// Method arguments.
        args: Vec<Spanned<Expr>>,
    },

    /// Field access on a struct: `orbital.semi_major`, `state.position`.
    FieldAccess {
        /// Object to access.
        object: Box<Spanned<Expr>>,
        /// Field name.
        field: String,
    },

    /// Local binding: `let x = expr in body`.
    Let {
        /// Variable name.
        name: String,
        /// Value to bind.
        value: Box<Spanned<Expr>>,
        /// Body where binding is visible.
        body: Box<Spanned<Expr>>,
    },

    /// Conditional expression: `if cond { then } else { else }`.
    If {
        /// Condition to test.
        condition: Box<Spanned<Expr>>,
        /// Expression if true.
        then_branch: Box<Spanned<Expr>>,
        /// Optional expression if false.
        else_branch: Option<Box<Spanned<Expr>>>,
    },

    /// For loop over a sequence: `for x in items { body }`.
    For {
        /// Loop variable name.
        var: String,
        /// Sequence to iterate.
        iter: Box<Spanned<Expr>>,
        /// Loop body.
        body: Box<Spanned<Expr>>,
    },

    /// Block of sequential expressions: `{ expr1; expr2; result }`.
    Block(Vec<Spanned<Expr>>),

    /// Emit a value to a signal: `emit(signal.terra.temp, value)`.
    EmitSignal {
        /// Target signal path.
        target: Path,
        /// Value to emit.
        value: Box<Spanned<Expr>>,
    },

    /// Emit a positioned sample to a field: `emit_field(field.surface, pos, value)`.
    EmitField {
        /// Target field path.
        target: Path,
        /// Position for the sample.
        position: Box<Spanned<Expr>>,
        /// Value at that position.
        value: Box<Spanned<Expr>>,
    },

    /// Struct literal: `{ x: 1.0, y: 2.0 }`.
    Struct(Vec<(String, Spanned<Expr>)>),

    /// Accumulated inputs from Collect phase: `collected`.
    Collected,

    /// Mathematical constant: `pi`, `tau`, `e`.
    MathConst(MathConst),

    /// Map function over sequence: `map(items, fn(x) { x * 2 })`.
    Map {
        /// Sequence to map over.
        sequence: Box<Spanned<Expr>>,
        /// Function to apply.
        function: Box<Spanned<Expr>>,
    },

    /// Fold/reduce sequence: `fold(items, 0, fn(acc, x) { acc + x })`.
    Fold {
        /// Sequence to fold.
        sequence: Box<Spanned<Expr>>,
        /// Initial accumulator value.
        init: Box<Spanned<Expr>>,
        /// Folding function.
        function: Box<Spanned<Expr>>,
    },

    // === Entity expressions ===

    /// Reference to current entity instance field: `self.mass`.
    SelfField(String),

    /// Reference to an entity type: `entity.stellar.moon`.
    EntityRef(Path),

    /// Access entity instance by ID: `entity.moon["luna"]`.
    EntityAccess {
        /// Entity type path.
        entity: Path,
        /// Instance identifier expression.
        instance: Box<Spanned<Expr>>,
    },

    /// Aggregate operation over entity instances: `sum(entity.moon, self.mass)`.
    Aggregate {
        /// Aggregation operator.
        op: AggregateOp,
        /// Entity type to aggregate over.
        entity: Path,
        /// Expression evaluated per instance.
        body: Box<Spanned<Expr>>,
    },

    /// Other instances excluding current: `other(entity.moon)`.
    /// Used for N-body interactions where you need all instances except self.
    Other(Path),

    /// Pairwise iteration: `pairs(entity.moon)`.
    /// Generates all unique (i,j) combinations where i < j.
    Pairs(Path),

    /// Filter entity instances by predicate: `filter(entity.moon, self.mass > 1e20)`.
    Filter {
        /// Entity type to filter.
        entity: Path,
        /// Predicate expression.
        predicate: Box<Spanned<Expr>>,
    },

    /// First instance matching predicate: `first(entity.plate, self.type == Continental)`.
    First {
        /// Entity type to search.
        entity: Path,
        /// Predicate to match.
        predicate: Box<Spanned<Expr>>,
    },

    /// Nearest instance to position: `nearest(entity.plate, position)`.
    Nearest {
        /// Entity type to search.
        entity: Path,
        /// Position to measure from.
        position: Box<Spanned<Expr>>,
    },

    /// All instances within radius: `within(entity.moon, pos, 1e9)`.
    Within {
        /// Entity type to search.
        entity: Path,
        /// Center position.
        position: Box<Spanned<Expr>>,
        /// Search radius.
        radius: Box<Spanned<Expr>>,
    },
}

/// Literal value in the DSL.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// Integer literal: `42`, `-7`, `1000000`.
    Integer(i64),
    /// Floating-point literal: `3.14`, `1e-6`, `2.998e8`.
    Float(f64),
    /// String literal: `"hello"`, `"temperature"`.
    String(String),
    /// Boolean literal: `true`, `false`.
    Bool(bool),
}

/// Mathematical constants available in the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathConst {
    /// Pi (3.14159...): ratio of circumference to diameter.
    Pi,
    /// Tau (6.28318...): ratio of circumference to radius (2*pi).
    Tau,
    /// Euler's number (2.71828...): base of natural logarithm.
    E,
    /// Imaginary unit for complex numbers.
    I,
    /// Golden ratio (1.61803...): (1 + sqrt(5)) / 2.
    Phi,
}

/// Binary operators for arithmetic, comparison, and logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: `a + b`.
    Add,
    /// Subtraction: `a - b`.
    Sub,
    /// Multiplication: `a * b`.
    Mul,
    /// Division: `a / b`.
    Div,
    /// Exponentiation: `a ^ b` or `a ** b`.
    Pow,
    /// Equality: `a == b`.
    Eq,
    /// Inequality: `a != b`.
    Ne,
    /// Less than: `a < b`.
    Lt,
    /// Less than or equal: `a <= b`.
    Le,
    /// Greater than: `a > b`.
    Gt,
    /// Greater than or equal: `a >= b`.
    Ge,
    /// Logical and: `a && b`.
    And,
    /// Logical or: `a || b`.
    Or,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Numeric negation: `-x`.
    Neg,
    /// Logical not: `!x`.
    Not,
}

/// Aggregate operations over entity instances
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// Sum of values: `sum(entity.moon, self.mass)`
    Sum,
    /// Product of values: `product(entity.layer, self.transmittance)`
    Product,
    /// Minimum value: `min(entity.moon, self.orbit_radius)`
    Min,
    /// Maximum value: `max(entity.star, self.luminosity)`
    Max,
    /// Average value: `mean(entity.plate, self.age)`
    Mean,
    /// Count of instances: `count(entity.moon)`
    Count,
    /// Any instance matches predicate: `any(entity.moon, self.mass > 1e22)`
    Any,
    /// All instances match predicate: `all(entity.star, self.luminosity > 0)`
    All,
    /// No instance matches predicate: `none(entity.plate, self.age < 0)`
    None,
}
