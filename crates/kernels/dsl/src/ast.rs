//! Abstract Syntax Tree for Continuum DSL

use std::ops::Range as StdRange;

/// Source location as a byte range into the source text.
///
/// Used to map AST nodes back to their original source for error reporting.
pub type Span = StdRange<usize>;

/// An AST node with its source location attached.
///
/// Wraps any AST type `T` with the span where it appeared in the source code.
/// This enables precise error messages pointing to the exact location of issues.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    /// The AST node payload.
    pub node: T,
    /// The byte range in source where this node appeared.
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// The root AST node representing a parsed DSL source file.
///
/// A compilation unit contains all top-level declarations from a single
/// `.cdsl` file. Multiple units are merged when loading a world.
#[derive(Debug, Clone, Default)]
pub struct CompilationUnit {
    /// All top-level items declared in this file.
    pub items: Vec<Spanned<Item>>,
}

/// Top-level DSL declarations that can appear in a source file.
///
/// Each variant represents a different kind of declaration:
/// - Configuration: [`ConstBlock`], [`ConfigBlock`], [`TypeDef`]
/// - Structure: [`StrataDef`], [`EraDef`], [`EntityDef`]
/// - Signals: [`SignalDef`], [`FieldDef`], [`OperatorDef`]
/// - Events: [`ImpulseDef`], [`FractureDef`], [`ChronicleDef`]
/// - Functions: [`FnDef`]
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    ConstBlock(ConstBlock),
    ConfigBlock(ConfigBlock),
    TypeDef(TypeDef),
    FnDef(FnDef),
    StrataDef(StrataDef),
    EraDef(EraDef),
    SignalDef(SignalDef),
    FieldDef(FieldDef),
    OperatorDef(OperatorDef),
    ImpulseDef(ImpulseDef),
    FractureDef(FractureDef),
    ChronicleDef(ChronicleDef),
    EntityDef(EntityDef),
}

/// A dot-separated identifier path for referencing DSL entities.
///
/// Paths are used throughout the DSL to name signals, strata, config values,
/// and other entities. Examples: `terra.surface.temp`, `config.dt`, `const.physics.gravity`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    /// The individual name segments, e.g., `["terra", "surface", "temp"]`.
    pub segments: Vec<String>,
}

impl Path {
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

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

#[derive(Debug, Clone, PartialEq)]
pub struct ConstBlock {
    pub entries: Vec<ConstEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstEntry {
    pub path: Spanned<Path>,
    pub value: Spanned<Literal>,
    pub unit: Option<Spanned<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigBlock {
    pub entries: Vec<ConfigEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigEntry {
    pub path: Spanned<Path>,
    pub value: Spanned<Literal>,
    pub unit: Option<Spanned<String>>,
}

// === Types ===

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: Spanned<String>,
    pub fields: Vec<TypeField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeField {
    pub name: Spanned<String>,
    pub ty: Spanned<TypeExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    Scalar { unit: String, range: Option<Range> },
    Vector { dim: u8, unit: String, magnitude: Option<Range> },
    Named(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub min: f64,
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

#[derive(Debug, Clone, PartialEq)]
pub struct StrataDef {
    pub path: Spanned<Path>,
    pub title: Option<Spanned<String>>,
    pub symbol: Option<Spanned<String>>,
    pub stride: Option<Spanned<u32>>,
}

// === Era ===

#[derive(Debug, Clone, PartialEq)]
pub struct EraDef {
    pub name: Spanned<String>,
    pub is_initial: bool,
    pub is_terminal: bool,
    pub title: Option<Spanned<String>>,
    pub dt: Option<Spanned<ValueWithUnit>>,
    pub config_overrides: Vec<ConfigEntry>,
    pub strata_states: Vec<StrataState>,
    pub transitions: Vec<Transition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueWithUnit {
    pub value: Literal,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StrataState {
    pub strata: Spanned<Path>,
    pub state: StrataStateKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrataStateKind {
    Active,
    ActiveWithStride(u32),
    Gated,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub target: Spanned<Path>,
    pub conditions: Vec<Spanned<Expr>>,
}

// === Signal ===

#[derive(Debug, Clone, PartialEq)]
pub struct SignalDef {
    pub path: Spanned<Path>,
    pub ty: Option<Spanned<TypeExpr>>,
    pub strata: Option<Spanned<Path>>,
    pub title: Option<Spanned<String>>,
    pub symbol: Option<Spanned<String>>,
    pub dt_raw: bool,
    pub local_consts: Vec<ConstEntry>,
    pub local_config: Vec<ConfigEntry>,
    pub warmup: Option<WarmupBlock>,
    pub resolve: Option<ResolveBlock>,
    pub assertions: Option<AssertBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WarmupBlock {
    pub iterations: Spanned<u32>,
    pub convergence: Option<Spanned<f64>>,
    pub iterate: Spanned<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolveBlock {
    pub body: Spanned<Expr>,
}

// === Assertions ===

/// An assertion block containing one or more assertions
#[derive(Debug, Clone, PartialEq)]
pub struct AssertBlock {
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

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    pub path: Spanned<Path>,
    pub ty: Option<Spanned<TypeExpr>>,
    pub strata: Option<Spanned<Path>>,
    pub topology: Option<Spanned<Topology>>,
    pub title: Option<Spanned<String>>,
    pub symbol: Option<Spanned<String>>,
    pub measure: Option<MeasureBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Topology {
    SphereSurface,
    PointCloud,
    Volume,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasureBlock {
    pub body: Spanned<Expr>,
}

// === Operator ===

#[derive(Debug, Clone, PartialEq)]
pub struct OperatorDef {
    pub path: Spanned<Path>,
    pub strata: Option<Spanned<Path>>,
    pub phase: Option<Spanned<OperatorPhase>>,
    pub body: Option<OperatorBody>,
    pub assertions: Option<AssertBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperatorPhase {
    Warmup,
    Collect,
    Measure,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperatorBody {
    Warmup(Spanned<Expr>),
    Collect(Spanned<Expr>),
    Measure(Spanned<Expr>),
}

// === Impulse ===

#[derive(Debug, Clone, PartialEq)]
pub struct ImpulseDef {
    pub path: Spanned<Path>,
    pub payload_type: Option<Spanned<TypeExpr>>,
    pub local_config: Vec<ConfigEntry>,
    pub apply: Option<ApplyBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApplyBlock {
    pub body: Spanned<Expr>,
}

// === Fracture ===

#[derive(Debug, Clone, PartialEq)]
pub struct FractureDef {
    pub path: Spanned<Path>,
    pub conditions: Vec<Spanned<Expr>>,
    pub emit: Vec<EmitStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmitStatement {
    pub target: Spanned<Path>,
    pub value: Spanned<Expr>,
}

// === Chronicle ===

#[derive(Debug, Clone, PartialEq)]
pub struct ChronicleDef {
    pub path: Spanned<Path>,
    pub observe: Option<ObserveBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserveBlock {
    pub handlers: Vec<ObserveHandler>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserveHandler {
    pub condition: Spanned<Expr>,
    pub event_name: Spanned<Path>,
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

/// Count validation bounds for entity instances
#[derive(Debug, Clone, PartialEq)]
pub struct CountBounds {
    pub min: u32,
    pub max: u32,
}

/// A field in an entity schema
#[derive(Debug, Clone, PartialEq)]
pub struct EntitySchemaField {
    pub name: Spanned<String>,
    pub ty: Spanned<TypeExpr>,
}

/// A field definition nested within an entity (for observation)
#[derive(Debug, Clone, PartialEq)]
pub struct EntityFieldDef {
    pub name: Spanned<String>,
    pub ty: Option<Spanned<TypeExpr>>,
    pub topology: Option<Spanned<Topology>>,
    pub measure: Option<MeasureBlock>,
}

// === Expressions ===

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    LiteralWithUnit { value: Literal, unit: String },
    Path(Path),
    Prev,
    PrevField(String),
    DtRaw,
    Payload,
    PayloadField(String),
    SignalRef(Path),
    ConstRef(Path),
    ConfigRef(Path),
    FieldRef(Path),
    Binary { op: BinaryOp, left: Box<Spanned<Expr>>, right: Box<Spanned<Expr>> },
    Unary { op: UnaryOp, operand: Box<Spanned<Expr>> },
    Call { function: Box<Spanned<Expr>>, args: Vec<Spanned<Expr>> },
    /// Method call on an object: `obj.method(args...)`
    /// Preserves the distinction between method calls and free function calls.
    MethodCall { object: Box<Spanned<Expr>>, method: String, args: Vec<Spanned<Expr>> },
    FieldAccess { object: Box<Spanned<Expr>>, field: String },
    Let { name: String, value: Box<Spanned<Expr>>, body: Box<Spanned<Expr>> },
    If { condition: Box<Spanned<Expr>>, then_branch: Box<Spanned<Expr>>, else_branch: Option<Box<Spanned<Expr>>> },
    For { var: String, iter: Box<Spanned<Expr>>, body: Box<Spanned<Expr>> },
    Block(Vec<Spanned<Expr>>),
    EmitSignal { target: Path, value: Box<Spanned<Expr>> },
    EmitField { target: Path, position: Box<Spanned<Expr>>, value: Box<Spanned<Expr>> },
    Struct(Vec<(String, Spanned<Expr>)>),
    /// Accumulated inputs from Collect phase: `collected`
    Collected,
    MathConst(MathConst),
    Map { sequence: Box<Spanned<Expr>>, function: Box<Spanned<Expr>> },
    Fold { sequence: Box<Spanned<Expr>>, init: Box<Spanned<Expr>>, function: Box<Spanned<Expr>> },

    // === Entity expressions ===

    /// Reference to current entity instance field: `self.mass`
    SelfField(String),

    /// Reference to an entity type: `entity.stellar.moon`
    EntityRef(Path),

    /// Access entity instance by ID: `entity.moon["luna"]`
    EntityAccess {
        entity: Path,
        instance: Box<Spanned<Expr>>,
    },

    /// Aggregate operation over entity instances: `sum(entity.moon, self.mass)`
    Aggregate {
        op: AggregateOp,
        entity: Path,
        body: Box<Spanned<Expr>>,
    },

    /// Other instances (self-exclusion): `other(entity.moon)`
    /// Used for N-body interactions where you need all instances except current
    Other(Path),

    /// Pairwise iteration: `pairs(entity.moon)`
    /// Generates all unique (i,j) combinations where i < j
    Pairs(Path),

    /// Filter entity instances: `filter(entity.moon, self.mass > 1e20)`
    Filter {
        entity: Path,
        predicate: Box<Spanned<Expr>>,
    },

    /// First matching instance: `first(entity.plate, self.type == Continental)`
    First {
        entity: Path,
        predicate: Box<Spanned<Expr>>,
    },

    /// Nearest instance to position: `nearest(entity.plate, position)`
    Nearest {
        entity: Path,
        position: Box<Spanned<Expr>>,
    },

    /// All instances within radius: `within(entity.moon, pos, 1e9)`
    Within {
        entity: Path,
        position: Box<Spanned<Expr>>,
        radius: Box<Spanned<Expr>>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathConst {
    Pi,
    Tau,
    E,
    I,
    Phi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Pow,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
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
