//! Abstract Syntax Tree for Continuum DSL
//!
//! These types represent the parsed structure of .cdsl files.
//! They are later lowered to typed IR for DAG construction.

use logos::Span;

/// A span in the source file
pub type SourceSpan = Span;

/// A spanned AST node
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: SourceSpan,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: SourceSpan) -> Self {
        Self { node, span }
    }
}

/// A complete DSL compilation unit (all .cdsl files in a world)
#[derive(Debug, Clone, Default)]
pub struct CompilationUnit {
    pub items: Vec<Spanned<Item>>,
}

/// Top-level items in a .cdsl file
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// const { ... }
    ConstBlock(ConstBlock),
    /// config { ... }
    ConfigBlock(ConfigBlock),
    /// type.Name { ... }
    TypeDef(TypeDef),
    /// strata.path { ... }
    StrataDef(StrataDef),
    /// era.name { ... }
    EraDef(EraDef),
    /// signal.path { ... }
    SignalDef(SignalDef),
    /// field.path { ... }
    FieldDef(FieldDef),
    /// operator.path { ... }
    OperatorDef(OperatorDef),
    /// impulse.path { ... }
    ImpulseDef(ImpulseDef),
    /// fracture.path { ... }
    FractureDef(FractureDef),
    /// chronicle.path { ... }
    ChronicleDef(ChronicleDef),
}

// =============================================================================
// Path and Identifier Types
// =============================================================================

/// A dot-separated path like "terra.geophysics.core.temp"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    pub segments: Vec<String>,
}

impl Path {
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    pub fn single(name: String) -> Self {
        Self {
            segments: vec![name],
        }
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

// =============================================================================
// Const and Config Blocks
// =============================================================================

/// const { path: value, ... }
#[derive(Debug, Clone, PartialEq)]
pub struct ConstBlock {
    pub entries: Vec<ConstEntry>,
}

/// A single constant entry: path: value <unit>
#[derive(Debug, Clone, PartialEq)]
pub struct ConstEntry {
    pub path: Spanned<Path>,
    pub value: Spanned<Literal>,
    pub unit: Option<Spanned<String>>,
}

/// config { path: value, ... }
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigBlock {
    pub entries: Vec<ConfigEntry>,
}

/// A single config entry: path: value <unit>
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigEntry {
    pub path: Spanned<Path>,
    pub value: Spanned<Literal>,
    pub unit: Option<Spanned<String>>,
}

// =============================================================================
// Type Definitions
// =============================================================================

/// type.Name { field: Type, ... }
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: Spanned<String>,
    pub fields: Vec<TypeField>,
}

/// A field in a type definition
#[derive(Debug, Clone, PartialEq)]
pub struct TypeField {
    pub name: Spanned<String>,
    pub ty: Spanned<TypeExpr>,
}

/// A type expression
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// Scalar<unit> or Scalar<unit, range>
    Scalar {
        unit: String,
        range: Option<Range>,
    },
    /// Vec2<unit>, Vec3<unit>, Vec4<unit>
    Vector {
        dim: u8, // 2, 3, or 4
        unit: String,
        magnitude: Option<Range>,
    },
    /// Tensor<N, M, unit>
    Tensor {
        rows: u32,
        cols: u32,
        unit: String,
        constraints: Vec<TensorConstraint>,
    },
    /// Seq<T>
    Seq {
        element: Box<TypeExpr>,
        constraints: Vec<SeqConstraint>,
    },
    /// Grid<W, H, T>
    Grid {
        width: u32,
        height: u32,
        element: Box<TypeExpr>,
    },
    /// Reference to a user-defined type
    Named(String),
}

/// A numeric range like 100..10000
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub min: f64,
    pub max: f64,
}

/// Tensor constraints
#[derive(Debug, Clone, PartialEq)]
pub enum TensorConstraint {
    Symmetric,
    PositiveDefinite,
}

/// Sequence constraints
#[derive(Debug, Clone, PartialEq)]
pub enum SeqConstraint {
    Each(Range),
    Sum(Range),
}

// =============================================================================
// Strata Definition
// =============================================================================

/// strata.path { attributes... }
#[derive(Debug, Clone, PartialEq)]
pub struct StrataDef {
    pub path: Spanned<Path>,
    pub title: Option<Spanned<String>>,
    pub symbol: Option<Spanned<String>>,
    pub stride: Option<Spanned<u32>>,
}

// =============================================================================
// Era Definition
// =============================================================================

/// era.name { ... }
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

/// Value with unit like 1 <Myr>
#[derive(Debug, Clone, PartialEq)]
pub struct ValueWithUnit {
    pub value: Literal,
    pub unit: String,
}

/// Stratum state in an era
#[derive(Debug, Clone, PartialEq)]
pub struct StrataState {
    pub strata: Spanned<Path>,
    pub state: StrataStateKind,
}

/// Kind of stratum state
#[derive(Debug, Clone, PartialEq)]
pub enum StrataStateKind {
    Active,
    ActiveWithStride(u32),
    Gated,
}

/// Era transition
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub target: Spanned<Path>,
    pub conditions: Vec<Spanned<Expr>>,
}

// =============================================================================
// Signal Definition
// =============================================================================

/// signal.path { ... }
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
}

/// warmup { iterations(N), convergence(e), iterate { ... } }
#[derive(Debug, Clone, PartialEq)]
pub struct WarmupBlock {
    pub iterations: Spanned<u32>,
    pub convergence: Option<Spanned<f64>>,
    pub iterate: Spanned<Expr>,
}

/// resolve { expr }
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveBlock {
    pub body: Spanned<Expr>,
}

// =============================================================================
// Field Definition
// =============================================================================

/// field.path { ... }
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

/// Field topology
#[derive(Debug, Clone, PartialEq)]
pub enum Topology {
    SphereSurface,
    PointCloud,
    Volume,
}

/// measure { expr }
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureBlock {
    pub body: Spanned<Expr>,
}

// =============================================================================
// Operator Definition
// =============================================================================

/// operator.path { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorDef {
    pub path: Spanned<Path>,
    pub strata: Option<Spanned<Path>>,
    pub phase: Option<Spanned<OperatorPhase>>,
    pub body: Option<OperatorBody>,
}

/// Operator phase
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorPhase {
    Warmup,
    Collect,
    Measure,
}

/// Operator body (phase-specific block)
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorBody {
    Warmup(Spanned<Expr>),
    Collect(Spanned<Expr>),
    Measure(Spanned<Expr>),
}

// =============================================================================
// Impulse Definition
// =============================================================================

/// impulse.path { ... }
#[derive(Debug, Clone, PartialEq)]
pub struct ImpulseDef {
    pub path: Spanned<Path>,
    pub payload_type: Option<Spanned<TypeExpr>>,
    pub local_config: Vec<ConfigEntry>,
    pub apply: Option<ApplyBlock>,
}

/// apply { expr }
#[derive(Debug, Clone, PartialEq)]
pub struct ApplyBlock {
    pub body: Spanned<Expr>,
}

// =============================================================================
// Fracture Definition
// =============================================================================

/// fracture.path { when { ... } emit { ... } }
#[derive(Debug, Clone, PartialEq)]
pub struct FractureDef {
    pub path: Spanned<Path>,
    pub conditions: Vec<Spanned<Expr>>,
    pub emit: Vec<EmitStatement>,
}

/// signal.path <- value
#[derive(Debug, Clone, PartialEq)]
pub struct EmitStatement {
    pub target: Spanned<Path>,
    pub value: Spanned<Expr>,
}

// =============================================================================
// Chronicle Definition
// =============================================================================

/// chronicle.path { observe { ... } }
#[derive(Debug, Clone, PartialEq)]
pub struct ChronicleDef {
    pub path: Spanned<Path>,
    pub observe: Option<ObserveBlock>,
}

/// observe { when condition { emit event.name { ... } } }
#[derive(Debug, Clone, PartialEq)]
pub struct ObserveBlock {
    pub handlers: Vec<ObserveHandler>,
}

/// when condition { emit event.name { ... } }
#[derive(Debug, Clone, PartialEq)]
pub struct ObserveHandler {
    pub condition: Spanned<Expr>,
    pub event_name: Spanned<Path>,
    pub event_fields: Vec<(Spanned<String>, Spanned<Expr>)>,
}

// =============================================================================
// Expressions
// =============================================================================

/// Expression node
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value
    Literal(Literal),

    /// Literal with unit: 5000 <K>, 1 <Myr>
    LiteralWithUnit {
        value: Literal,
        unit: String,
    },

    /// Variable or path reference
    Path(Path),

    /// prev (previous signal value)
    Prev,

    /// prev.field
    PrevField(String),

    /// dt_raw (explicit raw timestep access - requires : dt_raw declaration)
    DtRaw,

    /// payload (impulse data)
    Payload,

    /// payload.field
    PayloadField(String),

    /// signal.path
    SignalRef(Path),

    /// const.path
    ConstRef(Path),

    /// config.path
    ConfigRef(Path),

    /// field.path (for emit in measure phase)
    FieldRef(Path),

    /// Binary operation: a + b, a * b, etc.
    Binary {
        op: BinaryOp,
        left: Box<Spanned<Expr>>,
        right: Box<Spanned<Expr>>,
    },

    /// Unary operation: -a, not a
    Unary {
        op: UnaryOp,
        operand: Box<Spanned<Expr>>,
    },

    /// Function call: kernel.fn(args) or fn(args)
    Call {
        function: Box<Spanned<Expr>>,
        args: Vec<Spanned<Expr>>,
    },

    /// Field access: expr.field
    FieldAccess {
        object: Box<Spanned<Expr>>,
        field: String,
    },

    /// Let binding: let name = value
    Let {
        name: String,
        value: Box<Spanned<Expr>>,
        body: Box<Spanned<Expr>>,
    },

    /// If expression: if cond { then } else { else }
    If {
        condition: Box<Spanned<Expr>>,
        then_branch: Box<Spanned<Expr>>,
        else_branch: Option<Box<Spanned<Expr>>>,
    },

    /// For loop: for item in seq { body }
    For {
        var: String,
        iter: Box<Spanned<Expr>>,
        body: Box<Spanned<Expr>>,
    },

    /// Block of expressions (last one is the result)
    Block(Vec<Spanned<Expr>>),

    /// Emit to signal: signal.path <- value
    EmitSignal {
        target: Path,
        value: Box<Spanned<Expr>>,
    },

    /// Emit to field: field.path <- position, value
    EmitField {
        target: Path,
        position: Box<Spanned<Expr>>,
        value: Box<Spanned<Expr>>,
    },

    /// Struct literal: { field: value, ... }
    Struct(Vec<(String, Spanned<Expr>)>),

    /// Sum of inputs: sum(inputs)
    SumInputs,

    /// Mathematical constant: PI, TAU, E, I, PHI
    MathConst(MathConst),

    /// Map over sequence: map(seq, fn)
    Map {
        sequence: Box<Spanned<Expr>>,
        function: Box<Spanned<Expr>>,
    },

    /// Fold over sequence: fold(seq, init, fn)
    Fold {
        sequence: Box<Spanned<Expr>>,
        init: Box<Spanned<Expr>>,
        function: Box<Spanned<Expr>>,
    },
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

/// Mathematical constants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathConst {
    /// π (pi) - ratio of circumference to diameter
    Pi,
    /// τ (tau) - 2π, the circle constant
    Tau,
    /// e - Euler's number, base of natural logarithm
    E,
    /// i - imaginary unit (√-1)
    I,
    /// φ (phi) - golden ratio
    Phi,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Pow,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}
