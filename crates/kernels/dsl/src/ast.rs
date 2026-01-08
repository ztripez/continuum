//! Abstract Syntax Tree for Continuum DSL

use std::ops::Range as StdRange;

/// Source span (byte range)
pub type Span = StdRange<usize>;

/// A spanned AST node
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// A complete DSL compilation unit
#[derive(Debug, Clone, Default)]
pub struct CompilationUnit {
    pub items: Vec<Spanned<Item>>,
}

/// Top-level items
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    ConstBlock(ConstBlock),
    ConfigBlock(ConfigBlock),
    TypeDef(TypeDef),
    StrataDef(StrataDef),
    EraDef(EraDef),
    SignalDef(SignalDef),
    FieldDef(FieldDef),
    OperatorDef(OperatorDef),
    ImpulseDef(ImpulseDef),
    FractureDef(FractureDef),
    ChronicleDef(ChronicleDef),
}

/// Dot-separated path
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
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
    FieldAccess { object: Box<Spanned<Expr>>, field: String },
    Let { name: String, value: Box<Spanned<Expr>>, body: Box<Spanned<Expr>> },
    If { condition: Box<Spanned<Expr>>, then_branch: Box<Spanned<Expr>>, else_branch: Option<Box<Spanned<Expr>>> },
    For { var: String, iter: Box<Spanned<Expr>>, body: Box<Spanned<Expr>> },
    Block(Vec<Spanned<Expr>>),
    EmitSignal { target: Path, value: Box<Spanned<Expr>> },
    EmitField { target: Path, position: Box<Spanned<Expr>>, value: Box<Spanned<Expr>> },
    Struct(Vec<(String, Spanned<Expr>)>),
    SumInputs,
    MathConst(MathConst),
    Map { sequence: Box<Spanned<Expr>>, function: Box<Spanned<Expr>> },
    Fold { sequence: Box<Spanned<Expr>>, init: Box<Spanned<Expr>>, function: Box<Spanned<Expr>> },
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
