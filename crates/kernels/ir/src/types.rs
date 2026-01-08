//! IR types
//!
//! These types represent the compiled simulation before DAG construction.

use indexmap::IndexMap;

use continuum_foundation::{EraId, FieldId, FractureId, ImpulseId, OperatorId, SignalId, StratumId};

/// Compiled world ready for DAG construction
#[derive(Debug)]
pub struct CompiledWorld {
    /// Global constants (evaluated at compile time)
    pub constants: IndexMap<String, f64>,
    /// Runtime configuration values
    pub config: IndexMap<String, f64>,
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
}

/// Compiled stratum
#[derive(Debug, Clone)]
pub struct CompiledStratum {
    pub id: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub default_stride: u32,
}

/// Compiled era
#[derive(Debug, Clone)]
pub struct CompiledEra {
    pub id: EraId,
    pub is_initial: bool,
    pub is_terminal: bool,
    pub title: Option<String>,
    /// Time step in seconds
    pub dt_seconds: f64,
    /// Stratum states for this era
    pub strata_states: IndexMap<StratumId, StratumStateIr>,
    /// Transitions to other eras
    pub transitions: Vec<CompiledTransition>,
}

/// Stratum state in an era
#[derive(Debug, Clone, Copy)]
pub enum StratumStateIr {
    Active,
    ActiveWithStride(u32),
    Gated,
}

/// Transition condition
#[derive(Debug, Clone)]
pub struct CompiledTransition {
    pub target_era: EraId,
    pub condition: CompiledExpr,
}

/// Compiled signal
#[derive(Debug, Clone)]
pub struct CompiledSignal {
    pub id: SignalId,
    pub stratum: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub value_type: ValueType,
    pub uses_dt_raw: bool,
    /// Signals this signal reads
    pub reads: Vec<SignalId>,
    /// The resolve expression
    pub resolve: Option<CompiledExpr>,
    /// Warmup configuration
    pub warmup: Option<CompiledWarmup>,
}

/// Compiled field
#[derive(Debug, Clone)]
pub struct CompiledField {
    pub id: FieldId,
    pub stratum: StratumId,
    pub title: Option<String>,
    pub topology: TopologyIr,
    pub value_type: ValueType,
    /// Signals this field reads
    pub reads: Vec<SignalId>,
    /// The measure expression
    pub measure: Option<CompiledExpr>,
}

/// Compiled operator
#[derive(Debug, Clone)]
pub struct CompiledOperator {
    pub id: OperatorId,
    pub stratum: StratumId,
    pub phase: OperatorPhaseIr,
    /// Signals this operator reads
    pub reads: Vec<SignalId>,
    /// The operator body
    pub body: Option<CompiledExpr>,
}

/// Compiled impulse
#[derive(Debug, Clone)]
pub struct CompiledImpulse {
    pub id: ImpulseId,
    pub payload_type: ValueType,
    /// The apply expression
    pub apply: Option<CompiledExpr>,
}

/// Compiled fracture
#[derive(Debug, Clone)]
pub struct CompiledFracture {
    pub id: FractureId,
    /// Signals this fracture reads
    pub reads: Vec<SignalId>,
    /// Condition expressions (all must be true)
    pub conditions: Vec<CompiledExpr>,
    /// Emit statements
    pub emits: Vec<CompiledEmit>,
}

/// Compiled emit statement
#[derive(Debug, Clone)]
pub struct CompiledEmit {
    pub target: SignalId,
    pub value: CompiledExpr,
}

/// Warmup configuration
#[derive(Debug, Clone)]
pub struct CompiledWarmup {
    pub iterations: u32,
    pub convergence: Option<f64>,
    pub iterate: CompiledExpr,
}

/// Value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
}

/// Topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyIr {
    SphereSurface,
    PointCloud,
    Volume,
}

/// Operator phases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorPhaseIr {
    Warmup,
    Collect,
    Measure,
}

/// Compiled expression - simplified for execution
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    /// Literal value
    Literal(f64),
    /// Previous value of current signal
    Prev,
    /// Raw dt value
    DtRaw,
    /// Sum of inputs for this signal
    SumInputs,
    /// Reference to a signal
    Signal(SignalId),
    /// Reference to a constant
    Const(String),
    /// Reference to a config value
    Config(String),
    /// Binary operation
    Binary {
        op: BinaryOpIr,
        left: Box<CompiledExpr>,
        right: Box<CompiledExpr>,
    },
    /// Unary operation
    Unary {
        op: UnaryOpIr,
        operand: Box<CompiledExpr>,
    },
    /// Function call
    Call {
        function: String,
        args: Vec<CompiledExpr>,
    },
    /// Field access
    FieldAccess {
        object: Box<CompiledExpr>,
        field: String,
    },
    /// Conditional
    If {
        condition: Box<CompiledExpr>,
        then_branch: Box<CompiledExpr>,
        else_branch: Box<CompiledExpr>,
    },
    /// Let binding
    Let {
        name: String,
        value: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpIr {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpIr {
    Neg,
    Not,
}
