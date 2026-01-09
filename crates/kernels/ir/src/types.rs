//! IR types
//!
//! These types represent the compiled simulation before DAG construction.

use indexmap::IndexMap;

use continuum_foundation::{EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, OperatorId, SignalId, StratumId};

/// Compiled world ready for DAG construction
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
}

/// Compiled stratum
#[derive(Debug, Clone)]
pub struct CompiledStratum {
    pub id: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub default_stride: u32,
}

/// Compiled user-defined function
///
/// Functions are pure, inlined at call sites. They cannot access `prev`, `dt`, or write to signals.
/// They can access `const.*` and `config.*`, and call other functions.
#[derive(Debug, Clone)]
pub struct CompiledFn {
    pub id: FnId,
    /// Parameter names in order
    pub params: Vec<String>,
    /// Function body expression
    pub body: CompiledExpr,
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
    /// Assertions to validate after resolution
    pub assertions: Vec<CompiledAssertion>,
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
    /// Assertions to validate after execution
    pub assertions: Vec<CompiledAssertion>,
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

/// Compiled entity - a collection of structured instances
#[derive(Debug, Clone)]
pub struct CompiledEntity {
    pub id: EntityId,
    pub stratum: StratumId,
    /// Count source from config (e.g., "stellar.moon_count")
    pub count_source: Option<String>,
    /// Count validation bounds
    pub count_bounds: Option<(u32, u32)>,
    /// Schema fields for each instance
    pub schema: Vec<CompiledSchemaField>,
    /// Signals this entity reads
    pub reads: Vec<SignalId>,
    /// Entities this entity reads (for other(), cross-entity access)
    pub entity_reads: Vec<EntityId>,
    /// Resolution logic (executed per instance)
    pub resolve: Option<CompiledExpr>,
    /// Entity-level assertions
    pub assertions: Vec<CompiledAssertion>,
    /// Nested field definitions for observation
    pub fields: Vec<CompiledEntityField>,
}

/// A field in an entity schema
#[derive(Debug, Clone)]
pub struct CompiledSchemaField {
    pub name: String,
    pub value_type: ValueType,
}

/// A field definition nested within an entity (for observation)
#[derive(Debug, Clone)]
pub struct CompiledEntityField {
    pub name: String,
    pub value_type: ValueType,
    pub topology: TopologyIr,
    /// The measure expression
    pub measure: Option<CompiledExpr>,
}

/// Warmup configuration
#[derive(Debug, Clone)]
pub struct CompiledWarmup {
    pub iterations: u32,
    pub convergence: Option<f64>,
    pub iterate: CompiledExpr,
}

/// Compiled assertion
#[derive(Debug, Clone)]
pub struct CompiledAssertion {
    /// The condition that must be true
    pub condition: CompiledExpr,
    /// Severity of the assertion failure
    pub severity: AssertionSeverity,
    /// Optional message to emit on failure
    pub message: Option<String>,
}

/// Severity of an assertion failure
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

/// Value types
#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Scalar { range: Option<ValueRange> },
    Vec2,
    Vec3,
    Vec4,
}

/// Value range constraint
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueRange {
    pub min: f64,
    pub max: f64,
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
    /// Local variable reference
    Local(String),

    // === Entity expressions ===

    /// Access current entity instance field: self.mass
    SelfField(String),

    /// Access entity instance by ID: entity.moon["luna"].mass
    EntityAccess {
        entity: EntityId,
        instance: InstanceId,
        field: String,
    },

    /// Aggregate operation over entity instances: sum(entity.moon, self.mass)
    Aggregate {
        op: AggregateOpIr,
        entity: EntityId,
        body: Box<CompiledExpr>,
    },

    /// Other instances (self-exclusion): sum(other(entity.moon), ...)
    /// Used within entity resolve for N-body interactions
    Other {
        entity: EntityId,
        body: Box<CompiledExpr>,
    },

    /// Pairwise iteration: for (a, b) in pairs(entity.moon)
    Pairs {
        entity: EntityId,
        body: Box<CompiledExpr>,
    },

    /// Filter entity instances: filter(entity.moon, self.mass > 1e20)
    Filter {
        entity: EntityId,
        predicate: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },

    /// First matching instance: first(entity.plate, self.type == Continental)
    First {
        entity: EntityId,
        predicate: Box<CompiledExpr>,
    },

    /// Nearest instance to position: nearest(entity.plate, position)
    Nearest {
        entity: EntityId,
        position: Box<CompiledExpr>,
    },

    /// All instances within radius: within(entity.moon, pos, 1e9)
    Within {
        entity: EntityId,
        position: Box<CompiledExpr>,
        radius: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
}

/// Aggregate operations over entity instances
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOpIr {
    Sum,
    Product,
    Min,
    Max,
    Mean,
    Count,
    Any,
    All,
    None,
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
