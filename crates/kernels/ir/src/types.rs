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

use continuum_dsl::ast::Span;
use continuum_foundation::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, MemberId,
    OperatorId, Path, SignalId, StratumId, TypeId,
};

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
/// - **Unified nodes**: All DSL constructs in a single, indexed collection
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
/// println!("Signals defined: {}", world.signals().len());
/// for (id, signal) in world.signals() {
///     println!("  {} reads {:?}", id.0, signal.reads);
/// }
/// ```
#[derive(Debug)]
pub struct CompiledWorld {
    /// Global constants (evaluated at compile time)
    pub constants: IndexMap<String, f64>,
    /// Runtime configuration values
    pub config: IndexMap<String, f64>,

    /// **Unified node architecture**
    /// All DSL nodes in unified form for tooling and analysis
    pub nodes: IndexMap<Path, super::unified_nodes::CompiledNode>,
}

impl CompiledWorld {
    /// Get all signals as an IndexMap (backward compatibility)
    pub fn signals(&self) -> IndexMap<SignalId, CompiledSignal> {
        let mut signals = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Signal(props) = &node.kind {
                let signal = CompiledSignal {
                    span: node.span.clone(),
                    id: SignalId::from(path.clone()),
                    stratum: node
                        .stratum
                        .clone()
                        .unwrap_or_else(|| StratumId::from("default")),
                    title: props.title.clone(),
                    symbol: props.symbol.clone(),
                    value_type: props.value_type.clone(),
                    uses_dt_raw: props.uses_dt_raw,
                    reads: node.reads.clone(),
                    resolve: props.resolve.clone(),
                    resolve_components: props.resolve_components.clone(),
                    warmup: props.warmup.clone(),
                    assertions: props.assertions.clone(),
                };
                signals.insert(SignalId::from(path.clone()), signal);
            }
        }
        signals
    }

    /// Get all fields as an IndexMap (backward compatibility)
    pub fn fields(&self) -> IndexMap<FieldId, CompiledField> {
        let mut fields = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Field(props) = &node.kind {
                let field = CompiledField {
                    span: node.span.clone(),
                    id: FieldId::from(path.clone()),
                    stratum: node
                        .stratum
                        .clone()
                        .unwrap_or_else(|| StratumId::from("default")),
                    title: props.title.clone(),
                    topology: props.topology,
                    value_type: props.value_type.clone(),
                    reads: node.reads.clone(),
                    measure: props.measure.clone(),
                };
                fields.insert(FieldId::from(path.clone()), field);
            }
        }
        fields
    }

    /// Get all operators as an IndexMap (backward compatibility)
    pub fn operators(&self) -> IndexMap<OperatorId, CompiledOperator> {
        let mut operators = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Operator(props) = &node.kind {
                let operator = CompiledOperator {
                    span: node.span.clone(),
                    id: OperatorId::from(path.clone()),
                    stratum: node
                        .stratum
                        .clone()
                        .unwrap_or_else(|| StratumId::from("default")),
                    phase: props.phase,
                    reads: node.reads.clone(),
                    body: props.body.clone(),
                    assertions: props.assertions.clone(),
                };
                operators.insert(OperatorId::from(path.clone()), operator);
            }
        }
        operators
    }

    /// Get all eras as an IndexMap (backward compatibility)
    pub fn eras(&self) -> IndexMap<EraId, CompiledEra> {
        let mut eras = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Era(props) = &node.kind {
                let era = CompiledEra {
                    span: node.span.clone(),
                    id: EraId::from(path.clone()),
                    is_initial: props.is_initial,
                    is_terminal: props.is_terminal,
                    title: props.title.clone(),
                    dt_seconds: props.dt_seconds,
                    strata_states: props.strata_states.clone(),
                    transitions: props.transitions.clone(),
                };
                eras.insert(EraId::from(path.clone()), era);
            }
        }
        eras
    }

    /// Get all strata as an IndexMap (backward compatibility)
    pub fn strata(&self) -> IndexMap<StratumId, CompiledStratum> {
        let mut strata = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Stratum(props) = &node.kind {
                let stratum = CompiledStratum {
                    span: node.span.clone(),
                    id: StratumId::from(path.clone()),
                    title: props.title.clone(),
                    symbol: props.symbol.clone(),
                    default_stride: props.default_stride,
                };
                strata.insert(StratumId::from(path.clone()), stratum);
            }
        }
        strata
    }

    /// Get all members as an IndexMap (backward compatibility)
    pub fn members(&self) -> IndexMap<MemberId, CompiledMember> {
        let mut members = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Member(props) = &node.kind {
                let member = CompiledMember {
                    span: node.span.clone(),
                    id: MemberId::from(path.clone()),
                    entity_id: props.entity_id.clone(),
                    signal_name: props.signal_name.clone(),
                    stratum: node
                        .stratum
                        .clone()
                        .unwrap_or_else(|| StratumId::from("default")),
                    title: props.title.clone(),
                    symbol: props.symbol.clone(),
                    value_type: props.value_type.clone(),
                    uses_dt_raw: props.uses_dt_raw,
                    reads: node.reads.clone(),
                    member_reads: node.member_reads.clone(),
                    initial: props.initial.clone(),
                    resolve: props.resolve.clone(),
                    assertions: props.assertions.clone(),
                };
                members.insert(MemberId::from(path.clone()), member);
            }
        }
        members
    }

    /// Get all fractures as an IndexMap (backward compatibility)
    pub fn fractures(&self) -> IndexMap<FractureId, CompiledFracture> {
        let mut fractures = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Fracture(props) = &node.kind {
                let fracture = CompiledFracture {
                    span: node.span.clone(),
                    id: FractureId::from(path.clone()),
                    stratum: node
                        .stratum
                        .clone()
                        .unwrap_or_else(|| StratumId::from("default")),
                    reads: node.reads.clone(),
                    conditions: props.conditions.clone(),
                    emits: props.emits.clone(),
                };
                fractures.insert(FractureId::from(path.clone()), fracture);
            }
        }
        fractures
    }

    /// Get all entities as an IndexMap (backward compatibility)
    pub fn entities(&self) -> IndexMap<EntityId, CompiledEntity> {
        let mut entities = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Entity(props) = &node.kind {
                let entity = CompiledEntity {
                    span: node.span.clone(),
                    id: EntityId::from(path.clone()),
                    count_source: props.count_source.clone(),
                    count_bounds: props.count_bounds,
                };
                entities.insert(EntityId::from(path.clone()), entity);
            }
        }
        entities
    }

    /// Get all chronicles as an IndexMap (backward compatibility)  
    pub fn chronicles(&self) -> IndexMap<ChronicleId, CompiledChronicle> {
        let mut chronicles = IndexMap::new();
        for (path, node) in &self.nodes {
            if let super::unified_nodes::NodeKind::Chronicle(props) = &node.kind {
                let chronicle = CompiledChronicle {
                    span: node.span.clone(),
                    id: ChronicleId::from(path.clone()),
                    reads: node.reads.clone(),
                    handlers: props.handlers.clone(),
                };
                chronicles.insert(ChronicleId::from(path.clone()), chronicle);
            }
        }
        chronicles
    }

    /// Get a specific signal by id
    pub fn get_signal(&self, id: &SignalId) -> Option<CompiledSignal> {
        self.nodes.get(id.path()).and_then(|node| {
            if let super::unified_nodes::NodeKind::Signal(props) = &node.kind {
                Some(CompiledSignal {
                    span: node.span.clone(),
                    id: id.clone(),
                    stratum: node
                        .stratum
                        .clone()
                        .unwrap_or_else(|| StratumId::from("default")),
                    title: props.title.clone(),
                    symbol: props.symbol.clone(),
                    value_type: props.value_type.clone(),
                    uses_dt_raw: props.uses_dt_raw,
                    reads: node.reads.clone(),
                    resolve: props.resolve.clone(),
                    resolve_components: props.resolve_components.clone(),
                    warmup: props.warmup.clone(),
                    assertions: props.assertions.clone(),
                })
            } else {
                None
            }
        })
    }
}

// Rest of file with all the other type definitions...
// I'll continue with a minimal subset to get compilation working

/// A compiled stratum definition representing a simulation layer.
#[derive(Debug, Clone)]
pub struct CompiledStratum {
    pub span: Span,
    pub id: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub default_stride: u32,
}

/// A compiled user-defined function declaration.
#[derive(Debug, Clone)]
pub struct CompiledFn {
    pub span: Span,
    pub id: FnId,
    pub params: Vec<String>,
    pub body: CompiledExpr,
}

/// A compiled era definition representing a distinct time phase.
#[derive(Debug, Clone)]
pub struct CompiledEra {
    pub span: Span,
    pub id: EraId,
    pub is_initial: bool,
    pub is_terminal: bool,
    pub title: Option<String>,
    pub dt_seconds: f64,
    pub strata_states: IndexMap<StratumId, StratumState>,
    pub transitions: Vec<CompiledTransition>,
}

/// A compiled transition between eras.
#[derive(Debug, Clone)]
pub struct CompiledTransition {
    pub target_era: EraId,
    pub condition: CompiledExpr,
}

/// A compiled signal definition representing authoritative simulation state.
#[derive(Debug, Clone)]
pub struct CompiledSignal {
    pub span: Span,
    pub id: SignalId,
    pub stratum: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub value_type: ValueType,
    pub uses_dt_raw: bool,
    pub reads: Vec<SignalId>,
    pub resolve: Option<CompiledExpr>,
    pub resolve_components: Option<Vec<CompiledExpr>>,
    pub warmup: Option<CompiledWarmup>,
    pub assertions: Vec<CompiledAssertion>,
}

/// A compiled field definition for observable (non-causal) data.
#[derive(Debug, Clone)]
pub struct CompiledField {
    pub span: Span,
    pub id: FieldId,
    pub stratum: StratumId,
    pub title: Option<String>,
    pub topology: TopologyIr,
    pub value_type: ValueType,
    pub reads: Vec<SignalId>,
    pub measure: Option<CompiledExpr>,
}

/// A compiled operator definition for phase-specific computation.
#[derive(Debug, Clone)]
pub struct CompiledOperator {
    pub span: Span,
    pub id: OperatorId,
    pub stratum: StratumId,
    pub phase: OperatorPhaseIr,
    pub reads: Vec<SignalId>,
    pub body: Option<CompiledExpr>,
    pub assertions: Vec<CompiledAssertion>,
}

/// A compiled impulse definition for external causal input.
#[derive(Debug, Clone)]
pub struct CompiledImpulse {
    pub span: Span,
    pub id: ImpulseId,
    pub payload_type: ValueType,
    pub apply: Option<CompiledExpr>,
}

/// A compiled fracture definition for emergent tension detection.
#[derive(Debug, Clone)]
pub struct CompiledFracture {
    pub span: Span,
    pub id: FractureId,
    pub stratum: StratumId,
    pub reads: Vec<SignalId>,
    pub conditions: Vec<CompiledExpr>,
    pub emits: Vec<CompiledEmit>,
}

/// A signal emission from a fracture.
#[derive(Debug, Clone)]
pub struct CompiledEmit {
    pub target: SignalId,
    pub value: CompiledExpr,
}

/// A compiled entity definition representing a pure index space.
#[derive(Debug, Clone)]
pub struct CompiledEntity {
    pub span: Span,
    pub id: EntityId,
    pub count_source: Option<String>,
    pub count_bounds: Option<(u32, u32)>,
}

/// A compiled member signal definition representing per-entity authoritative state.
#[derive(Debug, Clone)]
pub struct CompiledMember {
    pub span: Span,
    pub id: MemberId,
    pub entity_id: EntityId,
    pub signal_name: String,
    pub stratum: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub value_type: ValueType,
    pub uses_dt_raw: bool,
    pub reads: Vec<SignalId>,
    pub member_reads: Vec<MemberId>,
    pub initial: Option<CompiledExpr>,
    pub resolve: Option<CompiledExpr>,
    pub assertions: Vec<CompiledAssertion>,
}

/// A compiled chronicle definition for observer-only event recording.
#[derive(Debug, Clone)]
pub struct CompiledChronicle {
    pub span: Span,
    pub id: ChronicleId,
    pub reads: Vec<SignalId>,
    pub handlers: Vec<CompiledObserveHandler>,
}

/// An observation handler within a chronicle.
#[derive(Debug, Clone)]
pub struct CompiledObserveHandler {
    pub condition: CompiledExpr,
    pub event_name: String,
    pub event_fields: Vec<CompiledEventField>,
}

/// A field within a chronicle event payload.
#[derive(Debug, Clone)]
pub struct CompiledEventField {
    pub name: String,
    pub value: CompiledExpr,
}

/// Warmup configuration for signal initialization.
#[derive(Debug, Clone)]
pub struct CompiledWarmup {
    pub iterations: u32,
    pub convergence: Option<f64>,
    pub iterate: CompiledExpr,
}

/// A compiled assertion for runtime validation.
#[derive(Debug, Clone)]
pub struct CompiledAssertion {
    pub condition: CompiledExpr,
    pub severity: AssertionSeverity,
    pub message: Option<String>,
}

/// The severity level of an assertion failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssertionSeverity {
    Warn,
    #[default]
    Error,
    Fatal,
}

/// The type of a signal or field value.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Scalar {
        unit: Option<String>,
        dimension: Option<crate::units::Unit>,
        range: Option<ValueRange>,
    },
    Vec2 {
        unit: Option<String>,
        dimension: Option<crate::units::Unit>,
        magnitude: Option<ValueRange>,
    },
    Vec3 {
        unit: Option<String>,
        dimension: Option<crate::units::Unit>,
        magnitude: Option<ValueRange>,
    },
    Vec4 {
        unit: Option<String>,
        dimension: Option<crate::units::Unit>,
        magnitude: Option<ValueRange>,
    },
    Tensor {
        rows: u8,
        cols: u8,
        unit: Option<String>,
        dimension: Option<crate::units::Unit>,
        constraints: Vec<TensorConstraintIr>,
    },
    Grid {
        width: u32,
        height: u32,
        element_type: Box<ValueType>,
    },
    Seq {
        element_type: Box<ValueType>,
        constraints: Vec<SeqConstraintIr>,
    },
}

/// A numeric range constraint for scalar values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueRange {
    pub min: f64,
    pub max: f64,
}

/// Tensor mathematical constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorConstraintIr {
    Symmetric,
    PositiveDefinite,
}

/// Sequence aggregate constraint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeqConstraintIr {
    Each(ValueRange),
    Sum(ValueRange),
}

/// Spatial topology for field data distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyIr {
    SphereSurface,
    PointCloud,
    Volume,
}

/// The execution phase for an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorPhaseIr {
    Warmup,
    Collect,
    Measure,
}

/// A compiled expression tree ready for bytecode generation or interpretation.
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    Literal(f64),
    Prev,
    DtRaw,
    SimTime,
    Collected,
    Signal(SignalId),
    Const(String),
    Config(String),
    Binary {
        op: BinaryOpIr,
        left: Box<CompiledExpr>,
        right: Box<CompiledExpr>,
    },
    Unary {
        op: UnaryOpIr,
        operand: Box<CompiledExpr>,
    },
    Call {
        function: String,
        args: Vec<CompiledExpr>,
    },
    KernelCall {
        function: String,
        args: Vec<CompiledExpr>,
    },
    DtRobustCall {
        operator: DtRobustOperator,
        args: Vec<CompiledExpr>,
        method: IntegrationMethod,
    },
    FieldAccess {
        object: Box<CompiledExpr>,
        field: String,
    },
    If {
        condition: Box<CompiledExpr>,
        then_branch: Box<CompiledExpr>,
        else_branch: Box<CompiledExpr>,
    },
    Let {
        name: String,
        value: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
    Local(String),
    Payload,
    PayloadField(String),
    EmitSignal {
        target: SignalId,
        value: Box<CompiledExpr>,
    },
    SelfField(String),
    EntityAccess {
        entity: EntityId,
        instance: InstanceId,
        field: String,
    },
    Aggregate {
        op: AggregateOpIr,
        entity: EntityId,
        body: Box<CompiledExpr>,
    },
    Other {
        entity: EntityId,
        body: Box<CompiledExpr>,
    },
    Pairs {
        entity: EntityId,
        body: Box<CompiledExpr>,
    },
    Filter {
        entity: EntityId,
        predicate: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
    First {
        entity: EntityId,
        predicate: Box<CompiledExpr>,
    },
    Nearest {
        entity: EntityId,
        position: Box<CompiledExpr>,
    },
    Within {
        entity: EntityId,
        position: Box<CompiledExpr>,
        radius: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
}

/// Aggregate operations over collections of entity instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Binary operators for two-operand expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Unary operators for single-operand expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpIr {
    Neg,
    Not,
}

/// dt-robust operators that provide numerically stable time integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtRobustOperator {
    Integrate,
    Decay,
    Relax,
    Accumulate,
    AdvancePhase,
    Smooth,
    Damp,
}

/// Integration method for dt-robust operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IntegrationMethod {
    #[default]
    Euler,
    Rk4,
    Verlet,
}

/// A compiled custom type definition.
#[derive(Debug, Clone)]
pub struct CompiledType {
    pub span: Span,
    pub id: TypeId,
    pub fields: Vec<CompiledTypeField>,
}

/// A field within a compiled custom type.
#[derive(Debug, Clone)]
pub struct CompiledTypeField {
    pub name: String,
    pub value_type: ValueType,
}
