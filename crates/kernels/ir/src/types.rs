use std::path::PathBuf;

use indexmap::IndexMap;

use continuum_dsl::ast::Span;
use continuum_foundation::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, MemberId,
    OperatorId, Path, SignalId, StratumId, TypeId,
};

// Re-export StratumState from foundation for backwards compatibility
pub use continuum_foundation::StratumState;

/// Trait for items that have a source location (file and span).
pub trait Locatable {
    fn file(&self) -> Option<&std::path::Path>;
    fn span(&self) -> &Span;
}

macro_rules! impl_locatable {
    ($t:ty) => {
        impl Locatable for $t {
            fn file(&self) -> Option<&std::path::Path> {
                self.file.as_deref()
            }
            fn span(&self) -> &Span {
                &self.span
            }
        }
    };
}

/// A compiled simulation world, ready for DAG construction.
#[derive(Debug)]
pub struct CompiledWorld {
    /// Global constants (evaluated at compile time)
    pub constants: IndexMap<String, (f64, Option<crate::units::Unit>)>,
    /// Runtime configuration values
    pub config: IndexMap<String, (f64, Option<crate::units::Unit>)>,

    /// **Unified node architecture**
    /// All DSL nodes in unified form for tooling and analysis
    pub nodes: IndexMap<Path, crate::unified_nodes::CompiledNode>,
}

impl CompiledWorld {
    /// Get all signal nodes from the unified node map.
    pub fn signals(&self) -> IndexMap<SignalId, CompiledSignal> {
        let mut signals = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Signal(props) = &node.kind {
                let signal = CompiledSignal {
                    file: node.file.clone(),
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

    /// Get all field nodes from the unified node map.
    pub fn fields(&self) -> IndexMap<FieldId, CompiledField> {
        let mut fields = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Field(props) = &node.kind {
                let field = CompiledField {
                    file: node.file.clone(),
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

    /// Get all operator nodes from the unified node map.
    pub fn operators(&self) -> IndexMap<OperatorId, CompiledOperator> {
        let mut operators = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Operator(props) = &node.kind {
                let operator = CompiledOperator {
                    file: node.file.clone(),
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

    /// Get all era nodes from the unified node map.
    pub fn eras(&self) -> IndexMap<EraId, CompiledEra> {
        let mut eras = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Era(props) = &node.kind {
                let era = CompiledEra {
                    file: node.file.clone(),
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

    /// Get all stratum nodes from the unified node map.
    pub fn strata(&self) -> IndexMap<StratumId, CompiledStratum> {
        let mut strata = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Stratum(props) = &node.kind {
                let stratum = CompiledStratum {
                    file: node.file.clone(),
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

    /// Get all member nodes from the unified node map.
    pub fn members(&self) -> IndexMap<MemberId, CompiledMember> {
        let mut members = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Member(props) = &node.kind {
                let member = CompiledMember {
                    file: node.file.clone(),
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

    /// Get all fracture nodes from the unified node map.
    pub fn fractures(&self) -> IndexMap<FractureId, CompiledFracture> {
        let mut fractures = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Fracture(props) = &node.kind {
                let fracture = CompiledFracture {
                    file: node.file.clone(),
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

    /// Get all entity nodes from the unified node map.
    pub fn entities(&self) -> IndexMap<EntityId, CompiledEntity> {
        let mut entities = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Entity(props) = &node.kind {
                let entity = CompiledEntity {
                    file: node.file.clone(),
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

    /// Get all chronicle nodes from the unified node map.
    pub fn chronicles(&self) -> IndexMap<ChronicleId, CompiledChronicle> {
        let mut chronicles = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Chronicle(props) = &node.kind {
                let chronicle = CompiledChronicle {
                    file: node.file.clone(),
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

    /// Get all function nodes from the unified node map.
    pub fn functions(&self) -> IndexMap<FnId, CompiledFn> {
        let mut functions = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Function(props) = &node.kind {
                let func = CompiledFn {
                    file: node.file.clone(),
                    span: node.span.clone(),
                    id: FnId::from(path.clone()),
                    params: props.params.clone(),
                    body: props.body.clone(),
                };
                functions.insert(FnId::from(path.clone()), func);
            }
        }
        functions
    }

    /// Get all type nodes from the unified node map.
    pub fn types(&self) -> IndexMap<TypeId, CompiledType> {
        let mut types = IndexMap::new();
        for (path, node) in &self.nodes {
            if let crate::unified_nodes::NodeKind::Type(props) = &node.kind {
                let ty = CompiledType {
                    file: node.file.clone(),
                    span: node.span.clone(),
                    id: TypeId::from(path.clone()),
                    fields: props.fields.clone(),
                };
                types.insert(TypeId::from(path.clone()), ty);
            }
        }
        types
    }
}

/// A compiled stratum definition representing a simulation layer.
#[derive(Debug, Clone)]
pub struct CompiledStratum {
    pub file: Option<PathBuf>,
    pub span: Span,
    pub id: StratumId,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub default_stride: u32,
}

/// A compiled user-defined function declaration.
#[derive(Debug, Clone)]
pub struct CompiledFn {
    pub file: Option<PathBuf>,
    pub span: Span,
    pub id: FnId,
    pub params: Vec<String>,
    pub body: CompiledExpr,
}

/// A compiled era definition representing a distinct time phase.
#[derive(Debug, Clone)]
pub struct CompiledEra {
    pub file: Option<PathBuf>,
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
    pub file: Option<PathBuf>,
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
    pub file: Option<PathBuf>,
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
    pub file: Option<PathBuf>,
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
    pub file: Option<PathBuf>,
    pub span: Span,
    pub id: ImpulseId,
    pub payload_type: ValueType,
    pub apply: Option<CompiledExpr>,
}

/// A compiled fracture definition for emergent tension detection.
#[derive(Debug, Clone)]
pub struct CompiledFracture {
    pub file: Option<PathBuf>,
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
    pub file: Option<PathBuf>,
    pub span: Span,
    pub id: EntityId,
    pub count_source: Option<String>,
    pub count_bounds: Option<(u32, u32)>,
}

/// A compiled member signal definition representing per-entity authoritative state.
#[derive(Debug, Clone)]
pub struct CompiledMember {
    pub file: Option<PathBuf>,
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
    pub file: Option<PathBuf>,
    pub span: Span,
    pub id: ChronicleId,
    pub reads: Vec<SignalId>,
    pub handlers: Vec<CompiledObserveHandler>,
}

/// A compiled custom type definition.
#[derive(Debug, Clone)]
pub struct CompiledType {
    pub file: Option<PathBuf>,
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
    Quat {
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
    Literal(f64, Option<crate::units::Unit>),
    Prev,
    DtRaw,
    SimTime,
    Collected,
    Signal(SignalId),
    Const(String, Option<crate::units::Unit>),
    Config(String, Option<crate::units::Unit>),
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
        namespace: String,
        function: String,
        args: Vec<CompiledExpr>,
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

impl CompiledExpr {
    /// Extract all signal dependencies from this expression.
    pub fn signal_dependencies(&self) -> Vec<SignalId> {
        let mut deps = Vec::new();
        self.collect_signal_dependencies(&mut deps);
        deps
    }

    fn collect_signal_dependencies(&self, deps: &mut Vec<SignalId>) {
        match self {
            CompiledExpr::Signal(id) => deps.push(id.clone()),
            CompiledExpr::Binary { left, right, .. } => {
                left.collect_signal_dependencies(deps);
                right.collect_signal_dependencies(deps);
            }
            CompiledExpr::Unary { operand, .. } => operand.collect_signal_dependencies(deps),
            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                condition.collect_signal_dependencies(deps);
                then_branch.collect_signal_dependencies(deps);
                else_branch.collect_signal_dependencies(deps);
            }
            CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
                for arg in args {
                    arg.collect_signal_dependencies(deps);
                }
            }
            CompiledExpr::FieldAccess { object, .. } => object.collect_signal_dependencies(deps),
            CompiledExpr::Let { value, body, .. } => {
                value.collect_signal_dependencies(deps);
                body.collect_signal_dependencies(deps);
            }
            CompiledExpr::EmitSignal { target, value } => {
                deps.push(target.clone());
                value.collect_signal_dependencies(deps);
            }
            CompiledExpr::Aggregate { body, .. } => body.collect_signal_dependencies(deps),
            CompiledExpr::Other { body, .. } | CompiledExpr::Pairs { body, .. } => {
                body.collect_signal_dependencies(deps)
            }
            CompiledExpr::Filter {
                predicate, body, ..
            } => {
                predicate.collect_signal_dependencies(deps);
                body.collect_signal_dependencies(deps);
            }
            CompiledExpr::First { predicate, .. } => predicate.collect_signal_dependencies(deps),
            CompiledExpr::Nearest { position, .. } => position.collect_signal_dependencies(deps),
            CompiledExpr::Within {
                position,
                radius,
                body,
                ..
            } => {
                position.collect_signal_dependencies(deps);
                radius.collect_signal_dependencies(deps);
                body.collect_signal_dependencies(deps);
            }
            _ => {}
        }
    }
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

impl_locatable!(CompiledSignal);
impl_locatable!(CompiledField);
impl_locatable!(CompiledOperator);
impl_locatable!(CompiledImpulse);
impl_locatable!(CompiledFracture);
impl_locatable!(CompiledEntity);
impl_locatable!(CompiledMember);
impl_locatable!(CompiledChronicle);
impl_locatable!(CompiledStratum);
impl_locatable!(CompiledEra);
impl_locatable!(CompiledFn);
impl_locatable!(CompiledType);

impl Locatable for crate::unified_nodes::CompiledNode {
    fn file(&self) -> Option<&std::path::Path> {
        self.file.as_deref()
    }
    fn span(&self) -> &Span {
        &self.span
    }
}
