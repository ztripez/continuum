//! Unified Node Architecture for Continuum IR
//!
//! This module provides a unified representation for all compilation nodes,
//! replacing the fragmented approach of separate `IndexMap`s for each node type.

use indexmap::IndexMap;
use std::ops::Range;

use std::path::PathBuf;

use continuum_foundation::{EntityId, FieldId, MemberId, Path, SignalId, StratumId};

use super::{
    CompiledAssertion, CompiledEmit, CompiledExpr, CompiledObserveHandler, CompiledTransition,
    CompiledTypeField, CompiledWarmup, OperatorPhaseIr, StratumState, TopologyIr, ValueType,
};

use serde::{Deserialize, Serialize};

/// A unified compilation node representing any DSL construct.
///
/// `CompiledNode` unifies all compilation results (`CompiledSignal`, `CompiledField`, etc.)
/// into a single type with common properties extracted. This enables:
///
/// - **Unified indexing**: Single `IndexMap<Path, CompiledNode>` instead of 8+ separate maps
/// - **Simplified tooling**: LSP, analysis, and validation can work with one node type
/// - **Consistent dependency tracking**: All nodes expose `reads` in the same way
/// - **Uniform span information**: Every node has source location for error reporting
///
/// The `kind` field contains node-specific properties and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledNode {
    /// The unique path identifier for this node
    pub id: Path,
    /// Source file path where this node was defined
    pub file: Option<PathBuf>,
    /// Source span for error reporting and IDE navigation  
    pub span: Range<usize>,
    /// Stratum binding for scheduling (if applicable)
    pub stratum: Option<StratumId>,
    /// Signals this node depends on for execution ordering
    pub reads: Vec<SignalId>,
    /// Member signals this node depends on (for entity nodes)
    pub member_reads: Vec<MemberId>,
    /// Node-specific properties and behavior
    pub kind: NodeKind,
}

/// The specific kind of compilation node and its properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    /// Signal: authoritative simulation state
    Signal(SignalProperties),
    /// Field: observable (non-causal) data
    Field(FieldProperties),
    /// Operator: phase-specific computation
    Operator(OperatorProperties),
    /// Impulse: external causal input
    Impulse(ImpulseProperties),
    /// Fracture: emergent tension detection
    Fracture(FractureProperties),
    /// Entity: pure index space for collections
    Entity(EntityProperties),
    /// Member: per-entity authoritative state
    Member(MemberProperties),
    /// Chronicle: observer-only event recording
    Chronicle(ChronicleProperties),
    /// Analyzer: observer-only analysis query
    Analyzer(AnalyzerProperties),
    /// Function: user-defined pure function
    Function(FunctionProperties),
    /// Type: custom type definition
    Type(TypeProperties),
    /// Stratum: simulation layer definition
    Stratum(StratumProperties),
    /// Era: time phase definition
    Era(EraProperties),
}

/// Properties specific to signal nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Human-readable title
    pub title: Option<String>,
    /// Unicode symbol for display
    pub symbol: Option<String>,
    /// Value type with optional bounds
    pub value_type: ValueType,
    /// Whether `dt_raw` is explicitly used
    pub uses_dt_raw: bool,
    /// Initial value expression, evaluated once at simulation start
    pub initial: Option<CompiledExpr>,
    /// The resolve expression (all types, vectors handled via runtime dispatch)
    pub resolve: Option<CompiledExpr>,
    /// Warmup configuration
    pub warmup: Option<CompiledWarmup>,
    /// Assertions to validate after resolution
    pub assertions: Vec<CompiledAssertion>,
}

/// Properties specific to field nodes  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Human-readable title
    pub title: Option<String>,
    /// Unicode symbol for display
    pub symbol: Option<String>,
    /// Spatial topology for reconstruction
    pub topology: TopologyIr,
    /// Value type at each sample point
    pub value_type: ValueType,
    /// The measure expression
    pub measure: Option<CompiledExpr>,
}

/// Properties specific to operator nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Execution phase
    pub phase: OperatorPhaseIr,
    /// The operator body
    pub body: Option<CompiledExpr>,
    /// Assertions to validate after execution
    pub assertions: Vec<CompiledAssertion>,
}

/// Properties specific to impulse nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpulseProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Human-readable title
    pub title: Option<String>,
    /// Unicode symbol for display
    pub symbol: Option<String>,
    /// Type of data carried by the impulse
    pub payload_type: ValueType,
    /// The apply expression
    pub apply: Option<CompiledExpr>,
}

/// Properties specific to fracture nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractureProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Condition expressions (all must be true)
    pub conditions: Vec<CompiledExpr>,
    /// Emit statements
    pub emits: Vec<CompiledEmit>,
}

/// Properties specific to entity nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Count source from config (e.g., "stellar.moon_count")
    pub count_source: Option<String>,
    /// Count validation bounds
    pub count_bounds: Option<(u32, u32)>,
}

/// Properties specific to member nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// The entity this member belongs to
    pub entity_id: EntityId,
    /// The signal name within the entity (e.g., "age" from "human.person.age")
    pub signal_name: String,
    /// Human-readable title for display
    pub title: Option<String>,
    /// Unicode symbol for visualization
    pub symbol: Option<String>,
    /// Value type with optional bounds
    pub value_type: ValueType,
    /// Whether `dt_raw` is explicitly used
    pub uses_dt_raw: bool,
    /// Initial value expression (evaluated once at entity creation)
    pub initial: Option<CompiledExpr>,
    /// The resolve expression
    pub resolve: Option<CompiledExpr>,
    /// Assertions to validate after resolution
    pub assertions: Vec<CompiledAssertion>,
}

/// Properties specific to chronicle nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Observation handlers that emit events when conditions are met
    pub handlers: Vec<CompiledObserveHandler>,
}

/// Properties specific to analyzer nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Required fields this analyzer depends on
    pub required_fields: Vec<FieldId>,
    /// Compute expression that produces the analysis result
    pub compute: CompiledExpr,
    /// Output schema describing the result structure
    pub output_schema: super::OutputSchema,
    /// Validation checks to run on results
    pub validations: Vec<super::CompiledValidation>,
}

/// Properties specific to function nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Parameter names in order
    pub params: Vec<String>,
    /// Function body expression
    pub body: CompiledExpr,
}

/// Properties specific to type nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Named fields with their value types
    pub fields: Vec<CompiledTypeField>,
}

/// Properties specific to stratum nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Human-readable title for display
    pub title: Option<String>,
    /// Unicode symbol for visualization
    pub symbol: Option<String>,
    /// Default stride (ticks between updates)
    pub default_stride: u32,
}

/// Properties specific to era nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EraProperties {
    /// Documentation comment from CDSL source
    pub doc: Option<String>,
    /// Whether this is the starting era
    pub is_initial: bool,
    /// Whether this era ends simulation
    pub is_terminal: bool,
    /// Human-readable title
    pub title: Option<String>,
    /// Time step in seconds
    pub dt_seconds: f64,
    /// Stratum states for this era
    pub strata_states: IndexMap<StratumId, StratumState>,
    /// Transitions to other eras
    pub transitions: Vec<CompiledTransition>,
}
