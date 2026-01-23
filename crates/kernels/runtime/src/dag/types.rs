//! Core DAG node types and executable DAG structure.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::reductions::ReductionOp;
use crate::types::{EntityId, Phase, SignalId, StratumId};
use crate::vectorized::MemberSignalId;

/// A single execution unit in the dependency graph.
///
/// Each node represents work to be done (signal resolution, operator execution,
/// etc.) along with its dependencies (signals it reads) and outputs (signals it writes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    /// Unique identifier for this node (e.g., `"sig.terra.temp"`).
    pub id: NodeId,
    /// Signal IDs this node reads during execution.
    pub reads: HashSet<SignalId>,
    /// Signal ID this node writes, if any (only for resolve nodes).
    pub writes: Option<SignalId>,
    /// The kind of work this node performs.
    pub kind: NodeKind,
}

/// Unique identifier for a node within a DAG.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

/// The type of work a DAG node performs during execution.
///
/// Different node kinds execute in different phases and have different
/// effects on simulation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    /// Compute a new signal value from its resolver expression.
    SignalResolve {
        /// The signal being resolved.
        signal: SignalId,
        /// Index into the resolver function table.
        resolver_idx: usize,
    },
    /// Execute a collect-phase operator that accumulates inputs.
    OperatorCollect {
        /// Index into the operator function table.
        operator_idx: usize,
    },
    /// Execute a measure-phase operator that emits observations.
    OperatorMeasure {
        /// Index into the operator function table.
        operator_idx: usize,
    },
    /// Emit a field value for external observation.
    FieldEmit {
        /// Index into the field emitter table.
        field_idx: usize,
    },
    /// Evaluate a fracture condition and potentially emit a response.
    Fracture {
        /// Index into the fracture detector table.
        fracture_idx: usize,
    },
    /// Observe signals and conditionally emit chronicle events.
    ///
    /// Chronicles are observer-only constructs that execute in the Measure phase.
    /// They read resolved signals and emit events for logging and analytics.
    /// Removing all chronicles must not change simulation results.
    ChronicleObserve {
        /// Index into the chronicle handler table.
        chronicle_idx: usize,
    },
    /// Execute a lane kernel for member signal resolution (L1/L2/L3).
    ///
    /// This is the two-level execution model where a DAG node can expand
    /// internally to a vectorized lane kernel operating over all instances.
    MemberSignalResolve {
        /// The member signal being resolved.
        member_signal: MemberSignalId,
        /// Index into the lane kernel registry.
        kernel_idx: usize,
    },
    /// Compute a population aggregate over an entity's member signal.
    ///
    /// This node acts as a **scheduling barrier**: it reads from all instances
    /// of a member signal and produces a scalar result. All member signal
    /// resolution for the entity must complete before this node executes.
    ///
    /// # Barrier Semantics
    ///
    /// ```text
    /// Entity instances:  [e0, e1, e2, ..., eN]
    ///                          ↓
    /// MemberSignal resolution (parallel)
    ///                          ↓
    ///                    ══════════════
    ///                    BARRIER: Aggregate
    ///                    ══════════════
    ///                          ↓
    /// Dependent signals (after barrier)
    /// ```
    ///
    /// # Available Operations
    ///
    /// - `agg.sum(entity, field)` - Sum all values
    /// - `agg.mean(entity, field)` - Average of all values
    /// - `agg.min(entity, field)` - Minimum value (lowest index wins ties)
    /// - `agg.max(entity, field)` - Maximum value (lowest index wins ties)
    /// - `agg.count(entity, predicate)` - Count matching instances
    PopulationAggregate {
        /// The entity type being aggregated over.
        entity_id: EntityId,
        /// The member signal to aggregate.
        member_signal: MemberSignalId,
        /// The reduction operation to apply.
        reduction_op: ReductionOp,
        /// Signal that stores the aggregate result.
        output_signal: SignalId,
        /// Index into the aggregate function table.
        aggregate_idx: usize,
    },
}

/// A set of DAG nodes with no inter-dependencies that can execute in parallel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Level {
    /// Nodes in this level, all of which can execute in parallel.
    pub nodes: Vec<DagNode>,
}

/// An executable DAG for a specific (phase, stratum, era) combination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableDag {
    /// The phase this DAG executes in
    pub phase: Phase,
    /// The stratum this DAG belongs to
    pub stratum: StratumId,
    /// Execution levels in order
    pub levels: Vec<Level>,
}

impl ExecutableDag {
    /// Total number of nodes
    pub fn node_count(&self) -> usize {
        self.levels.iter().map(|l| l.nodes.len()).sum()
    }

    /// Check if DAG is empty
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}
