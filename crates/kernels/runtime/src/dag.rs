//! Execution DAG for Simulation Scheduling.
//!
//! This module defines the directed acyclic graph (DAG) structure used to
//! schedule signal resolution and operator execution within each tick.
//!
//! # Structure
//!
//! - [`DagNode`] - A single execution unit (signal resolver, operator, field emitter)
//! - [`Level`] - A set of nodes with no inter-dependencies (can execute in parallel)
//! - [`ExecutableDag`] - A complete DAG for one (phase, stratum) combination
//! - [`EraDags`] - All DAGs for a single era
//! - [`DagSet`] - All DAGs across all eras
//!
//! # Execution Model
//!
//! DAGs are organized into topological levels. Within a level, all nodes can
//! execute in parallel since they have no dependencies on each other. Levels
//! are separated by barriers—all nodes in level N must complete before any
//! node in level N+1 begins.
//!
//! # Building DAGs
//!
//! Use [`DagBuilder`] to construct DAGs. Nodes are added with their dependencies,
//! then [`DagBuilder::build`] performs topological sorting to create levels.
//! If a cycle is detected, [`CycleError`] is returned.

use std::collections::HashSet;

use indexmap::{IndexMap, IndexSet};

use crate::reductions::ReductionOp;
use crate::types::{EntityId, EraId, Phase, SignalId, StratumId};
use crate::vectorized::MemberSignalId;

/// A single execution unit in the dependency graph.
///
/// Each node represents work to be done (signal resolution, operator execution,
/// etc.) along with its dependencies (signals it reads) and outputs (signals it writes).
#[derive(Debug, Clone)]
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
///
/// Node IDs follow conventions like `"sig.terra.temp"` for signals,
/// `"op.collect_heat"` for operators, `"frac.overheat"` for fractures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

/// The type of work a DAG node performs during execution.
///
/// Different node kinds execute in different phases and have different
/// effects on simulation state.
#[derive(Debug, Clone)]
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
///
/// Levels are the output of topological sorting. All nodes in a level can
/// run concurrently because none depends on another within the same level.
#[derive(Debug, Clone)]
pub struct Level {
    /// Nodes in this level, all of which can execute in parallel.
    pub nodes: Vec<DagNode>,
}

/// An executable DAG for a specific (phase, stratum, era) combination
#[derive(Debug, Clone)]
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

/// Collection of DAGs for an era
#[derive(Debug, Default)]
pub struct EraDags {
    /// DAGs indexed by (phase, stratum)
    dags: IndexMap<(Phase, StratumId), ExecutableDag>,
}

impl EraDags {
    /// Insert a DAG for a specific phase and stratum.
    pub fn insert(&mut self, dag: ExecutableDag) {
        let key = (dag.phase, dag.stratum.clone());
        self.dags.insert(key, dag);
    }

    /// Get a DAG for a specific phase and stratum
    pub fn get(&self, phase: Phase, stratum: &StratumId) -> Option<&ExecutableDag> {
        self.dags.get(&(phase, stratum.clone()))
    }

    /// Iterate over all DAGs for a phase.
    pub fn for_phase(&self, phase: Phase) -> impl Iterator<Item = &ExecutableDag> {
        self.dags
            .iter()
            .filter(move |((p, _), _)| *p == phase)
            .map(|(_, dag)| dag)
    }
}

/// All DAGs for all eras
#[derive(Debug, Default)]
pub struct DagSet {
    /// DAGs per era
    pub(crate) eras: IndexMap<EraId, EraDags>,
}

impl DagSet {
    /// Register all DAGs for a specific era.
    pub fn insert_era(&mut self, era: EraId, dags: EraDags) {
        self.eras.insert(era, dags);
    }

    /// Get DAGs for an era
    pub fn get_era(&self, era: &EraId) -> Option<&EraDags> {
        self.eras.get(era)
    }

    /// Number of eras
    pub fn era_count(&self) -> usize {
        self.eras.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.eras.is_empty()
    }
}

/// Builder for constructing DAGs from dependency information
pub struct DagBuilder {
    nodes: Vec<DagNode>,
    phase: Phase,
    stratum: StratumId,
}

impl DagBuilder {
    /// Create a new builder for a specific phase and stratum.
    pub fn new(phase: Phase, stratum: StratumId) -> Self {
        Self {
            nodes: Vec::new(),
            phase,
            stratum,
        }
    }

    /// Add a node to the DAG.
    pub fn add_node(&mut self, node: DagNode) {
        self.nodes.push(node);
    }

    /// Build the DAG with topological leveling.
    pub fn build(self) -> Result<ExecutableDag, CycleError> {
        let levels = topological_levels(&self.nodes)?;

        Ok(ExecutableDag {
            phase: self.phase,
            stratum: self.stratum,
            levels,
        })
    }
}

/// Error returned when a dependency cycle is detected during DAG construction.
///
/// Cycles prevent topological sorting, which is required for deterministic
/// execution ordering. This error contains the node IDs involved in the cycle
/// to help diagnose the circular dependency.
///
/// # Example
///
/// If signal A depends on signal B, and signal B depends on signal A,
/// both nodes will appear in `involved_nodes`.
#[derive(Debug)]
pub struct CycleError {
    /// Node identifiers involved in the dependency cycle.
    ///
    /// These are the nodes that could not be scheduled because they
    /// form a circular dependency chain.
    pub involved_nodes: Vec<NodeId>,
}

// ============================================================================
// Barrier-Aware DAG Builder
// ============================================================================

/// A dependency on a member signal, used for barrier scheduling.
///
/// When a signal reads from an aggregate over a member signal, we need to
/// ensure all member signal resolution happens before the aggregate computes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberSignalDependency {
    /// The member signal family being depended on.
    pub member_signal: MemberSignalId,
}

/// Aggregate barrier definition for population-level reduction.
///
/// This represents a barrier point in the DAG where all instances of a member
/// signal must be resolved before the aggregate can compute.
#[derive(Debug, Clone)]
pub struct AggregateBarrier {
    /// Unique identifier for this barrier.
    pub id: NodeId,
    /// The member signal being aggregated.
    pub member_signal: MemberSignalId,
    /// The reduction operation.
    pub reduction_op: ReductionOp,
    /// Output signal storing the aggregate result.
    pub output_signal: SignalId,
    /// Index into aggregate function table.
    pub aggregate_idx: usize,
}

/// Builder for constructing DAGs with barrier awareness.
///
/// This builder understands the relationship between:
/// 1. Member signal resolution nodes (per-entity)
/// 2. Population aggregate barriers
/// 3. Dependent signals that read aggregates
///
/// # Barrier Scheduling
///
/// When an aggregate barrier is added, the builder automatically:
/// - Creates a dependency on all member signal resolution nodes for that entity
/// - Schedules the aggregate after all instances are resolved
/// - Allows dependent signals to read the aggregate result
///
/// # Example
///
/// ```ignore
/// let mut builder = BarrierDagBuilder::new(Phase::Resolve, "physics".into());
///
/// // Add member signal resolution for all persons
/// builder.add_member_signal_resolve("person.age".parse().unwrap(), 0);
///
/// // Add aggregate barrier that computes mean age
/// builder.add_aggregate_barrier(AggregateBarrier {
///     id: NodeId("mean_age_barrier".to_string()),
///     member_signal: "person.age".parse().unwrap(),
///     reduction_op: ReductionOp::Mean,
///     output_signal: "population.mean_age".into(),
///     aggregate_idx: 0,
/// });
///
/// // Add signal that reads the aggregate
/// builder.add_signal_resolve(
///     "policy.retirement_age".into(),
///     0,
///     &["population.mean_age".into()], // Reads aggregate
/// );
/// ```
pub struct BarrierDagBuilder {
    nodes: Vec<DagNode>,
    phase: Phase,
    stratum: StratumId,
    /// Track which member signals have resolution nodes.
    member_signal_nodes: IndexMap<MemberSignalId, NodeId>,
    /// Track aggregate barriers for dependency resolution.
    aggregate_barriers: Vec<AggregateBarrier>,
}

impl BarrierDagBuilder {
    /// Create a new barrier-aware DAG builder.
    pub fn new(phase: Phase, stratum: StratumId) -> Self {
        Self {
            nodes: Vec::new(),
            phase,
            stratum,
            member_signal_nodes: IndexMap::new(),
            aggregate_barriers: Vec::new(),
        }
    }

    /// Add a member signal resolution node.
    ///
    /// This node resolves all instances of a member signal and must complete
    /// before any aggregate over this signal can be computed.
    pub fn add_member_signal_resolve(&mut self, member_signal: MemberSignalId, kernel_idx: usize) {
        let node_id = NodeId(format!("member.{}", member_signal));

        // Track this member signal node for barrier dependencies
        self.member_signal_nodes
            .insert(member_signal.clone(), node_id.clone());

        self.nodes.push(DagNode {
            id: node_id,
            reads: HashSet::new(), // Member signals read from previous tick
            writes: None,          // Writes to population storage, not global signal
            kind: NodeKind::MemberSignalResolve {
                member_signal,
                kernel_idx,
            },
        });
    }

    /// Add an aggregate barrier that computes a population reduction.
    ///
    /// This barrier automatically depends on the corresponding member signal
    /// resolution node, ensuring all instances are resolved before aggregation.
    pub fn add_aggregate_barrier(&mut self, barrier: AggregateBarrier) {
        // The barrier reads from the member signal (dependency on member resolve)
        let mut reads = HashSet::new();

        // If there's a corresponding member signal node, create a synthetic
        // signal ID for dependency tracking
        let member_dep_signal = SignalId::from(format!("__member.{}", barrier.member_signal));
        reads.insert(member_dep_signal.clone());

        // Update the member signal node to "write" this synthetic signal
        // so the topological sort understands the dependency
        if let Some(node_id) = self.member_signal_nodes.get(&barrier.member_signal) {
            // Find and update the node
            for node in &mut self.nodes {
                if &node.id == node_id {
                    node.writes = Some(member_dep_signal);
                    break;
                }
            }
        }

        self.nodes.push(DagNode {
            id: barrier.id.clone(),
            reads,
            writes: Some(barrier.output_signal.clone()),
            kind: NodeKind::PopulationAggregate {
                entity_id: barrier.member_signal.entity_id.clone(),
                member_signal: barrier.member_signal.clone(),
                reduction_op: barrier.reduction_op,
                output_signal: barrier.output_signal.clone(),
                aggregate_idx: barrier.aggregate_idx,
            },
        });

        self.aggregate_barriers.push(barrier);
    }

    /// Add a global signal resolution node.
    ///
    /// If this signal reads from an aggregate output, it will be scheduled
    /// after the aggregate barrier.
    pub fn add_signal_resolve(
        &mut self,
        signal: SignalId,
        resolver_idx: usize,
        reads: &[SignalId],
    ) {
        self.nodes.push(DagNode {
            id: NodeId(format!("sig.{}", signal)),
            reads: reads.iter().cloned().collect(),
            writes: Some(signal.clone()),
            kind: NodeKind::SignalResolve {
                signal,
                resolver_idx,
            },
        });
    }

    /// Build the DAG with barrier-aware topological leveling.
    pub fn build(self) -> Result<ExecutableDag, CycleError> {
        let levels = topological_levels(&self.nodes)?;

        Ok(ExecutableDag {
            phase: self.phase,
            stratum: self.stratum,
            levels,
        })
    }

    /// Get statistics about the barrier structure.
    pub fn barrier_stats(&self) -> BarrierStats {
        BarrierStats {
            member_signal_count: self.member_signal_nodes.len(),
            aggregate_barrier_count: self.aggregate_barriers.len(),
            total_node_count: self.nodes.len(),
        }
    }
}

/// Statistics about barrier structure in a DAG.
#[derive(Debug, Clone, Copy)]
pub struct BarrierStats {
    /// Number of member signal resolution nodes.
    pub member_signal_count: usize,
    /// Number of aggregate barrier nodes.
    pub aggregate_barrier_count: usize,
    /// Total number of nodes.
    pub total_node_count: usize,
}

// ============================================================================
// Gated Stratum Optimization
// ============================================================================

/// Marker for stratum execution eligibility.
///
/// Used to optimize gated and strided strata by avoiding work entirely
/// when the stratum is not eligible for the current tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StratumEligibility {
    /// Stratum is eligible and should execute.
    Eligible,
    /// Stratum is gated and should skip all work.
    Gated,
    /// Stratum has stride > 1 and tick is not aligned.
    StrideSkipped,
}

impl StratumEligibility {
    /// Determine eligibility from stratum state and tick.
    ///
    /// This is an O(1) check that avoids touching any entity arrays or
    /// signal storage when the stratum should skip execution.
    pub fn from_state(state: crate::types::StratumState, tick: u64) -> Self {
        match state {
            crate::types::StratumState::Active => StratumEligibility::Eligible,
            crate::types::StratumState::Gated => StratumEligibility::Gated,
            crate::types::StratumState::ActiveWithStride(stride) => {
                if tick.is_multiple_of(stride as u64) {
                    StratumEligibility::Eligible
                } else {
                    StratumEligibility::StrideSkipped
                }
            }
        }
    }

    /// Check if stratum should execute.
    pub fn should_execute(&self) -> bool {
        matches!(self, StratumEligibility::Eligible)
    }
}

// ============================================================================
// Barrier Verification
// ============================================================================

/// Verify that a DAG respects aggregate barrier semantics.
///
/// This checks that:
/// 1. All aggregates are scheduled after their source member signals
/// 2. All signals reading aggregates are scheduled after the aggregate
///
/// Returns `Ok(())` if valid, or `Err` with a description of the violation.
pub fn verify_barrier_semantics(dag: &ExecutableDag) -> Result<(), BarrierViolation> {
    let mut aggregate_levels: IndexMap<SignalId, usize> = IndexMap::new();
    let mut member_signal_levels: IndexMap<MemberSignalId, usize> = IndexMap::new();

    // First pass: record level of each node
    for (level_idx, level) in dag.levels.iter().enumerate() {
        for node in &level.nodes {
            match &node.kind {
                NodeKind::MemberSignalResolve { member_signal, .. } => {
                    member_signal_levels.insert(member_signal.clone(), level_idx);
                }
                NodeKind::PopulationAggregate {
                    member_signal,
                    output_signal,
                    ..
                } => {
                    aggregate_levels.insert(output_signal.clone(), level_idx);

                    // Verify: aggregate must be after member signal resolution
                    if let Some(&member_level) = member_signal_levels.get(member_signal) {
                        if level_idx <= member_level {
                            return Err(BarrierViolation::AggregateBeforeMemberSignal {
                                aggregate_signal: output_signal.clone(),
                                member_signal: member_signal.clone(),
                                aggregate_level: level_idx,
                                member_level,
                            });
                        }
                    }
                }
                NodeKind::SignalResolve { signal, .. } => {
                    // Check if this signal reads from any aggregate
                    for read_signal in &node.reads {
                        if let Some(&agg_level) = aggregate_levels.get(read_signal) {
                            if level_idx <= agg_level {
                                return Err(BarrierViolation::SignalBeforeAggregate {
                                    signal: signal.clone(),
                                    aggregate_signal: read_signal.clone(),
                                    signal_level: level_idx,
                                    aggregate_level: agg_level,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Error type for barrier semantic violations.
#[derive(Debug, Clone)]
pub enum BarrierViolation {
    /// An aggregate is scheduled before or at the same level as its source.
    AggregateBeforeMemberSignal {
        aggregate_signal: SignalId,
        member_signal: MemberSignalId,
        aggregate_level: usize,
        member_level: usize,
    },
    /// A signal reading an aggregate is scheduled before or at the aggregate level.
    SignalBeforeAggregate {
        signal: SignalId,
        aggregate_signal: SignalId,
        signal_level: usize,
        aggregate_level: usize,
    },
}

/// Compute topological levels using Kahn's algorithm
fn topological_levels(nodes: &[DagNode]) -> Result<Vec<Level>, CycleError> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Build adjacency and in-degree maps
    let mut in_degree: IndexMap<&NodeId, usize> = IndexMap::new();
    let mut dependents: IndexMap<&SignalId, Vec<&DagNode>> = IndexMap::new();

    // Initialize in-degrees
    for node in nodes {
        in_degree.insert(&node.id, 0);
    }

    // Map signal writes to nodes for dependency lookup
    let mut signal_to_node: IndexMap<&SignalId, &DagNode> = IndexMap::new();
    for node in nodes {
        if let Some(ref signal) = node.writes {
            signal_to_node.insert(signal, node);
        }
    }

    // Compute in-degrees based on signal reads
    // Deduplicate reads per node to avoid incrementing in-degree multiple times
    // for the same dependency
    for node in nodes {
        let mut seen_signals: IndexSet<&SignalId> = IndexSet::new();
        for read_signal in &node.reads {
            if signal_to_node.contains_key(read_signal) && seen_signals.insert(read_signal) {
                *in_degree.get_mut(&node.id).unwrap() += 1;
                dependents.entry(read_signal).or_default().push(node);
            }
        }
    }

    // Kahn's algorithm with level tracking
    let mut levels = Vec::new();
    let mut current_level: Vec<&DagNode> = nodes.iter().filter(|n| in_degree[&n.id] == 0).collect();

    let mut processed = 0;

    while !current_level.is_empty() {
        // Sort for determinism
        current_level.sort_by_key(|n| &n.id.0);

        let level = Level {
            nodes: current_level.iter().map(|n| (*n).clone()).collect(),
        };
        processed += level.nodes.len();

        // Find next level
        let mut next_level = Vec::new();
        for node in &current_level {
            if let Some(ref signal) = node.writes
                && let Some(deps) = dependents.get(signal)
            {
                for dep in deps {
                    let degree = in_degree.get_mut(&dep.id).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        next_level.push(*dep);
                    }
                }
            }
        }

        levels.push(level);
        current_level = next_level;
    }

    // Check for cycles
    if processed != nodes.len() {
        let cycle_nodes: Vec<NodeId> = nodes
            .iter()
            .filter(|n| in_degree[&n.id] > 0)
            .map(|n| n.id.clone())
            .collect();
        return Err(CycleError {
            involved_nodes: cycle_nodes,
        });
    }

    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, reads: &[&str], writes: Option<&str>) -> DagNode {
        DagNode {
            id: NodeId(id.to_string()),
            reads: reads.iter().map(|s| SignalId::from(*s)).collect(),
            writes: writes.map(|s| SignalId::from(s)),
            kind: NodeKind::SignalResolve {
                signal: SignalId::from(writes.unwrap_or(id)),
                resolver_idx: 0,
            },
        }
    }

    #[test]
    fn test_topological_levels_simple() {
        // A -> B -> C
        let nodes = vec![
            make_node("a", &[], Some("sig.a")),
            make_node("b", &["sig.a"], Some("sig.b")),
            make_node("c", &["sig.b"], Some("sig.c")),
        ];

        let levels = topological_levels(&nodes).unwrap();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].nodes.len(), 1);
        assert_eq!(levels[0].nodes[0].id.0, "a");
        assert_eq!(levels[1].nodes[0].id.0, "b");
        assert_eq!(levels[2].nodes[0].id.0, "c");
    }

    #[test]
    fn test_topological_levels_parallel() {
        // A, B (parallel) -> C
        let nodes = vec![
            make_node("a", &[], Some("sig.a")),
            make_node("b", &[], Some("sig.b")),
            make_node("c", &["sig.a", "sig.b"], Some("sig.c")),
        ];

        let levels = topological_levels(&nodes).unwrap();
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].nodes.len(), 2); // a and b parallel
        assert_eq!(levels[1].nodes.len(), 1); // c after both
    }

    #[test]
    fn test_cycle_detection() {
        // A -> B -> A (cycle)
        let nodes = vec![
            make_node("a", &["sig.b"], Some("sig.a")),
            make_node("b", &["sig.a"], Some("sig.b")),
        ];

        let result = topological_levels(&nodes);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_cycle_detection() {
        // A -> B -> C -> D -> B (cycle in longer chain)
        let nodes = vec![
            make_node("a", &[], Some("sig.a")),
            make_node("b", &["sig.a", "sig.d"], Some("sig.b")),
            make_node("c", &["sig.b"], Some("sig.c")),
            make_node("d", &["sig.c"], Some("sig.d")),
        ];

        let result = topological_levels(&nodes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        // b, c, d should be in the cycle (a has no incoming deps from the cycle)
        assert!(err.involved_nodes.len() >= 3);
    }

    #[test]
    fn test_diamond_no_false_positive() {
        // Diamond pattern: A -> B -> D, A -> C -> D (NOT a cycle)
        let nodes = vec![
            make_node("a", &[], Some("sig.a")),
            make_node("b", &["sig.a"], Some("sig.b")),
            make_node("c", &["sig.a"], Some("sig.c")),
            make_node("d", &["sig.b", "sig.c"], Some("sig.d")),
        ];

        let levels = topological_levels(&nodes).expect("diamond should not be a cycle");
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].nodes.len(), 1); // a
        assert_eq!(levels[1].nodes.len(), 2); // b, c parallel
        assert_eq!(levels[2].nodes.len(), 1); // d
    }

    #[test]
    fn test_deterministic_ordering() {
        // Same nodes added in different order should produce same levels
        let nodes1 = vec![
            make_node("z", &[], Some("sig.z")),
            make_node("a", &[], Some("sig.a")),
            make_node("m", &[], Some("sig.m")),
        ];
        let nodes2 = vec![
            make_node("a", &[], Some("sig.a")),
            make_node("m", &[], Some("sig.m")),
            make_node("z", &[], Some("sig.z")),
        ];

        let levels1 = topological_levels(&nodes1).unwrap();
        let levels2 = topological_levels(&nodes2).unwrap();

        assert_eq!(levels1.len(), levels2.len());
        let ids1: Vec<_> = levels1[0].nodes.iter().map(|n| &n.id.0).collect();
        let ids2: Vec<_> = levels2[0].nodes.iter().map(|n| &n.id.0).collect();
        assert_eq!(ids1, ids2); // Should be sorted: a, m, z
    }

    #[test]
    fn test_empty_dag() {
        let nodes: Vec<DagNode> = vec![];
        let levels = topological_levels(&nodes).unwrap();
        assert!(levels.is_empty());
    }

    #[test]
    fn test_single_node() {
        let nodes = vec![make_node("only", &[], Some("sig.only"))];
        let levels = topological_levels(&nodes).unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].nodes.len(), 1);
    }

    #[test]
    fn test_dag_builder_creates_correct_structure() {
        let mut builder = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
        builder.add_node(make_node("a", &[], Some("sig.a")));
        builder.add_node(make_node("b", &["sig.a"], Some("sig.b")));

        let dag = builder.build().unwrap();
        assert_eq!(dag.phase, Phase::Resolve);
        assert_eq!(dag.stratum.0, "terra");
        assert_eq!(dag.levels.len(), 2);
        assert_eq!(dag.node_count(), 2);
        assert!(!dag.is_empty());
    }

    #[test]
    fn test_era_dags_phase_iteration() {
        let mut era_dags = EraDags::default();

        // Add DAGs for different phases
        let mut collect_builder = DagBuilder::new(Phase::Collect, StratumId::from("terra"));
        collect_builder.add_node(make_node("collect_op", &[], None));
        era_dags.insert(collect_builder.build().unwrap());

        let mut resolve_builder = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
        resolve_builder.add_node(make_node("resolver", &[], Some("sig.x")));
        era_dags.insert(resolve_builder.build().unwrap());

        // Iterate by phase
        let collect_dags: Vec<_> = era_dags.for_phase(Phase::Collect).collect();
        let resolve_dags: Vec<_> = era_dags.for_phase(Phase::Resolve).collect();

        assert_eq!(collect_dags.len(), 1);
        assert_eq!(resolve_dags.len(), 1);
        assert_eq!(collect_dags[0].phase, Phase::Collect);
        assert_eq!(resolve_dags[0].phase, Phase::Resolve);
    }

    #[test]
    fn test_dag_set_multiple_eras() {
        let mut dag_set = DagSet::default();

        // Era 1
        let mut era1 = EraDags::default();
        let builder1 = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
        era1.insert(builder1.build().unwrap());
        dag_set.insert_era(EraId::from("era1"), era1);

        // Era 2
        let mut era2 = EraDags::default();
        let builder2 = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
        era2.insert(builder2.build().unwrap());
        dag_set.insert_era(EraId::from("era2"), era2);

        assert_eq!(dag_set.era_count(), 2);
        assert!(!dag_set.is_empty());
        assert!(dag_set.get_era(&EraId::from("era1")).is_some());
        assert!(dag_set.get_era(&EraId::from("era2")).is_some());
        assert!(dag_set.get_era(&EraId::from("era3")).is_none());
    }

    #[test]
    fn test_stratum_separation() {
        let mut era_dags = EraDags::default();

        // Same phase, different strata
        let builder_terra = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
        let builder_climate = DagBuilder::new(Phase::Resolve, StratumId::from("climate"));

        era_dags.insert(builder_terra.build().unwrap());
        era_dags.insert(builder_climate.build().unwrap());

        // Should be able to get each separately
        assert!(
            era_dags
                .get(Phase::Resolve, &StratumId::from("terra"))
                .is_some()
        );
        assert!(
            era_dags
                .get(Phase::Resolve, &StratumId::from("climate"))
                .is_some()
        );
        assert!(
            era_dags
                .get(Phase::Resolve, &StratumId::from("nonexistent"))
                .is_none()
        );
    }

    #[test]
    fn test_cycle_error_contains_all_cycle_nodes() {
        // Self-cycle: A depends on itself
        // This is actually impossible in our model since writes != reads for same node
        // But we can test a simple 2-node cycle
        let nodes = vec![
            make_node("x", &["sig.y"], Some("sig.x")),
            make_node("y", &["sig.x"], Some("sig.y")),
        ];

        let err = topological_levels(&nodes).unwrap_err();
        assert_eq!(err.involved_nodes.len(), 2);
        let ids: Vec<_> = err.involved_nodes.iter().map(|n| &n.0).collect();
        assert!(ids.contains(&&"x".to_string()));
        assert!(ids.contains(&&"y".to_string()));
    }

    // ========================================================================
    // Barrier-Aware DAG Builder Tests
    // ========================================================================

    use crate::reductions::ReductionOp;
    use crate::types::EntityId;
    use crate::vectorized::MemberSignalId;

    #[test]
    fn test_barrier_dag_builder_member_signal_before_aggregate() {
        let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("physics"));

        // Add member signal resolution for person.age
        let member_signal = MemberSignalId {
            entity_id: EntityId::from("person"),
            signal_name: "age".to_string(),
        };
        builder.add_member_signal_resolve(member_signal.clone(), 0);

        // Add aggregate barrier that computes mean age
        builder.add_aggregate_barrier(AggregateBarrier {
            id: NodeId("mean_age_barrier".to_string()),
            member_signal: member_signal.clone(),
            reduction_op: ReductionOp::Mean,
            output_signal: SignalId::from("population.mean_age"),
            aggregate_idx: 0,
        });

        let dag = builder.build().expect("should build without cycles");

        // Verify: member signal is in level 0, aggregate is in level 1
        assert_eq!(dag.levels.len(), 2);
        assert_eq!(dag.levels[0].nodes.len(), 1);
        assert_eq!(dag.levels[1].nodes.len(), 1);

        // Check level 0 is the member signal
        assert!(matches!(
            dag.levels[0].nodes[0].kind,
            NodeKind::MemberSignalResolve { .. }
        ));

        // Check level 1 is the aggregate
        assert!(matches!(
            dag.levels[1].nodes[0].kind,
            NodeKind::PopulationAggregate { .. }
        ));
    }

    #[test]
    fn test_barrier_dag_builder_signal_after_aggregate() {
        let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("physics"));

        // Add member signal resolution
        let member_signal = MemberSignalId {
            entity_id: EntityId::from("particle"),
            signal_name: "energy".to_string(),
        };
        builder.add_member_signal_resolve(member_signal.clone(), 0);

        // Add aggregate barrier
        builder.add_aggregate_barrier(AggregateBarrier {
            id: NodeId("total_energy_barrier".to_string()),
            member_signal: member_signal.clone(),
            reduction_op: ReductionOp::Sum,
            output_signal: SignalId::from("system.total_energy"),
            aggregate_idx: 0,
        });

        // Add a global signal that reads the aggregate
        builder.add_signal_resolve(
            SignalId::from("system.energy_ratio"),
            1,
            &[SignalId::from("system.total_energy")],
        );

        let dag = builder.build().expect("should build without cycles");

        // Verify: 3 levels - member signal, aggregate, dependent signal
        assert_eq!(dag.levels.len(), 3);
        assert!(matches!(
            dag.levels[0].nodes[0].kind,
            NodeKind::MemberSignalResolve { .. }
        ));
        assert!(matches!(
            dag.levels[1].nodes[0].kind,
            NodeKind::PopulationAggregate { .. }
        ));
        assert!(matches!(
            dag.levels[2].nodes[0].kind,
            NodeKind::SignalResolve { .. }
        ));
    }

    #[test]
    fn test_barrier_dag_builder_multiple_aggregates() {
        let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("physics"));

        let member_signal = MemberSignalId {
            entity_id: EntityId::from("cell"),
            signal_name: "temperature".to_string(),
        };
        builder.add_member_signal_resolve(member_signal.clone(), 0);

        // Multiple aggregates over the same member signal
        builder.add_aggregate_barrier(AggregateBarrier {
            id: NodeId("min_temp".to_string()),
            member_signal: member_signal.clone(),
            reduction_op: ReductionOp::Min,
            output_signal: SignalId::from("grid.min_temp"),
            aggregate_idx: 0,
        });

        builder.add_aggregate_barrier(AggregateBarrier {
            id: NodeId("max_temp".to_string()),
            member_signal: member_signal.clone(),
            reduction_op: ReductionOp::Max,
            output_signal: SignalId::from("grid.max_temp"),
            aggregate_idx: 1,
        });

        let dag = builder.build().expect("should build");

        // Both aggregates should be at level 1 (they can run in parallel)
        assert_eq!(dag.levels.len(), 2);
        assert_eq!(dag.levels[0].nodes.len(), 1); // member signal
        assert_eq!(dag.levels[1].nodes.len(), 2); // both aggregates
    }

    #[test]
    fn test_barrier_stats() {
        let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("test"));

        let ms1 = MemberSignalId {
            entity_id: EntityId::from("a"),
            signal_name: "x".to_string(),
        };
        let ms2 = MemberSignalId {
            entity_id: EntityId::from("b"),
            signal_name: "y".to_string(),
        };

        builder.add_member_signal_resolve(ms1.clone(), 0);
        builder.add_member_signal_resolve(ms2.clone(), 1);

        builder.add_aggregate_barrier(AggregateBarrier {
            id: NodeId("agg1".to_string()),
            member_signal: ms1,
            reduction_op: ReductionOp::Sum,
            output_signal: SignalId::from("out1"),
            aggregate_idx: 0,
        });

        let stats = builder.barrier_stats();
        assert_eq!(stats.member_signal_count, 2);
        assert_eq!(stats.aggregate_barrier_count, 1);
        assert_eq!(stats.total_node_count, 3);
    }

    // ========================================================================
    // Barrier Verification Tests
    // ========================================================================

    #[test]
    fn test_verify_barrier_semantics_valid() {
        let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("test"));

        let member_signal = MemberSignalId {
            entity_id: EntityId::from("entity"),
            signal_name: "value".to_string(),
        };
        builder.add_member_signal_resolve(member_signal.clone(), 0);

        builder.add_aggregate_barrier(AggregateBarrier {
            id: NodeId("agg".to_string()),
            member_signal,
            reduction_op: ReductionOp::Sum,
            output_signal: SignalId::from("sum"),
            aggregate_idx: 0,
        });

        builder.add_signal_resolve(SignalId::from("derived"), 0, &[SignalId::from("sum")]);

        let dag = builder.build().unwrap();
        assert!(verify_barrier_semantics(&dag).is_ok());
    }

    #[test]
    fn test_verify_barrier_semantics_rejects_aggregate_before_member() {
        // Manually construct a DAG where aggregate is at the SAME level
        // as member signal resolve (violation: aggregate should be AFTER member)
        // The member node must appear first in the level so it's registered
        // before the aggregate is processed.
        let member_signal = MemberSignalId {
            entity_id: EntityId::from("entity"),
            signal_name: "value".to_string(),
        };
        let output_signal = SignalId::from("sum");

        // Member signal resolve - processed first within level 0
        let member_node = DagNode {
            id: NodeId("member.entity.value".to_string()),
            reads: HashSet::new(),
            writes: None,
            kind: NodeKind::MemberSignalResolve {
                member_signal: member_signal.clone(),
                kernel_idx: 0,
            },
        };

        // Aggregate at same level - WRONG: should be at a later level
        let aggregate_node = DagNode {
            id: NodeId("agg.sum".to_string()),
            reads: HashSet::new(),
            writes: Some(output_signal.clone()),
            kind: NodeKind::PopulationAggregate {
                entity_id: EntityId::from("entity"),
                member_signal: member_signal.clone(),
                reduction_op: ReductionOp::Sum,
                output_signal: output_signal.clone(),
                aggregate_idx: 0,
            },
        };

        let dag = ExecutableDag {
            phase: Phase::Resolve,
            stratum: StratumId::from("test"),
            levels: vec![Level {
                // Member first so it's registered before aggregate is checked
                nodes: vec![member_node, aggregate_node],
            }],
        };

        let result = verify_barrier_semantics(&dag);
        assert!(result.is_err());

        match result.unwrap_err() {
            BarrierViolation::AggregateBeforeMemberSignal {
                aggregate_signal,
                member_signal: err_member,
                aggregate_level,
                member_level,
            } => {
                assert_eq!(aggregate_signal, output_signal);
                assert_eq!(err_member, member_signal);
                // Both at level 0 - aggregate is not AFTER member
                assert_eq!(aggregate_level, 0);
                assert_eq!(member_level, 0);
            }
            other => panic!("Expected AggregateBeforeMemberSignal, got {:?}", other),
        }
    }

    #[test]
    fn test_verify_barrier_semantics_rejects_signal_before_aggregate() {
        // Construct a DAG where a signal reading from aggregate
        // is at the same level as the aggregate (wrong order)
        let member_signal = MemberSignalId {
            entity_id: EntityId::from("entity"),
            signal_name: "value".to_string(),
        };
        let aggregate_signal = SignalId::from("sum");
        let derived_signal = SignalId::from("derived");

        // Level 0: Member signal resolve
        let member_node = DagNode {
            id: NodeId("member.entity.value".to_string()),
            reads: HashSet::new(),
            writes: None,
            kind: NodeKind::MemberSignalResolve {
                member_signal: member_signal.clone(),
                kernel_idx: 0,
            },
        };

        // Level 1: Both aggregate AND signal reading aggregate (WRONG - signal should be after)
        let aggregate_node = DagNode {
            id: NodeId("agg.sum".to_string()),
            reads: HashSet::new(),
            writes: Some(aggregate_signal.clone()),
            kind: NodeKind::PopulationAggregate {
                entity_id: EntityId::from("entity"),
                member_signal: member_signal.clone(),
                reduction_op: ReductionOp::Sum,
                output_signal: aggregate_signal.clone(),
                aggregate_idx: 0,
            },
        };

        let derived_node = DagNode {
            id: NodeId("sig.derived".to_string()),
            reads: {
                let mut reads = HashSet::new();
                reads.insert(aggregate_signal.clone());
                reads
            },
            writes: Some(derived_signal.clone()),
            kind: NodeKind::SignalResolve {
                signal: derived_signal.clone(),
                resolver_idx: 0,
            },
        };

        let dag = ExecutableDag {
            phase: Phase::Resolve,
            stratum: StratumId::from("test"),
            levels: vec![
                Level {
                    nodes: vec![member_node],
                },
                Level {
                    nodes: vec![aggregate_node, derived_node],
                },
            ],
        };

        let result = verify_barrier_semantics(&dag);
        assert!(result.is_err());

        match result.unwrap_err() {
            BarrierViolation::SignalBeforeAggregate {
                signal,
                aggregate_signal: err_agg,
                signal_level,
                aggregate_level,
            } => {
                assert_eq!(signal, derived_signal);
                assert_eq!(err_agg, aggregate_signal);
                assert_eq!(signal_level, 1);
                assert_eq!(aggregate_level, 1);
            }
            other => panic!("Expected SignalBeforeAggregate, got {:?}", other),
        }
    }

    // ========================================================================
    // Stratum Eligibility Tests
    // ========================================================================

    #[test]
    fn test_stratum_eligibility_active() {
        let eligibility = StratumEligibility::from_state(crate::types::StratumState::Active, 0);
        assert_eq!(eligibility, StratumEligibility::Eligible);
        assert!(eligibility.should_execute());
    }

    #[test]
    fn test_stratum_eligibility_gated() {
        let eligibility = StratumEligibility::from_state(crate::types::StratumState::Gated, 0);
        assert_eq!(eligibility, StratumEligibility::Gated);
        assert!(!eligibility.should_execute());
    }

    #[test]
    fn test_stratum_eligibility_stride_aligned() {
        // Stride 4, tick 8 is aligned (8 % 4 == 0)
        let eligibility =
            StratumEligibility::from_state(crate::types::StratumState::ActiveWithStride(4), 8);
        assert_eq!(eligibility, StratumEligibility::Eligible);
        assert!(eligibility.should_execute());
    }

    #[test]
    fn test_stratum_eligibility_stride_not_aligned() {
        // Stride 4, tick 7 is not aligned (7 % 4 != 0)
        let eligibility =
            StratumEligibility::from_state(crate::types::StratumState::ActiveWithStride(4), 7);
        assert_eq!(eligibility, StratumEligibility::StrideSkipped);
        assert!(!eligibility.should_execute());
    }

    #[test]
    fn test_stratum_eligibility_stride_zero_tick() {
        // Tick 0 should always be aligned (0 % n == 0 for any n)
        let eligibility =
            StratumEligibility::from_state(crate::types::StratumState::ActiveWithStride(10), 0);
        assert_eq!(eligibility, StratumEligibility::Eligible);
        assert!(eligibility.should_execute());
    }
}
