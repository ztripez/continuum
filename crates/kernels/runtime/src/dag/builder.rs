//! DAG builder types for constructing execution graphs.

use std::collections::HashSet;

use indexmap::IndexMap;

use crate::reductions::ReductionOp;
use crate::types::{Phase, SignalId, StratumId};
use crate::vectorized::MemberSignalId;

use super::topology::{topological_levels, CycleError};
use super::types::{DagNode, ExecutableDag, NodeId, NodeKind};

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
