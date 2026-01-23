//! Barrier semantic verification for DAG correctness.

use indexmap::IndexMap;

use crate::types::SignalId;
use crate::vectorized::MemberSignalId;

use super::types::{ExecutableDag, NodeKind};

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
