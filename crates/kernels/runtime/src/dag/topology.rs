//! Topological sorting and cycle detection for DAG construction.

use indexmap::{IndexMap, IndexSet};

use crate::types::SignalId;

use super::types::{DagNode, Level, NodeId};

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

/// Compute topological levels using Kahn's algorithm
pub(super) fn topological_levels(nodes: &[DagNode]) -> Result<Vec<Level>, CycleError> {
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
