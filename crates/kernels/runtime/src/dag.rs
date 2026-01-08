//! Execution DAG
//!
//! Represents the execution graph with topological levels.

use std::collections::HashSet;

use indexmap::IndexMap;

use crate::types::{SignalId, StratumId, EraId, Phase};

/// A node in the execution DAG
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Signals this node reads
    pub reads: HashSet<SignalId>,
    /// Signal this node writes (for signal resolve nodes)
    pub writes: Option<SignalId>,
    /// The node's execution behavior
    pub kind: NodeKind,
}

/// Unique identifier for a DAG node
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

/// The kind of work a node performs
#[derive(Debug, Clone)]
pub enum NodeKind {
    /// Resolve a signal value
    SignalResolve {
        signal: SignalId,
        /// Index into the resolver function table
        resolver_idx: usize,
    },
    /// Execute a collect-phase operator
    OperatorCollect {
        /// Index into the operator function table
        operator_idx: usize,
    },
    /// Execute a measure-phase operator
    OperatorMeasure {
        operator_idx: usize,
    },
    /// Emit a field value
    FieldEmit {
        field_idx: usize,
    },
    /// Evaluate and potentially emit fracture
    Fracture {
        fracture_idx: usize,
    },
}

/// A topological level - nodes that can execute in parallel
#[derive(Debug, Clone)]
pub struct Level {
    /// Nodes in this level (no dependencies between them)
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
    pub fn insert(&mut self, dag: ExecutableDag) {
        let key = (dag.phase, dag.stratum.clone());
        self.dags.insert(key, dag);
    }

    /// Get a DAG for a specific phase and stratum
    pub fn get(&self, phase: Phase, stratum: &StratumId) -> Option<&ExecutableDag> {
        self.dags.get(&(phase, stratum.clone()))
    }

    /// Iterate over all DAGs for a phase
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
    pub fn insert_era(&mut self, era: EraId, dags: EraDags) {
        self.eras.insert(era, dags);
    }

    /// Get DAGs for an era
    pub fn get_era(&self, era: &EraId) -> Option<&EraDags> {
        self.eras.get(era)
    }
}

/// Builder for constructing DAGs from dependency information
pub struct DagBuilder {
    nodes: Vec<DagNode>,
    phase: Phase,
    stratum: StratumId,
}

impl DagBuilder {
    pub fn new(phase: Phase, stratum: StratumId) -> Self {
        Self {
            nodes: Vec::new(),
            phase,
            stratum,
        }
    }

    /// Add a node to the DAG
    pub fn add_node(&mut self, node: DagNode) {
        self.nodes.push(node);
    }

    /// Build the DAG with topological leveling
    pub fn build(self) -> Result<ExecutableDag, CycleError> {
        let levels = topological_levels(&self.nodes)?;

        Ok(ExecutableDag {
            phase: self.phase,
            stratum: self.stratum,
            levels,
        })
    }
}

/// Error when a cycle is detected
#[derive(Debug)]
pub struct CycleError {
    pub involved_nodes: Vec<NodeId>,
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
    for node in nodes {
        for read_signal in &node.reads {
            if signal_to_node.contains_key(read_signal) {
                *in_degree.get_mut(&node.id).unwrap() += 1;
                dependents.entry(read_signal).or_default().push(node);
            }
        }
    }

    // Kahn's algorithm with level tracking
    let mut levels = Vec::new();
    let mut current_level: Vec<&DagNode> = nodes
        .iter()
        .filter(|n| in_degree[&n.id] == 0)
        .collect();

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
            reads: reads.iter().map(|s| SignalId(s.to_string())).collect(),
            writes: writes.map(|s| SignalId(s.to_string())),
            kind: NodeKind::SignalResolve {
                signal: SignalId(writes.unwrap_or(id).to_string()),
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
}
