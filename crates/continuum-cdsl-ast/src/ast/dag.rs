//! Execution DAG structures for compiled worlds.

use crate::foundation::{Path, Phase, StratumId};
use indexmap::IndexMap;

/// An execution graph for a specific phase and stratum.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionDag {
    /// The phase this DAG executes in.
    pub phase: Phase,
    /// The stratum this DAG belongs to.
    pub stratum: StratumId,
    /// Execution levels in topological order.
    /// Nodes in the same level can execute in parallel.
    pub levels: Vec<ExecutionLevel>,
}

/// A level in the execution DAG.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionLevel {
    /// Paths to the nodes in this level.
    pub nodes: Vec<Path>,
}

/// Collection of DAGs for a simulation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DagSet {
    /// DAGs indexed by (phase, stratum).
    /// Using a Map for efficient lookup by the runtime.
    pub dags: IndexMap<(Phase, StratumId), ExecutionDag>,
}
