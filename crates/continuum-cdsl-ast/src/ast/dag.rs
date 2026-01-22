//! Execution DAG structures for compiled worlds.

use crate::foundation::{EraId, Path, Phase, StratumId};
use indexmap::IndexMap;

/// An execution graph for a specific era, phase, and stratum.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionDag {
    /// The era this DAG belongs to.
    pub era: EraId,
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
///
/// DAGs are indexed by (era, phase, stratum) to support different execution
/// policies across eras (gated strata, different cadences, etc.).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DagSet {
    /// DAGs indexed by (era, phase, stratum).
    /// Using a Map for efficient lookup by the runtime.
    pub dags: IndexMap<(EraId, Phase, StratumId), ExecutionDag>,
}

impl DagSet {
    /// Get a DAG for a specific era, phase, and stratum.
    pub fn get(&self, era: &EraId, phase: Phase, stratum: &StratumId) -> Option<&ExecutionDag> {
        self.dags.get(&(era.clone(), phase, stratum.clone()))
    }

    /// Insert a DAG into the set.
    pub fn insert(&mut self, dag: ExecutionDag) {
        let key = (dag.era.clone(), dag.phase, dag.stratum.clone());
        self.dags.insert(key, dag);
    }

    /// Iterate over all DAGs for a specific era.
    pub fn iter_era(&self, era: &EraId) -> impl Iterator<Item = &ExecutionDag> {
        self.dags
            .iter()
            .filter(move |((e, _, _), _)| e == era)
            .map(|(_, dag)| dag)
    }

    /// Iterate over all DAGs for a specific era and phase.
    pub fn iter_era_phase(&self, era: &EraId, phase: Phase) -> impl Iterator<Item = &ExecutionDag> {
        self.dags
            .iter()
            .filter(move |((e, p, _), _)| e == era && *p == phase)
            .map(|(_, dag)| dag)
    }
}
