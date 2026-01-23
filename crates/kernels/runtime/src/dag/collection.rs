//! DAG collection types for organizing DAGs across eras and strata.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::types::{EraId, Phase, StratumId};

use super::types::ExecutableDag;

/// Collection of DAGs for an era
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
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
#[derive(Debug, Default, Serialize, Deserialize)]
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
