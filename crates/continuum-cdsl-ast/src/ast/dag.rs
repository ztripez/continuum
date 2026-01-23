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
    /// Retrieves a DAG for a specific era, phase, and stratum combination.
    ///
    /// Used by the runtime to look up the execution graph for the current
    /// simulation context. Each DAG defines the topological execution order
    /// for a specific (era, phase, stratum) triple.
    ///
    /// # Parameters
    ///
    /// * `era` - Era identifier (e.g., "early", "formation", "stabilization")
    /// * `phase` - Execution phase (Configure, Collect, Resolve, Fracture, Measure)
    /// * `stratum` - Stratum identifier (execution lane, e.g., "thermal", "tectonic")
    ///
    /// # Returns
    ///
    /// - `Some(&ExecutionDag)`: DAG exists for this combination
    /// - `None`: No DAG exists (stratum not active in this era/phase, or phase has no nodes)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl_ast::{DagSet, Phase};
    /// use continuum_cdsl_ast::foundation::{EraId, StratumId};
    ///
    /// let dag_set = DagSet::default();
    /// let era = EraId::new("early");
    /// let stratum = StratumId::new("thermal");
    ///
    /// if let Some(dag) = dag_set.get(&era, Phase::Resolve, &stratum) {
    ///     // Execute nodes in dag.levels
    /// }
    /// ```
    pub fn get(&self, era: &EraId, phase: Phase, stratum: &StratumId) -> Option<&ExecutionDag> {
        self.dags.get(&(era.clone(), phase, stratum.clone()))
    }

    /// Inserts a DAG into the set, replacing any existing DAG with the same key.
    ///
    /// The key is derived from the DAG's `(era, phase, stratum)` fields.
    /// If a DAG already exists for this combination, it is silently replaced.
    ///
    /// # Parameters
    ///
    /// * `dag` - Execution DAG to insert
    ///
    /// # Behavior
    ///
    /// - Extracts key from `dag.era`, `dag.phase`, `dag.stratum`
    /// - Replaces existing DAG if key collision occurs (silent overwrite)
    /// - Preserves insertion order determinism via `IndexMap`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl_ast::{DagSet, ExecutionDag, Phase};
    /// use continuum_cdsl_ast::foundation::{EraId, StratumId};
    ///
    /// let mut dag_set = DagSet::default();
    /// let dag = ExecutionDag {
    ///     era: EraId::new("early"),
    ///     phase: Phase::Resolve,
    ///     stratum: StratumId::new("thermal"),
    ///     levels: vec![],
    /// };
    ///
    /// dag_set.insert(dag);
    /// ```
    pub fn insert(&mut self, dag: ExecutionDag) {
        let key = (dag.era.clone(), dag.phase, dag.stratum.clone());
        self.dags.insert(key, dag);
    }

    /// Iterates over all DAGs for a specific era, across all phases and strata.
    ///
    /// Returns DAGs in **deterministic insertion order** (guaranteed by `IndexMap`).
    /// This is critical for deterministic execution: the runtime must process
    /// DAGs in a consistent order across runs for the same world.
    ///
    /// # Parameters
    ///
    /// * `era` - Era identifier to filter by
    ///
    /// # Returns
    ///
    /// Iterator yielding `&ExecutionDag` for all (phase, stratum) combinations
    /// in the specified era, in insertion order.
    ///
    /// # Determinism Guarantee
    ///
    /// Iteration order is deterministic and stable across:
    /// - Multiple runs of the same world
    /// - Same compilation from the same source
    /// - Serialization/deserialization round-trips
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl_ast::DagSet;
    /// use continuum_cdsl_ast::foundation::EraId;
    ///
    /// let dag_set = DagSet::default();
    /// let era = EraId::new("formation");
    ///
    /// for dag in dag_set.iter_era(&era) {
    ///     println!("Phase: {:?}, Stratum: {:?}", dag.phase, dag.stratum);
    /// }
    /// ```
    pub fn iter_era(&self, era: &EraId) -> impl Iterator<Item = &ExecutionDag> {
        self.dags
            .iter()
            .filter(move |((e, _, _), _)| e == era)
            .map(|(_, dag)| dag)
    }

    /// Iterates over all DAGs for a specific era and phase, across all strata.
    ///
    /// Returns DAGs in **deterministic insertion order** (guaranteed by `IndexMap`).
    /// Used by the runtime to execute all strata for a given phase in a consistent order.
    ///
    /// # Parameters
    ///
    /// * `era` - Era identifier to filter by
    /// * `phase` - Execution phase to filter by
    ///
    /// # Returns
    ///
    /// Iterator yielding `&ExecutionDag` for all strata in the specified era
    /// and phase, in insertion order.
    ///
    /// # Determinism Guarantee
    ///
    /// Iteration order is deterministic and stable. Strata are returned in the
    /// order they were inserted during compilation, which is derived from the
    /// sorted order of source files and declaration order within files.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl_ast::{DagSet, Phase};
    /// use continuum_cdsl_ast::foundation::EraId;
    ///
    /// let dag_set = DagSet::default();
    /// let era = EraId::new("formation");
    ///
    /// // Execute all strata for the Resolve phase
    /// for dag in dag_set.iter_era_phase(&era, Phase::Resolve) {
    ///     println!("Executing stratum: {:?}", dag.stratum);
    ///     // Execute nodes in dag.levels
    /// }
    /// ```
    pub fn iter_era_phase(&self, era: &EraId, phase: Phase) -> impl Iterator<Item = &ExecutionDag> {
        self.dags
            .iter()
            .filter(move |((e, p, _), _)| e == era && *p == phase)
            .map(|(_, dag)| dag)
    }
}
