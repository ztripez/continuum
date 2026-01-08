//! IR to Runtime DAG compilation
//!
//! Transforms CompiledWorld into executable runtime DAGs.

use indexmap::IndexMap;
use thiserror::Error;

use continuum_foundation::{EraId, SignalId, StratumId};
use continuum_runtime::dag::{
    CycleError, DagBuilder, DagNode, DagSet, EraDags, ExecutableDag, NodeId, NodeKind,
};
use continuum_runtime::types::Phase;

use crate::{CompiledWorld, OperatorPhaseIr};

/// Errors during DAG compilation
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("cycle detected in signal dependencies: {nodes:?}")]
    CycleDetected { nodes: Vec<String> },

    #[error("undefined stratum: {0}")]
    UndefinedStratum(String),

    #[error("undefined era: {0}")]
    UndefinedEra(String),
}

impl From<CycleError> for CompileError {
    fn from(e: CycleError) -> Self {
        CompileError::CycleDetected {
            nodes: e.involved_nodes.into_iter().map(|n| n.0).collect(),
        }
    }
}

/// Result of compilation
pub struct CompilationResult {
    /// Executable DAG set for all eras
    pub dags: DagSet,
    /// Resolver function indices (signal_id -> index)
    pub resolver_indices: IndexMap<SignalId, usize>,
    /// Operator function indices
    pub operator_indices: IndexMap<String, usize>,
    /// Field emitter indices
    pub field_indices: IndexMap<String, usize>,
    /// Fracture detector indices
    pub fracture_indices: IndexMap<String, usize>,
}

/// Compile a world to executable DAGs
pub fn compile(world: &CompiledWorld) -> Result<CompilationResult, CompileError> {
    let compiler = Compiler::new(world);
    compiler.compile()
}

struct Compiler<'a> {
    world: &'a CompiledWorld,
    resolver_indices: IndexMap<SignalId, usize>,
    operator_indices: IndexMap<String, usize>,
    field_indices: IndexMap<String, usize>,
    fracture_indices: IndexMap<String, usize>,
}

impl<'a> Compiler<'a> {
    fn new(world: &'a CompiledWorld) -> Self {
        Self {
            world,
            resolver_indices: IndexMap::new(),
            operator_indices: IndexMap::new(),
            field_indices: IndexMap::new(),
            fracture_indices: IndexMap::new(),
        }
    }

    fn compile(mut self) -> Result<CompilationResult, CompileError> {
        // Assign indices to all entities
        self.assign_indices();

        // Build DAGs for each era
        let mut dag_set = DagSet::default();

        for (era_id, _era) in &self.world.eras {
            let era_dags = self.compile_era(era_id)?;
            dag_set.insert_era(era_id.clone(), era_dags);
        }

        Ok(CompilationResult {
            dags: dag_set,
            resolver_indices: self.resolver_indices,
            operator_indices: self.operator_indices,
            field_indices: self.field_indices,
            fracture_indices: self.fracture_indices,
        })
    }

    fn assign_indices(&mut self) {
        // Assign resolver indices
        for (idx, (signal_id, _)) in self.world.signals.iter().enumerate() {
            self.resolver_indices.insert(signal_id.clone(), idx);
        }

        // Assign operator indices
        for (idx, (op_id, _)) in self.world.operators.iter().enumerate() {
            self.operator_indices.insert(op_id.0.clone(), idx);
        }

        // Assign field indices (only for fields with measure expressions)
        let mut field_idx = 0;
        for (field_id, field) in &self.world.fields {
            if field.measure.is_some() {
                self.field_indices.insert(field_id.0.clone(), field_idx);
                field_idx += 1;
            }
        }

        // Assign fracture indices
        for (idx, (fracture_id, _)) in self.world.fractures.iter().enumerate() {
            self.fracture_indices.insert(fracture_id.0.clone(), idx);
        }
    }

    fn compile_era(&self, _era_id: &EraId) -> Result<EraDags, CompileError> {
        let mut era_dags = EraDags::default();

        // Collect active strata for this era
        let active_strata: Vec<&StratumId> = self.world.strata.keys().collect();

        // Build DAGs per (phase, stratum)
        for stratum_id in active_strata {
            // Collect phase: operators
            if let Some(dag) = self.build_collect_dag(stratum_id)? {
                era_dags.insert(dag);
            }

            // Resolve phase: signals
            if let Some(dag) = self.build_resolve_dag(stratum_id)? {
                era_dags.insert(dag);
            }

            // Fracture phase: fracture detectors
            if let Some(dag) = self.build_fracture_dag(stratum_id)? {
                era_dags.insert(dag);
            }

            // Measure phase: fields
            if let Some(dag) = self.build_measure_dag(stratum_id)? {
                era_dags.insert(dag);
            }
        }

        Ok(era_dags)
    }

    fn build_collect_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Collect, (*stratum_id).clone());

        for (op_id, operator) in &self.world.operators {
            if operator.stratum != *stratum_id {
                continue;
            }

            if operator.phase != OperatorPhaseIr::Collect {
                continue;
            }

            let node = DagNode {
                id: NodeId(format!("op.{}", op_id.0)),
                reads: operator
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: None, // Operators don't write signals directly
                kind: NodeKind::OperatorCollect {
                    operator_idx: self.operator_indices[&op_id.0],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    fn build_resolve_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Resolve, (*stratum_id).clone());

        for (signal_id, signal) in &self.world.signals {
            if signal.stratum != *stratum_id {
                continue;
            }

            // Skip signals without resolve expressions
            if signal.resolve.is_none() {
                continue;
            }

            let node = DagNode {
                id: NodeId(format!("sig.{}", signal_id.0)),
                reads: signal
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: Some(continuum_runtime::SignalId(signal_id.0.clone())),
                kind: NodeKind::SignalResolve {
                    signal: continuum_runtime::SignalId(signal_id.0.clone()),
                    resolver_idx: self.resolver_indices[signal_id],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    fn build_fracture_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Fracture, (*stratum_id).clone());

        // Fractures aren't bound to a specific stratum in IR
        // Add all fractures to the first stratum only (to avoid duplicating execution)
        let first_stratum = self.world.strata.keys().next();
        if first_stratum != Some(stratum_id) {
            let dag = builder.build()?;
            return Ok(if dag.is_empty() { None } else { Some(dag) });
        }

        for (fracture_id, fracture) in &self.world.fractures {
            let node = DagNode {
                id: NodeId(format!("frac.{}", fracture_id.0)),
                reads: fracture
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: None,
                kind: NodeKind::Fracture {
                    fracture_idx: self.fracture_indices[&fracture_id.0],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    fn build_measure_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Measure, (*stratum_id).clone());

        // Fields with measure expressions become OperatorMeasure nodes
        // Fields without measure expressions would be FieldEmit nodes (for dependency tracking only)
        for (field_id, field) in &self.world.fields {
            if field.stratum != *stratum_id {
                continue;
            }

            // Only create nodes for fields that have measure expressions
            if field.measure.is_some() {
                let node = DagNode {
                    id: NodeId(format!("field.{}", field_id.0)),
                    reads: field
                        .reads
                        .iter()
                        .map(|s| continuum_runtime::SignalId(s.0.clone()))
                        .collect(),
                    writes: None,
                    kind: NodeKind::OperatorMeasure {
                        operator_idx: self.field_indices[&field_id.0],
                    },
                };
                builder.add_node(node);
            }
        }

        // Also add measure-phase operators
        for (op_id, operator) in &self.world.operators {
            if operator.stratum != *stratum_id {
                continue;
            }

            if operator.phase != OperatorPhaseIr::Measure {
                continue;
            }

            let node = DagNode {
                id: NodeId(format!("op.{}", op_id.0)),
                reads: operator
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: None,
                kind: NodeKind::OperatorMeasure {
                    operator_idx: self.operator_indices[&op_id.0],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower, CompiledWorld};
    use continuum_dsl::parse;

    fn parse_and_lower(src: &str) -> CompiledWorld {
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        lower(&unit.unwrap()).unwrap()
    }

    #[test]
    fn test_compile_empty() {
        let world = CompiledWorld {
            constants: IndexMap::new(),
            config: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
        };

        let result = compile(&world).unwrap();
        assert!(result.dags.is_empty());
    }

    #[test]
    fn test_compile_simple_signal() {
        let src = r#"
            strata.terra {
                : title("Terra")
            }

            era.hadean {
                : initial
            }

            signal.terra.temp {
                : strata(terra)
                resolve { prev + 1.0 }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Should have one era
        assert_eq!(result.dags.era_count(), 1);

        // Signal should have resolver index
        let sig_id = SignalId::from("terra.temp");
        assert!(result.resolver_indices.contains_key(&sig_id));
    }

    #[test]
    fn test_compile_signal_dependencies() {
        let src = r#"
            strata.terra {}

            era.main {
                : initial
            }

            signal.terra.a {
                : strata(terra)
                resolve { 1.0 }
            }

            signal.terra.b {
                : strata(terra)
                resolve { signal.terra.a * 2.0 }
            }

            signal.terra.c {
                : strata(terra)
                resolve { signal.terra.b + signal.terra.a }
            }
        "#;

        let world = parse_and_lower(src);
        let _result = compile(&world).unwrap();

        // Check that signal c depends on both a and b
        let sig_c = world.signals.get(&SignalId::from("terra.c")).unwrap();
        assert_eq!(sig_c.reads.len(), 2);
    }
}
