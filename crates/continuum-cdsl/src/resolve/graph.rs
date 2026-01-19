//! Execution graph compilation pass.
//!
//! This module transforms a resolved [`World`] into a set of deterministic
//! execution DAGs, one for each (Phase, Stratum, Era) combination.
//!
//! # Graph Construction Rules
//!
//! 1. **Phase Isolation** - Each phase has its own discrete DAG.
//! 2. **Stratum Partitioning** - Nodes are grouped by their assigned stratum.
//! 3. **Causal Dependencies** - Edges are established based on `reads` and `emits`.
//! 4. **Topological Order** - Nodes are sorted into levels for parallel execution.
//! 5. **Cycle Detection** - Circular dependencies are detected and reported as errors.

use crate::ast::World;
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Path, Phase, StratumId};
use indexmap::IndexMap;
use std::collections::HashMap;

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

/// A set of nodes that can execute in parallel.
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

/// Compiles execution DAGs for all phases and strata in the world.
pub fn compile_graphs(world: &World) -> Result<DagSet, Vec<CompileError>> {
    let mut dag_set = DagSet::default();
    let mut errors = Vec::new();

    // The set of all strata defined in the world
    let strata: Vec<_> = world.strata.keys().cloned().collect();

    // Phases that have execution DAGs
    let phases = [
        Phase::Configure,
        Phase::Collect,
        Phase::Resolve,
        Phase::Fracture,
        Phase::Measure,
    ];

    for phase in phases {
        for stratum_path in &strata {
            let stratum_id = StratumId::new(stratum_path.to_string());
            match build_dag(world, phase, &stratum_id) {
                Ok(Some(dag)) => {
                    dag_set.dags.insert((phase, stratum_id), dag);
                }
                Ok(None) => {
                    // Empty DAG, skip
                }
                Err(mut e) => {
                    errors.append(&mut e);
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(dag_set)
    } else {
        Err(errors)
    }
}

/// Builds a DAG for a specific phase and stratum.
fn build_dag(
    world: &World,
    phase: Phase,
    stratum: &StratumId,
) -> Result<Option<ExecutionDag>, Vec<CompileError>> {
    let mut nodes = Vec::new();

    // 1. Collect all nodes that execute in this phase and stratum
    for node in world.globals.values() {
        if node.stratum.as_ref() == Some(stratum) {
            if let Some(exec) = node.executions.iter().find(|e| e.phase == phase) {
                nodes.push((node.path.clone(), exec));
            }
        }
    }

    for node in world.members.values() {
        if node.stratum.as_ref() == Some(stratum) {
            if let Some(exec) = node.executions.iter().find(|e| e.phase == phase) {
                nodes.push((node.path.clone(), exec));
            }
        }
    }

    if nodes.is_empty() {
        return Ok(None);
    }

    // 2. Build dependency graph
    // We need to map which node "produces" a signal/field to know the edges.
    // In Resolve phase, a signal node produces itself.
    // In Collect phase, operators produce signal inputs.

    let mut producers: HashMap<Path, Path> = HashMap::new();
    for (path, exec) in &nodes {
        if phase == Phase::Resolve {
            // Signal nodes produce themselves in Resolve phase
            producers.insert(path.clone(), path.clone());
        } else {
            // In other phases, we look at 'emits'
            for emit in &exec.emits {
                producers.insert(emit.clone(), path.clone());
            }
        }
    }

    // Adjacency list: node -> [nodes it depends on]
    let mut adj: HashMap<Path, Vec<Path>> = HashMap::new();
    let mut in_degree: HashMap<Path, usize> = HashMap::new();

    for (path, _) in &nodes {
        in_degree.insert(path.clone(), 0);
        adj.insert(path.clone(), Vec::new());
    }

    for (path, exec) in &nodes {
        for read in &exec.reads {
            if let Some(producer_path) = producers.get(read) {
                // Self-dependency is allowed if it's 'prev', but extract_dependencies
                // should have filtered that out or marked it as such.
                // For now, if producer is same as current, it's either a bug in CDSL
                // or a valid intra-block dependency (not an edge in the DAG).
                if producer_path != path {
                    adj.entry(producer_path.clone())
                        .or_default()
                        .push(path.clone());
                    *in_degree.entry(path.clone()).or_default() += 1;
                }
            }
        }
    }

    // 3. Topological sort (Kahn's algorithm)
    let mut levels = Vec::new();
    let mut current_level: Vec<Path> = in_degree
        .iter()
        .filter(|&(_, &deg)| deg == 0)
        .map(|(p, _)| p.clone())
        .collect();

    let mut processed_count = 0;

    while !current_level.is_empty() {
        // Sort level for determinism
        current_level.sort();

        let level_nodes = current_level.clone();
        processed_count += level_nodes.len();
        levels.push(ExecutionLevel { nodes: level_nodes });

        let mut next_level = Vec::new();
        for node_path in &current_level {
            if let Some(dependents) = adj.get(node_path) {
                for dep in dependents {
                    let degree = in_degree.get_mut(dep).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        next_level.push(dep.clone());
                    }
                }
            }
        }
        current_level = next_level;
    }

    // 4. Cycle detection
    if processed_count != nodes.len() {
        let mut cycle_nodes: Vec<Path> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg > 0)
            .map(|(p, _)| p.clone())
            .collect();
        cycle_nodes.sort();

        return Err(vec![CompileError::new(
            ErrorKind::CyclicDependency,
            crate::foundation::Span::new(0, 0, 0, 0), // Should ideally find a better span
            format!(
                "Circular dependency detected in {:?} phase, stratum {}: {:?}",
                phase, stratum, cycle_nodes
            ),
        )]);
    }

    Ok(Some(ExecutionDag {
        phase,
        stratum: stratum.clone(),
        levels,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Execution, ExecutionBody, Node, RoleData};
    use crate::foundation::{Phase, Span, StratumId};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    #[test]
    fn test_dag_compilation_simple() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        // signal a { resolve { 1.0 } }
        let mut node_a = Node::new(Path::from_str("a"), span, RoleData::Signal, ());
        node_a.stratum = Some(StratumId::new("default"));
        node_a.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                crate::foundation::Type::Bool, // Dummy type
                span,
            )),
            vec![],
            vec![],
            span,
        )];

        // signal b { resolve { signal.a } }
        let mut node_b = Node::new(Path::from_str("b"), span, RoleData::Signal, ());
        node_b.stratum = Some(StratumId::new("default"));
        node_b.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_str("a")),
                crate::foundation::Type::Bool, // Dummy type
                span,
            )),
            vec![Path::from_str("a")],
            vec![],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_str("default"),
            crate::ast::Stratum::new(StratumId::new("default"), Path::from_str("default"), span),
        );
        world.globals.insert(node_a.path.clone(), node_a);
        world.globals.insert(node_b.path.clone(), node_b);

        let dag_set = compile_graphs(&world).expect("Failed to compile graphs");
        let dag = dag_set
            .dags
            .get(&(Phase::Resolve, StratumId::new("default")))
            .expect("DAG not found");

        assert_eq!(dag.levels.len(), 2);
        assert_eq!(dag.levels[0].nodes, vec![Path::from_str("a")]);
        assert_eq!(dag.levels[1].nodes, vec![Path::from_str("b")]);
    }

    #[test]
    fn test_dag_compilation_cycle() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        // signal a { resolve { signal.b } }
        let mut node_a = Node::new(Path::from_str("a"), span, RoleData::Signal, ());
        node_a.stratum = Some(StratumId::new("default"));
        node_a.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_str("b")),
                crate::foundation::Type::Bool,
                span,
            )),
            vec![Path::from_str("b")],
            vec![],
            span,
        )];

        // signal b { resolve { signal.a } }
        let mut node_b = Node::new(Path::from_str("b"), span, RoleData::Signal, ());
        node_b.stratum = Some(StratumId::new("default"));
        node_b.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_str("a")),
                crate::foundation::Type::Bool,
                span,
            )),
            vec![Path::from_str("a")],
            vec![],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_str("default"),
            crate::ast::Stratum::new(StratumId::new("default"), Path::from_str("default"), span),
        );
        world.globals.insert(node_a.path.clone(), node_a);
        world.globals.insert(node_b.path.clone(), node_b);

        let result = compile_graphs(&world);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
    }
}
