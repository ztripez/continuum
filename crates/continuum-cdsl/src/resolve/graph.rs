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

use crate::ast::{RoleId, World};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Path, Phase, Span, StratumId};
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

/// Traces a concrete dependency path through a cycle.
///
/// Given a set of nodes known to be in a cycle and an adjacency list,
/// this function performs DFS to find one complete cycle path.
///
/// # Returns
/// A path like `[a, b, c, a]` showing the dependency chain.
fn trace_cycle_path(cycle_nodes: &[Path], adj: &IndexMap<Path, Vec<Path>>) -> Vec<Path> {
    use std::collections::HashSet;

    if cycle_nodes.is_empty() {
        return Vec::new();
    }

    let cycle_set: HashSet<_> = cycle_nodes.iter().cloned().collect();
    let start = &cycle_nodes[0];
    let mut path = vec![start.clone()];
    let mut visited = HashSet::new();
    visited.insert(start.clone());

    let mut current = start;
    loop {
        // Find the next node in the cycle by looking at adjacency list
        if let Some(neighbors) = adj.get(current) {
            // Find a neighbor that's in the cycle
            if let Some(next) = neighbors.iter().find(|n| cycle_set.contains(n)) {
                if visited.contains(next) {
                    // Found the cycle completion
                    path.push(next.clone());
                    break;
                } else {
                    // Continue tracing
                    visited.insert(next.clone());
                    path.push(next.clone());
                    current = next;
                }
            } else {
                // Dead end, shouldn't happen but defend against it
                break;
            }
        } else {
            // No neighbors, shouldn't happen
            break;
        }
    }

    path
}

/// Builds a DAG for a specific phase and stratum.
fn build_dag(
    world: &World,
    phase: Phase,
    stratum: &StratumId,
) -> Result<Option<ExecutionDag>, Vec<CompileError>> {
    // Store (Path, Execution, Span) to enable accurate error reporting
    let mut nodes = Vec::new();
    let mut node_spans: IndexMap<Path, Span> = IndexMap::new();

    // 1. Collect all nodes that execute in this phase and stratum
    for node in world.globals.values() {
        if node.stratum.as_ref() == Some(stratum) {
            if let Some(exec) = node.executions.iter().find(|e| e.phase == phase) {
                // Observer boundary enforcement: Fields may only execute in Measure phase
                if node.role_id() == RoleId::Field && phase != Phase::Measure {
                    return Err(vec![CompileError::new(
                        ErrorKind::PhaseBoundaryViolation,
                        node.span,
                        format!(
                            "Field '{}' has execution in {:?} phase. Fields are observation-only and may only execute in Measure phase.",
                            node.path, phase
                        ),
                    )]);
                }

                nodes.push((node.path.clone(), exec));
                node_spans.insert(node.path.clone(), exec.span);
            }
        }
    }

    for node in world.members.values() {
        if node.stratum.as_ref() == Some(stratum) {
            if let Some(exec) = node.executions.iter().find(|e| e.phase == phase) {
                // Observer boundary enforcement: Fields may only execute in Measure phase
                if node.role_id() == RoleId::Field && phase != Phase::Measure {
                    return Err(vec![CompileError::new(
                        ErrorKind::PhaseBoundaryViolation,
                        node.span,
                        format!(
                            "Field '{}' has execution in {:?} phase. Fields are observation-only and may only execute in Measure phase.",
                            node.path, phase
                        ),
                    )]);
                }

                nodes.push((node.path.clone(), exec));
                node_spans.insert(node.path.clone(), exec.span);
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

    let mut producers: IndexMap<Path, Path> = IndexMap::new();
    for (path, exec) in &nodes {
        // In all phases, we look at 'emits' to determine causal links.
        // For Resolve phase, the compiler explicitly populates 'emits' with the node's own path.
        for emit in &exec.emits {
            if let Some(existing_producer) = producers.insert(emit.clone(), path.clone()) {
                // Conflict: Multiple nodes emitting to the same signal in the same phase/stratum.
                // This is only allowed in Collect phase (accumulation) where nodes contribute
                // to a signal via += or similar (which are resolved as separate emits in IR).
                if phase != Phase::Collect {
                    return Err(vec![CompileError::new(
                        ErrorKind::Conflict,
                        exec.span,
                        format!(
                            "Multiple nodes emitting to '{}' in {:?} phase: '{}' and '{}'. \
                             Only one producer is allowed per signal in this phase.",
                            emit, phase, existing_producer, path
                        ),
                    )]);
                }
            }
        }
    }

    // Adjacency list: node -> [nodes it depends on]
    let mut adj: IndexMap<Path, Vec<Path>> = IndexMap::new();
    let mut in_degree: IndexMap<Path, usize> = IndexMap::new();

    for (path, _) in &nodes {
        in_degree.insert(path.clone(), 0);
        adj.insert(path.clone(), Vec::new());
    }

    for (path, exec) in &nodes {
        for read in &exec.reads {
            if let Some(producer_path) = producers.get(read) {
                if producer_path == path {
                    // Self-dependency detected: node reads from itself
                    // This should only happen via temporal access (prev), which should have
                    // been filtered out during dependency extraction. If we see it here,
                    // it's a non-temporal self-read, which is a circular dependency.
                    let span = node_spans
                        .get(path)
                        .copied()
                        .unwrap_or_else(|| Span::new(0, 0, 0, 0));
                    return Err(vec![CompileError::new(
                        ErrorKind::CyclicDependency,
                        span,
                        format!(
                            "Node '{}' reads from itself in {:?} phase. \
                             Self-dependencies are only allowed via temporal access (prev). \
                             This appears to be a non-temporal circular dependency.",
                            path, phase
                        ),
                    )
                    .with_note(
                        "If you intended to read the previous tick's value, use 'prev' context. \
                         Otherwise, this is a circular dependency that must be broken."
                            .to_string(),
                    )]);
                } else {
                    // Normal dependency: add edge
                    adj.entry(producer_path.clone())
                        .or_default()
                        .push(path.clone());
                    *in_degree.entry(path.clone()).or_default() += 1;
                }
            } else {
                // If the read is a signal/field that exists but isn't in this stratum's producers,
                // it's a cross-stratum dependency violation.
                // We check world globals and members for stratum mismatch.
                let target_info = world
                    .globals
                    .get(read)
                    .map(|n| (n.role_id(), &n.stratum))
                    .or_else(|| world.members.get(read).map(|n| (n.role_id(), &n.stratum)));

                if let Some((role_id, target_stratum)) = target_info {
                    // Only Signal and Field roles have strata that matter for DAG construction.
                    // Config and Const are globally available and don't create DAG edges here
                    // because they are immutable during execution (resolved in Configure or earlier).
                    if matches!(role_id, RoleId::Signal | RoleId::Field) {
                        if target_stratum.as_ref() != Some(stratum) {
                            return Err(vec![CompileError::new(
                                ErrorKind::InvalidDependency,
                                exec.span,
                                format!(
                                    "Node '{}' (stratum {:?}) reads from '{}' (stratum {:?}). \
                                     Cross-stratum dependencies are forbidden to preserve strict determinism.",
                                    path, stratum, read, target_stratum
                                ),
                            )]);
                        }
                    }
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

        // Trace the actual dependency path through the cycle
        let cycle_path = trace_cycle_path(&cycle_nodes, &adj);

        // Use the span of the first node in the cycle
        let first_node_span = cycle_path
            .first()
            .and_then(|p| node_spans.get(p))
            .copied()
            .unwrap_or_else(|| Span::new(0, 0, 0, 0));

        // Format the cycle as a dependency chain: a → b → c → a
        let cycle_description = cycle_path
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" → ");

        let mut error = CompileError::new(
            ErrorKind::CyclicDependency,
            first_node_span,
            format!("Circular dependency: {}", cycle_description),
        );

        // Add labels for each step in the cycle
        for (i, node_path) in cycle_path.iter().enumerate() {
            if let Some(&span) = node_spans.get(node_path) {
                if i == 0 {
                    error = error.with_label(span, "cycle starts here".to_string());
                } else if i == cycle_path.len() - 1 {
                    error = error.with_label(span, "cycle completes here".to_string());
                } else {
                    error = error.with_label(
                        span,
                        format!("depends on '{}'", cycle_path.get(i + 1).unwrap()),
                    );
                }
            }
        }

        error = error.with_note(
            "Break the cycle by removing one of these dependencies or using temporal access (prev)"
                .to_string(),
        );

        return Err(vec![error]);
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
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // signal a { resolve { 1.0 } }
        let mut node_a = Node::new(Path::from_path_str("a"), span, RoleData::Signal, ());
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
            vec![node_a.path.clone()],
            span,
        )];

        // signal b { resolve { signal.a } }
        let mut node_b = Node::new(Path::from_path_str("b"), span, RoleData::Signal, ());
        node_b.stratum = Some(StratumId::new("default"));
        node_b.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_path_str("a")),
                crate::foundation::Type::Bool, // Dummy type
                span,
            )),
            vec![Path::from_path_str("a")],
            vec![],
            vec![node_b.path.clone()],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );
        world.globals.insert(node_a.path.clone(), node_a);
        world.globals.insert(node_b.path.clone(), node_b);

        let dag_set = compile_graphs(&world).expect("Failed to compile graphs");
        let dag = dag_set
            .dags
            .get(&(Phase::Resolve, StratumId::new("default")))
            .expect("DAG not found");

        assert_eq!(dag.levels.len(), 2);
        assert_eq!(dag.levels[0].nodes, vec![Path::from_path_str("a")]);
        assert_eq!(dag.levels[1].nodes, vec![Path::from_path_str("b")]);
    }

    #[test]
    fn test_dag_compilation_cycle() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // signal a { resolve { signal.b } }
        let mut node_a = Node::new(Path::from_path_str("a"), span, RoleData::Signal, ());
        node_a.stratum = Some(StratumId::new("default"));
        node_a.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_path_str("b")),
                crate::foundation::Type::Bool,
                span,
            )),
            vec![Path::from_path_str("b")],
            vec![],
            vec![node_a.path.clone()],
            span,
        )];

        // signal b { resolve { signal.a } }
        let mut node_b = Node::new(Path::from_path_str("b"), span, RoleData::Signal, ());
        node_b.stratum = Some(StratumId::new("default"));
        node_b.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_path_str("a")),
                crate::foundation::Type::Bool,
                span,
            )),
            vec![Path::from_path_str("a")],
            vec![],
            vec![node_b.path.clone()],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );
        world.globals.insert(node_a.path.clone(), node_a);
        world.globals.insert(node_b.path.clone(), node_b);

        let result = compile_graphs(&world);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
    }

    #[test]
    fn test_field_isolation_in_kernel_phases() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // field temperature { resolve { 1.0 } }  // INVALID: Fields can't execute in Resolve
        let mut field_node = Node::new(
            Path::from_path_str("temperature"),
            span,
            RoleData::Field {
                reconstruction: None,
            },
            (),
        );
        field_node.stratum = Some(StratumId::new("default"));
        field_node.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![field_node.path.clone()],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );
        world.globals.insert(field_node.path.clone(), field_node);

        // Should fail: Fields are observer-only, can't execute in Resolve phase
        let result = compile_graphs(&world);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].kind, ErrorKind::PhaseBoundaryViolation);
        assert!(errors[0]
            .message
            .contains("Fields are observation-only and may only execute in Measure phase"));
    }

    #[test]
    fn test_parallel_node_deterministic_ordering() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // Three signals with no dependencies - should sort alphabetically in same level
        // signal zebra { resolve { 1.0 } }
        // signal apple { resolve { 2.0 } }
        // signal banana { resolve { 3.0 } }

        let mut node_zebra = Node::new(Path::from_path_str("zebra"), span, RoleData::Signal, ());
        node_zebra.stratum = Some(StratumId::new("default"));
        node_zebra.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![node_zebra.path.clone()],
            span,
        )];

        let mut node_apple = Node::new(Path::from_path_str("apple"), span, RoleData::Signal, ());
        node_apple.stratum = Some(StratumId::new("default"));
        node_apple.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![node_apple.path.clone()],
            span,
        )];

        let mut node_banana = Node::new(Path::from_path_str("banana"), span, RoleData::Signal, ());
        node_banana.stratum = Some(StratumId::new("default"));
        node_banana.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 3.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![node_banana.path.clone()],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );

        // Insert in non-alphabetical order to verify sorting
        world.globals.insert(node_zebra.path.clone(), node_zebra);
        world.globals.insert(node_apple.path.clone(), node_apple);
        world.globals.insert(node_banana.path.clone(), node_banana);

        let dag_set = compile_graphs(&world).expect("Failed to compile graphs");
        let dag = dag_set
            .dags
            .get(&(Phase::Resolve, StratumId::new("default")))
            .expect("DAG not found");

        // All three should be in the same level (no dependencies)
        assert_eq!(dag.levels.len(), 1);
        // Nodes should be sorted alphabetically for determinism
        assert_eq!(
            dag.levels[0].nodes,
            vec![
                Path::from_path_str("apple"),
                Path::from_path_str("banana"),
                Path::from_path_str("zebra")
            ]
        );
    }

    #[test]
    fn test_empty_dag_handling() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );

        // No nodes in the world - all DAGs should be empty (None)
        let dag_set = compile_graphs(&world).expect("Failed to compile graphs");

        // Should have no DAGs for empty world
        assert!(dag_set.dags.is_empty());
    }

    #[test]
    fn test_cross_stratum_dependency_violation() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // signal slow_signal { : stratum(slow); resolve { 1.0 } }
        let mut node_slow = Node::new(
            Path::from_path_str("slow_signal"),
            span,
            RoleData::Signal,
            (),
        );
        node_slow.stratum = Some(StratumId::new("slow"));
        node_slow.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![node_slow.path.clone()],
            span,
        )];

        // signal fast_signal { : stratum(fast); resolve { signal.slow_signal } }
        let mut node_fast = Node::new(
            Path::from_path_str("fast_signal"),
            span,
            RoleData::Signal,
            (),
        );
        node_fast.stratum = Some(StratumId::new("fast"));
        node_fast.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Signal(Path::from_path_str("slow_signal")),
                crate::foundation::Type::Bool,
                span,
            )),
            vec![Path::from_path_str("slow_signal")],
            vec![],
            vec![node_fast.path.clone()],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("slow"),
            crate::ast::Stratum::new(StratumId::new("slow"), Path::from_path_str("slow"), span),
        );
        world.strata.insert(
            Path::from_path_str("fast"),
            crate::ast::Stratum::new(StratumId::new("fast"), Path::from_path_str("fast"), span),
        );
        world.globals.insert(node_slow.path.clone(), node_slow);
        world.globals.insert(node_fast.path.clone(), node_fast);

        // Should fail: fast_signal reads slow_signal from different stratum
        let result = compile_graphs(&world);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].kind, ErrorKind::InvalidDependency);
        assert!(errors[0]
            .message
            .contains("Cross-stratum dependencies are forbidden"));
    }

    #[test]
    fn test_multi_emitter_conflict() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // operator op1 { measure { field.a <- 1.0 } }
        let mut node_op1 = Node::new(Path::from_path_str("op1"), span, RoleData::Operator, ());
        node_op1.stratum = Some(StratumId::new("default"));
        node_op1.executions = vec![Execution::new(
            "measure".to_string(),
            Phase::Measure,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![Path::from_path_str("field.a")],
            span,
        )];

        let mut node_op2 = Node::new(Path::from_path_str("op2"), span, RoleData::Operator, ());
        node_op2.stratum = Some(StratumId::new("default"));
        node_op2.executions = vec![Execution::new(
            "measure".to_string(),
            Phase::Measure,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],
            vec![],
            vec![Path::from_path_str("field.a")],
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );
        world.globals.insert(node_op1.path.clone(), node_op1);
        world.globals.insert(node_op2.path.clone(), node_op2);

        // Should fail: two operators emitting to same field in Measure phase
        let result = compile_graphs(&world);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].kind, ErrorKind::Conflict);
        assert!(errors[0].message.contains("Multiple nodes emitting to"));
    }

    #[test]
    fn test_dag_compilation_temporal_success() {
        let span = test_span();
        let metadata = crate::ast::WorldDecl {
            path: Path::from_path_str("world"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: false,
        };

        // signal a { resolve { prev } }
        let mut node_a = Node::new(Path::from_path_str("a"), span, RoleData::Signal, ());
        node_a.stratum = Some(StratumId::new("default"));
        node_a.executions = vec![Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(crate::ast::TypedExpr::new(
                crate::ast::ExprKind::Prev,
                crate::foundation::Type::Bool,
                span,
            )),
            vec![],                    // No causal reads
            vec![node_a.path.clone()], // Temporal read of self
            vec![node_a.path.clone()], // Emits self
            span,
        )];

        let mut world = World::new(metadata);
        world.strata.insert(
            Path::from_path_str("default"),
            crate::ast::Stratum::new(
                StratumId::new("default"),
                Path::from_path_str("default"),
                span,
            ),
        );
        world.globals.insert(node_a.path.clone(), node_a);

        // Should SUCCEED: temporal read is not a DAG edge
        let result = compile_graphs(&world);
        assert!(
            result.is_ok(),
            "Temporal read should not cause cyclic dependency error: {:?}",
            result.err()
        );

        let dag_set = result.unwrap();
        let dag = dag_set
            .dags
            .get(&(Phase::Resolve, StratumId::new("default")))
            .unwrap();
        assert_eq!(dag.levels.len(), 1);
        assert_eq!(dag.levels[0].nodes, vec![Path::from_path_str("a")]);
    }
}
