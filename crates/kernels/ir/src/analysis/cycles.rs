use crate::CompiledWorld;
use continuum_foundation::Path;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// A circular dependency detected in the execution graph.
#[derive(Debug, Clone)]
pub struct Cycle {
    /// The path of nodes forming the cycle.
    pub path: Vec<Path>,
    /// The spans of each node in the cycle (for error reporting).
    pub spans: Vec<Range<usize>>,
}

/// Detect circular dependencies in the Resolve phase of each stratum.
pub fn find_cycles(world: &CompiledWorld) -> Vec<Cycle> {
    let mut cycles = Vec::new();
    let signals = world.signals();
    let members = world.members();

    // Map each signal to its dependencies (using strings for DFS)
    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    let mut spans: HashMap<String, Range<usize>> = HashMap::new();

    for (id, signal) in &signals {
        let id_str = id.to_string();
        deps.insert(
            id_str.clone(),
            signal.reads.iter().map(|r| r.to_string()).collect(),
        );
        spans.insert(id_str, signal.span.clone());
    }

    for (id, member) in &members {
        let id_str = id.to_string();
        deps.insert(
            id_str.clone(),
            member.reads.iter().map(|r| r.to_string()).collect(),
        );
        spans.insert(id_str, member.span.clone());
    }

    // DFS state
    let mut visited = HashSet::new();
    let mut on_stack = Vec::new();
    let mut on_stack_set = HashSet::new();

    let mut keys: Vec<_> = deps.keys().cloned().collect();
    keys.sort(); // Determinism

    for id in keys {
        if !visited.contains(&id) {
            check_cycles_dfs(
                &id,
                &deps,
                &spans,
                &mut visited,
                &mut on_stack,
                &mut on_stack_set,
                &mut cycles,
            );
        }
    }

    cycles
}

fn check_cycles_dfs(
    current: &String,
    deps: &HashMap<String, Vec<String>>,
    spans: &HashMap<String, Range<usize>>,
    visited: &mut HashSet<String>,
    on_stack: &mut Vec<String>,
    on_stack_set: &mut HashSet<String>,
    cycles: &mut Vec<Cycle>,
) {
    visited.insert(current.clone());
    on_stack.push(current.clone());
    on_stack_set.insert(current.clone());

    if let Some(current_deps) = deps.get(current) {
        for dep in current_deps {
            if on_stack_set.contains(dep) {
                // Cycle detected!
                let mut cycle_path = Vec::new();
                let mut cycle_spans = Vec::new();
                let mut found_start = false;

                for node in on_stack.iter() {
                    if node == dep {
                        found_start = true;
                    }
                    if found_start {
                        cycle_path.push(Path::from(node.clone()));
                        cycle_spans.push(spans.get(node).cloned().unwrap_or(0..0));
                    }
                }
                // Add the closing link
                cycle_path.push(Path::from(dep.clone()));
                cycle_spans.push(spans.get(dep).cloned().unwrap_or(0..0));

                cycles.push(Cycle {
                    path: cycle_path,
                    spans: cycle_spans,
                });
            } else if !visited.contains(dep) {
                check_cycles_dfs(dep, deps, spans, visited, on_stack, on_stack_set, cycles);
            }
        }
    }

    on_stack.pop();
    on_stack_set.remove(current);
}
