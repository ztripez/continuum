use continuum_foundation::{MemberId, Path};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::CompiledWorld;

/// A circular dependency detected in the execution graph.
#[derive(Debug, Clone)]
pub struct Cycle {
    /// The path of nodes forming the cycle.
    pub path: Vec<Path>,
    /// The spans of each node in the cycle (for error reporting).
    pub spans: Vec<Range<usize>>,
}

/// Detect circular dependencies in the Resolve phase.
///
/// Circular dependencies are only possible between nodes in the same stratum,
/// as cross-stratum dependencies are always resolved using temporal buffering
/// (reading the previous tick's value if the target stratum hasn't resolved yet).
pub fn find_cycles(world: &CompiledWorld) -> Vec<Cycle> {
    let mut cycles = Vec::new();
    let signals = world.signals();
    let members = world.members();

    // Map each signal to its dependencies (using strings for DFS)
    // ONLY include dependencies within the same stratum
    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    let mut spans: HashMap<String, Range<usize>> = HashMap::new();

    for (id, signal) in &signals {
        let id_str = id.to_string();
        let stratum = &signal.stratum;

        let same_stratum_reads: Vec<String> = signal
            .reads
            .iter()
            .filter(|read_id| {
                // Check if the referenced signal is in the same stratum
                if let Some(read_signal) = signals.get(*read_id) {
                    &read_signal.stratum == stratum
                } else if let Some(read_member) = members.get(&MemberId::from(read_id.to_string()))
                {
                    &read_member.stratum == stratum
                } else {
                    false
                }
            })
            .map(|r| r.to_string())
            .collect();

        deps.insert(id_str.clone(), same_stratum_reads);
        spans.insert(id_str, signal.span.clone());
    }

    for (id, member) in &members {
        let id_str = id.to_string();
        let stratum = &member.stratum;

        let same_stratum_reads: Vec<String> = member
            .reads
            .iter()
            .filter(|read_id| {
                if let Some(read_signal) = signals.get(*read_id) {
                    &read_signal.stratum == stratum
                } else if let Some(read_member) = members.get(&MemberId::from(read_id.to_string()))
                {
                    &read_member.stratum == stratum
                } else {
                    false
                }
            })
            .map(|r| r.to_string())
            .collect();

        // Also check member_reads (member-to-member dependencies)
        let same_stratum_member_reads: Vec<String> = member
            .member_reads
            .iter()
            .filter(|read_id| {
                if let Some(read_member) = members.get(*read_id) {
                    &read_member.stratum == stratum
                } else {
                    false
                }
            })
            .map(|r| r.to_string())
            .collect();

        let mut all_same_reads = same_stratum_reads;
        all_same_reads.extend(same_stratum_member_reads);

        deps.insert(id_str.clone(), all_same_reads);
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
