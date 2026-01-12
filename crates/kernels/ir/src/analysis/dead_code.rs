use crate::CompiledWorld;
use continuum_foundation::Path;
use std::collections::{HashSet, VecDeque};

/// Unused items in the IR.
#[derive(Debug, Clone)]
pub struct DeadCode {
    /// Signals that are never read.
    pub unused_signals: Vec<Path>,
}

/// Identify signals that are never read by any operator, field, fracture,
/// era transition, or another signal.
pub fn find_dead_code(world: &CompiledWorld) -> DeadCode {
    let signals = world.signals();
    let members = world.members();
    let fields = world.fields();
    let operators = world.operators();
    let fractures = world.fractures();
    let eras = world.eras();

    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();

    // 1. Collect all root dependencies

    // Operators
    for (_, op) in &operators {
        for read in &op.reads {
            let read_str = read.to_string();
            if !reachable.contains(&read_str) {
                reachable.insert(read_str.clone());
                queue.push_back(read_str);
            }
        }
    }

    // Fields
    for (_, field) in &fields {
        for read in &field.reads {
            let read_str = read.to_string();
            if !reachable.contains(&read_str) {
                reachable.insert(read_str.clone());
                queue.push_back(read_str);
            }
        }
    }

    // Fractures
    for (_, frac) in &fractures {
        for read in &frac.reads {
            let read_str = read.to_string();
            if !reachable.contains(&read_str) {
                reachable.insert(read_str.clone());
                queue.push_back(read_str);
            }
        }
    }

    // Era transitions
    for (_, era) in &eras {
        for trans in &era.transitions {
            for read in trans.condition.signal_dependencies() {
                let read_str = read.to_string();
                if !reachable.contains(&read_str) {
                    reachable.insert(read_str.clone());
                    queue.push_back(read_str);
                }
            }
        }
    }

    // 2. Propagate reachability through signals
    while let Some(current) = queue.pop_front() {
        // Try finding as signal
        use continuum_foundation::{MemberId, SignalId};
        if let Some(signal) = signals.get(&SignalId::from(current.clone())) {
            for read in &signal.reads {
                let read_str = read.to_string();
                if !reachable.contains(&read_str) {
                    reachable.insert(read_str.clone());
                    queue.push_back(read_str);
                }
            }
        }
        // Try finding as member
        if let Some(member) = members.get(&MemberId::from(current.clone())) {
            for read in &member.reads {
                let read_str = read.to_string();
                if !reachable.contains(&read_str) {
                    reachable.insert(read_str.clone());
                    queue.push_back(read_str);
                }
            }
        }
    }

    // 3. Identify unused signals
    let mut unused_signals = Vec::new();
    for id in signals.keys() {
        let id_str = id.to_string();
        if !reachable.contains(&id_str) {
            unused_signals.push(Path::from(id_str));
        }
    }
    for id in members.keys() {
        let id_str = id.to_string();
        if !reachable.contains(&id_str) {
            unused_signals.push(Path::from(id_str));
        }
    }

    unused_signals.sort();

    DeadCode { unused_signals }
}
