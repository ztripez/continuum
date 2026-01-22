//! Utility functions for resolution and compilation passes.

use std::collections::HashSet;
use std::hash::Hash;

/// Helper to convert a HashSet into a sorted unique Vec.
///
/// This satisfies the engine's core determinism invariant by ensuring
/// stable ordering of paths and other identifiers in the IR.
pub fn sort_unique<T: Ord + Hash + Eq>(set: HashSet<T>) -> Vec<T> {
    let mut vec: Vec<_> = set.into_iter().collect();
    vec.sort();
    vec
}
