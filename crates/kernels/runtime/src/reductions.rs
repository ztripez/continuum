//! Deterministic reduction operations for entity aggregates.
//!
//! This module provides reduction operations (`sum`, `mean`, `min`, `max`, etc.)
//! that produce identical results regardless of parallel execution order.
//!
//! # The Problem
//!
//! Naive parallel reduction can produce non-deterministic results due to:
//! - Floating-point associativity issues: `(a + b) + c ≠ a + (b + c)`
//! - Thread timing affecting reduction order
//! - Different tree structures per run
//!
//! # Solution: Fixed-Structure Tree Reduction
//!
//! We use a fixed binary tree structure where pairing is determined by index,
//! not by which computation finishes first:
//!
//! ```text
//! Entities: [e0, e1, e2, e3, e4, e5, e6, e7]
//!
//! Level 0: e0+e1  e2+e3  e4+e5  e6+e7
//! Level 1: (e0+e1)+(e2+e3)  (e4+e5)+(e6+e7)
//! Level 2: ((e0+e1)+(e2+e3))+((e4+e5)+(e6+e7))
//! ```
//!
//! # Non-Power-of-2 Handling
//!
//! For sizes that aren't powers of 2, odd elements propagate up:
//!
//! ```text
//! Entities: [e0, e1, e2, e3, e4]
//!
//! Level 0: e0+e1  e2+e3  e4
//! Level 1: (e0+e1)+(e2+e3)  e4
//! Level 2: ((e0+e1)+(e2+e3))+e4
//! ```
//!
//! # Available Operations
//!
//! | Function | Notes |
//! |----------|-------|
//! | [`sum`] | Fixed-tree reduction for deterministic floating-point |
//! | [`mean`] | Computed as sum/count |
//! | [`min`] | Lowest index wins ties |
//! | [`max`] | Lowest index wins ties |
//! | [`count`] | Count with optional predicate |
//! | [`product`] | Fixed-tree reduction |

use std::cmp::Ordering;

/// Result of a min/max reduction that tracks the winning index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndexedValue<T> {
    /// The index of the winning element
    pub index: usize,
    /// The value at that index
    pub value: T,
}

impl<T> IndexedValue<T> {
    /// Create a new indexed value.
    pub fn new(index: usize, value: T) -> Self {
        Self { index, value }
    }
}

// ============================================================================
// Core Tree Reduction
// ============================================================================

/// Perform a deterministic tree reduction with a binary operation.
///
/// The reduction follows a fixed binary tree structure where pairs are
/// determined by index, ensuring identical results regardless of execution order.
///
/// # Arguments
///
/// * `values` - The values to reduce
/// * `op` - Binary operation to combine values (must be associative for math correctness)
///
/// # Returns
///
/// The reduced value, or `None` if the slice is empty.
///
/// # Example
///
/// ```
/// use continuum_runtime::reductions::tree_reduce;
///
/// let values = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = tree_reduce(&values, |a, b| a + b);
/// assert!(result.is_some());
/// ```
pub fn tree_reduce<T, F>(values: &[T], op: F) -> Option<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if values.is_empty() {
        return None;
    }

    if values.len() == 1 {
        return Some(values[0]);
    }

    // Work buffer for reduction levels
    let mut current: Vec<T> = values.to_vec();
    let mut next: Vec<T> = Vec::with_capacity((values.len() + 1) / 2);

    while current.len() > 1 {
        next.clear();

        // Process pairs
        let mut i = 0;
        while i + 1 < current.len() {
            next.push(op(current[i], current[i + 1]));
            i += 2;
        }

        // Carry forward odd element
        if i < current.len() {
            next.push(current[i]);
        }

        std::mem::swap(&mut current, &mut next);
    }

    Some(current[0])
}

/// Perform a deterministic tree reduction that tracks indices.
///
/// Used for min/max operations where we need to know which element won.
/// When values are equal, the element with the lower index wins.
///
/// # Arguments
///
/// * `values` - The values to reduce
/// * `cmp` - Comparison function that returns true if first arg wins
///
/// # Returns
///
/// The winning `IndexedValue`, or `None` if empty.
pub fn tree_reduce_indexed<T, F>(values: &[T], cmp: F) -> Option<IndexedValue<T>>
where
    T: Copy,
    F: Fn(T, T) -> Ordering,
{
    if values.is_empty() {
        return None;
    }

    if values.len() == 1 {
        return Some(IndexedValue::new(0, values[0]));
    }

    // Initialize with indices
    let mut current: Vec<IndexedValue<T>> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| IndexedValue::new(i, v))
        .collect();

    let mut next: Vec<IndexedValue<T>> = Vec::with_capacity((values.len() + 1) / 2);

    while current.len() > 1 {
        next.clear();

        let mut i = 0;
        while i + 1 < current.len() {
            let a = current[i];
            let b = current[i + 1];

            // Winner is determined by comparison, ties go to lower index
            let winner = match cmp(a.value, b.value) {
                Ordering::Less => a,    // a wins (for min: a < b)
                Ordering::Greater => b, // b wins (for min: b < a)
                Ordering::Equal => {
                    // Tie: lower index wins
                    if a.index <= b.index {
                        a
                    } else {
                        b
                    }
                }
            };
            next.push(winner);
            i += 2;
        }

        // Carry forward odd element
        if i < current.len() {
            next.push(current[i]);
        }

        std::mem::swap(&mut current, &mut next);
    }

    Some(current[0])
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Deterministic sum of values using fixed-tree reduction.
///
/// Produces bitwise identical results regardless of parallel execution order.
///
/// # Example
///
/// ```
/// use continuum_runtime::reductions::sum;
///
/// let values = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(sum(&values), 10.0);
/// ```
pub fn sum(values: &[f64]) -> f64 {
    tree_reduce(values, |a, b| a + b).unwrap_or(0.0)
}

/// Deterministic sum of Vec3 values.
pub fn sum_vec3(values: &[[f64; 3]]) -> [f64; 3] {
    tree_reduce(values, |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        .unwrap_or([0.0, 0.0, 0.0])
}

/// Deterministic product of values using fixed-tree reduction.
pub fn product(values: &[f64]) -> f64 {
    tree_reduce(values, |a, b| a * b).unwrap_or(1.0)
}

/// Deterministic mean of values.
///
/// Computed as `sum / count` for consistency.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    sum(values) / values.len() as f64
}

/// Deterministic minimum with index tracking.
///
/// When multiple elements have the same minimum value, returns the one
/// with the lowest index for full determinism.
///
/// # Returns
///
/// `Some(IndexedValue)` with the minimum, or `None` if empty.
pub fn min_indexed(values: &[f64]) -> Option<IndexedValue<f64>> {
    tree_reduce_indexed(values, |a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
}

/// Deterministic minimum value.
///
/// Returns the minimum value, or `f64::INFINITY` if empty.
pub fn min(values: &[f64]) -> f64 {
    min_indexed(values)
        .map(|iv| iv.value)
        .unwrap_or(f64::INFINITY)
}

/// Deterministic maximum with index tracking.
///
/// When multiple elements have the same maximum value, returns the one
/// with the lowest index for full determinism.
pub fn max_indexed(values: &[f64]) -> Option<IndexedValue<f64>> {
    // For max, we flip the comparison: Greater means a wins
    tree_reduce_indexed(values, |a, b| {
        b.partial_cmp(&a).unwrap_or(Ordering::Equal)
    })
}

/// Deterministic maximum value.
///
/// Returns the maximum value, or `f64::NEG_INFINITY` if empty.
pub fn max(values: &[f64]) -> f64 {
    max_indexed(values)
        .map(|iv| iv.value)
        .unwrap_or(f64::NEG_INFINITY)
}

/// Count elements matching a predicate.
///
/// This is inherently deterministic as counting is both associative and commutative.
pub fn count<T, F>(values: &[T], predicate: F) -> usize
where
    F: Fn(&T) -> bool,
{
    values.iter().filter(|v| predicate(v)).count()
}

/// Count all elements (no predicate).
pub fn count_all<T>(values: &[T]) -> usize {
    values.len()
}

// ============================================================================
// Reduction Builder (for DSL integration)
// ============================================================================

/// Type of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// Sum of all values
    Sum,
    /// Product of all values
    Product,
    /// Arithmetic mean
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of elements
    Count,
}

impl ReductionOp {
    /// Execute this reduction on f64 values.
    pub fn execute(&self, values: &[f64]) -> f64 {
        match self {
            ReductionOp::Sum => sum(values),
            ReductionOp::Product => product(values),
            ReductionOp::Mean => mean(values),
            ReductionOp::Min => min(values),
            ReductionOp::Max => max(values),
            ReductionOp::Count => count_all(values) as f64,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Tree Reduce Tests
    // ========================================================================

    #[test]
    fn test_tree_reduce_empty() {
        let values: [f64; 0] = [];
        assert!(tree_reduce(&values, |a, b| a + b).is_none());
    }

    #[test]
    fn test_tree_reduce_single() {
        let values = [42.0];
        assert_eq!(tree_reduce(&values, |a, b| a + b), Some(42.0));
    }

    #[test]
    fn test_tree_reduce_power_of_2() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = tree_reduce(&values, |a, b| a + b);
        assert_eq!(result, Some(36.0));
    }

    #[test]
    fn test_tree_reduce_non_power_of_2() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = tree_reduce(&values, |a, b| a + b);
        assert_eq!(result, Some(15.0));
    }

    #[test]
    fn test_tree_reduce_three_elements() {
        let values = [1.0, 2.0, 3.0];
        // Tree: (1+2) + 3 = 6
        let result = tree_reduce(&values, |a, b| a + b);
        assert_eq!(result, Some(6.0));
    }

    // ========================================================================
    // Sum Tests
    // ========================================================================

    #[test]
    fn test_sum_empty() {
        assert_eq!(sum(&[]), 0.0);
    }

    #[test]
    fn test_sum_single() {
        assert_eq!(sum(&[42.0]), 42.0);
    }

    #[test]
    fn test_sum_multiple() {
        assert_eq!(sum(&[1.0, 2.0, 3.0, 4.0]), 10.0);
    }

    #[test]
    fn test_sum_deterministic() {
        // The same input should always produce the same output
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();

        let result1 = sum(&values);
        let result2 = sum(&values);

        // Bitwise identical
        assert_eq!(result1.to_bits(), result2.to_bits());
    }

    #[test]
    fn test_sum_vec3() {
        let values = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        assert_eq!(sum_vec3(&values), [1.0, 2.0, 3.0]);
    }

    // ========================================================================
    // Product Tests
    // ========================================================================

    #[test]
    fn test_product_empty() {
        assert_eq!(product(&[]), 1.0);
    }

    #[test]
    fn test_product_single() {
        assert_eq!(product(&[5.0]), 5.0);
    }

    #[test]
    fn test_product_multiple() {
        assert_eq!(product(&[2.0, 3.0, 4.0]), 24.0);
    }

    // ========================================================================
    // Mean Tests
    // ========================================================================

    #[test]
    fn test_mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_mean_single() {
        assert_eq!(mean(&[42.0]), 42.0);
    }

    #[test]
    fn test_mean_multiple() {
        assert_eq!(mean(&[2.0, 4.0, 6.0, 8.0]), 5.0);
    }

    // ========================================================================
    // Min/Max Tests
    // ========================================================================

    #[test]
    fn test_min_empty() {
        assert_eq!(min(&[]), f64::INFINITY);
    }

    #[test]
    fn test_min_single() {
        assert_eq!(min(&[42.0]), 42.0);
    }

    #[test]
    fn test_min_multiple() {
        assert_eq!(min(&[5.0, 2.0, 8.0, 1.0, 9.0]), 1.0);
    }

    #[test]
    fn test_min_indexed_tie_breaking() {
        // When there are ties, lowest index should win
        let values = [3.0, 1.0, 1.0, 2.0];
        let result = min_indexed(&values).unwrap();
        assert_eq!(result.value, 1.0);
        assert_eq!(result.index, 1); // First occurrence of 1.0
    }

    #[test]
    fn test_max_empty() {
        assert_eq!(max(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_max_single() {
        assert_eq!(max(&[42.0]), 42.0);
    }

    #[test]
    fn test_max_multiple() {
        assert_eq!(max(&[5.0, 2.0, 8.0, 1.0, 9.0]), 9.0);
    }

    #[test]
    fn test_max_indexed_tie_breaking() {
        // When there are ties, lowest index should win
        let values = [3.0, 9.0, 9.0, 2.0];
        let result = max_indexed(&values).unwrap();
        assert_eq!(result.value, 9.0);
        assert_eq!(result.index, 1); // First occurrence of 9.0
    }

    // ========================================================================
    // Count Tests
    // ========================================================================

    #[test]
    fn test_count_all() {
        assert_eq!(count_all(&[1, 2, 3, 4, 5]), 5);
    }

    #[test]
    fn test_count_with_predicate() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(count(&values, |&v| v > 3.0), 2);
    }

    #[test]
    fn test_count_empty() {
        let values: [f64; 0] = [];
        assert_eq!(count_all(&values), 0);
    }

    // ========================================================================
    // ReductionOp Tests
    // ========================================================================

    #[test]
    fn test_reduction_op_sum() {
        let values = [1.0, 2.0, 3.0];
        assert_eq!(ReductionOp::Sum.execute(&values), 6.0);
    }

    #[test]
    fn test_reduction_op_mean() {
        let values = [2.0, 4.0, 6.0];
        assert_eq!(ReductionOp::Mean.execute(&values), 4.0);
    }

    #[test]
    fn test_reduction_op_min() {
        let values = [5.0, 2.0, 8.0];
        assert_eq!(ReductionOp::Min.execute(&values), 2.0);
    }

    #[test]
    fn test_reduction_op_max() {
        let values = [5.0, 2.0, 8.0];
        assert_eq!(ReductionOp::Max.execute(&values), 8.0);
    }

    // ========================================================================
    // Determinism Verification
    // ========================================================================

    #[test]
    fn test_sum_tree_structure_is_fixed() {
        // Verify that the tree reduction structure is deterministic
        // by checking intermediate results match expected tree

        // For [1, 2, 3, 4]:
        // Level 0: 1+2=3, 3+4=7
        // Level 1: 3+7=10

        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(sum(&values), 10.0);

        // Compare with naive sum (which is left-to-right)
        // ((1+2)+3)+4 = 10
        // Both should give 10 here, but the tree structure matters for
        // floating-point determinism with more complex values
    }

    #[test]
    fn test_floating_point_determinism() {
        // Values that could expose floating-point ordering issues
        let values = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ];

        // Multiple runs must be bitwise identical
        let r1 = sum(&values);
        let r2 = sum(&values);
        let r3 = sum(&values);

        assert_eq!(r1.to_bits(), r2.to_bits());
        assert_eq!(r2.to_bits(), r3.to_bits());
    }

    #[test]
    fn test_indexed_reduction_preserves_original_indices() {
        // Even after tree reduction, the returned index should be
        // from the original input array

        let values = [10.0, 5.0, 15.0, 3.0, 20.0, 8.0, 12.0, 1.0];
        let result = min_indexed(&values).unwrap();

        // Minimum is 1.0 at index 7
        assert_eq!(result.value, 1.0);
        assert_eq!(result.index, 7);
        assert_eq!(values[result.index], result.value);
    }
}
