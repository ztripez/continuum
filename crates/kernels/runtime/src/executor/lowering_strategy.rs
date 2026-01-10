//! Lowering strategy selection for member signal execution.
//!
//! This module defines the available execution strategies and heuristics
//! for selecting the optimal strategy based on population characteristics.

// ============================================================================
// Lowering Strategy
// ============================================================================

/// Execution strategy for lowering member signal resolution.
///
/// Different strategies are optimal for different workload shapes:
///
/// | Strategy | Best For | Population | Complexity |
/// |----------|----------|------------|------------|
/// | L1 | General purpose | 2k-50k | Simple to medium |
/// | L2 | Dense SIMD | 50k+ | Simple expressions |
/// | L3 | Complex agents | <1k | Complex inter-dependencies |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoweringStrategy {
    /// L1: Instance-parallel tasks (chunked CPU parallelism via rayon).
    ///
    /// Divides the entity index space into chunks and processes them in
    /// parallel using rayon's work-stealing scheduler.
    ///
    /// Best for populations of 2k-50k with moderate expression complexity.
    InstanceParallel,

    /// L2: Vector kernel (SSA lowering, SIMD batching, GPU compute).
    ///
    /// Lowers resolver expressions to SSA IR, then generates SIMD or
    /// GPU compute code for efficient batch processing.
    ///
    /// Best for populations of 50k+ with simple expressions.
    VectorKernel,

    /// L3: Sub-DAG (internal dependency graph within the member signal).
    ///
    /// Constructs a dependency graph across member signals of the same
    /// entity, enabling fusion and sophisticated scheduling.
    ///
    /// Best for small populations (<1k) with complex inter-member dependencies.
    SubDag,
}

impl LoweringStrategy {
    /// Get the display name for this strategy.
    pub fn name(&self) -> &'static str {
        match self {
            LoweringStrategy::InstanceParallel => "L1:InstanceParallel",
            LoweringStrategy::VectorKernel => "L2:VectorKernel",
            LoweringStrategy::SubDag => "L3:SubDag",
        }
    }
}

impl std::fmt::Display for LoweringStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Strategy Selection Heuristics
// ============================================================================

/// Parameters for selecting a lowering strategy.
#[derive(Debug, Clone)]
pub struct LoweringHeuristics {
    /// Population threshold below which L3 (SubDag) is preferred.
    pub l3_threshold: usize,
    /// Population threshold above which L2 (VectorKernel) is preferred.
    pub l2_threshold: usize,
    /// Whether L2 is available (requires SSA lowering).
    pub l2_available: bool,
    /// Whether L3 is available (requires sub-DAG builder).
    pub l3_available: bool,
}

impl Default for LoweringHeuristics {
    fn default() -> Self {
        Self {
            l3_threshold: 1000,
            l2_threshold: 50000,
            l2_available: true, // L2 vectorized execution via SSA
            l3_available: true, // L3 sub-DAG execution for small populations
        }
    }
}

impl LoweringHeuristics {
    /// Select the optimal lowering strategy for a given population.
    ///
    /// The selection follows this priority:
    /// 1. If population < l3_threshold and L3 available → L3
    /// 2. If population > l2_threshold and L2 available → L2
    /// 3. Otherwise → L1 (always available)
    pub fn select(&self, population: usize) -> LoweringStrategy {
        if self.l3_available && population < self.l3_threshold {
            LoweringStrategy::SubDag
        } else if self.l2_available && population > self.l2_threshold {
            LoweringStrategy::VectorKernel
        } else {
            LoweringStrategy::InstanceParallel
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowering_strategy_display() {
        assert_eq!(
            LoweringStrategy::InstanceParallel.to_string(),
            "L1:InstanceParallel"
        );
        assert_eq!(
            LoweringStrategy::VectorKernel.to_string(),
            "L2:VectorKernel"
        );
        assert_eq!(LoweringStrategy::SubDag.to_string(), "L3:SubDag");
    }

    #[test]
    fn test_lowering_heuristics_default() {
        let h = LoweringHeuristics::default();

        // With defaults (L2 enabled, L3 enabled):
        // - Small populations (< 1000) use L3
        // - Medium populations use L1
        // - Large populations (> 50000) use L2
        assert_eq!(h.select(100), LoweringStrategy::SubDag);
        assert_eq!(h.select(500), LoweringStrategy::SubDag);
        assert_eq!(h.select(10000), LoweringStrategy::InstanceParallel);
        assert_eq!(h.select(60000), LoweringStrategy::VectorKernel);
        assert_eq!(h.select(100000), LoweringStrategy::VectorKernel);
    }

    #[test]
    fn test_lowering_heuristics_l2_disabled() {
        // Test behavior when L2 is explicitly disabled
        let h = LoweringHeuristics {
            l2_available: false,
            ..Default::default()
        };

        // All populations should use L1
        assert_eq!(h.select(10000), LoweringStrategy::InstanceParallel);
        assert_eq!(h.select(60000), LoweringStrategy::InstanceParallel);
    }

    #[test]
    fn test_lowering_heuristics_with_l3() {
        let h = LoweringHeuristics {
            l3_available: true,
            ..Default::default()
        };

        assert_eq!(h.select(500), LoweringStrategy::SubDag);
        assert_eq!(h.select(2000), LoweringStrategy::InstanceParallel);
    }
}
