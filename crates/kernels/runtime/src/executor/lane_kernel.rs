//! Lane kernel abstraction for two-level execution model.
//!
//! This module implements the second level of the two-level execution model:
//! - **Level 1: World-Level DAG** - Operators and signals scheduled by dependencies
//! - **Level 2: Lane Kernels** - Internal execution of entity/member signal nodes
//!
//! Member signal nodes in the DAG can be "lowered" to different execution strategies:
//! - L1: Instance-parallel tasks (chunked CPU parallelism)
//! - L2: Vector kernel (SSA lowering, SIMD, GPU compute)
//! - L3: Sub-DAG (internal dependency graph for complex logic)
//!
//! The lowering choice is transparent to the world-level DAG - a member signal
//! node completes before its dependents execute regardless of which strategy
//! was used internally.
//!
//! # Architecture
//!
//! ```text
//! World-Level DAG (Level 1)
//! ┌─────────────────────────────────────────────────────────┐
//! │  [SignalResolve] ──► [MemberSignalResolve] ──► [Measure]│
//! │                            │                            │
//! │                     ┌──────┴──────┐                     │
//! │                     │ Lane Kernel │ (Level 2)           │
//! │                     │ ┌─────────┐ │                     │
//! │                     │ │L1/L2/L3 │ │                     │
//! │                     │ └─────────┘ │                     │
//! │                     └─────────────┘                     │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use continuum_runtime::executor::{LaneKernel, LoweringStrategy, ScalarL1Kernel};
//!
//! // Create an L1 kernel for a scalar member signal
//! let kernel = ScalarL1Kernel::new(
//!     member_signal_id,
//!     Arc::new(|ctx| ctx.prev + 1.0),
//!     population_hint,
//! );
//!
//! // Execute the kernel (handles all instances in parallel)
//! kernel.execute(&signals, &mut population, dt)?;
//! ```

use std::sync::Arc;

use rayon::prelude::*;
use tracing::{debug, instrument, trace};

use crate::soa_storage::PopulationStorage;
use crate::storage::SignalStorage;
use crate::types::Dt;
use crate::vectorized::{EntityIndex, MemberSignalId};

use super::member_executor::{optimal_chunk_size, ScalarResolveContext, Vec3ResolveContext};

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
// Lane Kernel Trait
// ============================================================================

/// Result of lane kernel execution.
#[derive(Debug)]
pub struct LaneKernelResult {
    /// Number of instances processed.
    pub instances_processed: usize,
    /// Execution time in nanoseconds (if profiled).
    pub execution_ns: Option<u64>,
}

/// A compiled lane kernel that executes member signal resolution.
///
/// Lane kernels encapsulate a specific lowering strategy for a member signal.
/// They appear as single nodes in the world-level DAG but internally may
/// execute using parallel tasks, SIMD, GPU, or sub-DAG scheduling.
///
/// # Key Properties
///
/// - **Opaque to DAG** - The world-level scheduler only sees barriers
/// - **Deterministic** - Same inputs produce bitwise-identical outputs
/// - **Read-only inputs** - Kernels read signals/members, write to buffer
///
/// # Thread Safety
///
/// Lane kernels must be `Send + Sync` because the world-level DAG may
/// execute multiple strata/phases in parallel (where safe).
pub trait LaneKernel: Send + Sync {
    /// The lowering strategy this kernel uses.
    fn strategy(&self) -> LoweringStrategy;

    /// The member signal this kernel resolves.
    fn member_signal_id(&self) -> &MemberSignalId;

    /// Expected population size (for profiling/heuristics).
    fn population_hint(&self) -> usize;

    /// Execute the kernel, writing new values to the population storage.
    ///
    /// # Arguments
    ///
    /// * `signals` - Read-only access to global signals
    /// * `population` - Read-write access to population storage
    /// * `dt` - Time step for this tick
    ///
    /// # Returns
    ///
    /// Result containing execution statistics, or an error.
    fn execute(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError>;
}

/// Error during lane kernel execution.
#[derive(Debug)]
pub enum LaneKernelError {
    /// Member signal buffer not found in population storage.
    SignalNotFound(String),
    /// Numeric error during computation (NaN, Inf).
    NumericError { index: usize, message: String },
    /// Kernel execution failed.
    ExecutionFailed(String),
}

impl std::fmt::Display for LaneKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaneKernelError::SignalNotFound(name) => {
                write!(f, "member signal not found: {}", name)
            }
            LaneKernelError::NumericError { index, message } => {
                write!(f, "numeric error at index {}: {}", index, message)
            }
            LaneKernelError::ExecutionFailed(msg) => {
                write!(f, "kernel execution failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for LaneKernelError {}

// ============================================================================
// L1 Kernel: Scalar Instance-Parallel
// ============================================================================

/// Resolver function type for scalar member signals.
pub type ScalarKernelFn = Arc<dyn Fn(&ScalarResolveContext) -> f64 + Send + Sync>;

/// L1 lane kernel for scalar member signals using instance-parallel execution.
///
/// This kernel uses chunked parallel execution via rayon to resolve
/// all instances of a scalar member signal.
pub struct ScalarL1Kernel {
    member_signal_id: MemberSignalId,
    /// The signal name used for buffer access
    signal_name: String,
    resolver: ScalarKernelFn,
    population_hint: usize,
    /// Fixed chunk size, or None for auto-computed size
    fixed_chunk_size: Option<usize>,
}

impl ScalarL1Kernel {
    /// Create a new scalar L1 kernel.
    ///
    /// # Arguments
    ///
    /// * `member_signal_id` - The member signal this kernel resolves
    /// * `resolver` - The resolver function
    /// * `population_hint` - Expected population size
    pub fn new(
        member_signal_id: MemberSignalId,
        resolver: ScalarKernelFn,
        population_hint: usize,
    ) -> Self {
        let signal_name = member_signal_id.signal_name.clone();
        Self {
            member_signal_id,
            signal_name,
            resolver,
            population_hint,
            fixed_chunk_size: None,
        }
    }

    /// Create with a fixed chunk size.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.fixed_chunk_size = Some(chunk_size);
        self
    }
}

impl LaneKernel for ScalarL1Kernel {
    fn strategy(&self) -> LoweringStrategy {
        LoweringStrategy::InstanceParallel
    }

    fn member_signal_id(&self) -> &MemberSignalId {
        &self.member_signal_id
    }

    fn population_hint(&self) -> usize {
        self.population_hint
    }

    #[instrument(skip_all, name = "scalar_l1_kernel", fields(
        member = %self.member_signal_id,
        population = self.population_hint,
    ))]
    fn execute(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        let start = std::time::Instant::now();

        // Get previous values slice
        let prev_values = population
            .signals()
            .prev_scalar_slice(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        let population_size = prev_values.len();
        trace!(population_size, "executing scalar L1 kernel");

        // Determine chunk size
        let chunk_size = self
            .fixed_chunk_size
            .unwrap_or_else(|| optimal_chunk_size(population_size));

        // Clone prev_values for parallel iteration
        let prev_vec: Vec<f64> = prev_values.to_vec();

        // Execute in parallel chunks, collecting results with indices
        // We need to get a reference to the member signals for cross-member reads
        let member_signals = population.signals();
        let results: Vec<(usize, f64)> = prev_vec
            .par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;
                chunk
                    .iter()
                    .enumerate()
                    .map(|(local_idx, &prev)| {
                        let global_idx = base_idx + local_idx;
                        let ctx = ScalarResolveContext {
                            prev,
                            index: EntityIndex(global_idx),
                            signals,
                            members: member_signals,
                            dt,
                        };
                        (global_idx, (self.resolver)(&ctx))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Write results to current buffer in index order
        let current_slice = population
            .signals_mut()
            .scalar_slice_mut(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        for (idx, value) in results {
            current_slice[idx] = value;
        }

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        debug!(population_size, elapsed_ns, "scalar L1 kernel complete");

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: Some(elapsed_ns),
        })
    }
}

// ============================================================================
// L1 Kernel: Vec3 Instance-Parallel
// ============================================================================

/// Resolver function type for Vec3 member signals.
pub type Vec3KernelFn = Arc<dyn Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync>;

/// L1 lane kernel for Vec3 member signals using instance-parallel execution.
pub struct Vec3L1Kernel {
    member_signal_id: MemberSignalId,
    signal_name: String,
    resolver: Vec3KernelFn,
    population_hint: usize,
    /// Fixed chunk size, or None for auto-computed size
    fixed_chunk_size: Option<usize>,
}

impl Vec3L1Kernel {
    /// Create a new Vec3 L1 kernel.
    pub fn new(
        member_signal_id: MemberSignalId,
        resolver: Vec3KernelFn,
        population_hint: usize,
    ) -> Self {
        let signal_name = member_signal_id.signal_name.clone();
        Self {
            member_signal_id,
            signal_name,
            resolver,
            population_hint,
            fixed_chunk_size: None,
        }
    }

    /// Create with a fixed chunk size.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.fixed_chunk_size = Some(chunk_size);
        self
    }
}

impl LaneKernel for Vec3L1Kernel {
    fn strategy(&self) -> LoweringStrategy {
        LoweringStrategy::InstanceParallel
    }

    fn member_signal_id(&self) -> &MemberSignalId {
        &self.member_signal_id
    }

    fn population_hint(&self) -> usize {
        self.population_hint
    }

    #[instrument(skip_all, name = "vec3_l1_kernel", fields(
        member = %self.member_signal_id,
        population = self.population_hint,
    ))]
    fn execute(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        let start = std::time::Instant::now();

        // Get previous values slice
        let prev_values = population
            .signals()
            .prev_vec3_slice(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        let population_size = prev_values.len();
        trace!(population_size, "executing Vec3 L1 kernel");

        // Determine chunk size
        let chunk_size = self
            .fixed_chunk_size
            .unwrap_or_else(|| optimal_chunk_size(population_size));

        // Clone prev_values for parallel iteration
        let prev_vec: Vec<[f64; 3]> = prev_values.to_vec();

        // Execute in parallel chunks
        let member_signals = population.signals();
        let results: Vec<(usize, [f64; 3])> = prev_vec
            .par_chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;
                chunk
                    .iter()
                    .enumerate()
                    .map(|(local_idx, &prev)| {
                        let global_idx = base_idx + local_idx;
                        let ctx = Vec3ResolveContext {
                            prev,
                            index: EntityIndex(global_idx),
                            signals,
                            members: member_signals,
                            dt,
                        };
                        (global_idx, (self.resolver)(&ctx))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Write results to current buffer
        let current_slice = population
            .signals_mut()
            .vec3_slice_mut(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        for (idx, value) in results {
            current_slice[idx] = value;
        }

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        debug!(population_size, elapsed_ns, "Vec3 L1 kernel complete");

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: Some(elapsed_ns),
        })
    }
}

// ============================================================================
// Kernel Registry
// ============================================================================

/// Registry of lane kernels for dispatch during execution.
///
/// The registry maps member signal IDs to their compiled lane kernels,
/// enabling the phase executor to dispatch to the correct kernel.
pub struct LaneKernelRegistry {
    kernels: indexmap::IndexMap<MemberSignalId, Box<dyn LaneKernel>>,
}

impl LaneKernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: indexmap::IndexMap::new(),
        }
    }

    /// Register a lane kernel.
    pub fn register(&mut self, kernel: impl LaneKernel + 'static) {
        let id = kernel.member_signal_id().clone();
        self.kernels.insert(id, Box::new(kernel));
    }

    /// Get a kernel by member signal ID.
    pub fn get(&self, id: &MemberSignalId) -> Option<&dyn LaneKernel> {
        self.kernels.get(id).map(|k| k.as_ref())
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Iterate over all kernels.
    pub fn iter(&self) -> impl Iterator<Item = (&MemberSignalId, &dyn LaneKernel)> {
        self.kernels.iter().map(|(k, v)| (k, v.as_ref()))
    }
}

impl Default for LaneKernelRegistry {
    fn default() -> Self {
        Self::new()
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
            l2_available: false, // Not yet implemented
            l3_available: false, // Not yet implemented
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
    use crate::soa_storage::ValueType;
    use crate::storage::SignalStorage;
    use crate::types::EntityId;

    fn make_member_signal_id(entity: &str, signal: &str) -> MemberSignalId {
        MemberSignalId::new(EntityId::from(entity), signal)
    }

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

        // With defaults (L2/L3 not available), always select L1
        assert_eq!(h.select(100), LoweringStrategy::InstanceParallel);
        assert_eq!(h.select(10000), LoweringStrategy::InstanceParallel);
        assert_eq!(h.select(100000), LoweringStrategy::InstanceParallel);
    }

    #[test]
    fn test_lowering_heuristics_with_l2() {
        let h = LoweringHeuristics {
            l2_available: true,
            ..Default::default()
        };

        assert_eq!(h.select(10000), LoweringStrategy::InstanceParallel);
        assert_eq!(h.select(60000), LoweringStrategy::VectorKernel);
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

    #[test]
    fn test_lane_kernel_registry() {
        let mut registry = LaneKernelRegistry::new();
        assert!(registry.is_empty());

        let id = make_member_signal_id("test.entity", "value");
        let kernel = ScalarL1Kernel::new(id.clone(), Arc::new(|ctx| ctx.prev + 1.0), 100);

        registry.register(kernel);
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().strategy(),
            LoweringStrategy::InstanceParallel
        );
    }

    #[test]
    fn test_scalar_l1_kernel_properties() {
        let id = make_member_signal_id("test.entity", "age");
        let kernel = ScalarL1Kernel::new(id.clone(), Arc::new(|ctx| ctx.prev + 1.0), 1000);

        assert_eq!(kernel.strategy(), LoweringStrategy::InstanceParallel);
        assert_eq!(kernel.member_signal_id(), &id);
        assert_eq!(kernel.population_hint(), 1000);
    }

    #[test]
    fn test_scalar_l1_kernel_execution() {
        // Set up population storage
        let mut population = PopulationStorage::new("test.entity".into());
        population.register_signal("counter".to_string(), ValueType::Scalar);

        // Register instances
        for i in 0..10 {
            population.register_instance(format!("inst_{}", i));
        }
        population.finalize();

        // Initialize values via set_current then advance
        for i in 0..10 {
            population.set_current(&format!("inst_{}", i), "counter", crate::types::Value::Scalar(i as f64));
        }
        population.advance_tick();

        // Create kernel that adds 1 to each value
        let id = make_member_signal_id("test.entity", "counter");
        let kernel = ScalarL1Kernel::new(id, Arc::new(|ctx| ctx.prev + 1.0), 10);

        // Execute
        let signals = SignalStorage::default();
        let result = kernel
            .execute(&signals, &mut population, Dt(1.0))
            .unwrap();

        assert_eq!(result.instances_processed, 10);
        assert!(result.execution_ns.is_some());

        // Verify results
        for i in 0..10 {
            let value = population.get_current(&format!("inst_{}", i), "counter");
            assert_eq!(
                value,
                Some(crate::types::Value::Scalar(i as f64 + 1.0)),
                "value at index {} incorrect",
                i
            );
        }
    }

    #[test]
    fn test_vec3_l1_kernel_execution() {
        // Set up population storage
        let mut population = PopulationStorage::new("test.entity".into());
        population.register_signal("position".to_string(), ValueType::Vec3);

        // Register instances
        for i in 0..5 {
            population.register_instance(format!("inst_{}", i));
        }
        population.finalize();

        // Initialize values
        for i in 0..5 {
            population.set_current(
                &format!("inst_{}", i),
                "position",
                crate::types::Value::Vec3([i as f64, 0.0, 0.0]),
            );
        }
        population.advance_tick();

        // Create kernel that moves position by [1, 1, 1]
        let id = make_member_signal_id("test.entity", "position");
        let kernel = Vec3L1Kernel::new(
            id,
            Arc::new(|ctx| [ctx.prev[0] + 1.0, ctx.prev[1] + 1.0, ctx.prev[2] + 1.0]),
            5,
        );

        // Execute
        let signals = SignalStorage::default();
        let result = kernel
            .execute(&signals, &mut population, Dt(1.0))
            .unwrap();

        assert_eq!(result.instances_processed, 5);

        // Verify results
        for i in 0..5 {
            let value = population.get_current(&format!("inst_{}", i), "position");
            assert_eq!(
                value,
                Some(crate::types::Value::Vec3([i as f64 + 1.0, 1.0, 1.0])),
                "value at index {} incorrect",
                i
            );
        }
    }

    #[test]
    fn test_lane_kernel_error_display() {
        let err = LaneKernelError::SignalNotFound("test.sig".to_string());
        assert!(err.to_string().contains("not found"));

        let err = LaneKernelError::NumericError {
            index: 42,
            message: "NaN".to_string(),
        };
        assert!(err.to_string().contains("42"));

        let err = LaneKernelError::ExecutionFailed("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
