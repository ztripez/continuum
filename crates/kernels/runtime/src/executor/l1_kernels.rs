//! L1 lane kernels for instance-parallel member signal execution.
//!
//! L1 kernels use chunked CPU parallelism via rayon to resolve
//! member signals across all instances of an entity.

use std::sync::Arc;

use tracing::{debug, instrument, trace};

use crate::soa_storage::PopulationStorage;
use crate::storage::SignalStorage;
use crate::types::Dt;
use crate::vectorized::{EntityIndex, MemberSignalId};

use super::lane_kernel::{LaneKernel, LaneKernelError, LaneKernelResult};
use super::lowering_strategy::LoweringStrategy;
use super::member_executor::{
    optimal_chunk_size, parallel_chunked_map, ScalarResolveContext, Vec3ResolveContext,
};

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

        // Execute in parallel chunks using the generic helper
        // prev_values and member_signals are both immutable borrows that can coexist
        let member_signals = population.signals();
        let results = parallel_chunked_map(
            prev_values,
            |idx, &prev| {
                let ctx = ScalarResolveContext {
                    prev,
                    index: EntityIndex(idx),
                    signals,
                    members: member_signals,
                    dt,
                };
                (self.resolver)(&ctx)
            },
            chunk_size,
            64, // serial_threshold
        );

        // Write results to current buffer (order preserved by parallel_chunked_map)
        let current_slice = population
            .signals_mut()
            .scalar_slice_mut(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        current_slice.copy_from_slice(&results);

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

        // Execute in parallel chunks using the generic helper
        // prev_values and member_signals are both immutable borrows that can coexist
        let member_signals = population.signals();
        let results = parallel_chunked_map(
            prev_values,
            |idx, &prev| {
                let ctx = Vec3ResolveContext {
                    prev,
                    index: EntityIndex(idx),
                    signals,
                    members: member_signals,
                    dt,
                };
                (self.resolver)(&ctx)
            },
            chunk_size,
            64, // serial_threshold
        );

        // Write results to current buffer (order preserved by parallel_chunked_map)
        let current_slice = population
            .signals_mut()
            .vec3_slice_mut(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        current_slice.copy_from_slice(&results);

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        debug!(population_size, elapsed_ns, "Vec3 L1 kernel complete");

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: Some(elapsed_ns),
        })
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
            population.set_current(
                &format!("inst_{}", i),
                "counter",
                crate::types::Value::Scalar(i as f64),
            );
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
}
