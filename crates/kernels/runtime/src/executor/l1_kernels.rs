//! L1 lane kernels for instance-parallel member signal execution.
//!
//! L1 kernels use chunked CPU parallelism via rayon to resolve
//! member signals across all instances of an entity.
//!
//! # Generic Architecture
//!
//! The [`L1Kernel`] struct is generic over value type `T` which must implement
//! [`L1KernelValue`]. This trait provides the type-specific operations for
//! accessing SoA storage slices, enabling a single implementation to handle
//! Scalar, Vec2, Vec3, Vec4, and future types.

use std::sync::Arc;

use tracing::{debug, instrument, trace};

use crate::soa_storage::{MemberSignalBuffer, PopulationStorage};
use crate::storage::SignalStorage;
use crate::types::Dt;
use crate::vectorized::{EntityIndex, MemberSignalId};

use super::lane_kernel::{LaneKernel, LaneKernelError, LaneKernelResult};
use super::lowering_strategy::LoweringStrategy;
use super::member_executor::{optimal_chunk_size, parallel_chunked_map, MemberResolveContext};

// ============================================================================
// L1 Kernel Value Trait
// ============================================================================

/// Trait for value types that can be used with L1 kernels.
///
/// This trait abstracts over the type-specific slice access methods in
/// [`MemberSignalBuffer`], enabling generic L1 kernel implementation.
///
/// # Implementors
///
/// - `f64` (Scalar)
/// - `[f64; 2]` (Vec2)
/// - `[f64; 3]` (Vec3)
/// - `[f64; 4]` (Vec4/Quat)
///
/// # Example
///
/// ```ignore
/// // The trait enables generic kernel creation:
/// let scalar_kernel = L1Kernel::<f64>::new(id, resolver, 1000);
/// let vec3_kernel = L1Kernel::<[f64; 3]>::new(id, resolver, 1000);
/// ```
pub trait L1KernelValue: Copy + Send + Sync + 'static {
    /// Get the previous tick's slice for this value type.
    fn get_prev_slice<'a>(signals: &'a MemberSignalBuffer, name: &str) -> Option<&'a [Self]>;

    /// Get the current tick's mutable slice for this value type.
    fn get_current_slice_mut<'a>(
        signals: &'a mut MemberSignalBuffer,
        name: &str,
    ) -> Option<&'a mut [Self]>;

    /// Human-readable type name for logging.
    fn type_name() -> &'static str;
}

impl L1KernelValue for f64 {
    fn get_prev_slice<'a>(signals: &'a MemberSignalBuffer, name: &str) -> Option<&'a [Self]> {
        signals.prev_scalar_slice(name)
    }

    fn get_current_slice_mut<'a>(
        signals: &'a mut MemberSignalBuffer,
        name: &str,
    ) -> Option<&'a mut [Self]> {
        signals.scalar_slice_mut(name)
    }

    fn type_name() -> &'static str {
        "scalar"
    }
}

impl L1KernelValue for [f64; 2] {
    fn get_prev_slice<'a>(signals: &'a MemberSignalBuffer, name: &str) -> Option<&'a [Self]> {
        signals.prev_vec2_slice(name)
    }

    fn get_current_slice_mut<'a>(
        signals: &'a mut MemberSignalBuffer,
        name: &str,
    ) -> Option<&'a mut [Self]> {
        signals.vec2_slice_mut(name)
    }

    fn type_name() -> &'static str {
        "vec2"
    }
}

impl L1KernelValue for [f64; 3] {
    fn get_prev_slice<'a>(signals: &'a MemberSignalBuffer, name: &str) -> Option<&'a [Self]> {
        signals.prev_vec3_slice(name)
    }

    fn get_current_slice_mut<'a>(
        signals: &'a mut MemberSignalBuffer,
        name: &str,
    ) -> Option<&'a mut [Self]> {
        signals.vec3_slice_mut(name)
    }

    fn type_name() -> &'static str {
        "vec3"
    }
}

impl L1KernelValue for [f64; 4] {
    fn get_prev_slice<'a>(signals: &'a MemberSignalBuffer, name: &str) -> Option<&'a [Self]> {
        signals.prev_vec4_slice(name)
    }

    fn get_current_slice_mut<'a>(
        signals: &'a mut MemberSignalBuffer,
        name: &str,
    ) -> Option<&'a mut [Self]> {
        signals.vec4_slice_mut(name)
    }

    fn type_name() -> &'static str {
        "vec4"
    }
}

// ============================================================================
// Generic L1 Kernel
// ============================================================================

/// Resolver function type for member signals of type `T`.
pub type KernelFn<T> = Arc<dyn Fn(&MemberResolveContext<T>) -> T + Send + Sync>;

/// L1 lane kernel for member signals using instance-parallel execution.
///
/// This kernel uses chunked parallel execution via rayon to resolve
/// all instances of a member signal. It is generic over the value type `T`,
/// which must implement [`L1KernelValue`].
///
/// # Type Parameters
///
/// * `T` - The value type (f64, [f64; 2], [f64; 3], [f64; 4])
///
/// # Example
///
/// ```ignore
/// use continuum_runtime::executor::l1_kernels::{L1Kernel, KernelFn};
/// use std::sync::Arc;
///
/// // Scalar kernel
/// let scalar_kernel = L1Kernel::<f64>::new(
///     member_signal_id,
///     Arc::new(|ctx| ctx.prev + 1.0),
///     1000,
/// );
///
/// // Vec3 kernel
/// let vec3_kernel = L1Kernel::<[f64; 3]>::new(
///     member_signal_id,
///     Arc::new(|ctx| [ctx.prev[0] + 1.0, ctx.prev[1], ctx.prev[2]]),
///     1000,
/// );
/// ```
pub struct L1Kernel<T: L1KernelValue> {
    member_signal_id: MemberSignalId,
    /// The signal name used for buffer access
    signal_name: String,
    resolver: KernelFn<T>,
    population_hint: usize,
    /// Fixed chunk size, or None for auto-computed size
    fixed_chunk_size: Option<usize>,
}

impl<T: L1KernelValue> L1Kernel<T> {
    /// Create a new L1 kernel.
    ///
    /// # Arguments
    ///
    /// * `member_signal_id` - The member signal this kernel resolves
    /// * `resolver` - The resolver function
    /// * `population_hint` - Expected population size
    pub fn new(
        member_signal_id: MemberSignalId,
        resolver: KernelFn<T>,
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

impl<T: L1KernelValue> LaneKernel for L1Kernel<T> {
    fn strategy(&self) -> LoweringStrategy {
        LoweringStrategy::InstanceParallel
    }

    fn member_signal_id(&self) -> &MemberSignalId {
        &self.member_signal_id
    }

    fn population_hint(&self) -> usize {
        self.population_hint
    }

    #[instrument(skip_all, name = "l1_kernel", fields(
        member = %self.member_signal_id,
        value_type = T::type_name(),
        population = self.population_hint,
    ))]
    fn execute(
        &self,
        signals: &SignalStorage,
        entities: &crate::storage::EntityStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        let start = std::time::Instant::now();

        // Get previous values slice using the trait method
        let prev_values = T::get_prev_slice(population.signals(), &self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        let population_size = prev_values.len();
        trace!(
            population_size,
            value_type = T::type_name(),
            "executing L1 kernel"
        );

        // Determine chunk size
        let chunk_size = self
            .fixed_chunk_size
            .unwrap_or_else(|| optimal_chunk_size(population_size));

        // Execute in parallel chunks using the generic helper
        let member_signals = population.signals();
        let results = parallel_chunked_map(
            prev_values,
            |idx, &prev| {
                let ctx = MemberResolveContext {
                    prev,
                    index: EntityIndex(idx),
                    signals,
                    entities,
                    members: member_signals,
                    dt,
                    sim_time: 0.0, // TODO: Add sim_time to LaneKernel trait
                };
                (self.resolver)(&ctx)
            },
            chunk_size,
            64, // serial_threshold
        );

        // Write results to current buffer using the trait method
        let current_slice =
            T::get_current_slice_mut(population.signals_mut(), &self.signal_name)
                .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        current_slice.copy_from_slice(&results);

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        debug!(
            population_size,
            elapsed_ns,
            value_type = T::type_name(),
            "L1 kernel complete"
        );

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: Some(elapsed_ns),
        })
    }
}

// ============================================================================
// Type Aliases for Backward Compatibility
// ============================================================================

/// Scalar L1 kernel (backward compatibility alias).
pub type ScalarL1Kernel = L1Kernel<f64>;

/// Vec3 L1 kernel (backward compatibility alias).
pub type Vec3L1Kernel = L1Kernel<[f64; 3]>;

/// Vec2 L1 kernel.
pub type Vec2L1Kernel = L1Kernel<[f64; 2]>;

/// Vec4 L1 kernel.
pub type Vec4L1Kernel = L1Kernel<[f64; 4]>;

/// Resolver function type for scalar member signals (backward compatibility).
pub type ScalarKernelFn = KernelFn<f64>;

/// Resolver function type for Vec3 member signals (backward compatibility).
pub type Vec3KernelFn = KernelFn<[f64; 3]>;

// Re-export context types for convenience
pub use super::member_executor::{ScalarResolveContext, Vec3ResolveContext};

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
        population.register_signal("counter".to_string(), ValueType::scalar());

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
        let entities = crate::storage::EntityStorage::default();
        let result = kernel
            .execute(&signals, &entities, &mut population, Dt(1.0))
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
        population.register_signal("position".to_string(), ValueType::vec3());

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
        let entities = crate::storage::EntityStorage::default();
        let result = kernel
            .execute(&signals, &entities, &mut population, Dt(1.0))
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
