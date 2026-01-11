//! Member signal L1 executor: Instance-parallel chunked execution.
//!
//! This module implements the L1 lowering strategy for member signals,
//! which divides the entity index space into chunks and processes each
//! chunk in parallel using rayon.
//!
//! # Architecture
//!
//! ```text
//! Entity Index Space: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
//!                      └─chunk 0─┘ └─chunk 1─┘ └─chunk 2─┘
//!                           ↓           ↓           ↓
//!                      ┌─thread 0─┐ ┌─thread 1─┐ ┌─thread 2─┐
//!                      │ resolve  │ │ resolve  │ │ resolve  │
//!                      │ instances│ │ instances│ │ instances│
//!                      └────↓─────┘ └────↓─────┘ └────↓─────┘
//!                           └──────────┬───────────┘
//!                                      ↓
//!                              Deterministic
//!                              result vector
//! ```
//!
//! # Determinism
//!
//! Results are collected in deterministic index order regardless of
//! which thread processed which chunk. This is achieved by:
//!
//! 1. Using `par_iter().enumerate()` to preserve indices
//! 2. Collecting into a pre-sized Vec
//! 3. Using index-based writes instead of push
//!
//! # Example
//!
//! ```ignore
//! use continuum_runtime::executor::member_executor::{
//!     MemberResolveContext, resolve_member_signal_l1, ChunkConfig,
//! };
//!
//! let results = resolve_member_signal_l1(
//!     &prev_values,
//!     |ctx| ctx.prev + 1.0, // Simple increment
//!     &signals,
//!     dt,
//!     ChunkConfig::auto(prev_values.len()),
//! );
//! ```

use rayon::prelude::*;

use crate::soa_storage::MemberSignalBuffer;
use crate::storage::SignalStorage;
use crate::types::{Dt, Value};
use crate::vectorized::EntityIndex;

// ============================================================================
// Context Types
// ============================================================================

/// Read-only context for resolving a single member signal instance.
///
/// This context provides access to:
/// - `prev` - Previous tick's value for this instance
/// - `index` - The entity instance index
/// - `signals` - Read-only access to global signals
/// - `members` - Read-only access to other member signals for this entity
/// - `dt` - Time step
///
/// # Read-Only Guarantee
///
/// All references are immutable. The resolver body must be a pure function
/// that computes a new value from these inputs without side effects.
pub struct MemberResolveContext<'a, T> {
    /// Previous tick's value for this instance
    pub prev: T,
    /// Entity instance index
    pub index: EntityIndex,
    /// Read-only access to global signals
    pub signals: &'a SignalStorage,
    /// Read-only access to member signal buffer (for cross-member reads)
    pub members: &'a MemberSignalBuffer,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Typed context for scalar member signal resolution.
pub type ScalarResolveContext<'a> = MemberResolveContext<'a, f64>;

/// Typed context for Vec3 member signal resolution.
pub type Vec3ResolveContext<'a> = MemberResolveContext<'a, [f64; 3]>;

/// Function signature for scalar member signal resolvers.
pub type ScalarResolverFn = Box<dyn Fn(&ScalarResolveContext) -> f64 + Send + Sync>;

/// Function signature for Vec3 member signal resolvers.
pub type Vec3ResolverFn = Box<dyn Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync>;

// ============================================================================
// Chunk Configuration
// ============================================================================

/// Configuration for chunked parallel execution.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Number of instances per chunk
    pub chunk_size: usize,
    /// Minimum chunk size (avoid too-small chunks)
    pub min_chunk: usize,
    /// Maximum chunk size (limit per-task memory)
    pub max_chunk: usize,
}

impl ChunkConfig {
    /// Create a chunk configuration with explicit size.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            min_chunk: 64,
            max_chunk: 4096,
        }
    }

    /// Compute optimal chunk size based on population.
    ///
    /// Uses 4x oversubscription to balance work distribution
    /// against scheduling overhead.
    pub fn auto(population_size: usize) -> Self {
        let chunk_size = optimal_chunk_size(population_size);
        Self {
            chunk_size,
            min_chunk: 64,
            max_chunk: 4096,
        }
    }

    /// Get effective chunk size after clamping.
    pub fn effective_size(&self) -> usize {
        self.chunk_size.clamp(self.min_chunk, self.max_chunk)
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            min_chunk: 64,
            max_chunk: 4096,
        }
    }
}

/// Compute optimal chunk size for a given population.
///
/// The formula targets 4x oversubscription (4 chunks per thread) to:
/// - Balance work if some instances take longer
/// - Hide scheduling latency
/// - Avoid too-large chunks that hurt cache locality
///
/// # Bounds
///
/// - Minimum: 64 (avoid excessive scheduling overhead)
/// - Maximum: 4096 (limit per-task working set)
pub fn optimal_chunk_size(population_size: usize) -> usize {
    let num_threads = rayon::current_num_threads();
    let min_chunk = 64;
    let max_chunk = 4096;

    // Target 4x oversubscription
    let ideal = population_size / (num_threads * 4);

    ideal.clamp(min_chunk, max_chunk)
}

// ============================================================================
// Generic Parallel Execution Helper
// ============================================================================

/// Execute a mapping function over a slice in parallel chunks.
///
/// This is the core parallel execution pattern for L1 member signal resolution.
/// Results are collected in deterministic index order regardless of thread scheduling.
///
/// # Arguments
///
/// * `values` - Input slice to process
/// * `map_fn` - Function receiving (global_index, &value) and returning result
/// * `chunk_size` - Number of elements per parallel chunk
/// * `serial_threshold` - Execute serially if population <= this threshold
///
/// # Determinism
///
/// Results are always in index order because:
/// 1. `par_chunks` preserves chunk ordering
/// 2. Within each chunk, we iterate sequentially
///
/// # Example
///
/// ```ignore
/// let results = parallel_chunked_map(
///     &values,
///     |idx, &value| value * 2.0,
///     256,  // chunk_size
///     64,   // serial_threshold
/// );
/// ```
pub fn parallel_chunked_map<T, U, F>(
    values: &[T],
    map_fn: F,
    chunk_size: usize,
    serial_threshold: usize,
) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync,
{
    let population = values.len();

    // Small populations: execute serially
    if population <= serial_threshold {
        return values
            .iter()
            .enumerate()
            .map(|(idx, value)| map_fn(idx, value))
            .collect();
    }

    // Parallel chunked execution with deterministic ordering
    values
        .par_chunks(chunk_size)
        .enumerate()
        .flat_map(|(chunk_idx, chunk)| {
            let base_idx = chunk_idx * chunk_size;
            chunk
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let global_idx = base_idx + i;
                    map_fn(global_idx, value)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// ============================================================================
// L1 Execution Functions
// ============================================================================

/// Resolve all instances of a scalar member signal using L1 strategy.
///
/// This function divides the population into chunks and processes them
/// in parallel, collecting results in deterministic index order.
///
/// # Arguments
///
/// * `prev_values` - Previous tick values for all instances
/// * `resolver` - Function to compute new value from context
/// * `signals` - Global signal storage (read-only)
/// * `members` - Member signal buffer (read-only)
/// * `dt` - Time step
/// * `config` - Chunk configuration
///
/// # Returns
///
/// Vector of new values in index order (same length as `prev_values`).
///
/// # Determinism
///
/// Results are always in the same order as input indices, regardless of
/// thread scheduling. This is achieved using indexed parallel iteration.
pub fn resolve_scalar_l1<F>(
    prev_values: &[f64],
    resolver: F,
    signals: &SignalStorage,
    members: &MemberSignalBuffer,
    dt: Dt,
    sim_time: f64,
    config: ChunkConfig,
) -> Vec<f64>
where
    F: Fn(&ScalarResolveContext) -> f64 + Sync,
{
    parallel_chunked_map(
        prev_values,
        |idx, &prev| {
            let ctx = MemberResolveContext {
                prev,
                index: EntityIndex(idx),
                signals,
                members,
                dt,
                sim_time,
            };
            resolver(&ctx)
        },
        config.effective_size(),
        config.min_chunk,
    )
}

/// Resolve all instances of a Vec3 member signal using L1 strategy.
///
/// Same as [`resolve_scalar_l1`] but for Vec3 values.
pub fn resolve_vec3_l1<F>(
    prev_values: &[[f64; 3]],
    resolver: F,
    signals: &SignalStorage,
    members: &MemberSignalBuffer,
    dt: Dt,
    sim_time: f64,
    config: ChunkConfig,
) -> Vec<[f64; 3]>
where
    F: Fn(&Vec3ResolveContext) -> [f64; 3] + Sync,
{
    parallel_chunked_map(
        prev_values,
        |idx, &prev| {
            let ctx = MemberResolveContext {
                prev,
                index: EntityIndex(idx),
                signals,
                members,
                dt,
                sim_time,
            };
            resolver(&ctx)
        },
        config.effective_size(),
        config.min_chunk,
    )
}

/// Generic L1 resolution for any value type.
///
/// This function uses `Value` enum for flexibility at the cost of some overhead.
/// Prefer the typed versions ([`resolve_scalar_l1`], [`resolve_vec3_l1`]) for
/// hot paths.
pub fn resolve_member_signal_l1<F>(
    prev_values: &[Value],
    resolver: F,
    signals: &SignalStorage,
    members: &MemberSignalBuffer,
    dt: Dt,
    sim_time: f64,
    config: ChunkConfig,
) -> Vec<Value>
where
    F: Fn(&MemberResolveContext<Value>) -> Value + Sync,
{
    parallel_chunked_map(
        prev_values,
        |idx, prev| {
            let ctx = MemberResolveContext {
                prev: prev.clone(),
                index: EntityIndex(idx),
                signals,
                members,
                dt,
                sim_time,
            };
            resolver(&ctx)
        },
        config.effective_size(),
        config.min_chunk,
    )
}

// ============================================================================
// Member Signal Resolver Trait
// ============================================================================

/// Trait for member signal resolvers with different lowering strategies.
///
/// This trait abstracts over the execution strategy (L1, L2, L3) while
/// maintaining the core semantics of member signal resolution.
pub trait MemberSignalResolver: Send + Sync {
    /// The value type for this resolver.
    type Value: Clone + Send + Sync;

    /// Resolve all instances and return new values in index order.
    ///
    /// # Arguments
    ///
    /// * `prev_values` - Previous tick values for all instances
    /// * `signals` - Global signal storage
    /// * `members` - Member signal buffer
    /// * `dt` - Time step
    /// * `sim_time` - Accumulated simulation time in seconds
    ///
    /// # Returns
    ///
    /// Vector of new values, one per instance, in index order.
    fn resolve_all(
        &self,
        prev_values: &[Self::Value],
        signals: &SignalStorage,
        members: &MemberSignalBuffer,
        dt: Dt,
        sim_time: f64,
    ) -> Vec<Self::Value>;
}

/// L1 scalar member signal resolver.
///
/// Implements instance-parallel chunked execution for scalar member signals.
pub struct ScalarL1Resolver<F>
where
    F: Fn(&ScalarResolveContext) -> f64 + Send + Sync,
{
    resolver: F,
    config: ChunkConfig,
}

impl<F> ScalarL1Resolver<F>
where
    F: Fn(&ScalarResolveContext) -> f64 + Send + Sync,
{
    /// Create a new L1 resolver with default chunk configuration.
    pub fn new(resolver: F) -> Self {
        Self {
            resolver,
            config: ChunkConfig::default(),
        }
    }

    /// Create a new L1 resolver with custom chunk configuration.
    pub fn with_config(resolver: F, config: ChunkConfig) -> Self {
        Self { resolver, config }
    }
}

impl<F> MemberSignalResolver for ScalarL1Resolver<F>
where
    F: Fn(&ScalarResolveContext) -> f64 + Send + Sync,
{
    type Value = f64;

    fn resolve_all(
        &self,
        prev_values: &[f64],
        signals: &SignalStorage,
        members: &MemberSignalBuffer,
        dt: Dt,
        sim_time: f64,
    ) -> Vec<f64> {
        let config = ChunkConfig {
            chunk_size: optimal_chunk_size(prev_values.len()),
            ..self.config
        };
        resolve_scalar_l1(prev_values, &self.resolver, signals, members, dt, sim_time, config)
    }
}

/// L1 Vec3 member signal resolver.
pub struct Vec3L1Resolver<F>
where
    F: Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync,
{
    resolver: F,
    config: ChunkConfig,
}

impl<F> Vec3L1Resolver<F>
where
    F: Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync,
{
    /// Create a new L1 resolver with default chunk configuration.
    pub fn new(resolver: F) -> Self {
        Self {
            resolver,
            config: ChunkConfig::default(),
        }
    }

    /// Create a new L1 resolver with custom chunk configuration.
    pub fn with_config(resolver: F, config: ChunkConfig) -> Self {
        Self { resolver, config }
    }
}

impl<F> MemberSignalResolver for Vec3L1Resolver<F>
where
    F: Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync,
{
    type Value = [f64; 3];

    fn resolve_all(
        &self,
        prev_values: &[[f64; 3]],
        signals: &SignalStorage,
        members: &MemberSignalBuffer,
        dt: Dt,
        sim_time: f64,
    ) -> Vec<[f64; 3]> {
        let config = ChunkConfig {
            chunk_size: optimal_chunk_size(prev_values.len()),
            ..self.config
        };
        resolve_vec3_l1(prev_values, &self.resolver, signals, members, dt, sim_time, config)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signals() -> SignalStorage {
        let mut storage = SignalStorage::default();
        storage.init("test.signal".into(), Value::Scalar(42.0));
        storage
    }

    fn create_test_members(count: usize) -> MemberSignalBuffer {
        let mut buffer = MemberSignalBuffer::new();
        buffer.register_signal("age".to_string(), crate::soa_storage::ValueType::Scalar);
        buffer.init_instances(count);
        buffer
    }

    #[test]
    fn test_optimal_chunk_size_small() {
        // For small populations, should return min_chunk
        let size = optimal_chunk_size(100);
        assert!(size >= 64);
    }

    #[test]
    fn test_optimal_chunk_size_large() {
        // For large populations, should return reasonable chunk
        let size = optimal_chunk_size(100_000);
        assert!(size >= 64);
        assert!(size <= 4096);
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 256);
        assert_eq!(config.min_chunk, 64);
        assert_eq!(config.max_chunk, 4096);
    }

    #[test]
    fn test_chunk_config_auto() {
        let config = ChunkConfig::auto(10_000);
        assert!(config.chunk_size >= 64);
        assert!(config.chunk_size <= 4096);
    }

    #[test]
    fn test_resolve_scalar_l1_small() {
        let prev_values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let signals = create_test_signals();
        let members = create_test_members(10);
        let config = ChunkConfig::default();

        let results = resolve_scalar_l1(
            &prev_values,
            |ctx| ctx.prev + 1.0,
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        assert_eq!(results.len(), 10);
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, i as f64 + 1.0, "Index {} mismatch", i);
        }
    }

    #[test]
    fn test_resolve_scalar_l1_large() {
        let population = 10_000;
        let prev_values: Vec<f64> = (0..population).map(|i| i as f64).collect();
        let signals = create_test_signals();
        let members = create_test_members(population);
        let config = ChunkConfig::auto(population);

        let results = resolve_scalar_l1(
            &prev_values,
            |ctx| ctx.prev * 2.0,
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        assert_eq!(results.len(), population);
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, (i as f64) * 2.0, "Index {} mismatch", i);
        }
    }

    #[test]
    fn test_resolve_scalar_l1_deterministic() {
        let population = 1000;
        let prev_values: Vec<f64> = (0..population).map(|i| i as f64).collect();
        let signals = create_test_signals();
        let members = create_test_members(population);
        let config = ChunkConfig::auto(population);

        // Run multiple times and verify same results
        let results1 = resolve_scalar_l1(
            &prev_values,
            |ctx| ctx.prev + ctx.index.0 as f64,
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        let results2 = resolve_scalar_l1(
            &prev_values,
            |ctx| ctx.prev + ctx.index.0 as f64,
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        assert_eq!(results1, results2);
    }

    #[test]
    fn test_resolve_scalar_l1_index_order() {
        let population = 500;
        let prev_values: Vec<f64> = vec![0.0; population];
        let signals = create_test_signals();
        let members = create_test_members(population);
        let config = ChunkConfig::auto(population);

        // Resolver that returns the index itself
        let results = resolve_scalar_l1(
            &prev_values,
            |ctx| ctx.index.0 as f64,
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        // Verify indices are in order
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, i as f64, "Index {} not in order", i);
        }
    }

    #[test]
    fn test_resolve_vec3_l1() {
        let population = 100;
        let prev_values: Vec<[f64; 3]> = (0..population).map(|i| [i as f64, 0.0, 0.0]).collect();
        let signals = create_test_signals();
        let members = create_test_members(population);
        let config = ChunkConfig::auto(population);

        let results = resolve_vec3_l1(
            &prev_values,
            |ctx| [ctx.prev[0] + 1.0, ctx.prev[1] + 2.0, ctx.prev[2] + 3.0],
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        assert_eq!(results.len(), population);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result[0], i as f64 + 1.0);
            assert_eq!(result[1], 2.0);
            assert_eq!(result[2], 3.0);
        }
    }

    #[test]
    fn test_scalar_l1_resolver_trait() {
        let resolver = ScalarL1Resolver::new(|ctx: &ScalarResolveContext| ctx.prev + 1.0);

        let prev_values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signals = create_test_signals();
        let members = create_test_members(5);

        let results = resolver.resolve_all(&prev_values, &signals, &members, Dt(1.0), 0.0);

        assert_eq!(results, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scalar_l1_resolver_with_dt() {
        let resolver = ScalarL1Resolver::new(|ctx: &ScalarResolveContext| {
            ctx.prev + 10.0 * ctx.dt.seconds()
        });

        let prev_values: Vec<f64> = vec![0.0, 0.0, 0.0];
        let signals = create_test_signals();
        let members = create_test_members(3);

        let results = resolver.resolve_all(&prev_values, &signals, &members, Dt(0.5), 0.0);

        assert_eq!(results, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_resolve_with_signal_access() {
        let mut signals = SignalStorage::default();
        signals.init("multiplier".into(), Value::Scalar(3.0));
        let members = create_test_members(5);

        let prev_values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = ChunkConfig::default();

        let results = resolve_scalar_l1(
            &prev_values,
            |ctx| {
                let mult = ctx
                    .signals
                    .get(&"multiplier".into())
                    .and_then(|v| v.as_scalar())
                    .unwrap_or(1.0);
                ctx.prev * mult
            },
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        assert_eq!(results, vec![3.0, 6.0, 9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_generic_member_signal_l1() {
        let prev_values: Vec<Value> = vec![
            Value::Scalar(1.0),
            Value::Scalar(2.0),
            Value::Scalar(3.0),
        ];
        let signals = create_test_signals();
        let members = create_test_members(3);
        let config = ChunkConfig::default();

        let results = resolve_member_signal_l1(
            &prev_values,
            |ctx| match &ctx.prev {
                Value::Scalar(v) => Value::Scalar(v + 10.0),
                other => other.clone(),
            },
            &signals,
            &members,
            Dt(1.0),
            0.0,
            config,
        );

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Value::Scalar(11.0));
        assert_eq!(results[1], Value::Scalar(12.0));
        assert_eq!(results[2], Value::Scalar(13.0));
    }
}
