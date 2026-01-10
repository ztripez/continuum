//! SIMD-Optimized Vectorized Operations
//!
//! Provides SIMD-friendly implementations of common expression patterns for
//! L2 vectorized execution. These functions are structured to maximize LLVM
//! auto-vectorization opportunities.
//!
//! # Design Principles
//!
//! 1. **Chunk Processing**: Process data in chunks of 8 elements (matching AVX/AVX2)
//! 2. **Explicit Unrolling**: Use explicit loops that LLVM recognizes as vectorizable
//! 3. **Alignment Hints**: Arrays use 64-byte alignment for optimal cache line access
//! 4. **No Branches in Hot Loops**: Keep loop bodies branch-free for SIMD efficiency

/// SIMD lane width for processing (matches AVX 256-bit = 4 f64)
pub const SIMD_WIDTH: usize = 4;

/// Extended SIMD width for AVX-512 (8 f64)
pub const SIMD_WIDTH_WIDE: usize = 8;

// ============================================================================
// Simple Accumulator: prev + collected
// ============================================================================

/// SIMD-optimized simple accumulation: result[i] = prev[i] + collected[i]
///
/// This is the most common pattern (~30-40% of signals).
#[inline]
pub fn simd_accumulate(prev: &[f64], collected: &[f64], result: &mut [f64]) {
    debug_assert_eq!(prev.len(), collected.len());
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    // Process in chunks of 8 for AVX-512 / 2x AVX
    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        // Explicit unroll for SIMD auto-vectorization
        result[base] = prev[base] + collected[base];
        result[base + 1] = prev[base + 1] + collected[base + 1];
        result[base + 2] = prev[base + 2] + collected[base + 2];
        result[base + 3] = prev[base + 3] + collected[base + 3];
        result[base + 4] = prev[base + 4] + collected[base + 4];
        result[base + 5] = prev[base + 5] + collected[base + 5];
        result[base + 6] = prev[base + 6] + collected[base + 6];
        result[base + 7] = prev[base + 7] + collected[base + 7];
    }

    // Handle remainder
    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = prev[base + i] + collected[base + i];
    }
}

/// SIMD-optimized accumulation with broadcast: result[i] = prev[i] + scalar
#[inline]
pub fn simd_accumulate_broadcast(prev: &[f64], scalar: f64, result: &mut [f64]) {
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        result[base] = prev[base] + scalar;
        result[base + 1] = prev[base + 1] + scalar;
        result[base + 2] = prev[base + 2] + scalar;
        result[base + 3] = prev[base + 3] + scalar;
        result[base + 4] = prev[base + 4] + scalar;
        result[base + 5] = prev[base + 5] + scalar;
        result[base + 6] = prev[base + 6] + scalar;
        result[base + 7] = prev[base + 7] + scalar;
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = prev[base + i] + scalar;
    }
}

// ============================================================================
// Clamped Accumulator: clamp(prev + collected, min, max)
// ============================================================================

/// SIMD-optimized clamped accumulation with uniform bounds.
///
/// Common pattern (~20-30% of signals): clamp(prev + collected, min, max)
#[inline]
pub fn simd_clamp_accumulate(
    prev: &[f64],
    collected: &[f64],
    min: f64,
    max: f64,
    result: &mut [f64],
) {
    debug_assert_eq!(prev.len(), collected.len());
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        // Accumulate then clamp - ordered to maximize instruction-level parallelism
        let v0 = prev[base] + collected[base];
        let v1 = prev[base + 1] + collected[base + 1];
        let v2 = prev[base + 2] + collected[base + 2];
        let v3 = prev[base + 3] + collected[base + 3];
        let v4 = prev[base + 4] + collected[base + 4];
        let v5 = prev[base + 5] + collected[base + 5];
        let v6 = prev[base + 6] + collected[base + 6];
        let v7 = prev[base + 7] + collected[base + 7];

        result[base] = v0.clamp(min, max);
        result[base + 1] = v1.clamp(min, max);
        result[base + 2] = v2.clamp(min, max);
        result[base + 3] = v3.clamp(min, max);
        result[base + 4] = v4.clamp(min, max);
        result[base + 5] = v5.clamp(min, max);
        result[base + 6] = v6.clamp(min, max);
        result[base + 7] = v7.clamp(min, max);
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = (prev[base + i] + collected[base + i]).clamp(min, max);
    }
}

/// SIMD-optimized clamped accumulation with per-element bounds.
#[inline]
pub fn simd_clamp_accumulate_varying(
    prev: &[f64],
    collected: &[f64],
    min: &[f64],
    max: &[f64],
    result: &mut [f64],
) {
    debug_assert_eq!(prev.len(), collected.len());
    debug_assert_eq!(prev.len(), min.len());
    debug_assert_eq!(prev.len(), max.len());
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();

    // Simple loop - LLVM will vectorize this well
    for i in 0..n {
        result[i] = (prev[i] + collected[i]).clamp(min[i], max[i]);
    }
}

// ============================================================================
// Decay: prev * exp(-ln(2) * dt / half_life)
// ============================================================================

/// SIMD-optimized decay with uniform half-life.
///
/// Uses precomputed decay factor for efficiency.
#[inline]
pub fn simd_decay_uniform(prev: &[f64], decay_factor: f64, result: &mut [f64]) {
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        result[base] = prev[base] * decay_factor;
        result[base + 1] = prev[base + 1] * decay_factor;
        result[base + 2] = prev[base + 2] * decay_factor;
        result[base + 3] = prev[base + 3] * decay_factor;
        result[base + 4] = prev[base + 4] * decay_factor;
        result[base + 5] = prev[base + 5] * decay_factor;
        result[base + 6] = prev[base + 6] * decay_factor;
        result[base + 7] = prev[base + 7] * decay_factor;
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = prev[base + i] * decay_factor;
    }
}

/// Compute decay factor from half-life and dt.
#[inline]
pub fn decay_factor(dt: f64, half_life: f64) -> f64 {
    (-std::f64::consts::LN_2 * dt / half_life).exp()
}

/// SIMD-optimized decay accumulator: decay(prev, h) + collected
#[inline]
pub fn simd_decay_accumulate(
    prev: &[f64],
    collected: &[f64],
    decay_factor: f64,
    result: &mut [f64],
) {
    debug_assert_eq!(prev.len(), collected.len());
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        result[base] = prev[base] * decay_factor + collected[base];
        result[base + 1] = prev[base + 1] * decay_factor + collected[base + 1];
        result[base + 2] = prev[base + 2] * decay_factor + collected[base + 2];
        result[base + 3] = prev[base + 3] * decay_factor + collected[base + 3];
        result[base + 4] = prev[base + 4] * decay_factor + collected[base + 4];
        result[base + 5] = prev[base + 5] * decay_factor + collected[base + 5];
        result[base + 6] = prev[base + 6] * decay_factor + collected[base + 6];
        result[base + 7] = prev[base + 7] * decay_factor + collected[base + 7];
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = prev[base + i] * decay_factor + collected[base + i];
    }
}

// ============================================================================
// Linear Transform: a * prev + b * collected + c
// ============================================================================

/// SIMD-optimized linear transform with uniform coefficients.
#[inline]
pub fn simd_linear_transform(
    prev: &[f64],
    collected: &[f64],
    a: f64,
    b: f64,
    c: f64,
    result: &mut [f64],
) {
    debug_assert_eq!(prev.len(), collected.len());
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        result[base] = a * prev[base] + b * collected[base] + c;
        result[base + 1] = a * prev[base + 1] + b * collected[base + 1] + c;
        result[base + 2] = a * prev[base + 2] + b * collected[base + 2] + c;
        result[base + 3] = a * prev[base + 3] + b * collected[base + 3] + c;
        result[base + 4] = a * prev[base + 4] + b * collected[base + 4] + c;
        result[base + 5] = a * prev[base + 5] + b * collected[base + 5] + c;
        result[base + 6] = a * prev[base + 6] + b * collected[base + 6] + c;
        result[base + 7] = a * prev[base + 7] + b * collected[base + 7] + c;
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = a * prev[base + i] + b * collected[base + i] + c;
    }
}

// ============================================================================
// Integration: prev + rate * dt
// ============================================================================

/// SIMD-optimized Euler integration with uniform rate.
#[inline]
pub fn simd_integrate_uniform_rate(prev: &[f64], rate: f64, dt: f64, result: &mut [f64]) {
    let delta = rate * dt;
    simd_accumulate_broadcast(prev, delta, result);
}

/// SIMD-optimized Euler integration with per-element rate.
#[inline]
pub fn simd_integrate(prev: &[f64], rate: &[f64], dt: f64, result: &mut [f64]) {
    debug_assert_eq!(prev.len(), rate.len());
    debug_assert_eq!(prev.len(), result.len());

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        result[base] = prev[base] + rate[base] * dt;
        result[base + 1] = prev[base + 1] + rate[base + 1] * dt;
        result[base + 2] = prev[base + 2] + rate[base + 2] * dt;
        result[base + 3] = prev[base + 3] + rate[base + 3] * dt;
        result[base + 4] = prev[base + 4] + rate[base + 4] * dt;
        result[base + 5] = prev[base + 5] + rate[base + 5] * dt;
        result[base + 6] = prev[base + 6] + rate[base + 6] * dt;
        result[base + 7] = prev[base + 7] + rate[base + 7] * dt;
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = prev[base + i] + rate[base + i] * dt;
    }
}

// ============================================================================
// Passthrough and Constant
// ============================================================================

/// SIMD-optimized copy (for passthrough pattern).
#[inline]
pub fn simd_copy(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

/// Fill array with constant value.
#[inline]
pub fn simd_fill(value: f64, dst: &mut [f64]) {
    dst.fill(value);
}

// ============================================================================
// Smoothing / Relaxation: prev + (target - prev) * factor
// ============================================================================

/// SIMD-optimized exponential smoothing.
#[inline]
pub fn simd_smooth(prev: &[f64], target: &[f64], factor: f64, result: &mut [f64]) {
    debug_assert_eq!(prev.len(), target.len());
    debug_assert_eq!(prev.len(), result.len());

    let one_minus_factor = 1.0 - factor;

    let n = prev.len();
    let chunks = n / SIMD_WIDTH_WIDE;
    let remainder = n % SIMD_WIDTH_WIDE;

    for chunk in 0..chunks {
        let base = chunk * SIMD_WIDTH_WIDE;

        // Equivalent to: prev + (target - prev) * factor
        // Rewritten as: prev * (1 - factor) + target * factor for numerical stability
        result[base] = prev[base] * one_minus_factor + target[base] * factor;
        result[base + 1] = prev[base + 1] * one_minus_factor + target[base + 1] * factor;
        result[base + 2] = prev[base + 2] * one_minus_factor + target[base + 2] * factor;
        result[base + 3] = prev[base + 3] * one_minus_factor + target[base + 3] * factor;
        result[base + 4] = prev[base + 4] * one_minus_factor + target[base + 4] * factor;
        result[base + 5] = prev[base + 5] * one_minus_factor + target[base + 5] * factor;
        result[base + 6] = prev[base + 6] * one_minus_factor + target[base + 6] * factor;
        result[base + 7] = prev[base + 7] * one_minus_factor + target[base + 7] * factor;
    }

    let base = chunks * SIMD_WIDTH_WIDE;
    for i in 0..remainder {
        result[base + i] = prev[base + i] * one_minus_factor + target[base + i] * factor;
    }
}

/// Compute smoothing factor from dt and tau.
#[inline]
pub fn smooth_factor(dt: f64, tau: f64) -> f64 {
    1.0 - (-dt / tau).exp()
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if a slice is aligned for SIMD operations.
#[inline]
pub fn is_simd_aligned(slice: &[f64]) -> bool {
    let ptr = slice.as_ptr() as usize;
    ptr % 64 == 0 // 64-byte alignment for cache line / AVX-512
}

/// Calculate optimal chunk size based on population.
#[inline]
pub fn optimal_chunk_size(population: usize) -> usize {
    if population >= 1024 {
        SIMD_WIDTH_WIDE * 4 // Process 32 elements per iteration for large populations
    } else if population >= 64 {
        SIMD_WIDTH_WIDE
    } else {
        SIMD_WIDTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_accumulate() {
        let prev = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let collected = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut result = vec![0.0; 10];

        simd_accumulate(&prev, &collected, &mut result);

        assert!((result[0] - 1.1).abs() < 1e-10);
        assert!((result[4] - 5.5).abs() < 1e-10);
        assert!((result[9] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_clamp_accumulate() {
        let prev = vec![90.0, 95.0, 100.0, 50.0];
        let collected = vec![10.0, 10.0, 10.0, 10.0];
        let mut result = vec![0.0; 4];

        simd_clamp_accumulate(&prev, &collected, 0.0, 100.0, &mut result);

        assert!((result[0] - 100.0).abs() < 1e-10); // 90+10 = 100 (clamped)
        assert!((result[1] - 100.0).abs() < 1e-10); // 95+10 = 105 → 100
        assert!((result[2] - 100.0).abs() < 1e-10); // 100+10 = 110 → 100
        assert!((result[3] - 60.0).abs() < 1e-10); // 50+10 = 60 (no clamp)
    }

    #[test]
    fn test_simd_decay_uniform() {
        let prev = vec![100.0, 200.0, 300.0, 400.0];
        let factor = 0.5; // 50% decay
        let mut result = vec![0.0; 4];

        simd_decay_uniform(&prev, factor, &mut result);

        assert!((result[0] - 50.0).abs() < 1e-10);
        assert!((result[1] - 100.0).abs() < 1e-10);
        assert!((result[2] - 150.0).abs() < 1e-10);
        assert!((result[3] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_linear_transform() {
        let prev = vec![1.0, 2.0, 3.0, 4.0];
        let collected = vec![10.0, 20.0, 30.0, 40.0];
        let mut result = vec![0.0; 4];

        // result = 2 * prev + 0.1 * collected + 5
        simd_linear_transform(&prev, &collected, 2.0, 0.1, 5.0, &mut result);

        assert!((result[0] - 8.0).abs() < 1e-10); // 2*1 + 0.1*10 + 5 = 8
        assert!((result[1] - 11.0).abs() < 1e-10); // 2*2 + 0.1*20 + 5 = 11
    }

    #[test]
    fn test_simd_integrate() {
        let prev = vec![0.0, 10.0, 20.0, 30.0];
        let rate = vec![1.0, 2.0, 3.0, 4.0];
        let dt = 0.5;
        let mut result = vec![0.0; 4];

        simd_integrate(&prev, &rate, dt, &mut result);

        assert!((result[0] - 0.5).abs() < 1e-10); // 0 + 1*0.5
        assert!((result[1] - 11.0).abs() < 1e-10); // 10 + 2*0.5
        assert!((result[2] - 21.5).abs() < 1e-10); // 20 + 3*0.5
    }

    #[test]
    fn test_simd_smooth() {
        let prev = vec![0.0, 0.0, 0.0, 0.0];
        let target = vec![100.0, 100.0, 100.0, 100.0];
        let factor = 0.1; // Move 10% toward target
        let mut result = vec![0.0; 4];

        simd_smooth(&prev, &target, factor, &mut result);

        assert!((result[0] - 10.0).abs() < 1e-10); // 0 + 0.1 * (100 - 0) = 10
        assert!((result[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_factor() {
        let dt = 1.0;
        let half_life = 1.0;
        let factor = decay_factor(dt, half_life);

        // After one half-life, should be ~0.5
        assert!((factor - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_large_population() {
        // Test with a large population to verify SIMD chunking works
        let n = 10_000;
        let prev: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let collected: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let mut result = vec![0.0; n];

        simd_accumulate(&prev, &collected, &mut result);

        for i in 0..n {
            let expected = (i + i * 2) as f64;
            assert!(
                (result[i] - expected).abs() < 1e-10,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result[i]
            );
        }
    }
}
