//! Deterministic Random Number Generation
//!
//! Provides stable, reproducible pseudo-random number generation derived from
//! labeled seeds. All randomness in Continuum must be derived from the world
//! seed via labeled derivation to ensure determinism.
//!
//! # PRNG Algorithm
//!
//! Uses SplitMix64, a fast, high-quality PRNG that is:
//! - Deterministic and reproducible
//! - Portable (same results on all platforms)
//! - Fast for both scalar and SIMD execution
//! - Good statistical quality for simulation purposes
//!
//! # Stream Model
//!
//! ```text
//! world_seed
//!   └─> primitive_path ("terra.plate.velocity")
//!         └─> entity_id (for members)
//!               └─> advances with each rng call, never resets
//!                     └─> derive("label") creates substream via state mixing
//! ```

use crate::stable_hash::fnv1a64_str;
use std::f64::consts::PI;

/// A deterministic pseudo-random number stream.
///
/// Streams are created from seeds (typically derived from the world seed plus
/// a label) and produce a reproducible sequence of random values. Each call
/// to a generation method advances the stream state.
///
/// Streams never reset - they advance forever, ensuring determinism via
/// stable call sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RngStream {
    state: u64,
}

impl RngStream {
    /// Create a new RNG stream from a seed.
    ///
    /// The seed should be derived from the world seed combined with a
    /// semantic label to ensure reproducibility.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        // Ensure non-zero state (SplitMix64 requirement)
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    /// Create a new RNG stream from a string label.
    ///
    /// Uses stable FNV-1a hashing to convert the label to a seed.
    #[inline]
    pub fn from_label(label: &str) -> Self {
        Self::new(fnv1a64_str(label))
    }

    /// Create a new RNG stream by combining a parent seed with a label.
    ///
    /// This is the primary way to create derived streams for primitives:
    /// ```ignore
    /// let stream = RngStream::derive(world_seed, "terra.plate.velocity");
    /// ```
    #[inline]
    pub fn derive(parent_seed: u64, label: &str) -> Self {
        let label_hash = fnv1a64_str(label);
        // Mix parent seed with label hash using SplitMix64-style mixing
        let mixed = splitmix64_mix(parent_seed ^ label_hash);
        Self::new(mixed)
    }

    /// Create a substream by mixing a label into the current state.
    ///
    /// This creates an independent stream derived from the current state
    /// without advancing the parent stream.
    #[inline]
    pub fn substream(&self, label: &str) -> Self {
        let label_hash = fnv1a64_str(label);
        self.substream_from_hash(label_hash)
    }

    /// Create a substream by mixing a pre-computed hash into the current state.
    ///
    /// Use this when the label hash has already been computed (e.g., at compile time).
    #[inline]
    pub fn substream_from_hash(&self, label_hash: u64) -> Self {
        let mixed = splitmix64_mix(self.state ^ label_hash);
        Self::new(mixed)
    }

    /// Create a substream for a specific entity by mixing in the entity index.
    ///
    /// This ensures each entity gets a deterministic but different stream.
    #[inline]
    pub fn for_entity(&self, entity_index: u64) -> Self {
        let mixed = splitmix64_mix(self.state ^ entity_index);
        Self::new(mixed)
    }

    /// Get the current internal state (for debugging/testing).
    #[inline]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Generate the next random u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64_next(self.state);
        splitmix64_mix(self.state)
    }

    /// Generate a uniform random f64 in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        u64_to_f64_01(self.next_u64())
    }

    /// Generate a uniform random f64 in [min, max).
    #[inline]
    pub fn uniform_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.uniform() * (max - min)
    }

    /// Generate a standard normal (Gaussian) random value using Box-Muller.
    ///
    /// Returns a value from N(0, 1).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.uniform();
        let u2 = self.uniform();
        // Avoid log(0) by ensuring u1 > 0
        let u1 = if u1 == 0.0 { f64::MIN_POSITIVE } else { u1 };
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Generate a normal random value with given mean and standard deviation.
    #[inline]
    pub fn normal_with(&mut self, mean: f64, stddev: f64) -> f64 {
        mean + self.normal() * stddev
    }

    /// Generate a random boolean with given probability of being true.
    #[inline]
    pub fn bool_with_prob(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }

    /// Generate a random integer in [min, max] (inclusive).
    #[inline]
    pub fn int_range(&mut self, min: i64, max: i64) -> i64 {
        if min >= max {
            return min;
        }
        let range = (max - min + 1) as u64;
        let random = self.next_u64();
        min + (random % range) as i64
    }

    /// Select an index based on weights (returns index of selected item).
    ///
    /// Weights do not need to sum to 1.
    #[inline]
    pub fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        if weights.is_empty() {
            return 0;
        }

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return 0;
        }

        let threshold = self.uniform() * total;
        let mut cumulative = 0.0;

        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if threshold < cumulative {
                return i;
            }
        }

        // Fallback to last index (handles floating point edge cases)
        weights.len() - 1
    }

    /// Generate a random unit vector on the 2D unit circle.
    #[inline]
    pub fn unit_vec2(&mut self) -> [f64; 2] {
        let angle = self.uniform() * 2.0 * PI;
        [angle.cos(), angle.sin()]
    }

    /// Generate a random unit vector on the 3D unit sphere.
    ///
    /// Uses the spherical coordinate method for uniform distribution.
    #[inline]
    pub fn unit_vec3(&mut self) -> [f64; 3] {
        // Use rejection sampling for better uniformity
        loop {
            let x = self.uniform_range(-1.0, 1.0);
            let y = self.uniform_range(-1.0, 1.0);
            let z = self.uniform_range(-1.0, 1.0);
            let len_sq = x * x + y * y + z * z;

            if len_sq > 0.0 && len_sq <= 1.0 {
                let len = len_sq.sqrt();
                return [x / len, y / len, z / len];
            }
        }
    }

    /// Generate a random unit quaternion (uniform rotation).
    ///
    /// Uses the subgroup algorithm for uniform distribution over SO(3).
    #[inline]
    pub fn unit_quat(&mut self) -> [f64; 4] {
        // Ken Shoemake's method for uniform random quaternions
        let u1 = self.uniform();
        let u2 = self.uniform();
        let u3 = self.uniform();

        let sqrt_u1 = u1.sqrt();
        let sqrt_1_minus_u1 = (1.0 - u1).sqrt();

        let theta1 = 2.0 * PI * u2;
        let theta2 = 2.0 * PI * u3;

        // [w, x, y, z] format
        [
            sqrt_1_minus_u1 * theta1.sin(),
            sqrt_1_minus_u1 * theta1.cos(),
            sqrt_u1 * theta2.sin(),
            sqrt_u1 * theta2.cos(),
        ]
    }

    /// Generate a random point inside the 2D unit disk.
    #[inline]
    pub fn in_disk(&mut self) -> [f64; 2] {
        // Rejection sampling
        loop {
            let x = self.uniform_range(-1.0, 1.0);
            let y = self.uniform_range(-1.0, 1.0);
            if x * x + y * y <= 1.0 {
                return [x, y];
            }
        }
    }

    /// Generate a random point inside the 3D unit sphere.
    #[inline]
    pub fn in_sphere(&mut self) -> [f64; 3] {
        // Rejection sampling
        loop {
            let x = self.uniform_range(-1.0, 1.0);
            let y = self.uniform_range(-1.0, 1.0);
            let z = self.uniform_range(-1.0, 1.0);
            if x * x + y * y + z * z <= 1.0 {
                return [x, y, z];
            }
        }
    }
}

/// SplitMix64 state transition function.
///
/// This is the core PRNG algorithm - fast and high quality.
#[inline]
const fn splitmix64_next(state: u64) -> u64 {
    state.wrapping_add(0x9E3779B97F4A7C15)
}

/// SplitMix64 mixing function for deriving new states.
#[inline]
const fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Convert a u64 to a uniform f64 in [0, 1).
///
/// Uses the upper 53 bits for full f64 precision.
#[inline]
const fn u64_to_f64_01(x: u64) -> f64 {
    // Use upper 53 bits (f64 mantissa precision)
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let stream1 = RngStream::new(12345);
        let stream2 = RngStream::new(12345);
        assert_eq!(stream1.state(), stream2.state());

        let stream3 = RngStream::from_label("terra.plate.velocity");
        let stream4 = RngStream::from_label("terra.plate.velocity");
        assert_eq!(stream3.state(), stream4.state());
    }

    #[test]
    fn test_stream_determinism() {
        let mut stream1 = RngStream::new(42);
        let mut stream2 = RngStream::new(42);

        for _ in 0..1000 {
            assert_eq!(stream1.next_u64(), stream2.next_u64());
        }
    }

    #[test]
    fn test_uniform_range() {
        let mut stream = RngStream::new(12345);

        for _ in 0..1000 {
            let val = stream.uniform();
            assert!(val >= 0.0 && val < 1.0);
        }

        for _ in 0..1000 {
            let val = stream.uniform_range(10.0, 20.0);
            assert!(val >= 10.0 && val < 20.0);
        }
    }

    #[test]
    fn test_normal_distribution() {
        let mut stream = RngStream::new(12345);
        let mut sum = 0.0;
        let n = 10000;

        for _ in 0..n {
            sum += stream.normal();
        }

        let mean = sum / n as f64;
        // Mean should be close to 0 for standard normal
        assert!(mean.abs() < 0.1, "Mean {} too far from 0", mean);
    }

    #[test]
    fn test_int_range() {
        let mut stream = RngStream::new(12345);

        for _ in 0..1000 {
            let val = stream.int_range(5, 10);
            assert!(val >= 5 && val <= 10);
        }
    }

    #[test]
    fn test_unit_vec3_is_normalized() {
        let mut stream = RngStream::new(12345);

        for _ in 0..100 {
            let [x, y, z] = stream.unit_vec3();
            let len = (x * x + y * y + z * z).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "unit_vec3 not normalized: len = {}",
                len
            );
        }
    }

    #[test]
    fn test_unit_quat_is_normalized() {
        let mut stream = RngStream::new(12345);

        for _ in 0..100 {
            let [w, x, y, z] = stream.unit_quat();
            let len = (w * w + x * x + y * y + z * z).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "unit_quat not normalized: len = {}",
                len
            );
        }
    }

    #[test]
    fn test_in_sphere() {
        let mut stream = RngStream::new(12345);

        for _ in 0..100 {
            let [x, y, z] = stream.in_sphere();
            let len_sq = x * x + y * y + z * z;
            assert!(len_sq <= 1.0, "Point outside unit sphere");
        }
    }

    #[test]
    fn test_derive_creates_different_streams() {
        let parent_seed = 12345u64;
        let stream1 = RngStream::derive(parent_seed, "stream_a");
        let stream2 = RngStream::derive(parent_seed, "stream_b");
        assert_ne!(stream1.state(), stream2.state());
    }

    #[test]
    fn test_substream_independent() {
        let mut parent = RngStream::new(12345);
        let child1 = parent.substream("child");

        // Advance parent
        parent.next_u64();
        parent.next_u64();

        // Create another child - should be different because parent advanced
        // But wait, substream uses current state, not original...
        // Actually substream doesn't advance parent, just uses current state
        let child2 = parent.substream("child");

        // Child2 should be different because parent state changed
        assert_ne!(child1.state(), child2.state());
    }

    #[test]
    fn test_for_entity_determinism() {
        let stream = RngStream::new(12345);

        let entity_stream_0a = stream.for_entity(0);
        let entity_stream_0b = stream.for_entity(0);
        assert_eq!(entity_stream_0a.state(), entity_stream_0b.state());

        let entity_stream_1 = stream.for_entity(1);
        assert_ne!(entity_stream_0a.state(), entity_stream_1.state());
    }

    #[test]
    fn test_weighted_choice() {
        let mut stream = RngStream::new(12345);
        let weights = [0.7, 0.2, 0.1]; // 70%, 20%, 10%

        let mut counts = [0u32; 3];
        let n = 10000;

        for _ in 0..n {
            let idx = stream.weighted_choice(&weights);
            counts[idx] += 1;
        }

        // Check rough proportions (with some tolerance)
        let p0 = counts[0] as f64 / n as f64;
        let p1 = counts[1] as f64 / n as f64;
        let p2 = counts[2] as f64 / n as f64;

        assert!(
            (p0 - 0.7).abs() < 0.05,
            "Expected ~70%, got {}%",
            p0 * 100.0
        );
        assert!(
            (p1 - 0.2).abs() < 0.05,
            "Expected ~20%, got {}%",
            p1 * 100.0
        );
        assert!(
            (p2 - 0.1).abs() < 0.05,
            "Expected ~10%, got {}%",
            p2 * 100.0
        );
    }

    #[test]
    fn test_bool_with_prob() {
        let mut stream = RngStream::new(12345);
        let mut count = 0;
        let n = 10000;

        for _ in 0..n {
            if stream.bool_with_prob(0.3) {
                count += 1;
            }
        }

        let ratio = count as f64 / n as f64;
        assert!(
            (ratio - 0.3).abs() < 0.05,
            "Expected ~30%, got {}%",
            ratio * 100.0
        );
    }

    /// Regression test: ensure specific seeds produce specific values.
    /// If this test fails, determinism has been broken!
    #[test]
    fn test_determinism_regression() {
        let mut stream = RngStream::new(0xDEADBEEF);

        // These values must never change - computed from SplitMix64 algorithm
        // If this test fails after code changes, determinism has been broken!
        assert_eq!(stream.next_u64(), 0x4ADFB90F68C9EB9B);
        assert_eq!(stream.next_u64(), 0xDE586A3141A10922);
        assert_eq!(stream.next_u64(), 0x021FBC2F8E1CFC1D);
    }
}
