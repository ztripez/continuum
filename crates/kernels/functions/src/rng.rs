//! RNG Kernel Functions
//!
//! Deterministic random number generation functions for the DSL.
//!
//! # Design Note
//!
//! These kernel functions take a seed (u64) and a "call index" as parameters.
//! The call index is used to derive unique values for each call within a
//! primitive's execution. The execution context manages the call index
//! automatically, incrementing it for each RNG function call.
//!
//! This design keeps kernel functions pure while allowing the execution
//! layer to manage state advancement.

use continuum_foundation::rng::RngStream;
use continuum_kernel_macros::kernel_fn;

// ============================================================================
// STREAM DERIVATION
// ============================================================================

/// Derive a new RNG seed from a parent seed and label.
///
/// This creates a new deterministic seed by mixing the parent seed with
/// a hash of the label. Used for creating independent sub-streams.
///
/// # DSL Usage
/// ```cdsl
/// let child_seed = rng.derive(parent_seed, "velocity")
/// ```
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn derive(parent_seed: i64, label_hash: i64) -> i64 {
    let stream = RngStream::new(parent_seed as u64);
    let child = stream.substream_from_hash(label_hash as u64);
    child.state() as i64
}

/// Derive an entity-specific seed from a base seed and entity index.
///
/// Each entity gets a unique but deterministic seed derived from the
/// base primitive seed.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn derive_entity(base_seed: i64, entity_index: i64) -> i64 {
    let stream = RngStream::new(base_seed as u64);
    let entity_stream = stream.for_entity(entity_index as u64);
    entity_stream.state() as i64
}

// ============================================================================
// UNIFORM DISTRIBUTION
// ============================================================================

/// Generate a uniform random value in [0, 1).
///
/// Takes a seed and call index to produce a deterministic result.
/// The execution context manages the call index automatically.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn uniform(seed: i64, call_index: i64) -> f64 {
    let mut stream = RngStream::new(seed as u64);
    // Advance stream by call_index to get unique value for this call
    for _ in 0..call_index {
        stream.next_u64();
    }
    stream.uniform()
}

/// Generate a uniform random value in [min, max).
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitAny, UnitSameAs(2)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(2)
)]
pub fn uniform_range(seed: i64, call_index: i64, min: f64, max: f64) -> f64 {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    stream.uniform_range(min, max)
}

// ============================================================================
// NORMAL DISTRIBUTION
// ============================================================================

/// Generate a standard normal (Gaussian) random value.
///
/// Returns a value from N(0, 1).
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn normal(seed: i64, call_index: i64) -> f64 {
    let mut stream = RngStream::new(seed as u64);
    // Box-Muller uses 2 uniform values, so multiply call_index by 2
    for _ in 0..(call_index * 2) {
        stream.next_u64();
    }
    stream.normal()
}

/// Generate a normal random value with given mean and standard deviation.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitAny, UnitSameAs(2)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(2)
)]
pub fn normal_with(seed: i64, call_index: i64, mean: f64, stddev: f64) -> f64 {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..(call_index * 2) {
        stream.next_u64();
    }
    stream.normal_with(mean, stddev)
}

// ============================================================================
// DISCRETE SAMPLING
// ============================================================================

/// Generate a random boolean with given probability of being true.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn bool_prob(seed: i64, call_index: i64, probability: f64) -> bool {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    stream.bool_with_prob(probability)
}

/// Generate a random integer in [min, max] (inclusive).
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn int_range(seed: i64, call_index: i64, min: i64, max: i64) -> i64 {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    stream.int_range(min, max)
}

// ============================================================================
// GEOMETRIC SAMPLING
// ============================================================================

/// Generate a random unit vector on the 2D unit circle.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = Dimensionless
)]
pub fn unit_vec2(seed: i64, call_index: i64) -> [f64; 2] {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    stream.unit_vec2()
}

/// Generate a random unit vector on the 3D unit sphere.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = Dimensionless
)]
pub fn unit_vec3(seed: i64, call_index: i64) -> [f64; 3] {
    let stream = RngStream::new(seed as u64);
    // Rejection sampling uses variable number of calls, so we use a different approach:
    // Create a unique stream for this call
    let call_stream = stream.for_entity(call_index as u64);
    let mut call_rng = RngStream::new(call_stream.state());
    call_rng.unit_vec3()
}

/// Generate a random unit quaternion (uniform rotation).
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = Dimensionless
)]
pub fn unit_quat(seed: i64, call_index: i64) -> [f64; 4] {
    let mut stream = RngStream::new(seed as u64);
    // Use 3 uniform values
    for _ in 0..(call_index * 3) {
        stream.next_u64();
    }
    stream.unit_quat()
}

/// Generate a random point inside the 2D unit disk.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = Dimensionless
)]
pub fn in_disk(seed: i64, call_index: i64) -> [f64; 2] {
    let stream = RngStream::new(seed as u64);
    let call_stream = stream.for_entity(call_index as u64);
    let mut call_rng = RngStream::new(call_stream.state());
    call_rng.in_disk()
}

/// Generate a random point inside the 3D unit sphere.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = Dimensionless
)]
pub fn in_sphere(seed: i64, call_index: i64) -> [f64; 3] {
    let stream = RngStream::new(seed as u64);
    let call_stream = stream.for_entity(call_index as u64);
    let mut call_rng = RngStream::new(call_stream.state());
    call_rng.in_sphere()
}

// ============================================================================
// WEIGHTED CHOICE (variadic version would need special handling)
// ============================================================================

/// Select between two options based on weights.
///
/// Returns 0 or 1 based on relative weights.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn weighted_choice_2(seed: i64, call_index: i64, weight0: f64, weight1: f64) -> i64 {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    let weights = [weight0, weight1];
    stream.weighted_choice(&weights) as i64
}

/// Select between three options based on weights.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn weighted_choice_3(
    seed: i64,
    call_index: i64,
    weight0: f64,
    weight1: f64,
    weight2: f64,
) -> i64 {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    let weights = [weight0, weight1, weight2];
    stream.weighted_choice(&weights) as i64
}

/// Select between four options based on weights.
#[kernel_fn(
    namespace = "rng",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn weighted_choice_4(
    seed: i64,
    call_index: i64,
    weight0: f64,
    weight1: f64,
    weight2: f64,
    weight3: f64,
) -> i64 {
    let mut stream = RngStream::new(seed as u64);
    for _ in 0..call_index {
        stream.next_u64();
    }
    let weights = [weight0, weight1, weight2, weight3];
    stream.weighted_choice(&weights) as i64
}

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{Value, eval_in_namespace, is_known_in};

    #[test]
    fn test_rng_functions_registered() {
        assert!(is_known_in("rng", "derive"));
        assert!(is_known_in("rng", "derive_entity"));
        assert!(is_known_in("rng", "uniform"));
        assert!(is_known_in("rng", "uniform_range"));
        assert!(is_known_in("rng", "normal"));
        assert!(is_known_in("rng", "normal_with"));
        assert!(is_known_in("rng", "bool_prob"));
        assert!(is_known_in("rng", "int_range"));
        assert!(is_known_in("rng", "unit_vec2"));
        assert!(is_known_in("rng", "unit_vec3"));
        assert!(is_known_in("rng", "unit_quat"));
        assert!(is_known_in("rng", "in_disk"));
        assert!(is_known_in("rng", "in_sphere"));
        assert!(is_known_in("rng", "weighted_choice_2"));
        assert!(is_known_in("rng", "weighted_choice_3"));
        assert!(is_known_in("rng", "weighted_choice_4"));
    }

    #[test]
    fn test_uniform_determinism() {
        let seed = 12345i64;
        let call_idx = 0i64;

        let result1 = eval_in_namespace(
            "rng",
            "uniform",
            &[Value::Integer(seed), Value::Integer(call_idx)],
            0.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();

        let result2 = eval_in_namespace(
            "rng",
            "uniform",
            &[Value::Integer(seed), Value::Integer(call_idx)],
            0.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();

        // Same seed + call_idx = same result
        assert_eq!(result1, result2);

        // Different call_idx = different result
        let result3 = eval_in_namespace(
            "rng",
            "uniform",
            &[Value::Integer(seed), Value::Integer(1)],
            0.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();

        assert_ne!(result1, result3);
    }

    #[test]
    fn test_uniform_range() {
        let seed = 12345i64;

        for call_idx in 0..100 {
            let result = eval_in_namespace(
                "rng",
                "uniform_range",
                &[
                    Value::Integer(seed),
                    Value::Integer(call_idx),
                    Value::Scalar(10.0),
                    Value::Scalar(20.0),
                ],
                0.0,
            )
            .unwrap()
            .as_scalar()
            .unwrap();

            assert!(
                result >= 10.0 && result < 20.0,
                "Value {} out of range",
                result
            );
        }
    }

    #[test]
    fn test_unit_vec3_normalized() {
        let seed = 12345i64;

        for call_idx in 0..20 {
            let result = eval_in_namespace(
                "rng",
                "unit_vec3",
                &[Value::Integer(seed), Value::Integer(call_idx)],
                0.0,
            )
            .unwrap();

            if let Value::Vec3([x, y, z]) = result {
                let len = (x * x + y * y + z * z).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-10,
                    "unit_vec3 not normalized: len = {}",
                    len
                );
            } else {
                panic!("Expected Vec3");
            }
        }
    }

    #[test]
    fn test_int_range() {
        let seed = 12345i64;

        for call_idx in 0..100 {
            let result = eval_in_namespace(
                "rng",
                "int_range",
                &[
                    Value::Integer(seed),
                    Value::Integer(call_idx),
                    Value::Integer(5),
                    Value::Integer(10),
                ],
                0.0,
            )
            .unwrap()
            .as_int()
            .unwrap();

            assert!(result >= 5 && result <= 10, "Value {} out of range", result);
        }
    }

    #[test]
    fn test_derive_creates_different_seeds() {
        let parent = 12345i64;

        let child1 = eval_in_namespace(
            "rng",
            "derive",
            &[Value::Integer(parent), Value::Integer(111)],
            0.0,
        )
        .unwrap()
        .as_int()
        .unwrap();

        let child2 = eval_in_namespace(
            "rng",
            "derive",
            &[Value::Integer(parent), Value::Integer(222)],
            0.0,
        )
        .unwrap()
        .as_int()
        .unwrap();

        assert_ne!(child1, child2);
    }

    #[test]
    fn test_derive_entity_determinism() {
        let base_seed = 12345i64;

        let entity_0a = eval_in_namespace(
            "rng",
            "derive_entity",
            &[Value::Integer(base_seed), Value::Integer(0)],
            0.0,
        )
        .unwrap()
        .as_int()
        .unwrap();

        let entity_0b = eval_in_namespace(
            "rng",
            "derive_entity",
            &[Value::Integer(base_seed), Value::Integer(0)],
            0.0,
        )
        .unwrap()
        .as_int()
        .unwrap();

        // Same entity = same seed
        assert_eq!(entity_0a, entity_0b);

        let entity_1 = eval_in_namespace(
            "rng",
            "derive_entity",
            &[Value::Integer(base_seed), Value::Integer(1)],
            0.0,
        )
        .unwrap()
        .as_int()
        .unwrap();

        // Different entity = different seed
        assert_ne!(entity_0a, entity_1);
    }
}
