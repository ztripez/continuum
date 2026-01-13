//! Vector Constructors
//!
//! Functions for constructing vector types.

use continuum_foundation::Value;
use continuum_kernel_macros::kernel_fn;

/// Construct a 2D vector: `vec2(x, y)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn vec2(x: f64, y: f64) -> [f64; 2] {
    [x, y]
}

/// Construct a 3D vector: `vec3(x, y, z)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn vec3(x: f64, y: f64, z: f64) -> [f64; 3] {
    [x, y, z]
}

/// Vector length/magnitude: `length(x, y, z)` or `length(vec)`
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn length(args: &[Value]) -> f64 {
    if args.len() == 1 {
        // Single argument: expect vector type
        match &args[0] {
            Value::Vec2(v) => (v[0] * v[0] + v[1] * v[1]).sqrt(),
            Value::Vec3(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
            Value::Vec4(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt(),
            Value::Scalar(s) => s.abs(),
            _ => 0.0,
        }
    } else {
        // Multiple arguments: treat as components
        let mut sum_sq = 0.0;
        for arg in args {
            if let Some(v) = arg.as_scalar() {
                sum_sq += v * v;
            }
        }
        sum_sq.sqrt()
    }
}

/// Normalize a vector: `normalize(vec)` or `normalize(x, y, z)`
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn normalize(args: &[Value]) -> Value {
    if args.len() == 1 {
        match &args[0] {
            Value::Vec2(v) => {
                let mag = (v[0] * v[0] + v[1] * v[1]).sqrt();
                if mag > 0.0 {
                    Value::Vec2([v[0] / mag, v[1] / mag])
                } else {
                    Value::Vec2([0.0, 0.0])
                }
            }
            Value::Vec3(v) => {
                let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if mag > 0.0 {
                    Value::Vec3([v[0] / mag, v[1] / mag, v[2] / mag])
                } else {
                    Value::Vec3([0.0, 0.0, 0.0])
                }
            }
            Value::Vec4(v) => {
                let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                if mag > 0.0 {
                    Value::Vec4([v[0] / mag, v[1] / mag, v[2] / mag, v[3] / mag])
                } else {
                    Value::Vec4([0.0, 0.0, 0.0, 0.0])
                }
            }
            Value::Scalar(s) => Value::Scalar(s.signum()),
            v => v.clone(),
        }
    } else {
        // Multiple scalar args -> return normalized vector
        // Infer dimension from arg count
        let mut components = Vec::new();
        let mut sum_sq = 0.0;
        for arg in args {
            if let Some(v) = arg.as_scalar() {
                components.push(v);
                sum_sq += v * v;
            }
        }

        let mag = sum_sq.sqrt();
        if mag > 0.0 {
            for c in &mut components {
                *c /= mag;
            }
        }

        match components.len() {
            2 => Value::Vec2([components[0], components[1]]),
            3 => Value::Vec3([components[0], components[1], components[2]]),
            4 => Value::Vec4([components[0], components[1], components[2], components[3]]),
            _ => Value::Scalar(0.0), // Error or fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_kernel_registry::{Arity, Value, get_in_namespace, is_known_in};

    #[test]
    fn test_vec2_registered() {
        assert!(is_known_in("vector", "vec2"));
        let desc = get_in_namespace("vector", "vec2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_vec3_registered() {
        assert!(is_known_in("vector", "vec3"));
        let desc = get_in_namespace("vector", "vec3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
    }

    #[test]
    fn test_length_variadic() {
        let args = [Value::Scalar(3.0), Value::Scalar(4.0)];
        let res = length(&args);
        assert_eq!(res, 5.0);
    }

    #[test]
    fn test_length_vector() {
        let args = [Value::Vec3([1.0, 2.0, 2.0])];
        let res = length(&args);
        assert_eq!(res, 3.0);
    }
}
