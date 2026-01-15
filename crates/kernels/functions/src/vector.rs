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

/// Construct a 4D vector: `vec4(x, y, z, w)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn vec4(x: f64, y: f64, z: f64, w: f64) -> [f64; 4] {
    [x, y, z, w]
}

/// Construct a vector from 2-4 scalar components or pass through a vector.
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn vec(args: &[Value]) -> Value {
    if args.len() == 1 {
        return match &args[0] {
            Value::Vec2(v) => Value::Vec2(*v),
            Value::Vec3(v) => Value::Vec3(*v),
            Value::Vec4(v) => Value::Vec4(*v),
            _ => panic!("vector.vec expects a vector or 2-4 scalar components"),
        };
    }

    let mut components = Vec::new();
    for arg in args {
        if let Some(v) = arg.as_scalar() {
            components.push(v);
        } else {
            panic!("vector.vec expects scalar components");
        }
    }

    match components.len() {
        2 => Value::Vec2([components[0], components[1]]),
        3 => Value::Vec3([components[0], components[1], components[2]]),
        4 => Value::Vec4([components[0], components[1], components[2], components[3]]),
        _ => panic!("vector.vec expects 2-4 scalar components"),
    }
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

/// Dot product: `dot(a, b)` -> Scalar
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn dot(args: &[Value]) -> f64 {
    if args.len() != 2 {
        panic!("vector.dot expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Vec2(a), Value::Vec2(b)) => a[0] * b[0] + a[1] * b[1],
        (Value::Vec3(a), Value::Vec3(b)) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        (Value::Vec4(a), Value::Vec4(b)) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
        _ => panic!("vector.dot requires two vectors of same dimension"),
    }
}

/// Cross product (Vec3 only): `cross(a, b)` -> Vec3
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Reflect vector v around normal n: `reflect(v, n)`
/// Normal n is assumed to be normalized (unit vector)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn reflect(v: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let d = 2.0 * (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
}

/// Project a onto b: `project(a, onto)`
/// Returns the projection of vector a onto vector b
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn project(a: [f64; 3], onto: [f64; 3]) -> [f64; 3] {
    let dot_ab = a[0] * onto[0] + a[1] * onto[1] + a[2] * onto[2];
    let dot_bb = onto[0] * onto[0] + onto[1] * onto[1] + onto[2] * onto[2];
    if dot_bb < 1e-10 {
        panic!("vector.project: cannot project onto zero vector");
    }
    let scale = dot_ab / dot_bb;
    [onto[0] * scale, onto[1] * scale, onto[2] * scale]
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
    fn test_vec4_registered() {
        assert!(is_known_in("vector", "vec4"));
        let desc = get_in_namespace("vector", "vec4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(4));
    }

    #[test]
    fn test_vec_variadic_registered() {
        assert!(is_known_in("vector", "vec"));
        let desc = get_in_namespace("vector", "vec").unwrap();
        assert_eq!(desc.arity, Arity::Variadic);
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

    #[test]
    fn test_vec_variadic_value() {
        let args = [Value::Scalar(1.0), Value::Scalar(2.0), Value::Scalar(3.0)];
        let res = vec(&args);
        assert_eq!(res, Value::Vec3([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_dot_vec2() {
        let a = Value::Vec2([1.0, 2.0]);
        let b = Value::Vec2([3.0, 4.0]);
        let result = dot(&[a, b]);
        assert_eq!(result, 11.0); // 1*3 + 2*4 = 3 + 8 = 11
    }

    #[test]
    fn test_dot_vec3() {
        let a = Value::Vec3([1.0, 0.0, 0.0]);
        let b = Value::Vec3([0.0, 1.0, 0.0]);
        let result = dot(&[a, b]);
        assert_eq!(result, 0.0); // Perpendicular vectors
    }

    #[test]
    fn test_dot_vec3_parallel() {
        let a = Value::Vec3([2.0, 0.0, 0.0]);
        let b = Value::Vec3([3.0, 0.0, 0.0]);
        let result = dot(&[a, b]);
        assert_eq!(result, 6.0); // Parallel vectors
    }

    #[test]
    fn test_dot_vec4() {
        let a = Value::Vec4([1.0, 2.0, 3.0, 4.0]);
        let b = Value::Vec4([1.0, 1.0, 1.0, 1.0]);
        let result = dot(&[a, b]);
        assert_eq!(result, 10.0); // 1+2+3+4
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_dot_dimension_mismatch() {
        let a = Value::Vec2([1.0, 2.0]);
        let b = Value::Vec3([1.0, 2.0, 3.0]);
        let _ = dot(&[a, b]);
    }

    #[test]
    fn test_dot_registered() {
        assert!(is_known_in("vector", "dot"));
    }

    #[test]
    fn test_cross_basis_vectors() {
        // i × j = k
        let i = [1.0, 0.0, 0.0];
        let j = [0.0, 1.0, 0.0];
        let k = cross(i, j);
        assert_eq!(k, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cross_reverse() {
        // j × i = -k
        let i = [1.0, 0.0, 0.0];
        let j = [0.0, 1.0, 0.0];
        let result = cross(j, i);
        assert_eq!(result, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn test_cross_parallel() {
        // Parallel vectors have zero cross product
        let a = [2.0, 0.0, 0.0];
        let b = [4.0, 0.0, 0.0];
        let result = cross(a, b);
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cross_general() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = cross(a, b);
        // a × b = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
        assert_eq!(result, [-3.0, 6.0, -3.0]);
    }

    #[test]
    fn test_cross_registered() {
        assert!(is_known_in("vector", "cross"));
    }

    #[test]
    fn test_reflect_horizontal() {
        // Reflect downward vector off horizontal surface (normal points up)
        let v = [0.0, -1.0, 0.0]; // Going down
        let n = [0.0, 1.0, 0.0]; // Normal pointing up
        let result = reflect(v, n);
        assert_eq!(result, [0.0, 1.0, 0.0]); // Bounces up
    }

    #[test]
    fn test_reflect_45_degrees() {
        // Reflect at 45 degrees
        let v = [1.0, -1.0, 0.0]; // Down and right
        let n = [0.0, 1.0, 0.0]; // Normal pointing up
        let result = reflect(v, n);
        assert_eq!(result, [1.0, 1.0, 0.0]); // Up and right
    }

    #[test]
    fn test_reflect_parallel() {
        // Reflecting along the normal reverses direction
        let v = [0.0, 1.0, 0.0];
        let n = [0.0, 1.0, 0.0];
        let result = reflect(v, n);
        assert_eq!(result, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_project_onto_x() {
        // Project [3,4,0] onto x-axis
        let a = [3.0, 4.0, 0.0];
        let x_axis = [1.0, 0.0, 0.0];
        let result = project(a, x_axis);
        assert_eq!(result, [3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_project_general() {
        // Project [1,2,3] onto [1,1,0]
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 1.0, 0.0];
        let result = project(a, b);
        // dot(a,b) = 1+2 = 3, dot(b,b) = 1+1 = 2
        // proj = (3/2) * [1,1,0] = [1.5, 1.5, 0]
        assert_eq!(result, [1.5, 1.5, 0.0]);
    }

    #[test]
    fn test_project_parallel() {
        // Projecting a vector onto itself
        let a = [3.0, 4.0, 0.0];
        let result = project(a, a);
        assert_eq!(result, a);
    }

    #[test]
    #[should_panic(expected = "zero vector")]
    fn test_project_onto_zero() {
        let a = [1.0, 2.0, 3.0];
        let zero = [0.0, 0.0, 0.0];
        let _ = project(a, zero);
    }

    #[test]
    fn test_reflect_registered() {
        assert!(is_known_in("vector", "reflect"));
    }

    #[test]
    fn test_project_registered() {
        assert!(is_known_in("vector", "project"));
    }
}
