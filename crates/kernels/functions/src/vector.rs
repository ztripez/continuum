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

/// Distance between two 2D vectors: `distance(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn distance_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Distance between two 3D vectors: `distance(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn distance_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Distance between two 4D vectors: `distance(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn distance_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let dw = a[3] - b[3];
    (dx * dx + dy * dy + dz * dz + dw * dw).sqrt()
}

/// Squared distance between two 2D vectors: `distance_sq(a, b)` - cheaper than distance
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn distance_sq_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Squared distance between two 3D vectors: `distance_sq(a, b)` - cheaper than distance
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn distance_sq_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Squared distance between two 4D vectors: `distance_sq(a, b)` - cheaper than distance
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn distance_sq_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let dw = a[3] - b[3];
    dx * dx + dy * dy + dz * dz + dw * dw
}

/// Distance between two vectors: `distance(a, b)` (variadic)
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn distance(args: &[Value]) -> f64 {
    if args.len() != 2 {
        panic!("vector.distance expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Vec2(a), Value::Vec2(b)) => distance_vec2(*a, *b),
        (Value::Vec3(a), Value::Vec3(b)) => distance_vec3(*a, *b),
        (Value::Vec4(a), Value::Vec4(b)) => distance_vec4(*a, *b),
        _ => panic!("vector.distance requires two vectors of same dimension"),
    }
}

/// Squared distance between two vectors: `distance_sq(a, b)` (variadic) - cheaper than distance
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn distance_sq(args: &[Value]) -> f64 {
    if args.len() != 2 {
        panic!("vector.distance_sq expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Vec2(a), Value::Vec2(b)) => distance_sq_vec2(*a, *b),
        (Value::Vec3(a), Value::Vec3(b)) => distance_sq_vec3(*a, *b),
        (Value::Vec4(a), Value::Vec4(b)) => distance_sq_vec4(*a, *b),
        _ => panic!("vector.distance_sq requires two vectors of same dimension"),
    }
}

/// Linear interpolation for Vec2: `lerp(a, b, t)` → a + t * (b - a)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn lerp_vec2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// Linear interpolation for Vec3: `lerp(a, b, t)` → a + t * (b - a)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn lerp_vec3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Linear interpolation for Vec4: `lerp(a, b, t)` → a + t * (b - a)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn lerp_vec4(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
        a[3] + t * (b[3] - a[3]),
    ]
}

/// Linear interpolation: `lerp(a, b, t)` → a + t * (b - a) (variadic)
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn lerp(args: &[Value]) -> Value {
    if args.len() != 3 {
        panic!("vector.lerp expects exactly 3 arguments");
    }
    let t = match &args[2] {
        Value::Scalar(s) => *s,
        _ => panic!("vector.lerp: third argument must be a scalar"),
    };
    match (&args[0], &args[1]) {
        (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2(lerp_vec2(*a, *b, t)),
        (Value::Vec3(a), Value::Vec3(b)) => Value::Vec3(lerp_vec3(*a, *b, t)),
        (Value::Vec4(a), Value::Vec4(b)) => Value::Vec4(lerp_vec4(*a, *b, t)),
        _ => panic!("vector.lerp requires two vectors of same dimension"),
    }
}

/// Mix for Vec2 (GLSL alias for lerp): `mix(a, b, t)` → a + t * (b - a)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn mix_vec2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    lerp_vec2(a, b, t)
}

/// Mix for Vec3 (GLSL alias for lerp): `mix(a, b, t)` → a + t * (b - a)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn mix_vec3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    lerp_vec3(a, b, t)
}

/// Mix for Vec4 (GLSL alias for lerp): `mix(a, b, t)` → a + t * (b - a)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn mix_vec4(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    lerp_vec4(a, b, t)
}

/// Mix (GLSL alias for lerp): `mix(a, b, t)` → a + t * (b - a) (variadic)
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn mix(args: &[Value]) -> Value {
    lerp(args)
}

/// Component-wise clamp for Vec2: `clamp(v, min, max)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn clamp_vec2(v: [f64; 2], min: f64, max: f64) -> [f64; 2] {
    [v[0].clamp(min, max), v[1].clamp(min, max)]
}

/// Component-wise clamp for Vec3: `clamp(v, min, max)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn clamp_vec3(v: [f64; 3], min: f64, max: f64) -> [f64; 3] {
    [
        v[0].clamp(min, max),
        v[1].clamp(min, max),
        v[2].clamp(min, max),
    ]
}

/// Component-wise clamp for Vec4: `clamp(v, min, max)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn clamp_vec4(v: [f64; 4], min: f64, max: f64) -> [f64; 4] {
    [
        v[0].clamp(min, max),
        v[1].clamp(min, max),
        v[2].clamp(min, max),
        v[3].clamp(min, max),
    ]
}

/// Component-wise minimum for Vec2: `min(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn min_vec2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0].min(b[0]), a[1].min(b[1])]
}

/// Component-wise minimum for Vec3: `min(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn min_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

/// Component-wise minimum for Vec4: `min(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn min_vec4(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// Component-wise maximum for Vec2: `max(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn max_vec2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0].max(b[0]), a[1].max(b[1])]
}

/// Component-wise maximum for Vec3: `max(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn max_vec3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

/// Component-wise maximum for Vec4: `max(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn max_vec4(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// Component-wise absolute value for Vec2: `abs(v)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn abs_vec2(v: [f64; 2]) -> [f64; 2] {
    [v[0].abs(), v[1].abs()]
}

/// Component-wise absolute value for Vec3: `abs(v)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn abs_vec3(v: [f64; 3]) -> [f64; 3] {
    [v[0].abs(), v[1].abs(), v[2].abs()]
}

/// Component-wise absolute value for Vec4: `abs(v)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn abs_vec4(v: [f64; 4]) -> [f64; 4] {
    [v[0].abs(), v[1].abs(), v[2].abs(), v[3].abs()]
}

// Helper functions for internal use (not exposed via kernel_fn)
fn dot_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length_vec3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Angle between two vectors in radians: `angle(a, b)`
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = dot_vec3(a, b);
    let la = length_vec3(a);
    let lb = length_vec3(b);
    if la < 1e-10 || lb < 1e-10 {
        return 0.0; // Degenerate case: angle with zero vector
    }
    (d / (la * lb)).clamp(-1.0, 1.0).acos()
}

/// Refraction vector: `refract(I, N, eta)`
/// I: incident vector, N: normal, eta: refraction index ratio (eta1/eta2)
/// Returns refraction vector or zero vector for total internal reflection
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn refract(i: [f64; 3], n: [f64; 3], eta: f64) -> [f64; 3] {
    let cosi = -dot_vec3(n, i);
    let k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if k < 0.0 {
        // Total internal reflection
        [0.0, 0.0, 0.0]
    } else {
        let factor = eta * cosi - k.sqrt();
        [
            eta * i[0] + factor * n[0],
            eta * i[1] + factor * n[1],
            eta * i[2] + factor * n[2],
        ]
    }
}

/// Orient normal to face toward viewer: `faceforward(N, I, Nref)`
/// Returns N if dot(Nref, I) < 0, otherwise -N
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn faceforward(n: [f64; 3], i: [f64; 3], nref: [f64; 3]) -> [f64; 3] {
    if dot_vec3(nref, i) < 0.0 {
        n
    } else {
        [-n[0], -n[1], -n[2]]
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

    #[test]
    fn test_distance_vec2() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let result = distance_vec2(a, b);
        assert_eq!(result, 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_distance_vec3() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 2.0];
        let result = distance_vec3(a, b);
        assert_eq!(result, 3.0); // sqrt(1 + 4 + 4)
    }

    #[test]
    fn test_distance_vec4() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [2.0, 2.0, 1.0, 0.0];
        let result = distance_vec4(a, b);
        assert_eq!(result, 3.0); // sqrt(4 + 4 + 1)
    }

    #[test]
    fn test_distance_sq_vec2() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let result = distance_sq_vec2(a, b);
        assert_eq!(result, 25.0); // 9 + 16
    }

    #[test]
    fn test_distance_sq_vec3() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let result = distance_sq_vec3(a, b);
        assert_eq!(result, 25.0); // (3^2 + 4^2 + 0^2) = 9 + 16
    }

    #[test]
    fn test_distance_sq_vec4() {
        let a = [1.0, 1.0, 1.0, 1.0];
        let b = [2.0, 2.0, 2.0, 2.0];
        let result = distance_sq_vec4(a, b);
        assert_eq!(result, 4.0); // 4 * 1^2
    }

    #[test]
    fn test_distance_variadic_vec2() {
        let a = Value::Vec2([0.0, 0.0]);
        let b = Value::Vec2([3.0, 4.0]);
        let result = distance(&[a, b]);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_distance_variadic_vec3() {
        let a = Value::Vec3([0.0, 0.0, 0.0]);
        let b = Value::Vec3([1.0, 2.0, 2.0]);
        let result = distance(&[a, b]);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_distance_variadic_vec4() {
        let a = Value::Vec4([0.0, 0.0, 0.0, 0.0]);
        let b = Value::Vec4([2.0, 2.0, 1.0, 0.0]);
        let result = distance(&[a, b]);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_distance_sq_variadic_vec2() {
        let a = Value::Vec2([0.0, 0.0]);
        let b = Value::Vec2([3.0, 4.0]);
        let result = distance_sq(&[a, b]);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_distance_sq_variadic_vec3() {
        let a = Value::Vec3([1.0, 2.0, 3.0]);
        let b = Value::Vec3([4.0, 6.0, 3.0]);
        let result = distance_sq(&[a, b]);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_distance_sq_variadic_vec4() {
        let a = Value::Vec4([1.0, 1.0, 1.0, 1.0]);
        let b = Value::Vec4([2.0, 2.0, 2.0, 2.0]);
        let result = distance_sq(&[a, b]);
        assert_eq!(result, 4.0);
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_distance_dimension_mismatch() {
        let a = Value::Vec2([1.0, 2.0]);
        let b = Value::Vec3([1.0, 2.0, 3.0]);
        let _ = distance(&[a, b]);
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_distance_sq_dimension_mismatch() {
        let a = Value::Vec2([1.0, 2.0]);
        let b = Value::Vec3([1.0, 2.0, 3.0]);
        let _ = distance_sq(&[a, b]);
    }

    #[test]
    fn test_distance_registered() {
        assert!(is_known_in("vector", "distance"));
    }

    #[test]
    fn test_distance_sq_registered() {
        assert!(is_known_in("vector", "distance_sq"));
    }

    #[test]
    fn test_distance_vec2_registered() {
        assert!(is_known_in("vector", "distance_vec2"));
    }

    #[test]
    fn test_distance_vec3_registered() {
        assert!(is_known_in("vector", "distance_vec3"));
    }

    #[test]
    fn test_distance_vec4_registered() {
        assert!(is_known_in("vector", "distance_vec4"));
    }

    #[test]
    fn test_distance_sq_vec2_registered() {
        assert!(is_known_in("vector", "distance_sq_vec2"));
    }

    #[test]
    fn test_distance_sq_vec3_registered() {
        assert!(is_known_in("vector", "distance_sq_vec3"));
    }

    #[test]
    fn test_distance_sq_vec4_registered() {
        assert!(is_known_in("vector", "distance_sq_vec4"));
    }

    #[test]
    fn test_lerp_vec2_registered() {
        assert!(is_known_in("vector", "lerp_vec2"));
    }

    #[test]
    fn test_lerp_vec3_registered() {
        assert!(is_known_in("vector", "lerp_vec3"));
    }

    #[test]
    fn test_lerp_vec4_registered() {
        assert!(is_known_in("vector", "lerp_vec4"));
    }

    #[test]
    fn test_lerp_variadic_registered() {
        assert!(is_known_in("vector", "lerp"));
    }

    #[test]
    fn test_mix_vec2_registered() {
        assert!(is_known_in("vector", "mix_vec2"));
    }

    #[test]
    fn test_mix_vec3_registered() {
        assert!(is_known_in("vector", "mix_vec3"));
    }

    #[test]
    fn test_mix_vec4_registered() {
        assert!(is_known_in("vector", "mix_vec4"));
    }

    #[test]
    fn test_mix_variadic_registered() {
        assert!(is_known_in("vector", "mix"));
    }

    #[test]
    fn test_lerp_vec2_at_t0() {
        let a = [1.0, 2.0];
        let b = [5.0, 10.0];
        let result = lerp_vec2(a, b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_lerp_vec2_at_t1() {
        let a = [1.0, 2.0];
        let b = [5.0, 10.0];
        let result = lerp_vec2(a, b, 1.0);
        assert_eq!(result, b);
    }

    #[test]
    fn test_lerp_vec2_at_midpoint() {
        let a = [0.0, 0.0];
        let b = [10.0, 20.0];
        let result = lerp_vec2(a, b, 0.5);
        assert_eq!(result, [5.0, 10.0]);
    }

    #[test]
    fn test_lerp_vec3_at_t0() {
        let a = [1.0, 2.0, 3.0];
        let b = [5.0, 10.0, 15.0];
        let result = lerp_vec3(a, b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_lerp_vec3_at_t1() {
        let a = [1.0, 2.0, 3.0];
        let b = [5.0, 10.0, 15.0];
        let result = lerp_vec3(a, b, 1.0);
        assert_eq!(result, b);
    }

    #[test]
    fn test_lerp_vec3_at_midpoint() {
        let a = [0.0, 0.0, 0.0];
        let b = [10.0, 20.0, 30.0];
        let result = lerp_vec3(a, b, 0.5);
        assert_eq!(result, [5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_lerp_vec4_at_t0() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 10.0, 15.0, 20.0];
        let result = lerp_vec4(a, b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_lerp_vec4_at_t1() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 10.0, 15.0, 20.0];
        let result = lerp_vec4(a, b, 1.0);
        assert_eq!(result, b);
    }

    #[test]
    fn test_lerp_vec4_at_midpoint() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let result = lerp_vec4(a, b, 0.5);
        assert_eq!(result, [5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_lerp_variadic_vec2() {
        let a = Value::Vec2([0.0, 0.0]);
        let b = Value::Vec2([10.0, 20.0]);
        let t = Value::Scalar(0.5);
        let result = lerp(&[a, b, t]);
        assert_eq!(result, Value::Vec2([5.0, 10.0]));
    }

    #[test]
    fn test_lerp_variadic_vec3() {
        let a = Value::Vec3([0.0, 0.0, 0.0]);
        let b = Value::Vec3([10.0, 20.0, 30.0]);
        let t = Value::Scalar(0.5);
        let result = lerp(&[a, b, t]);
        assert_eq!(result, Value::Vec3([5.0, 10.0, 15.0]));
    }

    #[test]
    fn test_lerp_variadic_vec4() {
        let a = Value::Vec4([0.0, 0.0, 0.0, 0.0]);
        let b = Value::Vec4([10.0, 20.0, 30.0, 40.0]);
        let t = Value::Scalar(0.5);
        let result = lerp(&[a, b, t]);
        assert_eq!(result, Value::Vec4([5.0, 10.0, 15.0, 20.0]));
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_lerp_dimension_mismatch() {
        let a = Value::Vec2([1.0, 2.0]);
        let b = Value::Vec3([1.0, 2.0, 3.0]);
        let t = Value::Scalar(0.5);
        let _ = lerp(&[a, b, t]);
    }

    #[test]
    fn test_mix_vec2() {
        let a = [0.0, 0.0];
        let b = [10.0, 20.0];
        let result = mix_vec2(a, b, 0.5);
        assert_eq!(result, [5.0, 10.0]);
    }

    #[test]
    fn test_mix_vec3() {
        let a = [0.0, 0.0, 0.0];
        let b = [10.0, 20.0, 30.0];
        let result = mix_vec3(a, b, 0.5);
        assert_eq!(result, [5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_mix_vec4() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let result = mix_vec4(a, b, 0.5);
        assert_eq!(result, [5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_mix_variadic_vec3() {
        let a = Value::Vec3([0.0, 0.0, 0.0]);
        let b = Value::Vec3([10.0, 20.0, 30.0]);
        let t = Value::Scalar(0.5);
        let result = mix(&[a, b, t]);
        assert_eq!(result, Value::Vec3([5.0, 10.0, 15.0]));
    }

    // Component-wise clamp tests
    #[test]
    fn test_clamp_vec2() {
        let v = [-1.0, 5.0];
        let result = clamp_vec2(v, 0.0, 1.0);
        assert_eq!(result, [0.0, 1.0]);
    }

    #[test]
    fn test_clamp_vec3() {
        let v = [-1.0, 0.5, 5.0];
        let result = clamp_vec3(v, 0.0, 1.0);
        assert_eq!(result, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_clamp_vec4() {
        let v = [-2.0, -1.0, 0.5, 10.0];
        let result = clamp_vec4(v, 0.0, 1.0);
        assert_eq!(result, [0.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_clamp_vec2_registered() {
        assert!(is_known_in("vector", "clamp_vec2"));
    }

    #[test]
    fn test_clamp_vec3_registered() {
        assert!(is_known_in("vector", "clamp_vec3"));
    }

    #[test]
    fn test_clamp_vec4_registered() {
        assert!(is_known_in("vector", "clamp_vec4"));
    }

    // Component-wise min tests
    #[test]
    fn test_min_vec2() {
        let a = [1.0, 4.0];
        let b = [2.0, 3.0];
        let result = min_vec2(a, b);
        assert_eq!(result, [1.0, 3.0]);
    }

    #[test]
    fn test_min_vec3() {
        let a = [1.0, 4.0, 2.0];
        let b = [2.0, 3.0, 5.0];
        let result = min_vec3(a, b);
        assert_eq!(result, [1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_min_vec4() {
        let a = [1.0, 4.0, 2.0, 6.0];
        let b = [2.0, 3.0, 5.0, 1.0];
        let result = min_vec4(a, b);
        assert_eq!(result, [1.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_min_vec2_registered() {
        assert!(is_known_in("vector", "min_vec2"));
    }

    #[test]
    fn test_min_vec3_registered() {
        assert!(is_known_in("vector", "min_vec3"));
    }

    #[test]
    fn test_min_vec4_registered() {
        assert!(is_known_in("vector", "min_vec4"));
    }

    // Component-wise max tests
    #[test]
    fn test_max_vec2() {
        let a = [1.0, 4.0];
        let b = [2.0, 3.0];
        let result = max_vec2(a, b);
        assert_eq!(result, [2.0, 4.0]);
    }

    #[test]
    fn test_max_vec3() {
        let a = [1.0, 4.0, 2.0];
        let b = [2.0, 3.0, 5.0];
        let result = max_vec3(a, b);
        assert_eq!(result, [2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_max_vec4() {
        let a = [1.0, 4.0, 2.0, 6.0];
        let b = [2.0, 3.0, 5.0, 1.0];
        let result = max_vec4(a, b);
        assert_eq!(result, [2.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_max_vec2_registered() {
        assert!(is_known_in("vector", "max_vec2"));
    }

    #[test]
    fn test_max_vec3_registered() {
        assert!(is_known_in("vector", "max_vec3"));
    }

    #[test]
    fn test_max_vec4_registered() {
        assert!(is_known_in("vector", "max_vec4"));
    }

    // Component-wise abs tests
    #[test]
    fn test_abs_vec2() {
        let v = [-1.0, 2.0];
        let result = abs_vec2(v);
        assert_eq!(result, [1.0, 2.0]);
    }

    #[test]
    fn test_abs_vec3() {
        let v = [-1.0, 2.0, -3.0];
        let result = abs_vec3(v);
        assert_eq!(result, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_abs_vec4() {
        let v = [-1.0, 2.0, -3.0, 4.0];
        let result = abs_vec4(v);
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs_vec2_registered() {
        assert!(is_known_in("vector", "abs_vec2"));
    }

    #[test]
    fn test_abs_vec3_registered() {
        assert!(is_known_in("vector", "abs_vec3"));
    }

    #[test]
    fn test_abs_vec4_registered() {
        assert!(is_known_in("vector", "abs_vec4"));
    }
}
