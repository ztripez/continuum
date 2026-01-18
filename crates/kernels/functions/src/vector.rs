//! Vector Constructors
//!
//! Functions for constructing vector types.

use continuum_kernel_macros::kernel_fn;

/// Construct a 2D vector: `vec2(x, y)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn vec2(x: f64, y: f64) -> [f64; 2] {
    [x, y]
}

/// Construct a 3D vector: `vec3(x, y, z)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0)],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn vec3(x: f64, y: f64, z: f64) -> [f64; 3] {
    [x, y, z]
}

/// Construct a 4D vector: `vec4(x, y, z, w)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0), UnitSameAs(0)],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn vec4(x: f64, y: f64, z: f64, w: f64) -> [f64; 4] {
    [x, y, z, w]
}

/// Vector length/magnitude for Vec2: `length(vec)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(2))],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn length_vec2(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Vector length/magnitude for Vec3: `length(vec)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3))],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn length_vec3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Vector length/magnitude for Vec4: `length(vec)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4))],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn length_vec4(v: [f64; 4]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt()
}

/// Absolute value for scalar: `length(scalar)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn length_scalar(s: f64) -> f64 {
    s.abs()
}

/// Normalize a Vec2 to unit length: `normalize(vec)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(2))],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = Dimensionless
)]
pub fn normalize_vec2(v: [f64; 2]) -> [f64; 2] {
    let mag = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if mag > 0.0 {
        [v[0] / mag, v[1] / mag]
    } else {
        [0.0, 0.0]
    }
}

/// Normalize a Vec3 to unit length: `normalize(vec)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3))],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = Dimensionless
)]
pub fn normalize_vec3(v: [f64; 3]) -> [f64; 3] {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if mag > 0.0 {
        [v[0] / mag, v[1] / mag, v[2] / mag]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Normalize a Vec4 to unit length: `normalize(vec)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4))],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = Dimensionless
)]
pub fn normalize_vec4(v: [f64; 4]) -> [f64; 4] {
    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if mag > 0.0 {
        [v[0] / mag, v[1] / mag, v[2] / mag, v[3] / mag]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    }
}

/// Get sign of scalar: `normalize(scalar)` returns -1, 0, or 1
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn normalize_scalar(s: f64) -> f64 {
    s.signum()
}

/// Dot product for Vec2: `dot(a, b)` -> Scalar
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(2)), VectorDim(DimExact(2))],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 1])
)]
pub fn dot_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Dot product for Vec3: `dot(a, b)` -> Scalar
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 1])
)]
pub fn dot_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Dot product for Vec4: `dot(a, b)` -> Scalar
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4)), VectorDim(DimExact(4))],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 1])
)]
pub fn dot_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Cross product (Vec3 only): `cross(a, b)` -> Vec3
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = Multiply(&[0, 1])
)]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Reflect vector v around normal n: `reflect(v, n)`
/// Normal n is assumed to be normalized (unit vector)
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn reflect(v: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let d = 2.0 * (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]);
    [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]]
}

/// Project a onto b: `project(a, onto)`
/// Returns the projection of vector a onto vector b
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = UnitDerivSameAs(0)
)]
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
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(2)), VectorDim(DimExact(2))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn distance_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    continuum_foundation::vector_ops::distance_vec2(a, b)
}

/// Distance between two 3D vectors: `distance(a, b)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn distance_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    continuum_foundation::vector_ops::distance_vec3(a, b)
}

/// Distance between two 4D vectors: `distance(a, b)`
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4)), VectorDim(DimExact(4))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn distance_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    continuum_foundation::vector_ops::distance_vec4(a, b)
}

/// Squared distance between two 2D vectors: `distance_sq(a, b)` - cheaper than distance
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(2)), VectorDim(DimExact(2))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 0])
)]
pub fn distance_sq_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    continuum_foundation::vector_ops::distance_sq_vec2(a, b)
}

/// Squared distance between two 3D vectors: `distance_sq(a, b)` - cheaper than distance
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 0])
)]
pub fn distance_sq_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    continuum_foundation::vector_ops::distance_sq_vec3(a, b)
}

/// Squared distance between two 4D vectors: `distance_sq(a, b)` - cheaper than distance
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4)), VectorDim(DimExact(4))],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 0])
)]
pub fn distance_sq_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    continuum_foundation::vector_ops::distance_sq_vec4(a, b)
}

/// Linear interpolation for Vec2: `lerp(a, b, t)` → a + t * (b - a)
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(2)), VectorDim(DimExact(2)), AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn lerp_vec2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// Linear interpolation for Vec3: `lerp(a, b, t)` → a + t * (b - a)
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3)), AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn lerp_vec3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Linear interpolation for Vec4: `lerp(a, b, t)` → a + t * (b - a)
#[kernel_fn(
    namespace = "vector",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4)), VectorDim(DimExact(4)), AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitDimensionless],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn lerp_vec4(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
        a[3] + t * (b[3] - a[3]),
    ]
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

// ============================================================================
// NORMALIZED AND SPHERICAL INTERPOLATION
// ============================================================================

/// Normalized linear interpolation for Vec2: `nlerp(a, b, t)` → normalize(lerp(a, b, t))
///
/// Useful for interpolating directions. Result is always a unit vector.
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn nlerp_vec2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    let lerped = lerp_vec2(a, b, t);
    let len = (lerped[0] * lerped[0] + lerped[1] * lerped[1]).sqrt();
    if len > 1e-10 {
        [lerped[0] / len, lerped[1] / len]
    } else {
        a // Fallback to first input if result is degenerate
    }
}

/// Normalized linear interpolation for Vec3: `nlerp(a, b, t)` → normalize(lerp(a, b, t))
///
/// Useful for interpolating directions. Result is always a unit vector.
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn nlerp_vec3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let lerped = lerp_vec3(a, b, t);
    let len = (lerped[0] * lerped[0] + lerped[1] * lerped[1] + lerped[2] * lerped[2]).sqrt();
    if len > 1e-10 {
        [lerped[0] / len, lerped[1] / len, lerped[2] / len]
    } else {
        a // Fallback to first input if result is degenerate
    }
}

/// Spherical linear interpolation for Vec3: `slerp(a, b, t)`
///
/// Interpolates along the great circle arc between two unit vectors.
/// Both inputs should be unit vectors. Provides constant angular velocity.
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn slerp_vec3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    // Compute dot product (cosine of angle between vectors)
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    // Clamp to handle numerical errors
    let dot = dot.clamp(-1.0, 1.0);

    // If vectors are nearly parallel, use nlerp to avoid division by zero
    if dot.abs() > 0.9995 {
        return nlerp_vec3(a, b, t);
    }

    // Calculate the angle and its sine
    let theta = dot.acos();
    let sin_theta = theta.sin();

    // Calculate interpolation coefficients
    let s0 = ((1.0 - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;

    // Interpolate
    [
        s0 * a[0] + s1 * b[0],
        s0 * a[1] + s1 * b[1],
        s0 * a[2] + s1 * b[2],
    ]
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

    // Tests for angle, refract, faceforward

    #[test]
    fn test_angle_registered() {
        assert!(is_known_in("vector", "angle"));
    }

    #[test]
    fn test_angle_perpendicular() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let result = angle(a, b);
        assert!((result - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_angle_parallel() {
        let a = [1.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let result = angle(a, b);
        assert!(result.abs() < 1e-10); // 0 radians
    }

    #[test]
    fn test_angle_opposite() {
        let a = [1.0, 0.0, 0.0];
        let b = [-1.0, 0.0, 0.0];
        let result = angle(a, b);
        assert!((result - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_angle_zero_vector() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let result = angle(a, b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_refract_registered() {
        assert!(is_known_in("vector", "refract"));
    }

    #[test]
    fn test_refract_normal_incidence() {
        // Normal incidence (perpendicular)
        let i = [0.0, -1.0, 0.0];
        let n = [0.0, 1.0, 0.0];
        let eta = 1.5; // Glass (air -> glass)
        let result = refract(i, n, eta);
        // Should maintain direction, just scaled by eta
        assert!((result[0]).abs() < 1e-10);
        assert!(result[1] < 0.0); // Still going down
        assert!((result[2]).abs() < 1e-10);
    }

    #[test]
    fn test_refract_total_internal_reflection() {
        // Incident angle beyond critical angle
        let i = [0.8, -0.6, 0.0]; // Shallow angle
        let n = [0.0, 1.0, 0.0];
        let eta = 1.5; // Glass -> air (flipped ratio for TIR)
        let result = refract(i, n, eta);
        // Should return zero vector for total internal reflection
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_faceforward_registered() {
        assert!(is_known_in("vector", "faceforward"));
    }

    #[test]
    fn test_faceforward_toward_viewer() {
        // Normal already facing viewer (dot(nref, i) < 0)
        let n = [0.0, 1.0, 0.0];
        let i = [0.0, -1.0, 0.0]; // Incident going down
        let nref = [0.0, 1.0, 0.0]; // Reference normal up
        let result = faceforward(n, i, nref);
        assert_eq!(result, n); // Should keep normal as-is
    }

    #[test]
    fn test_faceforward_away_from_viewer() {
        // Normal facing away (dot(nref, i) >= 0)
        let n = [0.0, 1.0, 0.0];
        let i = [0.0, 1.0, 0.0]; // Incident going up (same direction as nref)
        let nref = [0.0, 1.0, 0.0];
        let result = faceforward(n, i, nref);
        assert_eq!(result, [0.0, -1.0, 0.0]); // Should flip normal
    }

    #[test]
    fn test_faceforward_perpendicular() {
        // Edge case: incident perpendicular to reference normal
        let n = [1.0, 0.0, 0.0];
        let i = [1.0, 0.0, 0.0]; // Perpendicular to normal
        let nref = [0.0, 1.0, 0.0];
        let result = faceforward(n, i, nref);
        assert_eq!(result, [-1.0, 0.0, 0.0]); // dot is 0, so >= 0, flip normal
    }

    // Tests for nlerp (normalized linear interpolation)

    #[test]
    fn test_nlerp_vec2_registered() {
        assert!(is_known_in("vector", "nlerp_vec2"));
    }

    #[test]
    fn test_nlerp_vec3_registered() {
        assert!(is_known_in("vector", "nlerp_vec3"));
    }

    #[test]
    fn test_nlerp_vec2_at_endpoints() {
        // At t=0, should return normalized a
        let a = [3.0, 4.0]; // Length 5
        let b = [0.0, 1.0]; // Already unit
        let result = nlerp_vec2(a, b, 0.0);
        assert!((result[0] - 0.6).abs() < 1e-10); // 3/5
        assert!((result[1] - 0.8).abs() < 1e-10); // 4/5

        // At t=1, should return b (already unit)
        let result = nlerp_vec2(a, b, 1.0);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nlerp_vec2_is_unit() {
        // Result should always be unit length
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = nlerp_vec2(a, b, t);
            let len = (result[0] * result[0] + result[1] * result[1]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "nlerp result should be unit at t={}",
                t
            );
        }
    }

    #[test]
    fn test_nlerp_vec3_at_endpoints() {
        // At t=0, should return normalized a
        let a = [1.0, 2.0, 2.0]; // Length 3
        let b = [0.0, 0.0, 1.0]; // Already unit
        let result = nlerp_vec3(a, b, 0.0);
        assert!((result[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((result[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result[2] - 2.0 / 3.0).abs() < 1e-10);

        // At t=1, should return b
        let result = nlerp_vec3(a, b, 1.0);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nlerp_vec3_is_unit() {
        // Result should always be unit length
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = nlerp_vec3(a, b, t);
            let len =
                (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "nlerp result should be unit at t={}",
                t
            );
        }
    }

    // Tests for slerp (spherical linear interpolation)
    #[test]
    fn test_slerp_vec3_registered() {
        assert!(is_known_in("vector", "slerp_vec3"));
    }

    #[test]
    fn test_slerp_vec3_at_endpoints() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];

        // At t=0, should return a
        let result = slerp_vec3(a, b, 0.0);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);

        // At t=1, should return b
        let result = slerp_vec3(a, b, 1.0);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_vec3_midpoint() {
        // Slerp between X and Y axis should give point at 45 degrees
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let result = slerp_vec3(a, b, 0.5);

        // At 45 degrees: (cos(45), sin(45), 0) = (sqrt(2)/2, sqrt(2)/2, 0)
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((result[0] - sqrt2_2).abs() < 1e-10);
        assert!((result[1] - sqrt2_2).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_vec3_is_unit() {
        // Result should always be unit length
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = slerp_vec3(a, b, t);
            let len =
                (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "slerp result should be unit at t={}",
                t
            );
        }
    }

    #[test]
    fn test_slerp_vec3_nearly_parallel() {
        // When vectors are nearly parallel, slerp should fall back to nlerp
        let a = [1.0, 0.0, 0.0];
        let b: [f64; 3] = [0.9999, 0.01, 0.0]; // Nearly parallel
        let b_len = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let b_norm = [b[0] / b_len, b[1] / b_len, b[2] / b_len];
        let result = slerp_vec3(a, b_norm, 0.5);

        // Should still produce unit result
        let len = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_vec3_constant_angular_velocity() {
        // Slerp maintains constant angular velocity
        // The angle from a to slerp(a,b,t) should be t * angle(a,b)
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let full_angle = std::f64::consts::FRAC_PI_2; // 90 degrees

        for &t in &[0.25, 0.5, 0.75] {
            let result = slerp_vec3(a, b, t);
            // Angle from a to result
            let cos_angle = a[0] * result[0] + a[1] * result[1] + a[2] * result[2];
            let actual_angle = cos_angle.acos();
            let expected_angle = t * full_angle;
            assert!(
                (actual_angle - expected_angle).abs() < 1e-10,
                "slerp should have constant angular velocity at t={}",
                t
            );
        }
    }
}
