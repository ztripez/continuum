//!
//! Quaternion operations.

use continuum_foundation::{Mat3, Mat4, Quat};
use continuum_kernel_macros::kernel_fn;

/// Construct a quaternion: `quat(w, x, y, z)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn quat(w: f64, x: f64, y: f64, z: f64) -> Quat {
    Quat([w, x, y, z])
}

/// Identity quaternion: `identity()`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn identity() -> Quat {
    Quat([1.0, 0.0, 0.0, 0.0])
}

/// Quaternion norm (magnitude): `norm(q)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn norm(q: Quat) -> f64 {
    let [w, x, y, z] = q.0;
    (w * w + x * x + y * y + z * z).sqrt()
}

/// Normalize a quaternion: `normalize(q)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn normalize(q: Quat) -> Quat {
    let [w, x, y, z] = q.0;
    let mag = (w * w + x * x + y * y + z * z).sqrt();
    if mag == 0.0 {
        panic!("quat.normalize requires non-zero quaternion");
    }
    Quat([w / mag, x / mag, y / mag, z / mag])
}

/// Conjugate a quaternion: `conjugate(q)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn conjugate(q: Quat) -> Quat {
    let [w, x, y, z] = q.0;
    Quat([w, -x, -y, -z])
}

/// Multiply quaternions: `mul(a, b)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn mul(a: Quat, b: Quat) -> Quat {
    let [aw, ax, ay, az] = a.0;
    let [bw, bx, by, bz] = b.0;
    Quat([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])
}

/// Quaternion inverse: `inverse(q)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn inverse(q: Quat) -> Quat {
    let [w, x, y, z] = q.0;
    let norm_sq = w * w + x * x + y * y + z * z;
    if norm_sq == 0.0 {
        panic!("quat.inverse requires non-zero quaternion");
    }
    Quat([w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq])
}

/// Quaternion dot product: `dot(a, b)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn dot(a: Quat, b: Quat) -> f64 {
    let [aw, ax, ay, az] = a.0;
    let [bw, bx, by, bz] = b.0;
    aw * bw + ax * bx + ay * by + az * bz
}

/// Construct a quaternion from axis-angle: `from_axis_angle(axis, angle)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Quat {
    let [x, y, z] = axis;
    let mag = (x * x + y * y + z * z).sqrt();
    if mag == 0.0 {
        panic!("quat.from_axis_angle requires non-zero axis");
    }

    let half = angle * 0.5;
    let s = half.sin() / mag;
    Quat([half.cos(), x * s, y * s, z * s])
}

/// Rotate a vector by a quaternion: `rotate(q, v)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn rotate(q: Quat, v: [f64; 3]) -> [f64; 3] {
    let [qw, qx, qy, qz] = q.0;
    let [vx, vy, vz] = v;

    let ix = qw * vx + qy * vz - qz * vy;
    let iy = qw * vy + qz * vx - qx * vz;
    let iz = qw * vz + qx * vy - qy * vx;
    let iw = -qx * vx - qy * vy - qz * vz;

    [
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    ]
}

/// Linear interpolation: `lerp(a, b, t)` (not normalized)
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn lerp(a: Quat, b: Quat, t: f64) -> Quat {
    let [aw, ax, ay, az] = a.0;
    let [bw, bx, by, bz] = b.0;
    Quat([
        aw + t * (bw - aw),
        ax + t * (bx - ax),
        ay + t * (by - ay),
        az + t * (bz - az),
    ])
}

/// Normalized linear interpolation: `nlerp(a, b, t)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn nlerp(a: Quat, b: Quat, t: f64) -> Quat {
    normalize(lerp(a, b, t))
}

/// Spherical linear interpolation: `slerp(a, b, t)`
#[kernel_fn(namespace = "quat", category = "quaternion")]
pub fn slerp(a: Quat, b: Quat, t: f64) -> Quat {
    let mut b = b;
    let mut d = dot(a, b);

    // Take shortest path
    if d < 0.0 {
        b = Quat([-b.0[0], -b.0[1], -b.0[2], -b.0[3]]);
        d = -d;
    }

    // Use nlerp for nearly parallel quaternions
    if d > 0.9995 {
        return nlerp(a, b, t);
    }

    let theta = d.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;

    let [aw, ax, ay, az] = a.0;
    let [bw, bx, by, bz] = b.0;
    Quat([
        wa * aw + wb * bw,
        wa * ax + wb * bx,
        wa * ay + wb * by,
        wa * az + wb * bz,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_kernel_registry::{Arity, get_in_namespace, is_known_in};

    #[test]
    fn test_quat_registered() {
        assert!(is_known_in("quat", "quat"));
        let desc = get_in_namespace("quat", "quat").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(4));
    }

    #[test]
    fn test_quat_normalize() {
        let q = quat(2.0, 0.0, 0.0, 0.0);
        let normed = normalize(q);
        assert_eq!(normed.0, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_inverse_registered() {
        assert!(is_known_in("quat", "inverse"));
        let desc = get_in_namespace("quat", "inverse").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_dot_registered() {
        assert!(is_known_in("quat", "dot"));
        let desc = get_in_namespace("quat", "dot").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_inverse_identity() {
        let q = identity();
        let inv = inverse(q);
        assert_eq!(inv.0, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_inverse_mul_yields_identity() {
        let q = quat(1.0, 2.0, 3.0, 4.0);
        let inv = inverse(q);
        let q2 = quat(1.0, 2.0, 3.0, 4.0);
        let result = mul(q2, inv);
        let [w, x, y, z] = result.0;
        assert!((w - 1.0).abs() < 1e-10);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_dot_identity() {
        let q1 = identity();
        let q2 = identity();
        let result = dot(q1, q2);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_dot_orthogonal() {
        let q1 = quat(1.0, 0.0, 0.0, 0.0);
        let q2 = quat(0.0, 1.0, 0.0, 0.0);
        let result = dot(q1, q2);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_commutative() {
        let q1 = quat(1.0, 2.0, 3.0, 4.0);
        let q2 = quat(5.0, 6.0, 7.0, 8.0);
        let d1 = dot(q1, q2);
        let q1b = quat(1.0, 2.0, 3.0, 4.0);
        let q2b = quat(5.0, 6.0, 7.0, 8.0);
        let d2 = dot(q2b, q1b);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_dot_with_self_is_norm_squared() {
        let q1 = quat(1.0, 2.0, 3.0, 4.0);
        let q2 = quat(1.0, 2.0, 3.0, 4.0);
        let d = dot(q1, q2);
        let q3 = quat(1.0, 2.0, 3.0, 4.0);
        let n = norm(q3);
        assert!((d - n * n).abs() < 1e-10);
    }

    // Interpolation function tests
    #[test]
    fn test_lerp_registered() {
        assert!(is_known_in("quat", "lerp"));
        let desc = get_in_namespace("quat", "lerp").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
    }

    #[test]
    fn test_nlerp_registered() {
        assert!(is_known_in("quat", "nlerp"));
        let desc = get_in_namespace("quat", "nlerp").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
    }

    #[test]
    fn test_slerp_registered() {
        assert!(is_known_in("quat", "slerp"));
        let desc = get_in_namespace("quat", "slerp").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
    }

    #[test]
    fn test_lerp_at_t0() {
        let q1 = quat(1.0, 2.0, 3.0, 4.0);
        let q2 = quat(5.0, 6.0, 7.0, 8.0);
        let result = lerp(q1, q2, 0.0);
        assert_eq!(result.0, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_lerp_at_t1() {
        let q1 = quat(1.0, 2.0, 3.0, 4.0);
        let q2 = quat(5.0, 6.0, 7.0, 8.0);
        let result = lerp(q1, q2, 1.0);
        assert_eq!(result.0, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_lerp_at_midpoint() {
        let q1 = quat(1.0, 0.0, 0.0, 0.0);
        let q2 = quat(0.0, 1.0, 0.0, 0.0);
        let result = lerp(q1, q2, 0.5);
        assert_eq!(result.0, [0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn test_nlerp_at_t0() {
        let q1 = normalize(quat(1.0, 2.0, 3.0, 4.0));
        let q2 = normalize(quat(5.0, 6.0, 7.0, 8.0));
        let result = nlerp(q1, q2, 0.0);
        let [w1, x1, y1, z1] = q1.0;
        let [wr, xr, yr, zr] = result.0;
        assert!((w1 - wr).abs() < 1e-10);
        assert!((x1 - xr).abs() < 1e-10);
        assert!((y1 - yr).abs() < 1e-10);
        assert!((z1 - zr).abs() < 1e-10);
    }

    #[test]
    fn test_nlerp_at_t1() {
        let q1 = normalize(quat(1.0, 2.0, 3.0, 4.0));
        let q2 = normalize(quat(5.0, 6.0, 7.0, 8.0));
        let result = nlerp(q1, q2, 1.0);
        let [w2, x2, y2, z2] = q2.0;
        let [wr, xr, yr, zr] = result.0;
        assert!((w2 - wr).abs() < 1e-10);
        assert!((x2 - xr).abs() < 1e-10);
        assert!((y2 - yr).abs() < 1e-10);
        assert!((z2 - zr).abs() < 1e-10);
    }

    #[test]
    fn test_nlerp_is_normalized() {
        let q1 = quat(1.0, 0.0, 0.0, 0.0);
        let q2 = quat(0.0, 1.0, 0.0, 0.0);
        let result = nlerp(q1, q2, 0.5);
        let n = norm(result);
        assert!((n - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_at_t0() {
        let q1 = normalize(quat(1.0, 2.0, 3.0, 4.0));
        let q2 = normalize(quat(5.0, 6.0, 7.0, 8.0));
        let result = slerp(q1, q2, 0.0);
        let [w1, x1, y1, z1] = q1.0;
        let [wr, xr, yr, zr] = result.0;
        assert!((w1 - wr).abs() < 1e-10);
        assert!((x1 - xr).abs() < 1e-10);
        assert!((y1 - yr).abs() < 1e-10);
        assert!((z1 - zr).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_at_t1() {
        let q1 = normalize(quat(1.0, 2.0, 3.0, 4.0));
        let q2 = normalize(quat(5.0, 6.0, 7.0, 8.0));
        let result = slerp(q1, q2, 1.0);
        let [w2, x2, y2, z2] = q2.0;
        let [wr, xr, yr, zr] = result.0;
        assert!((w2 - wr).abs() < 1e-10);
        assert!((x2 - xr).abs() < 1e-10);
        assert!((y2 - yr).abs() < 1e-10);
        assert!((z2 - zr).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_is_normalized() {
        let q1 = identity();
        let axis = [0.0, 0.0, 1.0];
        let q2 = from_axis_angle(axis, std::f64::consts::PI / 2.0);
        let result = slerp(q1, q2, 0.5);
        let n = norm(result);
        assert!((n - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_maintains_constant_speed() {
        // Slerp should interpolate at constant angular velocity
        let q1 = identity();
        let axis = [0.0, 0.0, 1.0];
        let q2 = from_axis_angle(axis, std::f64::consts::PI / 2.0);

        let r1 = slerp(q1, q2, 0.25);
        let r2 = slerp(q1, q2, 0.5);
        let r3 = slerp(q1, q2, 0.75);

        // Angular distances should be equal
        let d1 = dot(q1, r1).acos();
        let d2 = dot(r1, r2).acos();
        let d3 = dot(r2, r3).acos();

        assert!((d1 - d2).abs() < 1e-6);
        assert!((d2 - d3).abs() < 1e-6);
    }

    #[test]
    fn test_slerp_shortest_path() {
        // Test that slerp takes shortest path when quaternions are opposite hemisphere
        let q1 = identity();
        let q2 = Quat([-1.0, 0.0, 0.0, 0.0]); // Negative identity
        let result = slerp(q1, q2, 0.5);

        // Should still be close to identity, not interpolate through large angle
        let d = dot(q1, result);
        assert!(d > 0.9); // Should be close to identity
    }
}
