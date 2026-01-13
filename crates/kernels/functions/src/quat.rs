//!
//! Quaternion operations.

use continuum_foundation::Quat;
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
}
