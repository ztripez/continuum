//! Vector Operations (Shared Implementation)
//!
//! Low-level vector operations used by both VM executor and kernel functions.

/// Squared distance between two scalars.
#[inline]
pub fn distance_sq_scalar(a: f64, b: f64) -> f64 {
    let d = a - b;
    d * d
}

/// Squared distance between two 2D vectors.
#[inline]
pub fn distance_sq_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Squared distance between two 3D vectors.
#[inline]
pub fn distance_sq_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Squared distance between two 4D vectors.
#[inline]
pub fn distance_sq_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let dw = a[3] - b[3];
    dx * dx + dy * dy + dz * dz + dw * dw
}

/// Euclidean distance between two scalars.
#[inline]
pub fn distance_scalar(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

/// Euclidean distance between two 2D vectors.
#[inline]
pub fn distance_vec2(a: [f64; 2], b: [f64; 2]) -> f64 {
    distance_sq_vec2(a, b).sqrt()
}

/// Euclidean distance between two 3D vectors.
#[inline]
pub fn distance_vec3(a: [f64; 3], b: [f64; 3]) -> f64 {
    distance_sq_vec3(a, b).sqrt()
}

/// Euclidean distance between two 4D vectors.
#[inline]
pub fn distance_vec4(a: [f64; 4], b: [f64; 4]) -> f64 {
    distance_sq_vec4(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_sq_scalar() {
        assert_eq!(distance_sq_scalar(0.0, 3.0), 9.0);
        assert_eq!(distance_sq_scalar(3.0, 0.0), 9.0);
        assert_eq!(distance_sq_scalar(1.0, 1.0), 0.0);
    }

    #[test]
    fn test_distance_sq_vec2() {
        assert_eq!(distance_sq_vec2([0.0, 0.0], [3.0, 4.0]), 25.0); // 3-4-5 triangle
        assert_eq!(distance_sq_vec2([1.0, 1.0], [1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_distance_sq_vec3() {
        assert_eq!(distance_sq_vec3([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 1.0);
        assert_eq!(distance_sq_vec3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 3.0);
    }

    #[test]
    fn test_distance_sq_vec4() {
        assert_eq!(
            distance_sq_vec4([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            1.0
        );
        assert_eq!(
            distance_sq_vec4([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
            4.0
        );
    }

    #[test]
    fn test_distance_vec2() {
        assert_eq!(distance_vec2([0.0, 0.0], [3.0, 4.0]), 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_distance_vec3() {
        let d = distance_vec3([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_distance_vec4() {
        let d = distance_vec4([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
        assert!((d - 2.0).abs() < 1e-10); // sqrt(4) = 2
    }
}
