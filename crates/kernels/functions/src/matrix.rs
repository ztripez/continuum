//! Matrix Operations
//!
//! Functions for matrix operations: identity, transpose, determinant, inverse.

use continuum_foundation::{Mat2, Mat3, Mat4, Value};
use continuum_kernel_macros::kernel_fn;

/// Identity 2x2 matrix: `identity2()`
/// Returns column-major: [1, 0, 0, 1]
#[kernel_fn(namespace = "matrix", category = "matrix")]
pub fn identity2() -> Mat2 {
    Mat2([1.0, 0.0, 0.0, 1.0])
}

/// Identity 3x3 matrix: `identity3()`
/// Returns column-major: [1, 0, 0, 0, 1, 0, 0, 0, 1]
#[kernel_fn(namespace = "matrix", category = "matrix")]
pub fn identity3() -> Mat3 {
    Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

/// Identity 4x4 matrix: `identity4()`
/// Returns column-major: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
#[kernel_fn(namespace = "matrix", category = "matrix")]
pub fn identity4() -> Mat4 {
    Mat4([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// Transpose a matrix: `transpose(m)`
/// Converts column-major to row-major order (or vice versa)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn transpose(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.transpose expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(mat) => {
            // Column-major [m00, m10, m01, m11] -> [m00, m01, m10, m11]
            Value::Mat2([mat[0], mat[2], mat[1], mat[3]])
        }
        Value::Mat3(mat) => {
            // Column-major [m00, m10, m20, m01, m11, m21, m02, m12, m22]
            // -> [m00, m01, m02, m10, m11, m12, m20, m21, m22]
            Value::Mat3([
                mat[0], mat[3], mat[6], mat[1], mat[4], mat[7], mat[2], mat[5], mat[8],
            ])
        }
        Value::Mat4(mat) => {
            // Column-major to row-major transpose
            Value::Mat4([
                mat[0], mat[4], mat[8], mat[12], mat[1], mat[5], mat[9], mat[13], mat[2], mat[6],
                mat[10], mat[14], mat[3], mat[7], mat[11], mat[15],
            ])
        }
        _ => panic!("matrix.transpose expects Mat2, Mat3, or Mat4"),
    }
}

/// Determinant of a matrix: `determinant(m)` -> Scalar
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn determinant(args: &[Value]) -> f64 {
    if args.len() != 1 {
        panic!("matrix.determinant expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(mat) => {
            // Column-major: [m00, m10, m01, m11]
            // det = m00*m11 - m01*m10
            mat[0] * mat[3] - mat[2] * mat[1]
        }
        Value::Mat3(mat) => {
            // Column-major: [m00, m10, m20, m01, m11, m21, m02, m12, m22]
            // Using cofactor expansion along first row
            let m00 = mat[0];
            let m10 = mat[1];
            let m20 = mat[2];
            let m01 = mat[3];
            let m11 = mat[4];
            let m21 = mat[5];
            let m02 = mat[6];
            let m12 = mat[7];
            let m22 = mat[8];

            m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12)
                + m02 * (m10 * m21 - m20 * m11)
        }
        Value::Mat4(mat) => {
            // Column-major 4x4 determinant using cofactor expansion
            // This is more complex, using 3x3 subdeterminants
            let m00 = mat[0];
            let m10 = mat[1];
            let m20 = mat[2];
            let m30 = mat[3];
            let m01 = mat[4];
            let m11 = mat[5];
            let m21 = mat[6];
            let m31 = mat[7];
            let m02 = mat[8];
            let m12 = mat[9];
            let m22 = mat[10];
            let m32 = mat[11];
            let m03 = mat[12];
            let m13 = mat[13];
            let m23 = mat[14];
            let m33 = mat[15];

            // Cofactor expansion along first row
            let det0 = m11 * (m22 * m33 - m32 * m23) - m21 * (m12 * m33 - m32 * m13)
                + m31 * (m12 * m23 - m22 * m13);
            let det1 = m10 * (m22 * m33 - m32 * m23) - m20 * (m12 * m33 - m32 * m13)
                + m30 * (m12 * m23 - m22 * m13);
            let det2 = m10 * (m21 * m33 - m31 * m23) - m20 * (m11 * m33 - m31 * m13)
                + m30 * (m11 * m23 - m21 * m13);
            let det3 = m10 * (m21 * m32 - m31 * m22) - m20 * (m11 * m32 - m31 * m12)
                + m30 * (m11 * m22 - m21 * m12);

            m00 * det0 - m01 * det1 + m02 * det2 - m03 * det3
        }
        _ => panic!("matrix.determinant expects Mat2, Mat3, or Mat4"),
    }
}

/// Inverse of a matrix: `inverse(m)` -> Mat
/// Panics if matrix is singular (determinant = 0)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn inverse(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.inverse expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(mat) => {
            let det = mat[0] * mat[3] - mat[2] * mat[1];
            if det.abs() < 1e-10 {
                panic!("matrix.inverse: matrix is singular (determinant = 0)");
            }
            let inv_det = 1.0 / det;
            // Column-major inverse of 2x2
            Value::Mat2([
                mat[3] * inv_det,
                -mat[1] * inv_det,
                -mat[2] * inv_det,
                mat[0] * inv_det,
            ])
        }
        Value::Mat3(mat) => {
            let m00 = mat[0];
            let m10 = mat[1];
            let m20 = mat[2];
            let m01 = mat[3];
            let m11 = mat[4];
            let m21 = mat[5];
            let m02 = mat[6];
            let m12 = mat[7];
            let m22 = mat[8];

            let det = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12)
                + m02 * (m10 * m21 - m20 * m11);

            if det.abs() < 1e-10 {
                panic!("matrix.inverse: matrix is singular (determinant = 0)");
            }

            let inv_det = 1.0 / det;

            // Compute adjugate (cofactor matrix transposed) and multiply by 1/det
            Value::Mat3([
                (m11 * m22 - m21 * m12) * inv_det,
                (m20 * m12 - m10 * m22) * inv_det,
                (m10 * m21 - m20 * m11) * inv_det,
                (m21 * m02 - m01 * m22) * inv_det,
                (m00 * m22 - m20 * m02) * inv_det,
                (m20 * m01 - m00 * m21) * inv_det,
                (m01 * m12 - m11 * m02) * inv_det,
                (m10 * m02 - m00 * m12) * inv_det,
                (m00 * m11 - m10 * m01) * inv_det,
            ])
        }
        Value::Mat4(mat) => {
            // 4x4 matrix inversion using adjugate method
            // This is more complex - using the full cofactor matrix
            let m00 = mat[0];
            let m10 = mat[1];
            let m20 = mat[2];
            let m30 = mat[3];
            let m01 = mat[4];
            let m11 = mat[5];
            let m21 = mat[6];
            let m31 = mat[7];
            let m02 = mat[8];
            let m12 = mat[9];
            let m22 = mat[10];
            let m32 = mat[11];
            let m03 = mat[12];
            let m13 = mat[13];
            let m23 = mat[14];
            let m33 = mat[15];

            // Calculate determinant first
            let det0 = m11 * (m22 * m33 - m32 * m23) - m21 * (m12 * m33 - m32 * m13)
                + m31 * (m12 * m23 - m22 * m13);
            let det1 = m10 * (m22 * m33 - m32 * m23) - m20 * (m12 * m33 - m32 * m13)
                + m30 * (m12 * m23 - m22 * m13);
            let det2 = m10 * (m21 * m33 - m31 * m23) - m20 * (m11 * m33 - m31 * m13)
                + m30 * (m11 * m23 - m21 * m13);
            let det3 = m10 * (m21 * m32 - m31 * m22) - m20 * (m11 * m32 - m31 * m12)
                + m30 * (m11 * m22 - m21 * m12);

            let det = m00 * det0 - m01 * det1 + m02 * det2 - m03 * det3;

            if det.abs() < 1e-10 {
                panic!("matrix.inverse: matrix is singular (determinant = 0)");
            }

            let inv_det = 1.0 / det;

            // Compute cofactor matrix (adjugate) elements
            let c00 = (m11 * (m22 * m33 - m32 * m23) - m21 * (m12 * m33 - m32 * m13)
                + m31 * (m12 * m23 - m22 * m13))
                * inv_det;
            let c10 = -(m10 * (m22 * m33 - m32 * m23) - m20 * (m12 * m33 - m32 * m13)
                + m30 * (m12 * m23 - m22 * m13))
                * inv_det;
            let c20 = (m10 * (m21 * m33 - m31 * m23) - m20 * (m11 * m33 - m31 * m13)
                + m30 * (m11 * m23 - m21 * m13))
                * inv_det;
            let c30 = -(m10 * (m21 * m32 - m31 * m22) - m20 * (m11 * m32 - m31 * m12)
                + m30 * (m11 * m22 - m21 * m12))
                * inv_det;

            let c01 = -(m01 * (m22 * m33 - m32 * m23) - m21 * (m02 * m33 - m32 * m03)
                + m31 * (m02 * m23 - m22 * m03))
                * inv_det;
            let c11 = (m00 * (m22 * m33 - m32 * m23) - m20 * (m02 * m33 - m32 * m03)
                + m30 * (m02 * m23 - m22 * m03))
                * inv_det;
            let c21 = -(m00 * (m21 * m33 - m31 * m23) - m20 * (m01 * m33 - m31 * m03)
                + m30 * (m01 * m23 - m21 * m03))
                * inv_det;
            let c31 = (m00 * (m21 * m32 - m31 * m22) - m20 * (m01 * m32 - m31 * m02)
                + m30 * (m01 * m22 - m21 * m02))
                * inv_det;

            let c02 = (m01 * (m12 * m33 - m32 * m13) - m11 * (m02 * m33 - m32 * m03)
                + m31 * (m02 * m13 - m12 * m03))
                * inv_det;
            let c12 = -(m00 * (m12 * m33 - m32 * m13) - m10 * (m02 * m33 - m32 * m03)
                + m30 * (m02 * m13 - m12 * m03))
                * inv_det;
            let c22 = (m00 * (m11 * m33 - m31 * m13) - m10 * (m01 * m33 - m31 * m03)
                + m30 * (m01 * m13 - m11 * m03))
                * inv_det;
            let c32 = -(m00 * (m11 * m32 - m31 * m12) - m10 * (m01 * m32 - m31 * m02)
                + m30 * (m01 * m12 - m11 * m02))
                * inv_det;

            let c03 = -(m01 * (m12 * m23 - m22 * m13) - m11 * (m02 * m23 - m22 * m03)
                + m21 * (m02 * m13 - m12 * m03))
                * inv_det;
            let c13 = (m00 * (m12 * m23 - m22 * m13) - m10 * (m02 * m23 - m22 * m03)
                + m20 * (m02 * m13 - m12 * m03))
                * inv_det;
            let c23 = -(m00 * (m11 * m23 - m21 * m13) - m10 * (m01 * m23 - m21 * m03)
                + m20 * (m01 * m13 - m11 * m03))
                * inv_det;
            let c33 = (m00 * (m11 * m22 - m21 * m12) - m10 * (m01 * m22 - m21 * m02)
                + m20 * (m01 * m12 - m11 * m02))
                * inv_det;

            Value::Mat4([
                c00, c10, c20, c30, c01, c11, c21, c31, c02, c12, c22, c32, c03, c13, c23, c33,
            ])
        }
        _ => panic!("matrix.inverse expects Mat2, Mat3, or Mat4"),
    }
}

/// Matrix multiply: `mul(a, b)` -> Mat
/// Explicit function for matrix multiplication (alternative to a * b operator)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn mul(args: &[Value]) -> Value {
    if args.len() != 2 {
        panic!("matrix.mul expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Mat2(a), Value::Mat2(b)) => {
            let mut result = [0.0; 4];
            for col in 0..2 {
                for row in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += a[k * 2 + row] * b[col * 2 + k];
                    }
                    result[col * 2 + row] = sum;
                }
            }
            Value::Mat2(result)
        }
        (Value::Mat3(a), Value::Mat3(b)) => {
            let mut result = [0.0; 9];
            for col in 0..3 {
                for row in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += a[k * 3 + row] * b[col * 3 + k];
                    }
                    result[col * 3 + row] = sum;
                }
            }
            Value::Mat3(result)
        }
        (Value::Mat4(a), Value::Mat4(b)) => {
            let mut result = [0.0; 16];
            for col in 0..4 {
                for row in 0..4 {
                    let mut sum = 0.0;
                    for k in 0..4 {
                        sum += a[k * 4 + row] * b[col * 4 + k];
                    }
                    result[col * 4 + row] = sum;
                }
            }
            Value::Mat4(result)
        }
        _ => panic!("matrix.mul expects two matrices of the same size"),
    }
}

/// Transform vector by matrix: `transform(m, v)` -> Vec
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn transform(args: &[Value]) -> Value {
    if args.len() != 2 {
        panic!("matrix.transform expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Mat2(m), Value::Vec2(v)) => {
            let x = m[0] * v[0] + m[2] * v[1];
            let y = m[1] * v[0] + m[3] * v[1];
            Value::Vec2([x, y])
        }
        (Value::Mat3(m), Value::Vec3(v)) => {
            let x = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
            let y = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
            let z = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
            Value::Vec3([x, y, z])
        }
        (Value::Mat4(m), Value::Vec4(v)) => {
            let x = m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3];
            let y = m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3];
            let z = m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3];
            let w = m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3];
            Value::Vec4([x, y, z, w])
        }
        _ => panic!("matrix.transform expects (Mat2, Vec2), (Mat3, Vec3), or (Mat4, Vec4)"),
    }
}

/// Build rotation matrix from quaternion: `from_quat(q)` -> Mat3
#[kernel_fn(namespace = "matrix", category = "matrix")]
pub fn from_quat(q: [f64; 4]) -> Mat3 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);

    // Normalize quaternion
    let norm = (x * x + y * y + z * z + w * w).sqrt();
    let (x, y, z, w) = (x / norm, y / norm, z / norm, w / norm);

    // Compute rotation matrix elements (column-major)
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    Mat3([
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy),
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
        2.0 * (xz + wy),
        2.0 * (yz - wx),
        1.0 - 2.0 * (xx + yy),
    ])
}

/// Build rotation matrix from axis-angle: `from_axis_angle(axis, angle)` -> Mat3
#[kernel_fn(namespace = "matrix", category = "matrix")]
pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Mat3 {
    // Normalize axis
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / len, axis[1] / len, axis[2] / len);

    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;

    // Rodrigues' rotation formula (column-major)
    Mat3([
        t * x * x + c,
        t * x * y + s * z,
        t * x * z - s * y,
        t * x * y - s * z,
        t * y * y + c,
        t * y * z + s * x,
        t * x * z + s * y,
        t * y * z - s * x,
        t * z * z + c,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_kernel_registry::{Arity, get_in_namespace, is_known_in};

    #[test]
    fn test_identity2_registered() {
        assert!(is_known_in("matrix", "identity2"));
        let desc = get_in_namespace("matrix", "identity2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(0));
    }

    #[test]
    fn test_identity3_registered() {
        assert!(is_known_in("matrix", "identity3"));
        let desc = get_in_namespace("matrix", "identity3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(0));
    }

    #[test]
    fn test_identity4_registered() {
        assert!(is_known_in("matrix", "identity4"));
        let desc = get_in_namespace("matrix", "identity4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(0));
    }

    #[test]
    fn test_identity2_value() {
        let result = identity2();
        assert_eq!(result.0, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_identity3_value() {
        let result = identity3();
        assert_eq!(result.0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_transpose_mat2() {
        let m = Value::Mat2([1.0, 2.0, 3.0, 4.0]);
        let result = transpose(&[m]);
        assert_eq!(result, Value::Mat2([1.0, 3.0, 2.0, 4.0]));
    }

    #[test]
    fn test_transpose_mat3() {
        let m = Value::Mat3([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = transpose(&[m]);
        assert_eq!(
            result,
            Value::Mat3([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])
        );
    }

    #[test]
    fn test_determinant_mat2() {
        let m = Value::Mat2([1.0, 2.0, 3.0, 4.0]);
        let det = determinant(&[m]);
        // det = 1*4 - 3*2 = 4 - 6 = -2
        assert_eq!(det, -2.0);
    }

    #[test]
    fn test_determinant_mat3_identity() {
        let m = Value::Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let det = determinant(&[m]);
        assert_eq!(det, 1.0);
    }

    #[test]
    fn test_inverse_mat2() {
        let m = Value::Mat2([1.0, 0.0, 0.0, 1.0]); // Identity
        let inv = inverse(&[m]);
        assert_eq!(inv, Value::Mat2([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_inverse_mat2_scaled() {
        let m = Value::Mat2([2.0, 0.0, 0.0, 2.0]); // 2*Identity
        let inv = inverse(&[m]);
        assert_eq!(inv, Value::Mat2([0.5, 0.0, 0.0, 0.5]));
    }

    #[test]
    fn test_inverse_mat3_identity() {
        let m = Value::Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let inv = inverse(&[m]);
        assert_eq!(
            inv,
            Value::Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn test_inverse_mat2_singular() {
        let m = Value::Mat2([1.0, 2.0, 2.0, 4.0]); // Singular matrix (rows are linearly dependent)
        let _ = inverse(&[m]);
    }

    #[test]
    fn test_transpose_registered() {
        assert!(is_known_in("matrix", "transpose"));
    }

    #[test]
    fn test_determinant_registered() {
        assert!(is_known_in("matrix", "determinant"));
    }

    #[test]
    fn test_inverse_registered() {
        assert!(is_known_in("matrix", "inverse"));
    }

    #[test]
    fn test_mul_mat2_identity() {
        let identity = Value::Mat2([1.0, 0.0, 0.0, 1.0]);
        let m = Value::Mat2([1.0, 2.0, 3.0, 4.0]);
        let result = mul(&[identity, m.clone()]);
        assert_eq!(result, m);
    }

    #[test]
    fn test_mul_mat3() {
        let a = Value::Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]); // Identity
        let b = Value::Mat3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]); // 2*Identity
        let result = mul(&[a, b]);
        assert_eq!(
            result,
            Value::Mat3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
        );
    }

    #[test]
    fn test_transform_vec2() {
        let identity = Value::Mat2([1.0, 0.0, 0.0, 1.0]);
        let v = Value::Vec2([3.0, 4.0]);
        let result = transform(&[identity, v]);
        assert_eq!(result, Value::Vec2([3.0, 4.0]));
    }

    #[test]
    fn test_transform_vec3_scale() {
        let scale = Value::Mat3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        let v = Value::Vec3([1.0, 2.0, 3.0]);
        let result = transform(&[scale, v]);
        assert_eq!(result, Value::Vec3([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_from_quat_identity() {
        // Identity quaternion (w=1, x=y=z=0) produces identity rotation
        let q = [0.0, 0.0, 0.0, 1.0];
        let result = from_quat(q);
        // Should be close to identity matrix
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!((result.0[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_axis_angle_identity() {
        // Zero rotation produces identity
        let axis = [0.0, 0.0, 1.0];
        let angle = 0.0;
        let result = from_axis_angle(axis, angle);
        let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!((result.0[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_axis_angle_90_deg_z() {
        // 90 degree rotation around Z axis
        let axis = [0.0, 0.0, 1.0];
        let angle = std::f64::consts::FRAC_PI_2; // 90 degrees
        let result = from_axis_angle(axis, angle);
        // Should rotate (1,0,0) to (0,1,0)
        // Expected matrix (column-major): [[0,-1,0], [1,0,0], [0,0,1]]
        let expected = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!((result.0[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mul_registered() {
        assert!(is_known_in("matrix", "mul"));
    }

    #[test]
    fn test_transform_registered() {
        assert!(is_known_in("matrix", "transform"));
    }

    #[test]
    fn test_from_quat_registered() {
        assert!(is_known_in("matrix", "from_quat"));
    }

    #[test]
    fn test_from_axis_angle_registered() {
        assert!(is_known_in("matrix", "from_axis_angle"));
    }
}
