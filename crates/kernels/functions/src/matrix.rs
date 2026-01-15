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
}
