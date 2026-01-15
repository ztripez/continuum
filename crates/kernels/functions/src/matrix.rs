//! Matrix Operations
//!
//! Functions for matrix operations: identity, transpose, determinant, inverse, eigenvalues, SVD.

use continuum_foundation::{Mat2, Mat3, Mat4, Value};
use continuum_kernel_macros::kernel_fn;
use nalgebra as na;

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
    use continuum_foundation::matrix_ops::{mat2_mul, mat3_mul, mat4_mul};
    if args.len() != 2 {
        panic!("matrix.mul expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Mat2(a), Value::Mat2(b)) => Value::Mat2(mat2_mul(*a, *b)),
        (Value::Mat3(a), Value::Mat3(b)) => Value::Mat3(mat3_mul(*a, *b)),
        (Value::Mat4(a), Value::Mat4(b)) => Value::Mat4(mat4_mul(*a, *b)),
        _ => panic!("matrix.mul expects two matrices of the same size"),
    }
}

/// Transform vector by matrix: `transform(m, v)` -> Vec
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn transform(args: &[Value]) -> Value {
    use continuum_foundation::matrix_ops::{mat2_transform, mat3_transform, mat4_transform};
    if args.len() != 2 {
        panic!("matrix.transform expects exactly 2 arguments");
    }
    match (&args[0], &args[1]) {
        (Value::Mat2(m), Value::Vec2(v)) => Value::Vec2(mat2_transform(*m, *v)),
        (Value::Mat3(m), Value::Vec3(v)) => Value::Vec3(mat3_transform(*m, *v)),
        (Value::Mat4(m), Value::Vec4(v)) => Value::Vec4(mat4_transform(*m, *v)),
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

/// Eigenvalues of a symmetric matrix: `eigenvalues(m)` -> Vec
/// Returns eigenvalues sorted in descending order
/// Note: Only works for symmetric matrices. Non-symmetric matrices will give incorrect results.
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn eigenvalues(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.eigenvalues expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(arr) => {
            let mat = na::Matrix2::from_column_slice(arr);
            let eig = mat.symmetric_eigen();
            // Sort descending
            let mut vals = [eig.eigenvalues[0], eig.eigenvalues[1]];
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
            Value::Vec2(vals)
        }
        Value::Mat3(arr) => {
            let mat = na::Matrix3::from_column_slice(arr);
            let eig = mat.symmetric_eigen();
            let mut vals = [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]];
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
            Value::Vec3(vals)
        }
        Value::Mat4(arr) => {
            let mat = na::Matrix4::from_column_slice(arr);
            let eig = mat.symmetric_eigen();
            let mut vals = [
                eig.eigenvalues[0],
                eig.eigenvalues[1],
                eig.eigenvalues[2],
                eig.eigenvalues[3],
            ];
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
            Value::Vec4(vals)
        }
        _ => panic!("matrix.eigenvalues expects Mat2, Mat3, or Mat4"),
    }
}

/// Eigenvectors of a symmetric matrix: `eigenvectors(m)` -> Mat
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn eigenvectors(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.eigenvectors expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(arr) => {
            let mat = na::Matrix2::from_column_slice(arr);
            let eig = mat.symmetric_eigen();
            // Sort by eigenvalues descending
            let mut pairs: Vec<_> = eig
                .eigenvalues
                .iter()
                .zip(eig.eigenvectors.column_iter())
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

            // Extract eigenvectors as owned vectors
            let v0 = na::Vector2::from_iterator(pairs[0].1.iter().cloned());
            let v1 = na::Vector2::from_iterator(pairs[1].1.iter().cloned());
            let result = na::Matrix2::from_columns(&[v0, v1]);
            Value::Mat2(result.as_slice().try_into().unwrap())
        }
        Value::Mat3(arr) => {
            let mat = na::Matrix3::from_column_slice(arr);
            let eig = mat.symmetric_eigen();
            let mut pairs: Vec<_> = eig
                .eigenvalues
                .iter()
                .zip(eig.eigenvectors.column_iter())
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

            // Extract eigenvectors as owned vectors
            let v0 = na::Vector3::from_iterator(pairs[0].1.iter().cloned());
            let v1 = na::Vector3::from_iterator(pairs[1].1.iter().cloned());
            let v2 = na::Vector3::from_iterator(pairs[2].1.iter().cloned());
            let result = na::Matrix3::from_columns(&[v0, v1, v2]);
            Value::Mat3(result.as_slice().try_into().unwrap())
        }
        Value::Mat4(arr) => {
            let mat = na::Matrix4::from_column_slice(arr);
            let eig = mat.symmetric_eigen();
            let mut pairs: Vec<_> = eig
                .eigenvalues
                .iter()
                .zip(eig.eigenvectors.column_iter())
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

            // Extract eigenvectors as owned vectors
            let v0 = na::Vector4::from_iterator(pairs[0].1.iter().cloned());
            let v1 = na::Vector4::from_iterator(pairs[1].1.iter().cloned());
            let v2 = na::Vector4::from_iterator(pairs[2].1.iter().cloned());
            let v3 = na::Vector4::from_iterator(pairs[3].1.iter().cloned());
            let result = na::Matrix4::from_columns(&[v0, v1, v2, v3]);
            Value::Mat4(result.as_slice().try_into().unwrap())
        }
        _ => panic!("matrix.eigenvectors expects Mat2, Mat3, or Mat4"),
    }
}

/// SVD - U matrix: `svd_u(m)` -> Mat
/// Returns the left singular vectors (U in A = UΣV^T)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn svd_u(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.svd_u expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(arr) => {
            let mat = na::Matrix2::from_column_slice(arr);
            let svd = mat.svd(true, false);
            let u = svd.u.unwrap();
            Value::Mat2(u.as_slice().try_into().unwrap())
        }
        Value::Mat3(arr) => {
            let mat = na::Matrix3::from_column_slice(arr);
            let svd = mat.svd(true, false);
            let u = svd.u.unwrap();
            Value::Mat3(u.as_slice().try_into().unwrap())
        }
        Value::Mat4(arr) => {
            let mat = na::Matrix4::from_column_slice(arr);
            let svd = mat.svd(true, false);
            let u = svd.u.unwrap();
            Value::Mat4(u.as_slice().try_into().unwrap())
        }
        _ => panic!("matrix.svd_u expects Mat2, Mat3, or Mat4"),
    }
}

/// SVD - singular values: `svd_s(m)` -> Vec
/// Returns the singular values (diagonal of Σ in A = UΣV^T)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn svd_s(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.svd_s expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(arr) => {
            let mat = na::Matrix2::from_column_slice(arr);
            let svd = mat.svd(false, false);
            Value::Vec2([svd.singular_values[0], svd.singular_values[1]])
        }
        Value::Mat3(arr) => {
            let mat = na::Matrix3::from_column_slice(arr);
            let svd = mat.svd(false, false);
            Value::Vec3([
                svd.singular_values[0],
                svd.singular_values[1],
                svd.singular_values[2],
            ])
        }
        Value::Mat4(arr) => {
            let mat = na::Matrix4::from_column_slice(arr);
            let svd = mat.svd(false, false);
            Value::Vec4([
                svd.singular_values[0],
                svd.singular_values[1],
                svd.singular_values[2],
                svd.singular_values[3],
            ])
        }
        _ => panic!("matrix.svd_s expects Mat2, Mat3, or Mat4"),
    }
}

/// SVD - V^T matrix: `svd_vt(m)` -> Mat
/// Returns the transposed right singular vectors (V^T in A = UΣV^T)
#[kernel_fn(namespace = "matrix", category = "matrix", variadic)]
pub fn svd_vt(args: &[Value]) -> Value {
    if args.len() != 1 {
        panic!("matrix.svd_vt expects exactly 1 argument");
    }
    match &args[0] {
        Value::Mat2(arr) => {
            let mat = na::Matrix2::from_column_slice(arr);
            let svd = mat.svd(false, true);
            let vt = svd.v_t.unwrap();
            Value::Mat2(vt.as_slice().try_into().unwrap())
        }
        Value::Mat3(arr) => {
            let mat = na::Matrix3::from_column_slice(arr);
            let svd = mat.svd(false, true);
            let vt = svd.v_t.unwrap();
            Value::Mat3(vt.as_slice().try_into().unwrap())
        }
        Value::Mat4(arr) => {
            let mat = na::Matrix4::from_column_slice(arr);
            let svd = mat.svd(false, true);
            let vt = svd.v_t.unwrap();
            Value::Mat4(vt.as_slice().try_into().unwrap())
        }
        _ => panic!("matrix.svd_vt expects Mat2, Mat3, or Mat4"),
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

    #[test]
    fn test_eigenvalues_mat2_identity() {
        let identity = Value::Mat2([1.0, 0.0, 0.0, 1.0]);
        let result = eigenvalues(&[identity]);
        // Identity has eigenvalues [1, 1]
        if let Value::Vec2(vals) = result {
            assert!((vals[0] - 1.0).abs() < 1e-10);
            assert!((vals[1] - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Vec2");
        }
    }

    #[test]
    fn test_eigenvalues_mat3_diagonal() {
        // Diagonal matrix [[2, 0, 0], [0, 3, 0], [0, 0, 1]]
        let m = Value::Mat3([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        let result = eigenvalues(&[m]);
        // Eigenvalues should be [3, 2, 1] (sorted descending)
        if let Value::Vec3(vals) = result {
            assert!((vals[0] - 3.0).abs() < 1e-10);
            assert!((vals[1] - 2.0).abs() < 1e-10);
            assert!((vals[2] - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Vec3");
        }
    }

    #[test]
    fn test_eigenvectors_mat2_identity() {
        let identity = Value::Mat2([1.0, 0.0, 0.0, 1.0]);
        let result = eigenvectors(&[identity]);
        // Identity matrix has any orthonormal basis as eigenvectors
        // Just check it's a valid matrix
        if let Value::Mat2(_) = result {
            // Success - eigenvectors returned
        } else {
            panic!("Expected Mat2");
        }
    }

    #[test]
    fn test_svd_identity_mat2() {
        let identity = Value::Mat2([1.0, 0.0, 0.0, 1.0]);

        // Singular values should be [1, 1]
        let s = svd_s(&[identity.clone()]);
        if let Value::Vec2(vals) = s {
            assert!((vals[0] - 1.0).abs() < 1e-10);
            assert!((vals[1] - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Vec2");
        }

        // U should be approximately identity (or a rotation)
        let u = svd_u(&[identity.clone()]);
        assert!(matches!(u, Value::Mat2(_)));
    }

    #[test]
    fn test_svd_diagonal_mat3() {
        // Diagonal matrix [[2, 0, 0], [0, 3, 0], [0, 0, 1]]
        let m = Value::Mat3([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);

        // Singular values should be [3, 2, 1] (sorted descending by nalgebra)
        let s = svd_s(&[m.clone()]);
        if let Value::Vec3(vals) = s {
            // Singular values are sorted descending
            assert!((vals[0] - 3.0).abs() < 1e-10);
            assert!((vals[1] - 2.0).abs() < 1e-10);
            assert!((vals[2] - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Vec3");
        }
    }

    #[test]
    fn test_eigenvalues_registered() {
        assert!(is_known_in("matrix", "eigenvalues"));
    }

    #[test]
    fn test_eigenvectors_registered() {
        assert!(is_known_in("matrix", "eigenvectors"));
    }

    #[test]
    fn test_svd_u_registered() {
        assert!(is_known_in("matrix", "svd_u"));
    }

    #[test]
    fn test_svd_s_registered() {
        assert!(is_known_in("matrix", "svd_s"));
    }

    #[test]
    fn test_svd_vt_registered() {
        assert!(is_known_in("matrix", "svd_vt"));
    }
}
