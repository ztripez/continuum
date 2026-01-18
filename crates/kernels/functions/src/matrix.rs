//! Matrix Operations
//!
//! Functions for matrix operations: identity, transpose, determinant, inverse, eigenvalues, SVD.

use continuum_foundation::{Mat2, Mat3, Mat4};
use continuum_kernel_macros::kernel_fn;
use nalgebra as na;

/// Identity 2x2 matrix: `identity2()`
/// Returns column-major: [1, 0, 0, 1]
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [],
    unit_in = [],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = Dimensionless
)]
pub fn identity2() -> Mat2 {
    Mat2([1.0, 0.0, 0.0, 1.0])
}

/// Identity 3x3 matrix: `identity3()`
/// Returns column-major: [1, 0, 0, 0, 1, 0, 0, 0, 1]
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [],
    unit_in = [],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = Dimensionless
)]
pub fn identity3() -> Mat3 {
    Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

/// Identity 4x4 matrix: `identity4()`
/// Returns column-major: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [],
    unit_in = [],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn identity4() -> Mat4 {
    Mat4([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// Transpose a matrix: `transpose_mat2(m)`
/// Converts column-major to row-major order (or vice versa)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn transpose_mat2(mat: Mat2) -> Mat2 {
    Mat2([mat.0[0], mat.0[2], mat.0[1], mat.0[3]])
}

/// Transpose a matrix: `transpose_mat3(m)`
/// Converts column-major to row-major order (or vice versa)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn transpose_mat3(mat: Mat3) -> Mat3 {
    Mat3([
        mat.0[0], mat.0[3], mat.0[6], mat.0[1], mat.0[4], mat.0[7], mat.0[2], mat.0[5], mat.0[8],
    ])
}

/// Transpose a matrix: `transpose_mat4(m)`
/// Converts column-major to row-major order (or vice versa)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn transpose_mat4(mat: Mat4) -> Mat4 {
    Mat4([
        mat.0[0], mat.0[4], mat.0[8], mat.0[12], mat.0[1], mat.0[5], mat.0[9], mat.0[13], mat.0[2],
        mat.0[6], mat.0[10], mat.0[14], mat.0[3], mat.0[7], mat.0[11], mat.0[15],
    ])
}

/// Determinant of a matrix: `determinant_mat2(m)` -> Scalar
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 0])
)]
pub fn determinant_mat2(mat: Mat2) -> f64 {
    mat.0[0] * mat.0[3] - mat.0[2] * mat.0[1]
}

/// Determinant of a matrix: `determinant_mat3(m)` -> Scalar
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 0, 0])
)]
pub fn determinant_mat3(mat: Mat3) -> f64 {
    let m00 = mat.0[0];
    let m10 = mat.0[1];
    let m20 = mat.0[2];
    let m01 = mat.0[3];
    let m11 = mat.0[4];
    let m21 = mat.0[5];
    let m02 = mat.0[6];
    let m12 = mat.0[7];
    let m22 = mat.0[8];

    m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12) + m02 * (m10 * m21 - m20 * m11)
}

/// Determinant of a matrix: `determinant_mat4(m)` -> Scalar
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Multiply(&[0, 0, 0, 0])
)]
pub fn determinant_mat4(mat: Mat4) -> f64 {
    let m00 = mat.0[0];
    let m10 = mat.0[1];
    let m20 = mat.0[2];
    let m30 = mat.0[3];
    let m01 = mat.0[4];
    let m11 = mat.0[5];
    let m21 = mat.0[6];
    let m31 = mat.0[7];
    let m02 = mat.0[8];
    let m12 = mat.0[9];
    let m22 = mat.0[10];
    let m32 = mat.0[11];
    let m03 = mat.0[12];
    let m13 = mat.0[13];
    let m23 = mat.0[14];
    let m33 = mat.0[15];

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

/// Inverse of a matrix: `inverse_mat2(m)` -> Mat
/// Panics if matrix is singular (determinant = 0)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = Inverse(0)
)]
pub fn inverse_mat2(mat: Mat2) -> Mat2 {
    let det = mat.0[0] * mat.0[3] - mat.0[2] * mat.0[1];
    if det.abs() < 1e-10 {
        panic!("matrix.inverse: matrix is singular (determinant = 0)");
    }
    let inv_det = 1.0 / det;
    Mat2([
        mat.0[3] * inv_det,
        -mat.0[1] * inv_det,
        -mat.0[2] * inv_det,
        mat.0[0] * inv_det,
    ])
}

/// Inverse of a matrix: `inverse_mat3(m)` -> Mat
/// Panics if matrix is singular (determinant = 0)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = Inverse(0)
)]
pub fn inverse_mat3(mat: Mat3) -> Mat3 {
    let m00 = mat.0[0];
    let m10 = mat.0[1];
    let m20 = mat.0[2];
    let m01 = mat.0[3];
    let m11 = mat.0[4];
    let m21 = mat.0[5];
    let m02 = mat.0[6];
    let m12 = mat.0[7];
    let m22 = mat.0[8];

    let det = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12)
        + m02 * (m10 * m21 - m20 * m11);

    if det.abs() < 1e-10 {
        panic!("matrix.inverse: matrix is singular (determinant = 0)");
    }

    let inv_det = 1.0 / det;

    Mat3([
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

/// Inverse of a matrix: `inverse_mat4(m)` -> Mat
/// Panics if matrix is singular (determinant = 0)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Inverse(0)
)]
pub fn inverse_mat4(mat: Mat4) -> Mat4 {
    let m00 = mat.0[0];
    let m10 = mat.0[1];
    let m20 = mat.0[2];
    let m30 = mat.0[3];
    let m01 = mat.0[4];
    let m11 = mat.0[5];
    let m21 = mat.0[6];
    let m31 = mat.0[7];
    let m02 = mat.0[8];
    let m12 = mat.0[9];
    let m22 = mat.0[10];
    let m32 = mat.0[11];
    let m03 = mat.0[12];
    let m13 = mat.0[13];
    let m23 = mat.0[14];
    let m33 = mat.0[15];

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

    Mat4([
        c00, c10, c20, c30, c01, c11, c21, c31, c02, c12, c22, c32, c03, c13, c23, c33,
    ])
}

/// Matrix multiply: `mul_mat2(a, b)` -> Mat2
/// Explicit function for matrix multiplication (alternative to a * b operator)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [
        MatrixDims { rows: DimExact(2), cols: DimExact(2) },
        MatrixDims { rows: DimExact(2), cols: DimExact(2) }
    ],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = Multiply(&[0, 1])
)]
pub fn mul_mat2(a: Mat2, b: Mat2) -> Mat2 {
    use continuum_foundation::matrix_ops::mat2_mul;
    Mat2(mat2_mul(a.0, b.0))
}

/// Matrix multiply: `mul_mat3(a, b)` -> Mat3
/// Explicit function for matrix multiplication (alternative to a * b operator)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [
        MatrixDims { rows: DimExact(3), cols: DimExact(3) },
        MatrixDims { rows: DimExact(3), cols: DimExact(3) }
    ],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = Multiply(&[0, 1])
)]
pub fn mul_mat3(a: Mat3, b: Mat3) -> Mat3 {
    use continuum_foundation::matrix_ops::mat3_mul;
    Mat3(mat3_mul(a.0, b.0))
}

/// Matrix multiply: `mul_mat4(a, b)` -> Mat4
/// Explicit function for matrix multiplication (alternative to a * b operator)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [
        MatrixDims { rows: DimExact(4), cols: DimExact(4) },
        MatrixDims { rows: DimExact(4), cols: DimExact(4) }
    ],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Multiply(&[0, 1])
)]
pub fn mul_mat4(a: Mat4, b: Mat4) -> Mat4 {
    use continuum_foundation::matrix_ops::mat4_mul;
    Mat4(mat4_mul(a.0, b.0))
}

/// Transform vector by matrix: `transform_mat2_vec2(m, v)` -> Vec2
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }, VectorDim(DimExact(2))],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = Multiply(&[0, 1])
)]
pub fn transform_mat2_vec2(m: Mat2, v: [f64; 2]) -> [f64; 2] {
    use continuum_foundation::matrix_ops::mat2_transform;
    mat2_transform(m.0, v)
}

/// Transform vector by matrix: `transform_mat3_vec3(m, v)` -> Vec3
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }, VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = Multiply(&[0, 1])
)]
pub fn transform_mat3_vec3(m: Mat3, v: [f64; 3]) -> [f64; 3] {
    use continuum_foundation::matrix_ops::mat3_transform;
    mat3_transform(m.0, v)
}

/// Transform vector by matrix: `transform_mat4_vec4(m, v)` -> Vec4
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }, VectorDim(DimExact(4))],
    unit_in = [UnitAny, UnitAny],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = Multiply(&[0, 1])
)]
pub fn transform_mat4_vec4(m: Mat4, v: [f64; 4]) -> [f64; 4] {
    use continuum_foundation::matrix_ops::mat4_transform;
    mat4_transform(m.0, v)
}

/// Build rotation matrix from quaternion: `from_quat(q)` -> Mat3
///
/// Input quaternion is in [x, y, z, w] order and will be normalized before conversion.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [VectorDim(DimExact(4))],
    unit_in = [UnitDimensionless],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = Dimensionless
)]
pub fn from_quat(q: [f64; 4]) -> Mat3 {
    use continuum_foundation::Quat;

    // Convert [x, y, z, w] to Quat [w, x, y, z] order
    let quat = Quat([q[3], q[0], q[1], q[2]]);

    // Normalize and convert using quat module (One Truth)
    let normalized = crate::quat::normalize(quat);
    crate::quat::to_mat3(normalized)
}

/// Build rotation matrix from axis-angle: `from_axis_angle(axis, angle)` -> Mat3
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), AnyScalar],
    unit_in = [UnitDimensionless, Angle],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = Dimensionless
)]
pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Mat3 {
    // Normalize axis
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    assert!(
        len > f64::EPSILON,
        "cannot create rotation matrix from zero-length axis: [{}, {}, {}]",
        axis[0],
        axis[1],
        axis[2]
    );
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

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat2(m)` -> Vec2
/// Returns eigenvalues sorted in descending order
/// Note: Only works for symmetric matrices. Non-symmetric matrices will give incorrect results.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvalues_mat2(arr: Mat2) -> [f64; 2] {
    let mat = na::Matrix2::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let mut vals = [eig.eigenvalues[0], eig.eigenvalues[1]];
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    vals
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat3(m)` -> Vec3
/// Returns eigenvalues sorted in descending order
/// Note: Only works for symmetric matrices. Non-symmetric matrices will give incorrect results.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvalues_mat3(arr: Mat3) -> [f64; 3] {
    let mat = na::Matrix3::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let mut vals = [eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]];
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    vals
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat4(m)` -> Vec4
/// Returns eigenvalues sorted in descending order
/// Note: Only works for symmetric matrices. Non-symmetric matrices will give incorrect results.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvalues_mat4(arr: Mat4) -> [f64; 4] {
    let mat = na::Matrix4::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let mut vals = [
        eig.eigenvalues[0],
        eig.eigenvalues[1],
        eig.eigenvalues[2],
        eig.eigenvalues[3],
    ];
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    vals
}

/// Eigenvectors of a symmetric matrix: `eigenvectors_mat2(m)` -> Mat2
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvectors_mat2(arr: Mat2) -> Mat2 {
    let mat = na::Matrix2::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let mut pairs: Vec<_> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    let v0 = na::Vector2::from_iterator(pairs[0].1.iter().cloned());
    let v1 = na::Vector2::from_iterator(pairs[1].1.iter().cloned());
    let result = na::Matrix2::from_columns(&[v0, v1]);
    Mat2(result.as_slice().try_into().unwrap())
}

/// Eigenvectors of a symmetric matrix: `eigenvectors_mat3(m)` -> Mat3
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvectors_mat3(arr: Mat3) -> Mat3 {
    let mat = na::Matrix3::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let mut pairs: Vec<_> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    let v0 = na::Vector3::from_iterator(pairs[0].1.iter().cloned());
    let v1 = na::Vector3::from_iterator(pairs[1].1.iter().cloned());
    let v2 = na::Vector3::from_iterator(pairs[2].1.iter().cloned());
    let result = na::Matrix3::from_columns(&[v0, v1, v2]);
    Mat3(result.as_slice().try_into().unwrap())
}

/// Eigenvectors of a symmetric matrix: `eigenvectors_mat4(m)` -> Mat4
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvectors_mat4(arr: Mat4) -> Mat4 {
    let mat = na::Matrix4::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let mut pairs: Vec<_> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    let v0 = na::Vector4::from_iterator(pairs[0].1.iter().cloned());
    let v1 = na::Vector4::from_iterator(pairs[1].1.iter().cloned());
    let v2 = na::Vector4::from_iterator(pairs[2].1.iter().cloned());
    let v3 = na::Vector4::from_iterator(pairs[3].1.iter().cloned());
    let result = na::Matrix4::from_columns(&[v0, v1, v2, v3]);
    Mat4(result.as_slice().try_into().unwrap())
}

/// SVD - U matrix: `svd_u_mat2(m)` -> Mat2
/// Returns the left singular vectors (U in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_u_mat2(arr: Mat2) -> Mat2 {
    let mat = na::Matrix2::from_column_slice(&arr.0);
    let svd = mat.svd(true, false);
    let u = svd.u.unwrap();
    Mat2(u.as_slice().try_into().unwrap())
}

/// SVD - U matrix: `svd_u_mat3(m)` -> Mat3
/// Returns the left singular vectors (U in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_u_mat3(arr: Mat3) -> Mat3 {
    let mat = na::Matrix3::from_column_slice(&arr.0);
    let svd = mat.svd(true, false);
    let u = svd.u.unwrap();
    Mat3(u.as_slice().try_into().unwrap())
}

/// SVD - U matrix: `svd_u_mat4(m)` -> Mat4
/// Returns the left singular vectors (U in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_u_mat4(arr: Mat4) -> Mat4 {
    let mat = na::Matrix4::from_column_slice(&arr.0);
    let svd = mat.svd(true, false);
    let u = svd.u.unwrap();
    Mat4(u.as_slice().try_into().unwrap())
}

/// SVD - singular values: `svd_s_mat2(m)` -> Vec2
/// Returns the singular values (diagonal of Σ in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_s_mat2(arr: Mat2) -> [f64; 2] {
    let mat = na::Matrix2::from_column_slice(&arr.0);
    let svd = mat.svd(false, false);
    [svd.singular_values[0], svd.singular_values[1]]
}

/// SVD - singular values: `svd_s_mat3(m)` -> Vec3
/// Returns the singular values (diagonal of Σ in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(3)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_s_mat3(arr: Mat3) -> [f64; 3] {
    let mat = na::Matrix3::from_column_slice(&arr.0);
    let svd = mat.svd(false, false);
    [
        svd.singular_values[0],
        svd.singular_values[1],
        svd.singular_values[2],
    ]
}

/// SVD - singular values: `svd_s_mat4(m)` -> Vec4
/// Returns the singular values (diagonal of Σ in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(4)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_s_mat4(arr: Mat4) -> [f64; 4] {
    let mat = na::Matrix4::from_column_slice(&arr.0);
    let svd = mat.svd(false, false);
    [
        svd.singular_values[0],
        svd.singular_values[1],
        svd.singular_values[2],
        svd.singular_values[3],
    ]
}

/// SVD - V^T matrix: `svd_vt_mat2(m)` -> Mat2
/// Returns the transposed right singular vectors (V^T in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_vt_mat2(arr: Mat2) -> Mat2 {
    let mat = na::Matrix2::from_column_slice(&arr.0);
    let svd = mat.svd(false, true);
    let vt = svd.v_t.unwrap();
    Mat2(vt.as_slice().try_into().unwrap())
}

/// SVD - V^T matrix: `svd_vt_mat3(m)` -> Mat3
/// Returns the transposed right singular vectors (V^T in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(3), cols: DimExact(3) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_vt_mat3(arr: Mat3) -> Mat3 {
    let mat = na::Matrix3::from_column_slice(&arr.0);
    let svd = mat.svd(false, true);
    let vt = svd.v_t.unwrap();
    Mat3(vt.as_slice().try_into().unwrap())
}

/// SVD - V^T matrix: `svd_vt_mat4(m)` -> Mat4
/// Returns the transposed right singular vectors (V^T in A = UΣV^T)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn svd_vt_mat4(arr: Mat4) -> Mat4 {
    let mat = na::Matrix4::from_column_slice(&arr.0);
    let svd = mat.svd(false, true);
    let vt = svd.v_t.unwrap();
    Mat4(vt.as_slice().try_into().unwrap())
}

// ============================================================================
// Matrix Construction Functions
// ============================================================================

/// Trace of a matrix (sum of diagonal elements): `trace_mat2(m)` -> Scalar
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn trace_mat2(m: Mat2) -> f64 {
    m.0[0] + m.0[3]
}

/// Trace of a matrix (sum of diagonal elements): `trace_mat3(m)` -> Scalar
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(3), cols: DimExact(3) }],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn trace_mat3(m: Mat3) -> f64 {
    m.0[0] + m.0[4] + m.0[8]
}

/// Trace of a matrix (sum of diagonal elements): `trace_mat4(m)` -> Scalar
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(4), cols: DimExact(4) }],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn trace_mat4(m: Mat4) -> f64 {
    m.0[0] + m.0[5] + m.0[10] + m.0[15]
}

/// Scale matrix: `scale(x, y, z)` -> Mat4
/// Creates a 4x4 scaling transformation matrix.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitDimensionless, UnitDimensionless, UnitDimensionless],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn scale(x: f64, y: f64, z: f64) -> Mat4 {
    // Column-major 4x4:
    // [[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]]
    Mat4([
        x, 0.0, 0.0, 0.0, // column 0
        0.0, y, 0.0, 0.0, // column 1
        0.0, 0.0, z, 0.0, // column 2
        0.0, 0.0, 0.0, 1.0, // column 3
    ])
}

/// Translation matrix: `translation(x, y, z)` -> Mat4
/// Creates a 4x4 translation transformation matrix.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0)],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn translation(x: f64, y: f64, z: f64) -> Mat4 {
    // Column-major 4x4:
    // [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [x, y, z, 1]]
    Mat4([
        1.0, 0.0, 0.0, 0.0, // column 0
        0.0, 1.0, 0.0, 0.0, // column 1
        0.0, 0.0, 1.0, 0.0, // column 2
        x, y, z, 1.0, // column 3
    ])
}

/// Rotation around X axis: `rotation_x(angle)` -> Mat4
/// Creates a 4x4 rotation matrix around the X axis.
/// Angle is in radians.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [Angle],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn rotation_x(angle: f64) -> Mat4 {
    let c = angle.cos();
    let s = angle.sin();
    // Column-major 4x4:
    // [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]
    Mat4([
        1.0, 0.0, 0.0, 0.0, // column 0
        0.0, c, s, 0.0, // column 1
        0.0, -s, c, 0.0, // column 2
        0.0, 0.0, 0.0, 1.0, // column 3
    ])
}

/// Rotation around Y axis: `rotation_y(angle)` -> Mat4
/// Creates a 4x4 rotation matrix around the Y axis.
/// Angle is in radians.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [Angle],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn rotation_y(angle: f64) -> Mat4 {
    let c = angle.cos();
    let s = angle.sin();
    // Column-major 4x4:
    // [[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]]
    Mat4([
        c, 0.0, -s, 0.0, // column 0
        0.0, 1.0, 0.0, 0.0, // column 1
        s, 0.0, c, 0.0, // column 2
        0.0, 0.0, 0.0, 1.0, // column 3
    ])
}

/// Rotation around Z axis: `rotation_z(angle)` -> Mat4
/// Creates a 4x4 rotation matrix around the Z axis.
/// Angle is in radians.
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [Angle],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn rotation_z(angle: f64) -> Mat4 {
    let c = angle.cos();
    let s = angle.sin();
    // Column-major 4x4:
    // [[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Mat4([
        c, s, 0.0, 0.0, // column 0
        -s, c, 0.0, 0.0, // column 1
        0.0, 0.0, 1.0, 0.0, // column 2
        0.0, 0.0, 0.0, 1.0, // column 3
    ])
}

// ============================================================================
// Projection Matrix Functions
// ============================================================================

/// Perspective projection matrix: `perspective(fov_y, aspect, near, far)` -> Mat4
///
/// Creates a perspective projection matrix (OpenGL-style, right-handed, depth [-1, 1]).
///
/// # Arguments
/// * `fov_y` - Vertical field of view in radians
/// * `aspect` - Aspect ratio (width / height)
/// * `near` - Near clipping plane distance (positive)
/// * `far` - Far clipping plane distance (positive)
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [Angle, UnitDimensionless, UnitAny, UnitSameAs(2)],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> Mat4 {
    assert!(
        fov_y > 0.0 && fov_y < std::f64::consts::PI,
        "perspective: fov_y must be in range (0, π), got {}",
        fov_y
    );
    assert!(
        aspect > 0.0,
        "perspective: aspect ratio must be positive, got {}",
        aspect
    );
    assert!(
        near > 0.0,
        "perspective: near plane must be positive, got {}",
        near
    );
    assert!(
        far > near,
        "perspective: far plane ({}) must be greater than near plane ({})",
        far,
        near
    );

    let f = 1.0 / (fov_y / 2.0).tan();
    let nf = 1.0 / (near - far);

    // Column-major OpenGL-style perspective matrix
    Mat4([
        f / aspect,
        0.0,
        0.0,
        0.0, // column 0
        0.0,
        f,
        0.0,
        0.0, // column 1
        0.0,
        0.0,
        (far + near) * nf,
        -1.0, // column 2
        0.0,
        0.0,
        2.0 * far * near * nf,
        0.0, // column 3
    ])
}

/// Orthographic projection matrix: `orthographic(left, right, bottom, top, near, far)` -> Mat4
///
/// Creates an orthographic projection matrix (OpenGL-style, right-handed, depth [-1, 1]).
///
/// # Arguments
/// * `left` - Left clipping plane
/// * `right` - Right clipping plane
/// * `bottom` - Bottom clipping plane
/// * `top` - Top clipping plane
/// * `near` - Near clipping plane
/// * `far` - Far clipping plane
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, AnyScalar, AnyScalar, AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0), UnitSameAs(0), UnitAny, UnitSameAs(4)],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn orthographic(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Mat4 {
    assert!(
        (left - right).abs() > f64::EPSILON,
        "orthographic: left ({}) must not equal right ({})",
        left,
        right
    );
    assert!(
        (bottom - top).abs() > f64::EPSILON,
        "orthographic: bottom ({}) must not equal top ({})",
        bottom,
        top
    );
    assert!(
        (near - far).abs() > f64::EPSILON,
        "orthographic: near ({}) must not equal far ({})",
        near,
        far
    );

    let lr = 1.0 / (left - right);
    let bt = 1.0 / (bottom - top);
    let nf = 1.0 / (near - far);

    // Column-major OpenGL-style orthographic matrix
    Mat4([
        -2.0 * lr,
        0.0,
        0.0,
        0.0, // column 0
        0.0,
        -2.0 * bt,
        0.0,
        0.0, // column 1
        0.0,
        0.0,
        2.0 * nf,
        0.0, // column 2
        (left + right) * lr,
        (top + bottom) * bt,
        (far + near) * nf,
        1.0, // column 3
    ])
}

/// Look-at view matrix: `look_at(eye, target, up)` -> Mat4
///
/// Creates a view matrix that looks from `eye` position towards `target` with `up` vector.
///
/// # Arguments
/// * `eye` - Camera position as Vec3
/// * `target` - Target position to look at as Vec3
/// * `up` - Up direction as Vec3 (usually [0, 1, 0])
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [VectorDim(DimExact(3)), VectorDim(DimExact(3)), VectorDim(DimExact(3))],
    unit_in = [UnitAny, UnitSameAs(0), UnitDimensionless],
    shape_out = ShapeMatrixDims { rows: DimExact(4), cols: DimExact(4) },
    unit_out = Dimensionless
)]
pub fn look_at(eye: [f64; 3], target: [f64; 3], up: [f64; 3]) -> Mat4 {
    // Forward vector (from eye to target, normalized)
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    let f_len = (fx * fx + fy * fy + fz * fz).sqrt();
    assert!(
        f_len > f64::EPSILON,
        "look_at: eye and target positions are identical or too close: eye=[{}, {}, {}], target=[{}, {}, {}]",
        eye[0],
        eye[1],
        eye[2],
        target[0],
        target[1],
        target[2]
    );
    let fx = fx / f_len;
    let fy = fy / f_len;
    let fz = fz / f_len;

    // Right vector (cross product of forward and up, normalized)
    let rx = fy * up[2] - fz * up[1];
    let ry = fz * up[0] - fx * up[2];
    let rz = fx * up[1] - fy * up[0];
    let r_len = (rx * rx + ry * ry + rz * rz).sqrt();
    assert!(
        r_len > f64::EPSILON,
        "look_at: up vector is parallel to forward direction (cannot compute right vector): up=[{}, {}, {}], forward=[{}, {}, {}]",
        up[0],
        up[1],
        up[2],
        fx,
        fy,
        fz
    );
    let rx = rx / r_len;
    let ry = ry / r_len;
    let rz = rz / r_len;

    // Actual up vector (cross product of right and forward)
    let ux = ry * fz - rz * fy;
    let uy = rz * fx - rx * fz;
    let uz = rx * fy - ry * fx;

    // Translation (dot products)
    let tx = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
    let ty = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    let tz = fx * eye[0] + fy * eye[1] + fz * eye[2];

    // Column-major view matrix
    Mat4([
        rx, ux, -fx, 0.0, // column 0
        ry, uy, -fy, 0.0, // column 1
        rz, uz, -fz, 0.0, // column 2
        tx, ty, tz, 1.0, // column 3
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_kernel_registry::{Arity, get_in_namespace, is_known_in};

    #[test]
    fn test_transpose_mat2_registered() {
        assert!(is_known_in("matrix", "transpose_mat2"));
        let desc = get_in_namespace("matrix", "transpose_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_transpose_mat3_registered() {
        assert!(is_known_in("matrix", "transpose_mat3"));
        let desc = get_in_namespace("matrix", "transpose_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_transpose_mat4_registered() {
        assert!(is_known_in("matrix", "transpose_mat4"));
        let desc = get_in_namespace("matrix", "transpose_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_determinant_mat2_registered() {
        assert!(is_known_in("matrix", "determinant_mat2"));
        let desc = get_in_namespace("matrix", "determinant_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_determinant_mat3_registered() {
        assert!(is_known_in("matrix", "determinant_mat3"));
        let desc = get_in_namespace("matrix", "determinant_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_determinant_mat4_registered() {
        assert!(is_known_in("matrix", "determinant_mat4"));
        let desc = get_in_namespace("matrix", "determinant_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_inverse_mat2_registered() {
        assert!(is_known_in("matrix", "inverse_mat2"));
        let desc = get_in_namespace("matrix", "inverse_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_inverse_mat3_registered() {
        assert!(is_known_in("matrix", "inverse_mat3"));
        let desc = get_in_namespace("matrix", "inverse_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_inverse_mat4_registered() {
        assert!(is_known_in("matrix", "inverse_mat4"));
        let desc = get_in_namespace("matrix", "inverse_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_mul_mat2_registered() {
        assert!(is_known_in("matrix", "mul_mat2"));
        let desc = get_in_namespace("matrix", "mul_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_mul_mat3_registered() {
        assert!(is_known_in("matrix", "mul_mat3"));
        let desc = get_in_namespace("matrix", "mul_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_mul_mat4_registered() {
        assert!(is_known_in("matrix", "mul_mat4"));
        let desc = get_in_namespace("matrix", "mul_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_transform_mat2_vec2_registered() {
        assert!(is_known_in("matrix", "transform_mat2_vec2"));
        let desc = get_in_namespace("matrix", "transform_mat2_vec2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_transform_mat3_vec3_registered() {
        assert!(is_known_in("matrix", "transform_mat3_vec3"));
        let desc = get_in_namespace("matrix", "transform_mat3_vec3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_transform_mat4_vec4_registered() {
        assert!(is_known_in("matrix", "transform_mat4_vec4"));
        let desc = get_in_namespace("matrix", "transform_mat4_vec4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_eigenvalues_mat2_registered() {
        assert!(is_known_in("matrix", "eigenvalues_mat2"));
        let desc = get_in_namespace("matrix", "eigenvalues_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_eigenvalues_mat3_registered() {
        assert!(is_known_in("matrix", "eigenvalues_mat3"));
        let desc = get_in_namespace("matrix", "eigenvalues_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_eigenvalues_mat4_registered() {
        assert!(is_known_in("matrix", "eigenvalues_mat4"));
        let desc = get_in_namespace("matrix", "eigenvalues_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_eigenvectors_mat2_registered() {
        assert!(is_known_in("matrix", "eigenvectors_mat2"));
        let desc = get_in_namespace("matrix", "eigenvectors_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_eigenvectors_mat3_registered() {
        assert!(is_known_in("matrix", "eigenvectors_mat3"));
        let desc = get_in_namespace("matrix", "eigenvectors_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_eigenvectors_mat4_registered() {
        assert!(is_known_in("matrix", "eigenvectors_mat4"));
        let desc = get_in_namespace("matrix", "eigenvectors_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_u_mat2_registered() {
        assert!(is_known_in("matrix", "svd_u_mat2"));
        let desc = get_in_namespace("matrix", "svd_u_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_u_mat3_registered() {
        assert!(is_known_in("matrix", "svd_u_mat3"));
        let desc = get_in_namespace("matrix", "svd_u_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_u_mat4_registered() {
        assert!(is_known_in("matrix", "svd_u_mat4"));
        let desc = get_in_namespace("matrix", "svd_u_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_s_mat2_registered() {
        assert!(is_known_in("matrix", "svd_s_mat2"));
        let desc = get_in_namespace("matrix", "svd_s_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_s_mat3_registered() {
        assert!(is_known_in("matrix", "svd_s_mat3"));
        let desc = get_in_namespace("matrix", "svd_s_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_s_mat4_registered() {
        assert!(is_known_in("matrix", "svd_s_mat4"));
        let desc = get_in_namespace("matrix", "svd_s_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_vt_mat2_registered() {
        assert!(is_known_in("matrix", "svd_vt_mat2"));
        let desc = get_in_namespace("matrix", "svd_vt_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_vt_mat3_registered() {
        assert!(is_known_in("matrix", "svd_vt_mat3"));
        let desc = get_in_namespace("matrix", "svd_vt_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_svd_vt_mat4_registered() {
        assert!(is_known_in("matrix", "svd_vt_mat4"));
        let desc = get_in_namespace("matrix", "svd_vt_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_trace_mat2_registered() {
        assert!(is_known_in("matrix", "trace_mat2"));
        let desc = get_in_namespace("matrix", "trace_mat2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_trace_mat3_registered() {
        assert!(is_known_in("matrix", "trace_mat3"));
        let desc = get_in_namespace("matrix", "trace_mat3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

    #[test]
    fn test_trace_mat4_registered() {
        assert!(is_known_in("matrix", "trace_mat4"));
        let desc = get_in_namespace("matrix", "trace_mat4").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
    }

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
        let m = Mat2([1.0, 2.0, 3.0, 4.0]);
        let result = transpose_mat2(m);
        assert_eq!(result.0, [1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_mat3() {
        let m = Mat3([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = transpose_mat3(m);
        assert_eq!(result.0, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_determinant_mat2() {
        let m = Mat2([1.0, 2.0, 3.0, 4.0]);
        let det = determinant_mat2(m);
        // det = 1*4 - 3*2 = 4 - 6 = -2
        assert_eq!(det, -2.0);
    }

    #[test]
    fn test_determinant_mat3_identity() {
        let m = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let det = determinant_mat3(m);
        assert_eq!(det, 1.0);
    }

    #[test]
    fn test_inverse_mat2() {
        let m = Mat2([1.0, 0.0, 0.0, 1.0]); // Identity
        let inv = inverse_mat2(m);
        assert_eq!(inv.0, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_inverse_mat2_scaled() {
        let m = Mat2([2.0, 0.0, 0.0, 2.0]); // 2*Identity
        let inv = inverse_mat2(m);
        assert_eq!(inv.0, [0.5, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_inverse_mat3_identity() {
        let m = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let inv = inverse_mat3(m);
        assert_eq!(inv.0, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn test_inverse_mat2_singular() {
        let m = Mat2([1.0, 2.0, 2.0, 4.0]); // Singular matrix (rows are linearly dependent)
        let _ = inverse_mat2(m);
    }

    #[test]
    fn test_mul_mat2_identity() {
        let identity = Mat2([1.0, 0.0, 0.0, 1.0]);
        let m = Mat2([1.0, 2.0, 3.0, 4.0]);
        let result = mul_mat2(identity, Mat2([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(result.0, m.0);
    }

    #[test]
    fn test_mul_mat3() {
        let a = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]); // Identity
        let b = Mat3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]); // 2*Identity
        let result = mul_mat3(a, b);
        assert_eq!(result.0, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_transform_vec2() {
        let identity = Mat2([1.0, 0.0, 0.0, 1.0]);
        let v = [3.0, 4.0];
        let result = transform_mat2_vec2(identity, v);
        assert_eq!(result, [3.0, 4.0]);
    }

    #[test]
    fn test_transform_vec3_scale() {
        let scale = Mat3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        let v = [1.0, 2.0, 3.0];
        let result = transform_mat3_vec3(scale, v);
        assert_eq!(result, [2.0, 4.0, 6.0]);
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
    fn test_from_quat_registered() {
        assert!(is_known_in("matrix", "from_quat"));
    }

    #[test]
    fn test_from_axis_angle_registered() {
        assert!(is_known_in("matrix", "from_axis_angle"));
    }

    #[test]
    fn test_eigenvalues_mat2_identity() {
        let identity = Mat2([1.0, 0.0, 0.0, 1.0]);
        let result = eigenvalues_mat2(identity);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_mat3_diagonal() {
        // Diagonal matrix [[2, 0, 0], [0, 3, 0], [0, 0, 1]]
        let m = Mat3([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        let result = eigenvalues_mat3(m);
        // Eigenvalues should be [3, 2, 1] (sorted descending)
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvectors_mat2_identity() {
        let identity = Mat2([1.0, 0.0, 0.0, 1.0]);
        let result = eigenvectors_mat2(identity);
        // Identity matrix has any orthonormal basis as eigenvectors
        // Just check it's a valid matrix
        let _ = result;
    }

    #[test]
    fn test_svd_identity_mat2() {
        let identity = Mat2([1.0, 0.0, 0.0, 1.0]);

        // Singular values should be [1, 1]
        let s = svd_s_mat2(identity);
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 1.0).abs() < 1e-10);

        // U should be approximately identity (or a rotation)
        let u = svd_u_mat2(Mat2([1.0, 0.0, 0.0, 1.0]));
        assert!(matches!(u, Mat2(_)));
    }

    #[test]
    fn test_svd_diagonal_mat3() {
        // Diagonal matrix [[2, 0, 0], [0, 3, 0], [0, 0, 1]]
        let m = Mat3([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);

        // Singular values should be [3, 2, 1] (sorted descending by nalgebra)
        let s = svd_s_mat3(m);
        assert!((s[0] - 3.0).abs() < 1e-10);
        assert!((s[1] - 2.0).abs() < 1e-10);
        assert!((s[2] - 1.0).abs() < 1e-10);
    }

    // ============================================================================
    // Matrix Construction Tests
    // ============================================================================

    #[test]
    fn test_trace_mat2() {
        // Mat2 in column-major: [m00, m10, m01, m11]
        // Diagonal: m00 + m11 = 1 + 4 = 5
        let m = Mat2([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(trace_mat2(m), 5.0);
    }

    #[test]
    fn test_trace_mat3_identity() {
        let m = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(trace_mat3(m), 3.0);
    }

    #[test]
    fn test_trace_mat4_identity() {
        let m = Mat4([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(trace_mat4(m), 4.0);
    }

    #[test]
    fn test_scale_uniform() {
        let m = scale(2.0, 2.0, 2.0);
        // Diagonal should be [2, 2, 2, 1]
        assert_eq!(m.0[0], 2.0); // m00
        assert_eq!(m.0[5], 2.0); // m11
        assert_eq!(m.0[10], 2.0); // m22
        assert_eq!(m.0[15], 1.0); // m33
    }

    #[test]
    fn test_scale_non_uniform() {
        let m = scale(1.0, 2.0, 3.0);
        assert_eq!(m.0[0], 1.0);
        assert_eq!(m.0[5], 2.0);
        assert_eq!(m.0[10], 3.0);
    }

    #[test]
    fn test_scale_registered() {
        assert!(is_known_in("matrix", "scale"));
    }

    #[test]
    fn test_translation_basic() {
        let m = translation(10.0, 20.0, 30.0);
        // Translation is in column 3: [x, y, z, 1]
        assert_eq!(m.0[12], 10.0); // tx
        assert_eq!(m.0[13], 20.0); // ty
        assert_eq!(m.0[14], 30.0); // tz
        assert_eq!(m.0[15], 1.0); // w
        // Identity in upper-left 3x3
        assert_eq!(m.0[0], 1.0);
        assert_eq!(m.0[5], 1.0);
        assert_eq!(m.0[10], 1.0);
    }

    #[test]
    fn test_translation_registered() {
        assert!(is_known_in("matrix", "translation"));
    }

    #[test]
    fn test_rotation_x_zero() {
        let m = rotation_x(0.0);
        // Should be identity
        assert!((m.0[0] - 1.0).abs() < 1e-10);
        assert!((m.0[5] - 1.0).abs() < 1e-10);
        assert!((m.0[10] - 1.0).abs() < 1e-10);
        assert!((m.0[15] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_x_90_deg() {
        let m = rotation_x(std::f64::consts::FRAC_PI_2);
        // Rotating (0, 1, 0) by 90 deg around X should give (0, 0, 1)
        // Using matrix.transform would verify this
        assert!((m.0[5] - 0.0).abs() < 1e-10); // cos(90) = 0
        assert!((m.0[6] - 1.0).abs() < 1e-10); // sin(90) = 1
    }

    #[test]
    fn test_rotation_x_registered() {
        assert!(is_known_in("matrix", "rotation_x"));
    }

    #[test]
    fn test_rotation_y_zero() {
        let m = rotation_y(0.0);
        assert!((m.0[0] - 1.0).abs() < 1e-10);
        assert!((m.0[5] - 1.0).abs() < 1e-10);
        assert!((m.0[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_y_registered() {
        assert!(is_known_in("matrix", "rotation_y"));
    }

    #[test]
    fn test_rotation_z_zero() {
        let m = rotation_z(0.0);
        assert!((m.0[0] - 1.0).abs() < 1e-10);
        assert!((m.0[5] - 1.0).abs() < 1e-10);
        assert!((m.0[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_z_90_deg() {
        let m = rotation_z(std::f64::consts::FRAC_PI_2);
        // cos(90) = 0, sin(90) = 1
        assert!((m.0[0] - 0.0).abs() < 1e-10);
        assert!((m.0[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_z_registered() {
        assert!(is_known_in("matrix", "rotation_z"));
    }

    // ============================================================================
    // Projection Matrix Tests
    // ============================================================================

    #[test]
    fn test_perspective_basic() {
        let fov = std::f64::consts::FRAC_PI_4; // 45 degrees
        let m = perspective(fov, 1.0, 0.1, 100.0);
        // Check that it produces a valid perspective matrix
        // m.0[15] should be 0 for perspective (not 1 like affine transforms)
        assert_eq!(m.0[15], 0.0);
        // m.0[11] should be -1 (perspective divide flag)
        assert_eq!(m.0[11], -1.0);
    }

    #[test]
    fn test_perspective_aspect_ratio() {
        let fov = std::f64::consts::FRAC_PI_2; // 90 degrees
        let m1 = perspective(fov, 1.0, 0.1, 100.0);
        let m2 = perspective(fov, 2.0, 0.1, 100.0);
        // With aspect 2.0, horizontal FOV is wider
        // m.0[0] (x scaling) should be smaller for wider aspect
        assert!(m2.0[0] < m1.0[0]);
    }

    #[test]
    fn test_perspective_registered() {
        assert!(is_known_in("matrix", "perspective"));
    }

    #[test]
    fn test_orthographic_basic() {
        let m = orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        // Check that it produces a valid orthographic matrix
        // m.0[15] should be 1.0 for orthographic
        assert_eq!(m.0[15], 1.0);
        // m.0[11] should be 0.0 (no perspective divide)
        assert_eq!(m.0[11], 0.0);
    }

    #[test]
    fn test_orthographic_symmetric() {
        let m = orthographic(-10.0, 10.0, -10.0, 10.0, 1.0, 100.0);
        // For symmetric bounds, translation should be 0
        assert!((m.0[12] - 0.0).abs() < 1e-10); // tx
        assert!((m.0[13] - 0.0).abs() < 1e-10); // ty
    }

    #[test]
    fn test_orthographic_registered() {
        assert!(is_known_in("matrix", "orthographic"));
    }

    #[test]
    fn test_look_at_basic() {
        let eye = [0.0, 0.0, 5.0];
        let target = [0.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];
        let m = look_at(eye, target, up);
        // Looking down -Z axis from (0,0,5)
        // m.0[15] should be 1.0 (view matrix is affine)
        assert_eq!(m.0[15], 1.0);
    }

    #[test]
    fn test_look_at_identity_like() {
        // Eye at origin looking down -Z with Y up should be close to identity
        // (except for sign of Z column)
        let eye = [0.0, 0.0, 0.0];
        let target = [0.0, 0.0, -1.0];
        let up = [0.0, 1.0, 0.0];
        let m = look_at(eye, target, up);

        // Right vector should be (1, 0, 0)
        assert!((m.0[0] - 1.0).abs() < 1e-10);
        // Up vector should be (0, 1, 0)
        assert!((m.0[5] - 1.0).abs() < 1e-10);
        // Forward (negated) should be (0, 0, 1)
        assert!((m.0[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_look_at_registered() {
        assert!(is_known_in("matrix", "look_at"));
    }

    // === Zero-Norm Guard Tests ===

    #[test]
    #[should_panic(expected = "quat.normalize requires non-zero quaternion")]
    fn test_from_quat_zero() {
        let _ = from_quat([0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "cannot create rotation matrix from zero-length axis")]
    fn test_from_axis_angle_zero() {
        let _ = from_axis_angle([0.0, 0.0, 0.0], 1.0);
    }

    #[test]
    #[should_panic(expected = "eye and target positions are identical")]
    fn test_look_at_identical_eye_target() {
        let pos = [1.0, 2.0, 3.0];
        let _ = look_at(pos, pos, [0.0, 1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "up vector is parallel to forward direction")]
    fn test_look_at_parallel_up() {
        // Up and forward are parallel (both along Z axis)
        let _ = look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "fov_y must be in range")]
    fn test_perspective_invalid_fov_zero() {
        let _ = perspective(0.0, 1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "fov_y must be in range")]
    fn test_perspective_invalid_fov_too_large() {
        let _ = perspective(std::f64::consts::PI, 1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "aspect ratio must be positive")]
    fn test_perspective_invalid_aspect() {
        let _ = perspective(1.0, -1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "near plane must be positive")]
    fn test_perspective_invalid_near() {
        let _ = perspective(1.0, 1.0, -0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "far plane")]
    fn test_perspective_near_greater_than_far() {
        let _ = perspective(1.0, 1.0, 100.0, 0.1);
    }

    #[test]
    #[should_panic(expected = "left")]
    fn test_orthographic_left_equals_right() {
        let _ = orthographic(1.0, 1.0, 0.0, 1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "bottom")]
    fn test_orthographic_bottom_equals_top() {
        let _ = orthographic(0.0, 1.0, 1.0, 1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "near")]
    fn test_orthographic_near_equals_far() {
        let _ = orthographic(0.0, 1.0, 0.0, 1.0, 1.0, 1.0);
    }
}
