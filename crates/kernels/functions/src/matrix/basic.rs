//! Basic Matrix Operations
//!
//! Identity, transpose, determinant, inverse, multiplication, transformation, and trace operations.

use continuum_foundation::{Mat2, Mat3, Mat4};
use continuum_kernel_macros::kernel_fn;

use super::utils::{from_na_mat4, to_na_mat4};

/// Compute Frobenius norm of a 2x2 matrix (sqrt of sum of squares of all elements)
/// Used for scale-invariant singularity checks
fn frobenius_norm_mat2(m: &Mat2) -> f64 {
    m.0.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute Frobenius norm of a 3x3 matrix (sqrt of sum of squares of all elements)
/// Used for scale-invariant singularity checks
fn frobenius_norm_mat3(m: &Mat3) -> f64 {
    m.0.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Returns a 2x2 identity matrix.
///
/// The identity matrix has ones on the main diagonal and zeros elsewhere.
///
/// # Returns
/// A [`Mat2`] containing the identity matrix in column-major order: `[1.0, 0.0, 0.0, 1.0]`.
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

/// Returns a 3x3 identity matrix.
///
/// The identity matrix has ones on the main diagonal and zeros elsewhere.
///
/// # Returns
/// A [`Mat3`] containing the identity matrix in column-major order.
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

/// Returns a 4x4 identity matrix.
///
/// The identity matrix has ones on the main diagonal and zeros elsewhere.
///
/// # Returns
/// A [`Mat4`] containing the identity matrix in column-major order.
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

/// Computes the transpose of a 2x2 matrix.
///
/// Swaps the rows and columns of the input matrix.
///
/// # Parameters
/// - `mat`: The 2x2 matrix to transpose.
///
/// # Returns
/// The transposed 2x2 matrix.
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

/// Computes the transpose of a 3x3 matrix.
///
/// Swaps the rows and columns of the input matrix.
///
/// # Parameters
/// - `mat`: The 3x3 matrix to transpose.
///
/// # Returns
/// The transposed 3x3 matrix.
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

/// Computes the transpose of a 4x4 matrix.
///
/// Swaps the rows and columns of the input matrix.
///
/// # Parameters
/// - `mat`: The 4x4 matrix to transpose.
///
/// # Returns
/// The transposed 4x4 matrix.
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

/// Computes the determinant of a 2x2 matrix.
///
/// # Parameters
/// - `mat`: The 2x2 matrix to evaluate.
///
/// # Returns
/// The determinant as a scalar value.
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

/// Computes the determinant of a 3x3 matrix.
///
/// # Parameters
/// - `mat`: The 3x3 matrix to evaluate.
///
/// # Returns
/// The determinant as a scalar value.
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

/// Computes the determinant of a 4x4 matrix.
///
/// # Parameters
/// - `mat`: The 4x4 matrix to evaluate.
///
/// # Returns
/// The determinant as a scalar value.
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
    let det = determinant_mat2(Mat2(mat.0));
    let norm = frobenius_norm_mat2(&mat);
    let eps = f64::EPSILON * 100.0 * norm * norm;

    if det.abs() < eps {
        panic!(
            "matrix.inverse: matrix is singular (determinant = {}, tolerance = {})",
            det, eps
        );
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

    let det = determinant_mat3(Mat3(mat.0));
    let norm = frobenius_norm_mat3(&mat);
    let eps = f64::EPSILON * 100.0 * norm * norm * norm;

    if det.abs() < eps {
        panic!(
            "matrix.inverse: matrix is singular (determinant = {}, tolerance = {})",
            det, eps
        );
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
    let m = to_na_mat4(&mat);

    match m.try_inverse() {
        Some(inv) => from_na_mat4(inv),
        None => {
            let det = determinant_mat4(mat);
            panic!("matrix.inverse: matrix is singular (determinant = {})", det);
        }
    }
}

/// Matrix multiply: `mul_mat2(a, b)` -> Mat2
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
