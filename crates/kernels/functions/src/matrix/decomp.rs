//! Matrix Decomposition Operations
//!
//! Eigenvalues, eigenvectors, and Singular Value Decomposition (SVD).

use continuum_foundation::{Mat2, Mat3, Mat4};
use continuum_kernel_macros::kernel_fn;
use nalgebra as na;

use super::utils::{from_na_mat2, from_na_mat3, from_na_mat4, to_na_mat2, to_na_mat3, to_na_mat4};

/// Sort eigenvalues in descending order by magnitude
fn sort_eigenvalues_desc<const N: usize>(mut vals: [f64; N]) -> [f64; N] {
    vals.sort_by(|a, b| b.partial_cmp(a).expect("eigenvalues must not be NaN"));
    vals
}

/// Computes the eigenvalues of a symmetric 2x2 matrix.
///
/// Returns eigenvalues sorted in descending order.
///
/// Uses a direct analytic formula (quadratic characteristic equation) for O(1) performance
/// with no allocation overhead. For a symmetric 2x2 matrix `[[a, b], [b, c]]`:
/// - Characteristic polynomial: `λ² - (a+c)λ + (ac-b²) = 0`
/// - Eigenvalues: `λ = (a+c)/2 ± sqrt((a+c)²/4 - (ac-b²))`
///
/// # Parameters
/// - `m`: The symmetric 2x2 matrix.
///
/// # Returns
/// A 2D vector containing the eigenvalues `[λ1, λ2]` where `λ1 >= λ2`.
///
/// # Panics
/// Panics if the input matrix is not symmetric (within `1e-12` tolerance).
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeVectorDim(DimExact(2)),
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvalues_mat2(m: Mat2) -> [f64; 2] {
    // Validate symmetry
    const SYM_TOL: f64 = 1e-12;
    assert!(
        (m.0[1] - m.0[2]).abs() < SYM_TOL,
        "eigenvalues_mat2: matrix must be symmetric (m10 != m01)"
    );

    let a = m.0[0];
    let b = m.0[1];
    let c = m.0[3];

    let tr = a + c;
    let det = a * c - b * b;

    let gap = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let l1 = tr / 2.0 + gap;
    let l2 = tr / 2.0 - gap;

    [l1, l2]
}

/// Computes the eigenvalues of a symmetric 3x3 matrix.
///
/// Returns eigenvalues sorted in descending order.
///
/// # Parameters
/// - `arr`: The symmetric 3x3 matrix.
///
/// # Returns
/// A 3D vector containing the eigenvalues `[λ1, λ2, λ3]` where `λ1 >= λ2 >= λ3`.
///
/// # Panics
/// Panics if the input matrix is not symmetric (within `1e-12` tolerance).
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
    // Validate symmetry
    const SYM_TOL: f64 = 1e-12;
    assert!(
        (arr.0[1] - arr.0[3]).abs() < SYM_TOL,
        "eigenvalues_mat3: matrix must be symmetric (m10 != m01)"
    );
    assert!(
        (arr.0[2] - arr.0[6]).abs() < SYM_TOL,
        "eigenvalues_mat3: matrix must be symmetric (m20 != m02)"
    );
    assert!(
        (arr.0[5] - arr.0[7]).abs() < SYM_TOL,
        "eigenvalues_mat3: matrix must be symmetric (m21 != m12)"
    );

    let mat = na::Matrix3::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let vals: [f64; 3] = eig
        .eigenvalues
        .as_slice()
        .try_into()
        .expect("3x3 matrix must have exactly 3 eigenvalues");
    sort_eigenvalues_desc(vals)
}

/// Computes the eigenvalues of a symmetric 4x4 matrix.
///
/// Returns eigenvalues sorted in descending order.
///
/// # Parameters
/// - `arr`: The symmetric 4x4 matrix.
///
/// # Returns
/// A 4D vector containing the eigenvalues `[λ1, λ2, λ3, λ4]` where `λ1 >= λ2 >= λ3 >= λ4`.
///
/// # Panics
/// Panics if the input matrix is not symmetric (within `1e-12` tolerance).
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
    // Validate symmetry
    const SYM_TOL: f64 = 1e-12;
    assert!(
        (arr.0[1] - arr.0[4]).abs() < SYM_TOL,
        "eigenvalues_mat4: matrix must be symmetric (m10 != m01)"
    );
    assert!(
        (arr.0[2] - arr.0[8]).abs() < SYM_TOL,
        "eigenvalues_mat4: matrix must be symmetric (m20 != m02)"
    );
    assert!(
        (arr.0[3] - arr.0[12]).abs() < SYM_TOL,
        "eigenvalues_mat4: matrix must be symmetric (m30 != m03)"
    );
    assert!(
        (arr.0[6] - arr.0[9]).abs() < SYM_TOL,
        "eigenvalues_mat4: matrix must be symmetric (m21 != m12)"
    );
    assert!(
        (arr.0[7] - arr.0[13]).abs() < SYM_TOL,
        "eigenvalues_mat4: matrix must be symmetric (m31 != m13)"
    );
    assert!(
        (arr.0[11] - arr.0[14]).abs() < SYM_TOL,
        "eigenvalues_mat4: matrix must be symmetric (m32 != m23)"
    );

    let mat = na::Matrix4::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();
    let vals: [f64; 4] = eig
        .eigenvalues
        .as_slice()
        .try_into()
        .expect("4x4 matrix must have exactly 4 eigenvalues");
    sort_eigenvalues_desc(vals)
}

/// Computes the eigenvectors of a symmetric 2x2 matrix.
///
/// Returns eigenvectors as columns of a matrix, ordered by their corresponding
/// eigenvalues (descending).
///
/// # Parameters
/// - `m`: The symmetric 2x2 matrix.
///
/// # Returns
/// A 2x2 matrix where columns are the eigenvectors.
///
/// # Panics
/// Panics if the input matrix is not symmetric (within `1e-12` tolerance).
#[kernel_fn(
    namespace = "matrix",
    category = "matrix",
    purity = Pure,
    shape_in = [MatrixDims { rows: DimExact(2), cols: DimExact(2) }],
    unit_in = [UnitAny],
    shape_out = ShapeMatrixDims { rows: DimExact(2), cols: DimExact(2) },
    unit_out = UnitDerivSameAs(0)
)]
pub fn eigenvectors_mat2(m: Mat2) -> Mat2 {
    // Validate symmetry
    const SYM_TOL: f64 = 1e-12;
    assert!(
        (m.0[1] - m.0[2]).abs() < SYM_TOL,
        "eigenvectors_mat2: matrix must be symmetric (m10 != m01)"
    );

    let mat = na::Matrix2::from_column_slice(&m.0);
    let eig = mat.symmetric_eigen();

    // Sort eigenvectors by eigenvalue
    let mut pairs: Vec<_> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).expect("eigenvalues must not be NaN"));

    let v0 = na::Vector2::from_iterator(pairs[0].1.iter().cloned());
    let v1 = na::Vector2::from_iterator(pairs[1].1.iter().cloned());
    let result = na::Matrix2::from_columns(&[v0, v1]);
    Mat2(
        result
            .as_slice()
            .try_into()
            .expect("2x2 matrix must convert to [f64; 4]"),
    )
}

/// Computes the eigenvectors of a symmetric 3x3 matrix.
///
/// Returns eigenvectors as columns of a matrix, ordered by their corresponding
/// eigenvalues (descending).
///
/// # Parameters
/// - `arr`: The symmetric 3x3 matrix.
///
/// # Returns
/// A 3x3 matrix where columns are the eigenvectors.
///
/// # Panics
/// Panics if the input matrix is not symmetric (within `1e-12` tolerance).
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
    // Validate symmetry
    const SYM_TOL: f64 = 1e-12;
    assert!(
        (arr.0[1] - arr.0[3]).abs() < SYM_TOL,
        "eigenvectors_mat3: matrix must be symmetric (m10 != m01)"
    );
    assert!(
        (arr.0[2] - arr.0[6]).abs() < SYM_TOL,
        "eigenvectors_mat3: matrix must be symmetric (m20 != m02)"
    );
    assert!(
        (arr.0[5] - arr.0[7]).abs() < SYM_TOL,
        "eigenvectors_mat3: matrix must be symmetric (m21 != m12)"
    );

    let mat = na::Matrix3::from_column_slice(&arr.0);
    let eig = mat.symmetric_eigen();

    let mut pairs: Vec<_> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).expect("eigenvalues must not be NaN"));

    let v0 = na::Vector3::from_iterator(pairs[0].1.iter().cloned());
    let v1 = na::Vector3::from_iterator(pairs[1].1.iter().cloned());
    let v2 = na::Vector3::from_iterator(pairs[2].1.iter().cloned());
    let result = na::Matrix3::from_columns(&[v0, v1, v2]);
    Mat3(
        result
            .as_slice()
            .try_into()
            .expect("3x3 matrix must convert to [f64; 9]"),
    )
}

/// Computes the eigenvectors of a symmetric 4x4 matrix.
///
/// Returns eigenvectors as columns of a matrix, ordered by their corresponding
/// eigenvalues (descending).
///
/// # Parameters
/// - `arr`: The symmetric 4x4 matrix.
///
/// # Returns
/// A 4x4 matrix where columns are the eigenvectors.
///
/// # Panics
/// Panics if the input matrix is not symmetric (within `1e-12` tolerance).
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
    // Validate symmetry
    const SYM_TOL: f64 = 1e-12;
    assert!(
        (arr.0[1] - arr.0[4]).abs() < SYM_TOL,
        "eigenvectors_mat4: matrix must be symmetric (m10 != m01)"
    );
    assert!(
        (arr.0[2] - arr.0[8]).abs() < SYM_TOL,
        "eigenvectors_mat4: matrix must be symmetric (m20 != m02)"
    );
    assert!(
        (arr.0[3] - arr.0[12]).abs() < SYM_TOL,
        "eigenvectors_mat4: matrix must be symmetric (m30 != m03)"
    );
    assert!(
        (arr.0[6] - arr.0[9]).abs() < SYM_TOL,
        "eigenvectors_mat4: matrix must be symmetric (m21 != m12)"
    );
    assert!(
        (arr.0[7] - arr.0[13]).abs() < SYM_TOL,
        "eigenvectors_mat4: matrix must be symmetric (m31 != m13)"
    );
    assert!(
        (arr.0[11] - arr.0[14]).abs() < SYM_TOL,
        "eigenvectors_mat4: matrix must be symmetric (m32 != m23)"
    );

    let mat = na::Matrix4::from_column_slice(&arr.0);

    let eig = mat.symmetric_eigen();
    let mut pairs: Vec<_> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).expect("eigenvalues must not be NaN"));

    let v0 = na::Vector4::from_iterator(pairs[0].1.iter().cloned());
    let v1 = na::Vector4::from_iterator(pairs[1].1.iter().cloned());
    let v2 = na::Vector4::from_iterator(pairs[2].1.iter().cloned());
    let v3 = na::Vector4::from_iterator(pairs[3].1.iter().cloned());
    let result = na::Matrix4::from_columns(&[v0, v1, v2, v3]);
    Mat4(
        result
            .as_slice()
            .try_into()
            .expect("4x4 matrix must convert to [f64; 16]"),
    )
}

/// Computes the left singular vectors (U) of a 2x2 matrix.
///
/// In the decomposition `A = UΣV^T`, this returns `U`.
/// Columns of `U` are orthonormal.
///
/// # Parameters
/// - `arr`: The 2x2 matrix.
///
/// # Returns
/// A 2x2 matrix containing the left singular vectors as columns.
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both `(u, v)` and `(-u, -v)` are valid.
///   The returned signs are implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///
/// # Panics
/// Panics if the SVD failed to converge or compute `U`.
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
    let mat = to_na_mat2(&arr);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .expect("svd_u_mat2: SVD failed to converge or compute U");
    from_na_mat2(u)
}

/// Computes the left singular vectors (U) of a 3x3 matrix.
///
/// In the decomposition `A = UΣV^T`, this returns `U`.
/// Columns of `U` are orthonormal.
///
/// # Parameters
/// - `arr`: The 3x3 matrix.
///
/// # Returns
/// A 3x3 matrix containing the left singular vectors as columns.
///
/// # Determinism Warning
/// - **Sign ambiguity**: See [`svd_u_mat2`].
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///
/// # Panics
/// Panics if the SVD failed to converge or compute `U`.
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
    let mat = to_na_mat3(&arr);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .expect("svd_u_mat3: SVD failed to converge or compute U");
    from_na_mat3(u)
}

/// Computes the left singular vectors (U) of a 4x4 matrix.
///
/// In the decomposition `A = UΣV^T`, this returns `U`.
/// Columns of `U` are orthonormal.
///
/// # Parameters
/// - `arr`: The 4x4 matrix.
///
/// # Returns
/// A 4x4 matrix containing the left singular vectors as columns.
///
/// # Determinism Warning
/// - **Sign ambiguity**: See [`svd_u_mat2`].
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///
/// # Panics
/// Panics if the SVD failed to converge or compute `U`.
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
    let mat = to_na_mat4(&arr);
    let svd = mat.svd(true, false);
    let u = svd
        .u
        .expect("svd_u_mat4: SVD failed to converge or compute U");
    from_na_mat4(u)
}

/// Computes the singular values (Σ) of a 2x2 matrix.
///
/// Returns the diagonal elements of `Σ` in the decomposition `A = UΣV^T`.
/// Singular values are always non-negative and returned in descending order.
///
/// # Parameters
/// - `arr`: The 2x2 matrix.
///
/// # Returns
/// A 2D vector containing the singular values `[σ1, σ2]` where `σ1 >= σ2 >= 0`.
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
    let mat = to_na_mat2(&arr);
    let svd = mat.svd(false, false);
    [svd.singular_values[0], svd.singular_values[1]]
}

/// Computes the singular values (Σ) of a 3x3 matrix.
///
/// Returns the diagonal elements of `Σ` in the decomposition `A = UΣV^T`.
/// Singular values are always non-negative and returned in descending order.
///
/// # Parameters
/// - `arr`: The 3x3 matrix.
///
/// # Returns
/// A 3D vector containing the singular values `[σ1, σ2, σ3]` where `σ1 >= σ2 >= σ3 >= 0`.
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
    let mat = to_na_mat3(&arr);
    let svd = mat.svd(false, false);
    [
        svd.singular_values[0],
        svd.singular_values[1],
        svd.singular_values[2],
    ]
}

/// Computes the singular values (Σ) of a 4x4 matrix.
///
/// Returns the diagonal elements of `Σ` in the decomposition `A = UΣV^T`.
/// Singular values are always non-negative and returned in descending order.
///
/// # Parameters
/// - `arr`: The 4x4 matrix.
///
/// # Returns
/// A 4D vector containing the singular values `[σ1, σ2, σ3, σ4]` where `σ1 >= σ2 >= σ3 >= σ4 >= 0`.
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
    let mat = to_na_mat4(&arr);
    let svd = mat.svd(false, false);
    [
        svd.singular_values[0],
        svd.singular_values[1],
        svd.singular_values[2],
        svd.singular_values[3],
    ]
}

/// Computes the transposed right singular vectors (V^T) of a 2x2 matrix.
///
/// In the decomposition `A = UΣV^T`, this returns `V^T`.
/// Rows of `V^T` are orthonormal.
///
/// # Parameters
/// - `arr`: The 2x2 matrix.
///
/// # Returns
/// A 2x2 matrix containing the transposed right singular vectors.
///
/// # Determinism Warning
/// - **Sign ambiguity**: See [`svd_u_mat2`].
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///
/// # Panics
/// Panics if the SVD failed to converge or compute `V^T`.
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
    let mat = to_na_mat2(&arr);
    let svd = mat.svd(false, true);
    let vt = svd
        .v_t
        .expect("svd_vt_mat2: SVD failed to converge or compute V_T");
    from_na_mat2(vt)
}

/// Computes the transposed right singular vectors (V^T) of a 3x3 matrix.
///
/// In the decomposition `A = UΣV^T`, this returns `V^T`.
/// Rows of `V^T` are orthonormal.
///
/// # Parameters
/// - `arr`: The 3x3 matrix.
///
/// # Returns
/// A 3x3 matrix containing the transposed right singular vectors.
///
/// # Determinism Warning
/// - **Sign ambiguity**: See [`svd_u_mat2`].
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///
/// # Panics
/// Panics if the SVD failed to converge or compute `V^T`.
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
    let mat = to_na_mat3(&arr);
    let svd = mat.svd(false, true);
    let vt = svd
        .v_t
        .expect("svd_vt_mat3: SVD failed to converge or compute V_T");
    from_na_mat3(vt)
}

/// Computes the transposed right singular vectors (V^T) of a 4x4 matrix.
///
/// In the decomposition `A = UΣV^T`, this returns `V^T`.
/// Rows of `V^T` are orthonormal.
///
/// # Parameters
/// - `arr`: The 4x4 matrix.
///
/// # Returns
/// A 4x4 matrix containing the transposed right singular vectors.
///
/// # Determinism Warning
/// - **Sign ambiguity**: See [`svd_u_mat2`].
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///
/// # Panics
/// Panics if the SVD failed to converge or compute `V^T`.
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
    let mat = to_na_mat4(&arr);
    let svd = mat.svd(false, true);
    let vt = svd
        .v_t
        .expect("svd_vt_mat4: SVD failed to converge or compute V_T");
    from_na_mat4(vt)
}
