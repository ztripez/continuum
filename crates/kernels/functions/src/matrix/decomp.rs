//! Matrix Decomposition Operations
//!
//! Eigenvalues, eigenvectors, and Singular Value Decomposition (SVD).

use continuum_foundation::{Mat2, Mat3, Mat4};
use continuum_kernel_macros::kernel_fn;
use nalgebra as na;

/// Sort eigenvalues in descending order by magnitude
fn sort_eigenvalues_desc<const N: usize>(mut vals: [f64; N]) -> [f64; N] {
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    vals
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat2(m)` -> Vec2
///
/// Returns eigenvalues sorted in descending order.
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism Warning
/// Eigenvalue ordering is currently implementation-defined (descending by magnitude).
/// While stable within a single nalgebra version, ordering may change between versions.
/// Sign of eigenvectors is ambiguous (v and -v are both valid eigenvectors).
/// For deterministic simulations, consider these limitations carefully.
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
    sort_eigenvalues_desc([eig.eigenvalues[0], eig.eigenvalues[1]])
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat3(m)` -> Vec3
///
/// Returns eigenvalues sorted in descending order.
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism Warning
/// Eigenvalue ordering is currently implementation-defined (descending by magnitude).
/// While stable within a single nalgebra version, ordering may change between versions.
/// Sign of eigenvectors is ambiguous (v and -v are both valid eigenvectors).
/// For deterministic simulations, consider these limitations carefully.
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
    sort_eigenvalues_desc([eig.eigenvalues[0], eig.eigenvalues[1], eig.eigenvalues[2]])
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat4(m)` -> Vec4
///
/// Returns eigenvalues sorted in descending order.
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism Warning
/// Eigenvalue ordering is currently implementation-defined (descending by magnitude).
/// While stable within a single nalgebra version, ordering may change between versions.
/// Sign of eigenvectors is ambiguous (v and -v are both valid eigenvectors).
/// For deterministic simulations, consider these limitations carefully.
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
    sort_eigenvalues_desc([
        eig.eigenvalues[0],
        eig.eigenvalues[1],
        eig.eigenvalues[2],
        eig.eigenvalues[3],
    ])
}

/// Eigenvectors of a symmetric matrix: `eigenvectors_mat2(m)` -> Mat2
///
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues).
/// Eigenvectors are orthonormal and sorted to match eigenvalue order (descending).
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism Warning
/// - **Sign ambiguity**: Both v and -v are valid eigenvectors for eigenvalue λ.
///   The returned sign is nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Eigenvectors are sorted by eigenvalue magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations requiring stable signs/ordering, consider
///   implementing canonical normalization (e.g., first non-zero component positive).
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
///
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues).
/// Eigenvectors are orthonormal and sorted to match eigenvalue order (descending).
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism Warning
/// - **Sign ambiguity**: Both v and -v are valid eigenvectors for eigenvalue λ.
///   The returned sign is nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Eigenvectors are sorted by eigenvalue magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations requiring stable signs/ordering, consider
///   implementing canonical normalization (e.g., first non-zero component positive).
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
///
/// Returns matrix where columns are eigenvectors (corresponding to sorted eigenvalues).
/// Eigenvectors are orthonormal and sorted to match eigenvalue order (descending).
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism Warning
/// - **Sign ambiguity**: Both v and -v are valid eigenvectors for eigenvalue λ.
///   The returned sign is nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Eigenvectors are sorted by eigenvalue magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations requiring stable signs/ordering, consider
///   implementing canonical normalization (e.g., first non-zero component positive).
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
///
/// Returns the left singular vectors (U in A = UΣV^T).
/// Columns of U are orthonormal.
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both (u, v) and (-u, -v) are valid.
///   The returned signs are nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations, be aware these limitations may cause non-determinism
///   if nalgebra's SVD implementation changes.
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
///
/// Returns the left singular vectors (U in A = UΣV^T).
/// Columns of U are orthonormal.
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both (u, v) and (-u, -v) are valid.
///   The returned signs are nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations, be aware these limitations may cause non-determinism
///   if nalgebra's SVD implementation changes.
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
///
/// Returns the left singular vectors (U in A = UΣV^T).
/// Columns of U are orthonormal.
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both (u, v) and (-u, -v) are valid.
///   The returned signs are nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations, be aware these limitations may cause non-determinism
///   if nalgebra's SVD implementation changes.
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
///
/// Returns the singular values (diagonal of Σ in A = UΣV^T).
/// Singular values are always non-negative and returned in descending order.
///
/// # Determinism
/// Singular values themselves are mathematically unique (up to ordering).
/// Ordering is descending by magnitude, which is stable and deterministic.
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
///
/// Returns the singular values (diagonal of Σ in A = UΣV^T).
/// Singular values are always non-negative and returned in descending order.
///
/// # Determinism
/// Singular values themselves are mathematically unique (up to ordering).
/// Ordering is descending by magnitude, which is stable and deterministic.
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
///
/// Returns the singular values (diagonal of Σ in A = UΣV^T).
/// Singular values are always non-negative and returned in descending order.
///
/// # Determinism
/// Singular values themselves are mathematically unique (up to ordering).
/// Ordering is descending by magnitude, which is stable and deterministic.
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
///
/// Returns the transposed right singular vectors (V^T in A = UΣV^T).
/// Rows of V^T are orthonormal (equivalently, columns of V are orthonormal).
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both (u, v) and (-u, -v) are valid.
///   The returned signs are nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations, be aware these limitations may cause non-determinism
///   if nalgebra's SVD implementation changes.
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
///
/// Returns the transposed right singular vectors (V^T in A = UΣV^T).
/// Rows of V^T are orthonormal (equivalently, columns of V are orthonormal).
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both (u, v) and (-u, -v) are valid.
///   The returned signs are nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations, be aware these limitations may cause non-determinism
///   if nalgebra's SVD implementation changes.
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
///
/// Returns the transposed right singular vectors (V^T in A = UΣV^T).
/// Rows of V^T are orthonormal (equivalently, columns of V are orthonormal).
///
/// # Determinism Warning
/// - **Sign ambiguity**: For singular value σ, both (u, v) and (-u, -v) are valid.
///   The returned signs are nalgebra implementation-defined and may change between versions.
/// - **Ordering**: Singular vectors are sorted by singular value magnitude (descending).
///   While stable within a nalgebra version, ordering may change between versions.
/// - For deterministic simulations, be aware these limitations may cause non-determinism
///   if nalgebra's SVD implementation changes.
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
