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
/// Uses direct analytic formula (quadratic characteristic equation) for O(1) performance
/// with no allocation overhead. For a symmetric 2x2 matrix [[a, b], [b, c]]:
/// - Characteristic polynomial: λ² - (a+c)λ + (ac-b²) = 0
/// - Eigenvalues: λ = (a+c)/2 ± sqrt((a+c)²/4 - (ac-b²))
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism
/// Analytic formula is fully deterministic. Eigenvalues are sorted descending by magnitude.
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
    // Matrix [[a, b], [b, c]] in column-major: [a, b, b, c]
    let a = arr.0[0];
    let b = arr.0[1]; // off-diagonal
    let c = arr.0[3];

    // Characteristic polynomial: λ² - (a+c)λ + (ac-b²) = 0
    // Eigenvalues: λ = (a+c)/2 ± sqrt((a-c)²/4 + b²)
    // Use stable form to avoid catastrophic cancellation when eigenvalues are nearly equal
    let trace = a + c;
    let half_diff = (a - c) / 2.0;
    let disc_raw = half_diff * half_diff + b * b;

    // Fail loudly if discriminant is significantly negative (indicates non-symmetric input or numerical error)
    const DISC_TOL: f64 = 1e-12;
    assert!(
        disc_raw >= -DISC_TOL,
        "eigenvalues_mat2: negative discriminant {} (tolerance {}) indicates non-symmetric matrix or numerical error",
        disc_raw,
        DISC_TOL
    );

    let discriminant = disc_raw.max(0.0).sqrt();

    let lambda1 = trace / 2.0 + discriminant;
    let lambda2 = trace / 2.0 - discriminant;

    // Return sorted descending
    if lambda1 >= lambda2 {
        [lambda1, lambda2]
    } else {
        [lambda2, lambda1]
    }
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat3(m)` -> Vec3
///
/// Returns eigenvalues sorted in descending order.
///
/// Uses Cardano's formula for solving the cubic characteristic polynomial analytically,
/// providing O(1) performance with no allocation overhead. For a 3x3 real symmetric matrix,
/// all eigenvalues are guaranteed to be real.
///
/// # Preconditions
/// - **REQUIRES SYMMETRIC MATRIX**: Input must be symmetric (A = A^T).
///   Non-symmetric matrices will produce incorrect results without error.
///   No runtime validation is performed for performance reasons.
///
/// # Determinism
/// Analytic formula is fully deterministic. Eigenvalues are sorted descending by magnitude.
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
    // Extract elements (column-major: [m00, m10, m20, m01, m11, m21, m02, m12, m22])
    let m00 = arr.0[0];
    let m10 = arr.0[1];
    let m20 = arr.0[2];
    let m11 = arr.0[4];
    let m21 = arr.0[5];
    let m22 = arr.0[8];

    // Characteristic polynomial coefficients
    let trace = m00 + m11 + m22;
    let p1 = m10 * m10 + m20 * m20 + m21 * m21; // sum of squared off-diagonals
    let q = trace / 3.0; // mean of eigenvalues
    let p2 =
        ((m00 - q) * (m00 - q) + (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q) + 2.0 * p1) / 6.0;
    let p = p2.sqrt();

    // Degenerate case: all eigenvalues equal (matrix is scalar multiple of identity)
    // This is a valid mathematical case, not an error, but we make it explicit rather than silent
    const P_THRESHOLD: f64 = 1e-14;
    if p < P_THRESHOLD {
        // All eigenvalues equal to trace/3
        return [q, q, q];
    }

    // Compute determinant of (A - qI) / p for angle calculation
    let b00 = (m00 - q) / p;
    let b10 = m10 / p;
    let b20 = m20 / p;
    let b11 = (m11 - q) / p;
    let b21 = m21 / p;
    let b22 = (m22 - q) / p;

    let r = (b00 * (b11 * b22 - b21 * b21) - b10 * (b10 * b22 - b21 * b20)
        + b20 * (b10 * b21 - b11 * b20))
        / 2.0;

    // Fail loudly if r is outside valid range (indicates numerical instability)
    const R_TOL: f64 = 1e-10;
    assert!(
        r >= -1.0 - R_TOL && r <= 1.0 + R_TOL,
        "eigenvalues_mat3: r={} out of valid range [-1,1] (tolerance {}), indicates numerical instability or non-symmetric matrix",
        r,
        R_TOL
    );

    // Clamp to exact range for acos (acceptable after validation)
    let r = r.clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    // Eigenvalues (all real for symmetric matrices)
    let lambda1 = q + 2.0 * p * phi.cos();
    let lambda3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let lambda2 = 3.0 * q - lambda1 - lambda3; // trace = sum of eigenvalues

    // Sort descending
    sort_eigenvalues_desc([lambda1, lambda2, lambda3])
}

/// Eigenvalues of a symmetric matrix: `eigenvalues_mat4(m)` -> Vec4
///
/// Returns eigenvalues sorted in descending order.
///
/// **Implementation Note**: Uses nalgebra's iterative decomposition instead of analytic formula.
/// While 2x2 and 3x3 matrices have simple closed-form solutions (quadratic and Cardano's formula),
/// the 4x4 analytic solution (Ferrari's quartic formula) is extremely complex and numerically
/// unstable. Iterative methods are more practical for n≥4.
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
