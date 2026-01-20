//! Matrix conversion utilities between Continuum types and nalgebra.
//!
//! These utilities provide zero-copy or minimal-copy conversions between the
//! fixed-size matrix types in `continuum-foundation` and the `nalgebra`
//! linear algebra library. All conversions preserve the column-major layout
//! used by both systems.

use continuum_foundation::{Mat2, Mat3, Mat4};
use nalgebra as na;

/// Converts a Continuum [`Mat2`] into a nalgebra [`na::Matrix2`].
///
/// # Parameters
/// - `mat`: The source 2x2 matrix in column-major order.
///
/// # Returns
/// A `nalgebra` matrix populated with the same values.
#[inline]
pub fn to_na_mat2(mat: &Mat2) -> na::Matrix2<f64> {
    na::Matrix2::from_column_slice(&mat.0)
}

/// Converts a nalgebra [`na::Matrix2`] into a Continuum [`Mat2`].
///
/// # Parameters
/// - `mat`: The source `nalgebra` matrix.
///
/// # Returns
/// A `continuum-foundation` 2x2 matrix.
///
/// # Panics
/// Panics if the internal slice conversion fails or size is incorrect.
#[inline]
pub fn from_na_mat2(mat: na::Matrix2<f64>) -> Mat2 {
    let slice = mat.as_slice();
    assert_eq!(slice.len(), 4, "from_na_mat2: expected 4 elements");
    Mat2(
        slice
            .try_into()
            .expect("from_na_mat2: failed to convert slice to [f64; 4]"),
    )
}

/// Converts a Continuum [`Mat3`] into a nalgebra [`na::Matrix3`].
///
/// # Parameters
/// - `mat`: The source 3x3 matrix in column-major order.
///
/// # Returns
/// A `nalgebra` matrix populated with the same values.
#[inline]
pub fn to_na_mat3(mat: &Mat3) -> na::Matrix3<f64> {
    na::Matrix3::from_column_slice(&mat.0)
}

/// Converts a nalgebra [`na::Matrix3`] into a Continuum [`Mat3`].
///
/// # Parameters
/// - `mat`: The source `nalgebra` matrix.
///
/// # Returns
/// A `continuum-foundation` 3x3 matrix.
///
/// # Panics
/// Panics if the internal slice conversion fails or size is incorrect.
#[inline]
pub fn from_na_mat3(mat: na::Matrix3<f64>) -> Mat3 {
    let slice = mat.as_slice();
    assert_eq!(slice.len(), 9, "from_na_mat3: expected 9 elements");
    Mat3(
        slice
            .try_into()
            .expect("from_na_mat3: failed to convert slice to [f64; 9]"),
    )
}

/// Converts a Continuum [`Mat4`] into a nalgebra [`na::Matrix4`].
///
/// # Parameters
/// - `mat`: The source 4x4 matrix in column-major order.
///
/// # Returns
/// A `nalgebra` matrix populated with the same values.
#[inline]
pub fn to_na_mat4(mat: &Mat4) -> na::Matrix4<f64> {
    na::Matrix4::from_column_slice(&mat.0)
}

/// Converts a nalgebra [`na::Matrix4`] into a Continuum [`Mat4`].
///
/// # Parameters
/// - `mat`: The source `nalgebra` matrix.
///
/// # Returns
/// A `continuum-foundation` 4x4 matrix.
///
/// # Panics
/// Panics if the internal slice conversion fails or size is incorrect.
#[inline]
pub fn from_na_mat4(mat: na::Matrix4<f64>) -> Mat4 {
    let slice = mat.as_slice();
    assert_eq!(slice.len(), 16, "from_na_mat4: expected 16 elements");
    Mat4(
        slice
            .try_into()
            .expect("from_na_mat4: failed to convert slice to [f64; 16]"),
    )
}
