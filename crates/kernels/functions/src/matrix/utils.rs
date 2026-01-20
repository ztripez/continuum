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
/// Panics if the internal slice conversion fails. This is guaranteed to succeed
/// for fixed-size matrices of the same dimension.
#[inline]
pub fn from_na_mat2(mat: na::Matrix2<f64>) -> Mat2 {
    Mat2(mat.as_slice().try_into().unwrap())
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
/// Panics if the internal slice conversion fails. This is guaranteed to succeed
/// for fixed-size matrices of the same dimension.
#[inline]
pub fn from_na_mat3(mat: na::Matrix3<f64>) -> Mat3 {
    Mat3(mat.as_slice().try_into().unwrap())
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
/// Panics if the internal slice conversion fails. This is guaranteed to succeed
/// for fixed-size matrices of the same dimension.
#[inline]
pub fn from_na_mat4(mat: na::Matrix4<f64>) -> Mat4 {
    Mat4(mat.as_slice().try_into().unwrap())
}
