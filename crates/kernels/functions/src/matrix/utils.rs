//! Matrix conversion utilities between Continuum types and nalgebra.

use continuum_foundation::{Mat2, Mat3, Mat4};
use nalgebra as na;

/// Convert Mat2 to nalgebra Matrix2
#[inline]
pub fn to_na_mat2(mat: &Mat2) -> na::Matrix2<f64> {
    na::Matrix2::from_column_slice(&mat.0)
}

/// Convert nalgebra Matrix2 to Mat2
#[inline]
pub fn from_na_mat2(mat: na::Matrix2<f64>) -> Mat2 {
    Mat2(mat.as_slice().try_into().unwrap())
}

/// Convert Mat3 to nalgebra Matrix3
#[inline]
pub fn to_na_mat3(mat: &Mat3) -> na::Matrix3<f64> {
    na::Matrix3::from_column_slice(&mat.0)
}

/// Convert nalgebra Matrix3 to Mat3
#[inline]
pub fn from_na_mat3(mat: na::Matrix3<f64>) -> Mat3 {
    Mat3(mat.as_slice().try_into().unwrap())
}

/// Convert Mat4 to nalgebra Matrix4
#[inline]
pub fn to_na_mat4(mat: &Mat4) -> na::Matrix4<f64> {
    na::Matrix4::from_column_slice(&mat.0)
}

/// Convert nalgebra Matrix4 to Mat4
#[inline]
pub fn from_na_mat4(mat: na::Matrix4<f64>) -> Mat4 {
    Mat4(mat.as_slice().try_into().unwrap())
}
