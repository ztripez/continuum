//! Matrix Construction Functions
//!
//! Functions for building matrices from quaternions, axis-angles, scales, translations, and rotations.

use continuum_foundation::{Mat3, Mat4};
use continuum_kernel_macros::kernel_fn;

/// Builds a 3x3 rotation matrix from a quaternion.
///
/// # Parameters
/// - `q`: Input quaternion in `[x, y, z, w]` order. It will be normalized
///   before conversion to ensure a valid rotation matrix.
///
/// # Returns
/// A 3x3 rotation [`Mat3`].
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

/// Builds a 3x3 rotation matrix from an axis and an angle.
///
/// # Parameters
/// - `axis`: The rotation axis as a 3D vector. It will be normalized.
/// - `angle`: The rotation angle in radians.
///
/// # Returns
/// A 3x3 rotation [`Mat3`].
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
    // Convert axis-angle to quaternion, then to matrix (One Truth)
    let quat = crate::quat::from_axis_angle(axis, angle);
    crate::quat::to_mat3(quat)
}

/// Creates a 4x4 scaling transformation matrix.
///
/// # Parameters
/// - `x`: Scale factor along the X axis.
/// - `y`: Scale factor along the Y axis.
/// - `z`: Scale factor along the Z axis.
///
/// # Returns
/// A 4x4 scaling [`Mat4`].
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

/// Creates a 4x4 translation transformation matrix.
///
/// # Parameters
/// - `x`: Translation along the X axis.
/// - `y`: Translation along the Y axis.
/// - `z`: Translation along the Z axis.
///
/// # Returns
/// A 4x4 translation [`Mat4`].
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

/// Creates a 4x4 rotation matrix around the X axis.
///
/// # Parameters
/// - `angle`: Rotation angle in radians.
///
/// # Returns
/// A 4x4 rotation [`Mat4`].
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

/// Creates a 4x4 rotation matrix around the Y axis.
///
/// # Parameters
/// - `angle`: Rotation angle in radians.
///
/// # Returns
/// A 4x4 rotation [`Mat4`].
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

/// Creates a 4x4 rotation matrix around the Z axis.
///
/// # Parameters
/// - `angle`: Rotation angle in radians.
///
/// # Returns
/// A 4x4 rotation [`Mat4`].
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
