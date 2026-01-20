//! Projection Matrix Functions
//!
//! Perspective, orthographic, and look-at view matrices for 3D graphics.

use continuum_foundation::Mat4;
use continuum_kernel_macros::kernel_fn;

/// Creates a perspective projection matrix.
///
/// Generates a right-handed perspective matrix (OpenGL-style, depth range [-1, 1]).
///
/// # Parameters
/// - `fov_y`: Vertical field of view in radians. Must be in (0, π).
/// - `aspect`: Aspect ratio (width / height). Must be positive.
/// - `near`: Distance to the near clipping plane. Must be positive.
/// - `far`: Distance to the far clipping plane. Must be greater than `near`.
///
/// # Returns
/// A 4x4 perspective projection [`Mat4`].
///
/// # Panics
/// Panics if input constraints are violated.
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

/// Creates an orthographic projection matrix.
///
/// Generates a right-handed orthographic matrix (OpenGL-style, depth range [-1, 1]).
///
/// # Parameters
/// - `left`: Left clipping plane.
/// - `right`: Right clipping plane.
/// - `bottom`: Bottom clipping plane.
/// - `top`: Top clipping plane.
/// - `near`: Near clipping plane.
/// - `far`: Far clipping plane.
///
/// # Returns
/// A 4x4 orthographic projection [`Mat4`].
///
/// # Panics
/// Panics if parallel planes are provided (e.g., `left == right`).
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

/// Creates a look-at view matrix.
///
/// Generates a view matrix that looks from `eye` towards `target`, oriented with `up`.
///
/// # Parameters
/// - `eye`: Camera position as a 3D vector.
/// - `target`: Target position to look at as a 3D vector.
/// - `up`: Up direction vector (normalized direction, usually [0, 1, 0]).
///
/// # Returns
/// A 4x4 look-at view [`Mat4`].
///
/// # Panics
/// Panics if:
/// - `eye` and `target` are at the same position.
/// - `up` is parallel to the view direction.
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
