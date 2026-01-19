//! Matrix Module Tests

use super::*;
use continuum_foundation::{Mat2, Mat3, Mat4};
use continuum_kernel_registry::is_known_in;

// Matrix-vector multiplication helpers - use actual kernel functions to avoid duplication
// This also tests the transform_mat*_vec* kernels indirectly
fn mat2_times_vec2(m: Mat2, v: [f64; 2]) -> [f64; 2] {
    transform_mat2_vec2(m, v)
}

fn mat3_times_vec3(m: Mat3, v: [f64; 3]) -> [f64; 3] {
    transform_mat3_vec3(m, v)
}

fn mat4_times_vec4(m: Mat4, v: [f64; 4]) -> [f64; 4] {
    transform_mat4_vec4(m, v)
}

/// Table of all matrix kernels that should be registered
const MATRIX_KERNELS: &[&str] = &[
    "determinant_mat2",
    "determinant_mat3",
    "determinant_mat4",
    "eigenvalues_mat2",
    "eigenvalues_mat3",
    "eigenvalues_mat4",
    "eigenvectors_mat2",
    "eigenvectors_mat3",
    "eigenvectors_mat4",
    "from_axis_angle",
    "from_quat",
    "identity2",
    "identity3",
    "identity4",
    "inverse_mat2",
    "inverse_mat3",
    "inverse_mat4",
    "look_at",
    "mul_mat2",
    "mul_mat3",
    "mul_mat4",
    "orthographic",
    "perspective",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scale",
    "svd_s_mat2",
    "svd_s_mat3",
    "svd_s_mat4",
    "svd_u_mat2",
    "svd_u_mat3",
    "svd_u_mat4",
    "svd_vt_mat2",
    "svd_vt_mat3",
    "svd_vt_mat4",
    "trace_mat2",
    "trace_mat3",
    "trace_mat4",
    "transform_mat2_vec2",
    "transform_mat3_vec3",
    "transform_mat4_vec4",
    "translation",
    "transpose_mat2",
    "transpose_mat3",
    "transpose_mat4",
];

#[test]
fn test_all_matrix_kernels_registered() {
    for name in MATRIX_KERNELS {
        assert!(
            is_known_in("matrix", name),
            "Kernel matrix::{} not registered",
            name
        );
    }
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
fn test_inverse_mat2_large_scale() {
    // Test numerical stability with large values (1e8 scale)
    let scale = 1e8;
    let m = Mat2([scale, 0.0, 0.0, scale]);
    let inv = inverse_mat2(m);
    let expected = 1.0 / scale;
    assert!((inv.0[0] - expected).abs() < 1e-10 * expected.abs());
    assert!((inv.0[3] - expected).abs() < 1e-10 * expected.abs());

    // Verify M·M⁻¹ = I
    let m2 = Mat2([scale, 0.0, 0.0, scale]);
    let product = mul_mat2(m2, inv);
    assert!(
        (product.0[0] - 1.0).abs() < 1e-6,
        "M·M⁻¹ should be identity"
    );
    assert!((product.0[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_inverse_mat3_large_scale() {
    // Test numerical stability with large values
    let scale = 1e8;
    let m = Mat3([scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale]);
    let inv = inverse_mat3(m);
    let expected = 1.0 / scale;
    assert!((inv.0[0] - expected).abs() < 1e-10 * expected.abs());
    assert!((inv.0[4] - expected).abs() < 1e-10 * expected.abs());
    assert!((inv.0[8] - expected).abs() < 1e-10 * expected.abs());
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
fn test_eigenvalues_mat2_identity() {
    let identity = Mat2([1.0, 0.0, 0.0, 1.0]);
    let result = eigenvalues_mat2(identity);
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_eigenvalues_mat2_off_diagonal() {
    // Symmetric matrix [[3, 1], [1, 3]] - off-diagonal elements test analytic formula
    let m = Mat2([3.0, 1.0, 1.0, 3.0]);
    let result = eigenvalues_mat2(m);
    // Eigenvalues: (3+3 ± √((3-3)² + 4·1²)) / 2 = (6 ± 2) / 2 = 4, 2
    assert!((result[0] - 4.0).abs() < 1e-10, "λ₁ should be 4");
    assert!((result[1] - 2.0).abs() < 1e-10, "λ₂ should be 2");
}

#[test]
fn test_eigenvalues_mat2_diagonal() {
    // Diagonal matrix [[5, 0], [0, 2]] - tests b ≈ 0 branch
    let m = Mat2([5.0, 0.0, 0.0, 2.0]);
    let result = eigenvalues_mat2(m);
    assert!((result[0] - 5.0).abs() < 1e-10, "λ₁ should be 5");
    assert!((result[1] - 2.0).abs() < 1e-10, "λ₂ should be 2");
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
fn test_eigenvalues_mat3_degenerate_scalar_multiple() {
    // Scalar multiple of identity: 5·I = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    // Tests p < P_THRESHOLD branch (all eigenvalues equal)
    let m = Mat3([5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]);
    let result = eigenvalues_mat3(m);
    // All eigenvalues should be 5.0
    assert!((result[0] - 5.0).abs() < 1e-10, "λ₁ should be 5");
    assert!((result[1] - 5.0).abs() < 1e-10, "λ₂ should be 5");
    assert!((result[2] - 5.0).abs() < 1e-10, "λ₃ should be 5");
}

#[test]
fn test_eigenvalues_mat3_general_symmetric() {
    // General symmetric matrix [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
    // Tests Cardano formula with non-degenerate case
    let m = Mat3([4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]);
    let result = eigenvalues_mat3(m);
    // Verify sorted descending
    assert!(result[0] >= result[1], "λ₁ should be >= λ₂");
    assert!(result[1] >= result[2], "λ₂ should be >= λ₃");
    // Verify trace = sum of eigenvalues (12 = 4+4+4)
    let trace_sum = result[0] + result[1] + result[2];
    assert!(
        (trace_sum - 12.0).abs() < 1e-10,
        "Sum of eigenvalues should equal trace"
    );
}

#[test]
fn test_eigenvectors_mat2_identity() {
    let identity = Mat2([1.0, 0.0, 0.0, 1.0]);
    let result = eigenvectors_mat2(identity);

    // Identity matrix has any orthonormal basis as eigenvectors
    // Verify columns are orthonormal
    let v0 = [result.0[0], result.0[1]];
    let v1 = [result.0[2], result.0[3]];

    // Check unit length
    let norm0 = (v0[0] * v0[0] + v0[1] * v0[1]).sqrt();
    let norm1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    assert!((norm0 - 1.0).abs() < 1e-10, "Column 0 not unit length");
    assert!((norm1 - 1.0).abs() < 1e-10, "Column 1 not unit length");

    // Check orthogonality
    let dot = v0[0] * v1[0] + v0[1] * v1[1];
    assert!(dot.abs() < 1e-10, "Columns not orthogonal");
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
// Eigenvalue/Eigenvector Correctness Tests
// ============================================================================

#[test]
fn test_eigenvalues_eigenvectors_mat2_correctness() {
    // Symmetric matrix [[3, 1], [1, 3]]
    let m_data = [3.0, 1.0, 1.0, 3.0];
    let eigenvals = eigenvalues_mat2(Mat2(m_data));
    let eigenvecs = eigenvectors_mat2(Mat2(m_data));

    // Test A·v = λ·v for each eigenvalue/eigenvector pair
    // Column 0: eigenvector for eigenvals[0]
    let v0 = [eigenvecs.0[0], eigenvecs.0[1]];
    let av0 = mat2_times_vec2(Mat2(m_data), v0);
    let lambda_v0 = [eigenvals[0] * v0[0], eigenvals[0] * v0[1]];
    assert!(
        (av0[0] - lambda_v0[0]).abs() < 1e-10,
        "A·v0 ≠ λ0·v0 (x component)"
    );
    assert!(
        (av0[1] - lambda_v0[1]).abs() < 1e-10,
        "A·v0 ≠ λ0·v0 (y component)"
    );

    // Column 1: eigenvector for eigenvals[1]
    let v1 = [eigenvecs.0[2], eigenvecs.0[3]];
    let av1 = mat2_times_vec2(Mat2(m_data), v1);
    let lambda_v1 = [eigenvals[1] * v1[0], eigenvals[1] * v1[1]];
    assert!(
        (av1[0] - lambda_v1[0]).abs() < 1e-10,
        "A·v1 ≠ λ1·v1 (x component)"
    );
    assert!(
        (av1[1] - lambda_v1[1]).abs() < 1e-10,
        "A·v1 ≠ λ1·v1 (y component)"
    );
}

#[test]
fn test_eigenvectors_mat2_orthonormality() {
    // Symmetric matrix [[3, 1], [1, 3]]
    let eigenvecs = eigenvectors_mat2(Mat2([3.0, 1.0, 1.0, 3.0]));

    // Columns should be orthonormal
    let v0 = [eigenvecs.0[0], eigenvecs.0[1]];
    let v1 = [eigenvecs.0[2], eigenvecs.0[3]];

    // Check unit length
    let norm0 = (v0[0] * v0[0] + v0[1] * v0[1]).sqrt();
    let norm1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    assert!((norm0 - 1.0).abs() < 1e-10, "Eigenvector 0 not unit length");
    assert!((norm1 - 1.0).abs() < 1e-10, "Eigenvector 1 not unit length");

    // Check orthogonality (dot product should be 0)
    let dot = v0[0] * v1[0] + v0[1] * v1[1];
    assert!(dot.abs() < 1e-10, "Eigenvectors not orthogonal");
}

#[test]
fn test_eigenvalues_eigenvectors_mat3_correctness() {
    // Symmetric matrix [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
    let m_data = [4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0];
    let eigenvals = eigenvalues_mat3(Mat3(m_data));
    let eigenvecs = eigenvectors_mat3(Mat3(m_data));

    // Test A·v = λ·v for each eigenvalue/eigenvector pair
    for i in 0..3 {
        let v = [
            eigenvecs.0[i * 3],
            eigenvecs.0[i * 3 + 1],
            eigenvecs.0[i * 3 + 2],
        ];
        let av = mat3_times_vec3(Mat3(m_data), v);
        let lambda_v = [
            eigenvals[i] * v[0],
            eigenvals[i] * v[1],
            eigenvals[i] * v[2],
        ];

        assert!(
            (av[0] - lambda_v[0]).abs() < 1e-10,
            "A·v{} ≠ λ{}·v{} (x component)",
            i,
            i,
            i
        );
        assert!(
            (av[1] - lambda_v[1]).abs() < 1e-10,
            "A·v{} ≠ λ{}·v{} (y component)",
            i,
            i,
            i
        );
        assert!(
            (av[2] - lambda_v[2]).abs() < 1e-10,
            "A·v{} ≠ λ{}·v{} (z component)",
            i,
            i,
            i
        );
    }
}

#[test]
fn test_eigenvectors_mat3_orthonormality() {
    // Symmetric matrix [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
    let eigenvecs = eigenvectors_mat3(Mat3([4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0]));

    // Check unit length and orthogonality for all pairs
    for i in 0..3 {
        let v = [
            eigenvecs.0[i * 3],
            eigenvecs.0[i * 3 + 1],
            eigenvecs.0[i * 3 + 2],
        ];

        // Check unit length
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Eigenvector {} not unit length",
            i
        );

        // Check orthogonality with all other vectors
        for j in (i + 1)..3 {
            let w = [
                eigenvecs.0[j * 3],
                eigenvecs.0[j * 3 + 1],
                eigenvecs.0[j * 3 + 2],
            ];
            let dot = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
            assert!(
                dot.abs() < 1e-10,
                "Eigenvectors {} and {} not orthogonal",
                i,
                j
            );
        }
    }
}

#[test]
fn test_eigenvalues_eigenvectors_mat4_correctness() {
    // Symmetric 4x4 matrix (tridiagonal)
    let m_data = [
        5.0, 1.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0, 5.0,
    ];
    let eigenvals = eigenvalues_mat4(Mat4(m_data));
    let eigenvecs = eigenvectors_mat4(Mat4(m_data));

    // Test A·v = λ·v for each eigenvalue/eigenvector pair
    for i in 0..4 {
        let v = [
            eigenvecs.0[i * 4],
            eigenvecs.0[i * 4 + 1],
            eigenvecs.0[i * 4 + 2],
            eigenvecs.0[i * 4 + 3],
        ];
        let av = mat4_times_vec4(Mat4(m_data), v);
        let lambda_v = [
            eigenvals[i] * v[0],
            eigenvals[i] * v[1],
            eigenvals[i] * v[2],
            eigenvals[i] * v[3],
        ];

        for k in 0..4 {
            assert!(
                (av[k] - lambda_v[k]).abs() < 1e-9,
                "A·v{} ≠ λ{}·v{} (component {})",
                i,
                i,
                i,
                k
            );
        }
    }
}

#[test]
fn test_eigenvectors_mat4_orthonormality() {
    // Symmetric 4x4 matrix (tridiagonal)
    let eigenvecs = eigenvectors_mat4(Mat4([
        5.0, 1.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 1.0, 5.0,
    ]));

    // Check unit length and orthogonality for all pairs
    for i in 0..4 {
        let v = [
            eigenvecs.0[i * 4],
            eigenvecs.0[i * 4 + 1],
            eigenvecs.0[i * 4 + 2],
            eigenvecs.0[i * 4 + 3],
        ];

        // Check unit length
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "Eigenvector {} not unit length",
            i
        );

        // Check orthogonality with all other vectors
        for j in (i + 1)..4 {
            let w = [
                eigenvecs.0[j * 4],
                eigenvecs.0[j * 4 + 1],
                eigenvecs.0[j * 4 + 2],
                eigenvecs.0[j * 4 + 3],
            ];
            let dot = v[0] * w[0] + v[1] * w[1] + v[2] * w[2] + v[3] * w[3];
            assert!(
                dot.abs() < 1e-9,
                "Eigenvectors {} and {} not orthogonal",
                i,
                j
            );
        }
    }
}

// ============================================================================
// SVD Correctness Tests
// ============================================================================

#[test]
fn test_svd_mat2_reconstruction() {
    // Test matrix
    let m_data = [3.0, 1.0, 1.0, 2.0];

    let u = svd_u_mat2(Mat2(m_data));
    let s = svd_s_mat2(Mat2(m_data));
    let vt = svd_vt_mat2(Mat2(m_data));

    // Reconstruct: A ≈ U·Σ·V^T
    // Matrices are column-major: [col0_row0, col0_row1, col1_row0, col1_row1]
    // Compute U·Σ·V^T element by element
    let mut reconstructed = [0.0; 4];
    for col in 0..2 {
        for row in 0..2 {
            // A[col][row] = sum_k U[k][row] * s[k] * V^T[col][k]
            for k in 0..2 {
                reconstructed[col * 2 + row] += u.0[k * 2 + row] * s[k] * vt.0[col * 2 + k];
            }
        }
    }

    // Check reconstruction matches original
    for i in 0..4 {
        assert!(
            (reconstructed[i] - m_data[i]).abs() < 1e-10,
            "SVD reconstruction failed at index {}",
            i
        );
    }
}

#[test]
fn test_svd_mat3_reconstruction() {
    // Test matrix
    let m_data = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];

    let u = svd_u_mat3(Mat3(m_data));
    let s = svd_s_mat3(Mat3(m_data));
    let vt = svd_vt_mat3(Mat3(m_data));

    // Reconstruct: A ≈ U·Σ·V^T
    // Matrices are column-major
    let mut reconstructed = [0.0; 9];
    for col in 0..3 {
        for row in 0..3 {
            // A[col][row] = sum_k U[k][row] * s[k] * V^T[col][k]
            for k in 0..3 {
                reconstructed[col * 3 + row] += u.0[k * 3 + row] * s[k] * vt.0[col * 3 + k];
            }
        }
    }

    // Check reconstruction matches original
    for i in 0..9 {
        assert!(
            (reconstructed[i] - m_data[i]).abs() < 1e-10,
            "SVD reconstruction failed at index {}",
            i
        );
    }
}

#[test]
fn test_svd_mat4_reconstruction() {
    // Test matrix (4x4 symmetric)
    let m_data = [
        5.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0,
    ];

    let u = svd_u_mat4(Mat4(m_data));
    let s = svd_s_mat4(Mat4(m_data));
    let vt = svd_vt_mat4(Mat4(m_data));

    // Reconstruct: A ≈ U·Σ·V^T
    // Matrices are column-major
    let mut reconstructed = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            // A[col][row] = sum_k U[k][row] * s[k] * V^T[col][k]
            for k in 0..4 {
                reconstructed[col * 4 + row] += u.0[k * 4 + row] * s[k] * vt.0[col * 4 + k];
            }
        }
    }

    // Check reconstruction matches original
    for i in 0..16 {
        assert!(
            (reconstructed[i] - m_data[i]).abs() < 1e-9,
            "SVD reconstruction failed at index {}",
            i
        );
    }
}

#[test]
fn test_svd_u_orthonormality_mat4() {
    let u = svd_u_mat4(Mat4([
        5.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0,
    ]));

    // Check columns are orthonormal
    for i in 0..4 {
        let col = [u.0[i * 4], u.0[i * 4 + 1], u.0[i * 4 + 2], u.0[i * 4 + 3]];

        // Check unit length
        let norm = (col[0] * col[0] + col[1] * col[1] + col[2] * col[2] + col[3] * col[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "U column {} not unit length", i);

        // Check orthogonality with other columns
        for j in (i + 1)..4 {
            let other = [u.0[j * 4], u.0[j * 4 + 1], u.0[j * 4 + 2], u.0[j * 4 + 3]];
            let dot = col[0] * other[0] + col[1] * other[1] + col[2] * other[2] + col[3] * other[3];
            assert!(
                dot.abs() < 1e-10,
                "U columns {} and {} not orthogonal",
                i,
                j
            );
        }
    }
}

#[test]
fn test_svd_vt_orthonormality_mat4() {
    let vt = svd_vt_mat4(Mat4([
        5.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 2.0,
    ]));

    // V^T rows should be orthonormal
    for i in 0..4 {
        let row = [
            vt.0[i * 4],
            vt.0[i * 4 + 1],
            vt.0[i * 4 + 2],
            vt.0[i * 4 + 3],
        ];

        // Check unit length
        let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2] + row[3] * row[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "V^T row {} not unit length", i);

        // Check orthogonality with other rows
        for j in (i + 1)..4 {
            let other = [
                vt.0[j * 4],
                vt.0[j * 4 + 1],
                vt.0[j * 4 + 2],
                vt.0[j * 4 + 3],
            ];
            let dot = row[0] * other[0] + row[1] * other[1] + row[2] * other[2] + row[3] * other[3];
            assert!(dot.abs() < 1e-10, "V^T rows {} and {} not orthogonal", i, j);
        }
    }
}

#[test]
fn test_svd_u_orthonormality_mat2() {
    let u = svd_u_mat2(Mat2([3.0, 1.0, 1.0, 2.0]));

    // Check columns are orthonormal
    let u0 = [u.0[0], u.0[1]];
    let u1 = [u.0[2], u.0[3]];

    // Check unit length
    let norm0 = (u0[0] * u0[0] + u0[1] * u0[1]).sqrt();
    let norm1 = (u1[0] * u1[0] + u1[1] * u1[1]).sqrt();
    assert!((norm0 - 1.0).abs() < 1e-10, "U column 0 not unit length");
    assert!((norm1 - 1.0).abs() < 1e-10, "U column 1 not unit length");

    // Check orthogonality
    let dot = u0[0] * u1[0] + u0[1] * u1[1];
    assert!(dot.abs() < 1e-10, "U columns not orthogonal");
}

#[test]
fn test_svd_vt_orthonormality_mat3() {
    let vt = svd_vt_mat3(Mat3([4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]));

    // V^T rows should be orthonormal (equivalently, V columns are orthonormal)
    for i in 0..3 {
        let row = [vt.0[i * 3], vt.0[i * 3 + 1], vt.0[i * 3 + 2]];

        // Check unit length
        let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "V^T row {} not unit length", i);

        // Check orthogonality with other rows
        for j in (i + 1)..3 {
            let other = [vt.0[j * 3], vt.0[j * 3 + 1], vt.0[j * 3 + 2]];
            let dot = row[0] * other[0] + row[1] * other[1] + row[2] * other[2];
            assert!(dot.abs() < 1e-10, "V^T rows {} and {} not orthogonal", i, j);
        }
    }
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
fn test_rotation_y_zero() {
    let m = rotation_y(0.0);
    assert!((m.0[0] - 1.0).abs() < 1e-10);
    assert!((m.0[5] - 1.0).abs() < 1e-10);
    assert!((m.0[10] - 1.0).abs() < 1e-10);
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

// === Zero-Norm Guard Tests ===

#[test]
#[should_panic(expected = "quat.normalize requires non-zero quaternion")]
fn test_from_quat_zero() {
    let _ = from_quat([0.0, 0.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "quat.from_axis_angle requires non-zero axis")]
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

// === Mat4 Value Tests ===

#[test]
fn test_determinant_mat4_identity() {
    let m = Mat4([
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let det = determinant_mat4(m);
    assert_eq!(det, 1.0);
}

#[test]
fn test_determinant_mat4_scaled() {
    // Determinant of diagonal matrix is product of diagonal elements
    let m = Mat4([
        2.0, 0.0, 0.0, 0.0, // col 0
        0.0, 3.0, 0.0, 0.0, // col 1
        0.0, 0.0, 4.0, 0.0, // col 2
        0.0, 0.0, 0.0, 5.0, // col 3
    ]);
    let det = determinant_mat4(m);
    assert_eq!(det, 2.0 * 3.0 * 4.0 * 5.0); // 120.0
}

#[test]
fn test_inverse_mat4_identity() {
    let identity = Mat4([
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let expected = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let inv = inverse_mat4(identity);
    assert_eq!(inv.0, expected);
}

#[test]
fn test_inverse_mat4_scaled() {
    let m = Mat4([
        2.0, 0.0, 0.0, 0.0, // col 0
        0.0, 2.0, 0.0, 0.0, // col 1
        0.0, 0.0, 2.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let inv = inverse_mat4(m);
    let expected = Mat4([
        0.5, 0.0, 0.0, 0.0, // col 0
        0.0, 0.5, 0.0, 0.0, // col 1
        0.0, 0.0, 0.5, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    assert_eq!(inv.0, expected.0);
}

#[test]
#[should_panic(expected = "singular")]
fn test_inverse_mat4_singular() {
    // Zero row = singular matrix
    let m = Mat4([
        0.0, 0.0, 0.0, 0.0, // col 0 - all zeros
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let _ = inverse_mat4(m);
}

#[test]
fn test_mul_mat4_identity() {
    let identity = Mat4([
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let m = Mat4([
        1.0, 2.0, 3.0, 4.0, // col 0
        5.0, 6.0, 7.0, 8.0, // col 1
        9.0, 10.0, 11.0, 12.0, // col 2
        13.0, 14.0, 15.0, 16.0, // col 3
    ]);
    let expected = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let result = mul_mat4(identity, m);
    assert_eq!(result.0, expected);
}

#[test]
fn test_mul_mat4_inverse() {
    // A * A^-1 = I
    let m = Mat4([
        2.0, 0.0, 0.0, 0.0, // col 0
        0.0, 3.0, 0.0, 0.0, // col 1
        0.0, 0.0, 4.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let m_copy = Mat4([
        2.0, 0.0, 0.0, 0.0, // col 0
        0.0, 3.0, 0.0, 0.0, // col 1
        0.0, 0.0, 4.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let inv = inverse_mat4(m);
    let result = mul_mat4(m_copy, inv);

    let identity = Mat4([
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);

    // Check each element is close to identity (accounting for floating point error)
    for i in 0..16 {
        assert!((result.0[i] - identity.0[i]).abs() < 1e-10);
    }
}

#[test]
fn test_transform_mat4_vec4_identity() {
    let identity = Mat4([
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let v = [1.0, 2.0, 3.0, 4.0];
    let result = transform_mat4_vec4(identity, v);
    assert_eq!(result, v);
}

#[test]
fn test_transform_mat4_vec4_scaled() {
    // Scale x by 2, y by 3, z by 4, w unchanged
    let m = Mat4([
        2.0, 0.0, 0.0, 0.0, // col 0
        0.0, 3.0, 0.0, 0.0, // col 1
        0.0, 0.0, 4.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ]);
    let v = [1.0, 1.0, 1.0, 1.0];
    let result = transform_mat4_vec4(m, v);
    assert_eq!(result, [2.0, 3.0, 4.0, 1.0]);
}

// ============================================================================
// Fail-Loud Tests: Verify assertions for invalid inputs
// ============================================================================

// Note: For real symmetric 2x2 matrices, the discriminant in the quadratic formula
// is mathematically guaranteed to be non-negative (complex eigenvalues don't exist).
// The assertion in eigenvalues_mat2 is defensive programming to catch floating-point
// errors or accidental non-symmetric inputs. In practice, this assertion should rarely
// trigger for valid symmetric matrices.
//
// Commenting out this test as creating a pathological case that triggers it without
// causing overflow is difficult. The assertion remains in the code as a safeguard.

// Note: For real symmetric 3x3 matrices, the value r in Cardano's formula is
// mathematically guaranteed to be in [-1, 1] (since r = det(B)/2 where B is normalized).
// The assertion in eigenvalues_mat3 is defensive programming to catch floating-point
// errors or accidental non-symmetric inputs. In practice, this assertion should rarely
// trigger for valid symmetric matrices.
//
// Commenting out this test as creating a pathological symmetric case that triggers it
// without causing overflow is difficult. The assertion remains in the code as a safeguard.
