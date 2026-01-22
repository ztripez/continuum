//! Matrix Operations (Shared Implementation)
//!
//! Low-level matrix operations used by both VM executor and kernel functions.
//! All matrices use column-major storage.

/// Matrix-matrix multiplication for 2x2 matrices.
///
/// Both inputs and output are column-major: `[col0_row0, col0_row1, col1_row0, col1_row1]`
#[inline]
pub fn mat2_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[2] * b[1], // result col0 row0
        a[1] * b[0] + a[3] * b[1], // result col0 row1
        a[0] * b[2] + a[2] * b[3], // result col1 row0
        a[1] * b[2] + a[3] * b[3], // result col1 row1
    ]
}

/// Matrix-matrix multiplication for 3x3 matrices.
///
/// Column-major storage: `[m00, m10, m20, m01, m11, m21, m02, m12, m22]`
#[inline]
pub fn mat3_mul(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut result = [0.0; 9];
    for col in 0..3 {
        for row in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += a[k * 3 + row] * b[col * 3 + k];
            }
            result[col * 3 + row] = sum;
        }
    }
    result
}

/// Matrix-matrix multiplication for 4x4 matrices.
///
/// Column-major storage: `[m00, m10, m20, m30, m01, m11, m21, m31, ...]`
#[inline]
pub fn mat4_mul(a: [f64; 16], b: [f64; 16]) -> [f64; 16] {
    let mut result = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    result
}

/// Transform a 2D vector by a 2x2 matrix.
///
/// Matrix is column-major: `[col0_row0, col0_row1, col1_row0, col1_row1]`
#[inline]
pub fn mat2_transform(m: [f64; 4], v: [f64; 2]) -> [f64; 2] {
    [
        m[0] * v[0] + m[2] * v[1], // row 0
        m[1] * v[0] + m[3] * v[1], // row 1
    ]
}

/// Transform a 3D vector by a 3x3 matrix.
///
/// Matrix is column-major: `[m00, m10, m20, m01, m11, m21, m02, m12, m22]`
#[inline]
pub fn mat3_transform(m: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2], // row 0
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2], // row 1
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2], // row 2
    ]
}

/// Transform a 4D vector by a 4x4 matrix.
///
/// Matrix is column-major: `[m00, m10, m20, m30, m01, m11, m21, m31, ...]`
#[inline]
pub fn mat4_transform(m: [f64; 16], v: [f64; 4]) -> [f64; 4] {
    [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3], // row 0
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3], // row 1
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3], // row 2
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3], // row 3
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat2_mul_identity() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let m = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mat2_mul(identity, m), m);
        assert_eq!(mat2_mul(m, identity), m);
    }

    #[test]
    fn test_mat2_mul() {
        let a = [1.0, 2.0, 3.0, 4.0]; // [[1,3], [2,4]]
        let b = [5.0, 6.0, 7.0, 8.0]; // [[5,7], [6,8]]
                                      // Result: [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]] = [[23, 31], [34, 46]]
                                      // Column-major: [23, 34, 31, 46]
        assert_eq!(mat2_mul(a, b), [23.0, 34.0, 31.0, 46.0]);
    }

    #[test]
    fn test_mat3_mul_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(mat3_mul(identity, m), m);
        assert_eq!(mat3_mul(m, identity), m);
    }

    #[test]
    fn test_mat4_mul_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let m = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        assert_eq!(mat4_mul(identity, m), m);
        assert_eq!(mat4_mul(m, identity), m);
    }

    #[test]
    fn test_mat2_transform_identity() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let v = [3.0, 4.0];
        assert_eq!(mat2_transform(identity, v), v);
    }

    #[test]
    fn test_mat2_transform_scale() {
        let scale = [2.0, 0.0, 0.0, 2.0]; // 2x scale
        let v = [3.0, 4.0];
        assert_eq!(mat2_transform(scale, v), [6.0, 8.0]);
    }

    #[test]
    fn test_mat3_transform_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0];
        assert_eq!(mat3_transform(identity, v), v);
    }

    #[test]
    fn test_mat3_transform_scale() {
        let scale = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let v = [1.0, 2.0, 3.0];
        assert_eq!(mat3_transform(scale, v), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mat4_transform_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let v = [1.0, 2.0, 3.0, 1.0];
        assert_eq!(mat4_transform(identity, v), v);
    }

    #[test]
    fn test_mat4_transform_translation() {
        // Translation matrix: translate by (10, 20, 30)
        // Column-major: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [10,20,30,1]]
        let translate = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 20.0, 30.0, 1.0,
        ];
        let v = [1.0, 2.0, 3.0, 1.0]; // w=1 for point
        assert_eq!(mat4_transform(translate, v), [11.0, 22.0, 33.0, 1.0]);
    }
}
