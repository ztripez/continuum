//! Tensor Operations
//!
//! Functions for creating and manipulating dynamic tensors.

use continuum_foundation::tensor::TensorData;
use continuum_kernel_macros::kernel_fn;

/// Create tensor filled with zeros: `zeros(rows, cols)`
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn zeros(rows: f64, cols: f64) -> TensorData {
    TensorData::new(rows as usize, cols as usize)
}

/// Create tensor filled with ones: `ones(rows, cols)`
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn ones(rows: f64, cols: f64) -> TensorData {
    let r = rows as usize;
    let c = cols as usize;
    let data = vec![1.0; r * c];
    TensorData::from_vec(r, c, data)
}

/// Create identity tensor: `eye(size)`
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn eye(size: f64) -> TensorData {
    let n = size as usize;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    TensorData::from_vec(n, n, data)
}

/// Get element: `get(t, row, col)` -> Scalar
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn get(t: TensorData, row: f64, col: f64) -> f64 {
    t.get(row as usize, col as usize)
}

/// Set element (returns new tensor): `set(t, row, col, value)` -> Tensor
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn set(mut t: TensorData, row: f64, col: f64, value: f64) -> TensorData {
    t.set(row as usize, col as usize, value);
    t
}

/// Get number of rows: `rows(t)`
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn rows(t: TensorData) -> f64 {
    t.rows as f64
}

/// Get number of columns: `cols(t)`
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn cols(t: TensorData) -> f64 {
    t.cols as f64
}

/// Tensor multiply: `mul(a, b)` -> Tensor
/// Requires a.cols == b.rows
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn mul(a: TensorData, b: TensorData) -> TensorData {
    if a.cols != b.rows {
        panic!(
            "tensor.mul dimension mismatch: {}×{} * {}×{} (a.cols must equal b.rows)",
            a.rows, a.cols, b.rows, b.cols
        );
    }
    let mut result = TensorData::new(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    result
}

/// Transpose: `transpose(t)` -> Tensor
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn transpose(t: TensorData) -> TensorData {
    let mut result = TensorData::new(t.cols, t.rows);
    for i in 0..t.rows {
        for j in 0..t.cols {
            result.set(j, i, t.get(i, j));
        }
    }
    result
}

/// Element-wise addition: `add(a, b)` -> Tensor
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn add(a: TensorData, b: TensorData) -> TensorData {
    if a.rows != b.rows || a.cols != b.cols {
        panic!(
            "tensor.add dimension mismatch: {}×{} + {}×{}",
            a.rows, a.cols, b.rows, b.cols
        );
    }
    let mut result = TensorData::new(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) + b.get(i, j));
        }
    }
    result
}

/// Element-wise subtraction: `sub(a, b)` -> Tensor
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn sub(a: TensorData, b: TensorData) -> TensorData {
    if a.rows != b.rows || a.cols != b.cols {
        panic!(
            "tensor.sub dimension mismatch: {}×{} - {}×{}",
            a.rows, a.cols, b.rows, b.cols
        );
    }
    let mut result = TensorData::new(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(i, j, a.get(i, j) - b.get(i, j));
        }
    }
    result
}

/// Scalar multiplication: `scale(t, s)` -> Tensor
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn scale(t: TensorData, s: f64) -> TensorData {
    let mut result = TensorData::new(t.rows, t.cols);
    for i in 0..t.rows {
        for j in 0..t.cols {
            result.set(i, j, t.get(i, j) * s);
        }
    }
    result
}

// ============================================================================
// Tensor Property Functions
// ============================================================================

/// Trace (sum of diagonal): `trace(t)` -> Scalar
/// For non-square tensors, sums min(rows, cols) diagonal elements.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn trace(t: TensorData) -> f64 {
    let n = t.rows.min(t.cols);
    let mut sum = 0.0;
    for i in 0..n {
        sum += t.get(i, i);
    }
    sum
}

/// Frobenius norm: `norm(t)` -> Scalar
/// Square root of sum of squared elements.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn norm(t: TensorData) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..t.rows {
        for j in 0..t.cols {
            let v = t.get(i, j);
            sum_sq += v * v;
        }
    }
    sum_sq.sqrt()
}

/// Determinant: `det(t)` -> Scalar
/// Requires square tensor. Uses LU decomposition via nalgebra.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn det(t: TensorData) -> f64 {
    use nalgebra::DMatrix;

    if t.rows != t.cols {
        panic!(
            "tensor.det requires square tensor, got {}×{}",
            t.rows, t.cols
        );
    }

    let mat = DMatrix::from_row_slice(t.rows, t.cols, t.data());
    mat.determinant()
}

/// Matrix inverse: `inv(t)` -> Tensor
/// Requires square tensor. Panics if singular.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn inv(t: TensorData) -> TensorData {
    use nalgebra::DMatrix;

    if t.rows != t.cols {
        panic!(
            "tensor.inv requires square tensor, got {}×{}",
            t.rows, t.cols
        );
    }

    let mat = DMatrix::from_row_slice(t.rows, t.cols, t.data());
    match mat.try_inverse() {
        Some(inv_mat) => {
            // nalgebra stores in column-major, convert to row-major
            let mut data = Vec::with_capacity(t.rows * t.cols);
            for i in 0..t.rows {
                for j in 0..t.cols {
                    data.push(inv_mat[(i, j)]);
                }
            }
            TensorData::from_vec(t.rows, t.cols, data)
        }
        None => panic!("tensor.inv: matrix is singular (determinant = 0)"),
    }
}

// ============================================================================
// Tensor Manipulation Functions
// ============================================================================

/// Reshape tensor: `reshape(t, new_rows, new_cols)` -> Tensor
/// Total elements must match.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn reshape(t: TensorData, new_rows: f64, new_cols: f64) -> TensorData {
    let nr = new_rows as usize;
    let nc = new_cols as usize;

    if nr * nc != t.rows * t.cols {
        panic!(
            "tensor.reshape: cannot reshape {}×{} ({} elements) to {}×{} ({} elements)",
            t.rows,
            t.cols,
            t.rows * t.cols,
            nr,
            nc,
            nr * nc
        );
    }

    // Copy data in row-major order
    let mut data = Vec::with_capacity(t.rows * t.cols);
    for i in 0..t.rows {
        for j in 0..t.cols {
            data.push(t.get(i, j));
        }
    }

    TensorData::from_vec(nr, nc, data)
}

/// Extract sub-tensor: `slice(t, row_start, row_end, col_start, col_end)` -> Tensor
/// Indices are inclusive start, exclusive end.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn slice(
    t: TensorData,
    row_start: f64,
    row_end: f64,
    col_start: f64,
    col_end: f64,
) -> TensorData {
    let r0 = row_start as usize;
    let r1 = row_end as usize;
    let c0 = col_start as usize;
    let c1 = col_end as usize;

    if r1 > t.rows || c1 > t.cols {
        panic!(
            "tensor.slice: indices [{}:{}, {}:{}] out of bounds for {}×{} tensor",
            r0, r1, c0, c1, t.rows, t.cols
        );
    }

    if r0 >= r1 || c0 >= c1 {
        panic!("tensor.slice: invalid range [{}:{}, {}:{}]", r0, r1, c0, c1);
    }

    let new_rows = r1 - r0;
    let new_cols = c1 - c0;
    let mut result = TensorData::new(new_rows, new_cols);

    for i in 0..new_rows {
        for j in 0..new_cols {
            result.set(i, j, t.get(r0 + i, c0 + j));
        }
    }

    result
}

/// Solve linear system Ax = b: `solve(A, b)` -> Tensor
/// A must be square, b must have same number of rows as A.
/// Returns x such that Ax = b.
#[kernel_fn(namespace = "tensor", category = "tensor")]
pub fn solve(a: TensorData, b: TensorData) -> TensorData {
    use nalgebra::DMatrix;

    if a.rows != a.cols {
        panic!("tensor.solve: A must be square, got {}×{}", a.rows, a.cols);
    }

    if a.rows != b.rows {
        panic!(
            "tensor.solve: A rows ({}) must match b rows ({})",
            a.rows, b.rows
        );
    }

    let mat_a = DMatrix::from_row_slice(a.rows, a.cols, a.data());
    let mat_b = DMatrix::from_row_slice(b.rows, b.cols, b.data());

    match mat_a.lu().solve(&mat_b) {
        Some(x) => {
            // nalgebra stores in column-major, convert to row-major
            let mut data = Vec::with_capacity(b.rows * b.cols);
            for i in 0..b.rows {
                for j in 0..b.cols {
                    data.push(x[(i, j)]);
                }
            }
            TensorData::from_vec(b.rows, b.cols, data)
        }
        None => panic!("tensor.solve: system is singular (no solution)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_kernel_registry::{Arity, get_in_namespace, is_known_in};

    #[test]
    fn test_zeros_registered() {
        assert!(is_known_in("tensor", "zeros"));
        let desc = get_in_namespace("tensor", "zeros").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_zeros_value() {
        let t = zeros(2.0, 3.0);
        assert_eq!(t.shape(), (2, 3));
        assert_eq!(t.get(0, 0), 0.0);
        assert_eq!(t.get(1, 2), 0.0);
    }

    #[test]
    fn test_ones_value() {
        let t = ones(2.0, 2.0);
        assert_eq!(t.shape(), (2, 2));
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 1), 1.0);
    }

    #[test]
    fn test_eye_value() {
        let t = eye(3.0);
        assert_eq!(t.shape(), (3, 3));
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 1), 1.0);
        assert_eq!(t.get(2, 2), 1.0);
        assert_eq!(t.get(0, 1), 0.0);
        assert_eq!(t.get(1, 0), 0.0);
    }

    #[test]
    fn test_get_function() {
        let t = ones(2.0, 3.0);
        let val = get(t, 1.0, 2.0);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_set_function() {
        let t = zeros(2.0, 2.0);
        let t2 = set(t, 0.0, 1.0, 5.0);
        assert_eq!(get(t2, 0.0, 1.0), 5.0);
    }

    #[test]
    fn test_rows_cols() {
        let t = zeros(3.0, 4.0);
        assert_eq!(rows(t.clone()), 3.0);
        assert_eq!(cols(t), 4.0);
    }

    #[test]
    fn test_ones_registered() {
        assert!(is_known_in("tensor", "ones"));
    }

    #[test]
    fn test_eye_registered() {
        assert!(is_known_in("tensor", "eye"));
    }

    #[test]
    fn test_get_registered() {
        assert!(is_known_in("tensor", "get"));
    }

    #[test]
    fn test_set_registered() {
        assert!(is_known_in("tensor", "set"));
    }

    #[test]
    fn test_rows_registered() {
        assert!(is_known_in("tensor", "rows"));
    }

    #[test]
    fn test_cols_registered() {
        assert!(is_known_in("tensor", "cols"));
    }

    #[test]
    fn test_mul_identity() {
        // Multiply by identity
        let a = ones(2.0, 3.0);
        let identity = eye(2.0);
        let result = mul(identity, a.clone());
        assert_eq!(result.shape(), (2, 3));
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(1, 2), 1.0);
    }

    #[test]
    fn test_mul_general() {
        // 2x3 * 3x2 = 2x2
        let a = TensorData::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = TensorData::from_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = mul(a, b);
        assert_eq!(result.shape(), (2, 2));
        // First element: 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        assert_eq!(result.get(0, 0), 22.0);
        // Second element: 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        assert_eq!(result.get(0, 1), 28.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_mul_dimension_mismatch() {
        let a = zeros(2.0, 3.0);
        let b = zeros(2.0, 2.0); // Wrong dimensions
        let _ = mul(a, b);
    }

    #[test]
    fn test_transpose() {
        let t = TensorData::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = transpose(t);
        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 2.0);
        assert_eq!(result.get(2, 1), 6.0);
    }

    #[test]
    fn test_transpose_square() {
        // Input: [[1, 2], [3, 4]]
        let t = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = transpose(t);
        // Expected: [[1, 3], [2, 4]]
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 3.0);
        assert_eq!(result.get(1, 0), 2.0);
        assert_eq!(result.get(1, 1), 4.0);
    }

    #[test]
    fn test_add() {
        let a = ones(2.0, 2.0);
        let b = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = add(a, b);
        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 3.0);
        assert_eq!(result.get(1, 0), 4.0);
        assert_eq!(result.get(1, 1), 5.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_add_dimension_mismatch() {
        let a = zeros(2.0, 2.0);
        let b = zeros(2.0, 3.0);
        let _ = add(a, b);
    }

    #[test]
    fn test_sub() {
        let a = TensorData::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let b = ones(2.0, 2.0);
        let result = sub(a, b);
        assert_eq!(result.get(0, 0), 4.0);
        assert_eq!(result.get(0, 1), 5.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 7.0);
    }

    #[test]
    fn test_scale() {
        let t = ones(2.0, 2.0);
        let result = scale(t, 3.0);
        assert_eq!(result.get(0, 0), 3.0);
        assert_eq!(result.get(1, 1), 3.0);
    }

    #[test]
    fn test_mul_registered() {
        assert!(is_known_in("tensor", "mul"));
    }

    #[test]
    fn test_transpose_registered() {
        assert!(is_known_in("tensor", "transpose"));
    }

    #[test]
    fn test_add_registered() {
        assert!(is_known_in("tensor", "add"));
    }

    #[test]
    fn test_sub_registered() {
        assert!(is_known_in("tensor", "sub"));
    }

    #[test]
    fn test_scale_registered() {
        assert!(is_known_in("tensor", "scale"));
    }

    // ============================================================================
    // Property Function Tests
    // ============================================================================

    #[test]
    fn test_trace_identity() {
        let t = eye(3.0);
        assert_eq!(trace(t), 3.0);
    }

    #[test]
    fn test_trace_general() {
        // [[1, 2], [3, 4]] -> trace = 1 + 4 = 5
        let t = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(trace(t), 5.0);
    }

    #[test]
    fn test_trace_non_square() {
        // [[1, 2, 3], [4, 5, 6]] -> trace = 1 + 5 = 6 (min(2,3) = 2 elements)
        let t = TensorData::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(trace(t), 6.0);
    }

    #[test]
    fn test_trace_registered() {
        assert!(is_known_in("tensor", "trace"));
    }

    #[test]
    fn test_norm_ones() {
        // 2x2 of ones: norm = sqrt(4) = 2
        let t = ones(2.0, 2.0);
        assert_eq!(norm(t), 2.0);
    }

    #[test]
    fn test_norm_general() {
        // [[3, 4]] -> norm = sqrt(9 + 16) = 5
        let t = TensorData::from_slice(1, 2, &[3.0, 4.0]);
        assert_eq!(norm(t), 5.0);
    }

    #[test]
    fn test_norm_registered() {
        assert!(is_known_in("tensor", "norm"));
    }

    #[test]
    fn test_det_identity() {
        let t = eye(3.0);
        assert!((det(t) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_2x2() {
        // [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
        let t = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!((det(t) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_det_registered() {
        assert!(is_known_in("tensor", "det"));
    }

    #[test]
    fn test_inv_identity() {
        let t = eye(3.0);
        let t_inv = inv(t.clone());
        // Identity inverse is identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((t_inv.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_inv_2x2() {
        // [[2, 0], [0, 2]] -> inv = [[0.5, 0], [0, 0.5]]
        let t = TensorData::from_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let t_inv = inv(t);
        assert!((t_inv.get(0, 0) - 0.5).abs() < 1e-10);
        assert!((t_inv.get(1, 1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_inv_registered() {
        assert!(is_known_in("tensor", "inv"));
    }

    // ============================================================================
    // Manipulation Function Tests
    // ============================================================================

    #[test]
    fn test_reshape_basic() {
        // [[1, 2, 3, 4, 5, 6]] 1x6 -> 2x3
        let t = TensorData::from_slice(1, 6, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reshape(t, 2.0, 3.0);
        assert_eq!(r.shape(), (2, 3));
        assert_eq!(r.get(0, 0), 1.0);
        assert_eq!(r.get(0, 2), 3.0);
        assert_eq!(r.get(1, 0), 4.0);
    }

    #[test]
    fn test_reshape_square() {
        // 4x1 -> 2x2
        let t = TensorData::from_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let r = reshape(t, 2.0, 2.0);
        assert_eq!(r.shape(), (2, 2));
    }

    #[test]
    #[should_panic(expected = "cannot reshape")]
    fn test_reshape_invalid() {
        let t = TensorData::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = reshape(t, 2.0, 2.0); // 6 elements can't fit in 4
    }

    #[test]
    fn test_reshape_registered() {
        assert!(is_known_in("tensor", "reshape"));
    }

    #[test]
    fn test_slice_basic() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let t = TensorData::from_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Extract rows 0-2, cols 1-3 -> [[2, 3], [5, 6]]
        let s = slice(t, 0.0, 2.0, 1.0, 3.0);
        assert_eq!(s.shape(), (2, 2));
        assert_eq!(s.get(0, 0), 2.0);
        assert_eq!(s.get(0, 1), 3.0);
        assert_eq!(s.get(1, 0), 5.0);
        assert_eq!(s.get(1, 1), 6.0);
    }

    #[test]
    fn test_slice_single_row() {
        let t = TensorData::from_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Extract row 1
        let s = slice(t, 1.0, 2.0, 0.0, 3.0);
        assert_eq!(s.shape(), (1, 3));
        assert_eq!(s.get(0, 0), 4.0);
        assert_eq!(s.get(0, 1), 5.0);
        assert_eq!(s.get(0, 2), 6.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_slice_out_of_bounds() {
        let t = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let _ = slice(t, 0.0, 3.0, 0.0, 2.0);
    }

    #[test]
    fn test_slice_registered() {
        assert!(is_known_in("tensor", "slice"));
    }

    #[test]
    fn test_solve_identity() {
        // Identity * x = b -> x = b
        let a = eye(2.0);
        let b = TensorData::from_slice(2, 1, &[3.0, 4.0]);
        let x = solve(a, b);
        assert!((x.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((x.get(1, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_diagonal() {
        // [[2, 0], [0, 3]] * x = [4, 9] -> x = [2, 3]
        let a = TensorData::from_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let b = TensorData::from_slice(2, 1, &[4.0, 9.0]);
        let x = solve(a, b);
        assert!((x.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((x.get(1, 0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_registered() {
        assert!(is_known_in("tensor", "solve"));
    }
}
