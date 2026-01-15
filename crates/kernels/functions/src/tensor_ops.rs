//! Tensor Operations
//!
//! Functions for creating and manipulating dynamic tensors.

use continuum_foundation::TensorData;
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
}
