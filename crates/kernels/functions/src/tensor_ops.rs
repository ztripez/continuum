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
}
