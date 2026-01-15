//! Tensor Types
//!
//! Dynamic tensors using Arc<[f64]> for cheap cloning and GPU compatibility.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Dynamic tensor with Arc-based storage for cheap cloning
#[derive(Clone, Debug)]
pub struct TensorData {
    pub rows: usize,
    pub cols: usize,
    #[allow(dead_code)]
    pub(crate) data: Arc<[f64]>, // Row-major storage
}

// Custom Serialize implementation
impl Serialize for TensorData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TensorData", 3)?;
        state.serialize_field("rows", &self.rows)?;
        state.serialize_field("cols", &self.cols)?;
        state.serialize_field("data", self.data.as_ref())?;
        state.end()
    }
}

// Custom Deserialize implementation
impl<'de> Deserialize<'de> for TensorData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Rows,
            Cols,
            Data,
        }

        struct TensorDataVisitor;

        impl<'de> serde::de::Visitor<'de> for TensorDataVisitor {
            type Value = TensorData;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct TensorData")
            }

            fn visit_map<V>(self, mut map: V) -> Result<TensorData, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut rows = None;
                let mut cols = None;
                let mut data: Option<Vec<f64>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Rows => {
                            if rows.is_some() {
                                return Err(serde::de::Error::duplicate_field("rows"));
                            }
                            rows = Some(map.next_value()?);
                        }
                        Field::Cols => {
                            if cols.is_some() {
                                return Err(serde::de::Error::duplicate_field("cols"));
                            }
                            cols = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(serde::de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }

                let rows = rows.ok_or_else(|| serde::de::Error::missing_field("rows"))?;
                let cols = cols.ok_or_else(|| serde::de::Error::missing_field("cols"))?;
                let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;

                Ok(TensorData {
                    rows,
                    cols,
                    data: data.into(),
                })
            }
        }

        const FIELDS: &[&str] = &["rows", "cols", "data"];
        deserializer.deserialize_struct("TensorData", FIELDS, TensorDataVisitor)
    }
}

impl TensorData {
    /// Create a new tensor filled with zeros
    pub fn new(rows: usize, cols: usize) -> Self {
        let data: Arc<[f64]> = vec![0.0; rows * cols].into();
        Self { rows, cols, data }
    }

    /// Create a tensor from a slice (copies data)
    pub fn from_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} does not match dimensions {}×{}",
            data.len(),
            rows,
            cols
        );
        Self {
            rows,
            cols,
            data: data.into(),
        }
    }

    /// Create a tensor from a Vec (moves data)
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} does not match dimensions {}×{}",
            data.len(),
            rows,
            cols
        );
        Self {
            rows,
            cols,
            data: data.into(),
        }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.rows && col < self.cols,
            "Index ({}, {}) out of bounds for {}×{} tensor",
            row,
            col,
            self.rows,
            self.cols
        );
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col) using copy-on-write
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(
            row < self.rows && col < self.cols,
            "Index ({}, {}) out of bounds for {}×{} tensor",
            row,
            col,
            self.rows,
            self.cols
        );
        Arc::make_mut(&mut self.data)[row * self.cols + col] = value;
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get raw data slice (row-major order)
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.data.as_ref() == other.data.as_ref()
    }
}

impl fmt::Display for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({}×{}) [", self.rows, self.cols)?;
        for row in 0..self.rows {
            if row > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for col in 0..self.cols {
                if col > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(row, col))?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zeros() {
        let t = TensorData::new(2, 3);
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 3);
        assert_eq!(t.len(), 6);
        assert_eq!(t.get(0, 0), 0.0);
        assert_eq!(t.get(1, 2), 0.0);
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = TensorData::from_slice(2, 3, &data);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 2.0);
        assert_eq!(t.get(0, 2), 3.0);
        assert_eq!(t.get(1, 0), 4.0);
        assert_eq!(t.get(1, 1), 5.0);
        assert_eq!(t.get(1, 2), 6.0);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = TensorData::from_vec(2, 2, data);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 1), 4.0);
    }

    #[test]
    fn test_set() {
        let mut t = TensorData::new(2, 2);
        t.set(0, 1, 5.0);
        assert_eq!(t.get(0, 1), 5.0);
        t.set(1, 0, 10.0);
        assert_eq!(t.get(1, 0), 10.0);
    }

    #[test]
    fn test_clone_is_cheap() {
        let t1 = TensorData::from_slice(100, 100, &vec![1.0; 10000]);
        let t2 = t1.clone();
        // Cloning just bumps refcount - both share same data
        assert_eq!(t1, t2);
        assert_eq!(Arc::as_ptr(&t1.data), Arc::as_ptr(&t2.data));
    }

    #[test]
    fn test_copy_on_write() {
        let t1 = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut t2 = t1.clone();

        // Before mutation, they share data
        assert_eq!(Arc::as_ptr(&t1.data), Arc::as_ptr(&t2.data));

        // After mutation, t2 gets its own copy
        t2.set(0, 0, 10.0);
        assert_ne!(Arc::as_ptr(&t1.data), Arc::as_ptr(&t2.data));
        assert_eq!(t1.get(0, 0), 1.0); // t1 unchanged
        assert_eq!(t2.get(0, 0), 10.0); // t2 modified
    }

    #[test]
    fn test_partial_eq() {
        let t1 = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let t2 = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let t3 = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 5.0]);
        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_shape() {
        let t = TensorData::new(3, 4);
        assert_eq!(t.shape(), (3, 4));
    }

    #[test]
    fn test_is_empty() {
        let t1 = TensorData::new(0, 0);
        assert!(t1.is_empty());

        let t2 = TensorData::new(2, 3);
        assert!(!t2.is_empty());
    }

    #[test]
    #[should_panic(expected = "does not match dimensions")]
    fn test_from_slice_wrong_size() {
        let _ = TensorData::from_slice(2, 3, &[1.0, 2.0]); // Wrong size
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let t = TensorData::new(2, 3);
        let _ = t.get(2, 0); // Row out of bounds
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_set_out_of_bounds() {
        let mut t = TensorData::new(2, 3);
        t.set(0, 3, 1.0); // Col out of bounds
    }

    #[test]
    fn test_display() {
        let t = TensorData::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let display = format!("{}", t);
        assert!(display.contains("Tensor(2×3)"));
        assert!(display.contains("1"));
        assert!(display.contains("6"));
    }
}
