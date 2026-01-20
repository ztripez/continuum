use crate::tensor::TensorData;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Runtime value for signals and fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Single scalar value (e.g., temperature, pressure, density).
    Scalar(f64),
    /// Boolean value (logic).
    Boolean(bool),
    /// Integer value (counts, indices).
    Integer(i64),
    /// 2D vector (e.g., UV coordinates, 2D position).
    Vec2([f64; 2]),
    /// 3D vector (e.g., position, velocity, force).
    Vec3([f64; 3]),
    /// 4D vector (e.g., RGBA color).
    Vec4([f64; 4]),
    /// Quaternion (w, x, y, z).
    Quat([f64; 4]),
    /// 2x2 matrix (column-major: [m00, m10, m01, m11]).
    Mat2([f64; 4]),
    /// 3x3 matrix (column-major: [m00, m10, m20, m01, m11, m21, m02, m12, m22]).
    Mat3([f64; 9]),
    /// 4x4 matrix (column-major: 4 columns × 4 rows = 16 elements).
    Mat4([f64; 16]),
    /// Dynamic tensor (row-major storage with Arc for cheap cloning).
    Tensor(TensorData),
    /// Structured payload with named fields, wrapped in Arc for cheap cloning.
    Map(Arc<Vec<(String, Value)>>),
    // TODO: Grid, Seq
}

impl Value {
    /// Create a new map value from fields.
    pub fn map(fields: Vec<(String, Value)>) -> Self {
        Value::Map(Arc::new(fields))
    }
}

impl Value {
    /// Attempt to get the value as a scalar.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
            Value::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Attempt to get the value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 2D vector.
    pub fn as_vec2(&self) -> Option<[f64; 2]> {
        match self {
            Value::Vec2(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 3D vector.
    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 4D vector.
    pub fn as_vec4(&self) -> Option<[f64; 4]> {
        match self {
            Value::Vec4(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a quaternion.
    pub fn as_quat(&self) -> Option<[f64; 4]> {
        match self {
            Value::Quat(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 2x2 matrix (column-major).
    pub fn as_mat2(&self) -> Option<[f64; 4]> {
        match self {
            Value::Mat2(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 3x3 matrix (column-major).
    pub fn as_mat3(&self) -> Option<[f64; 9]> {
        match self {
            Value::Mat3(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 4x4 matrix (column-major).
    pub fn as_mat4(&self) -> Option<[f64; 16]> {
        match self {
            Value::Mat4(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a tensor.
    pub fn as_tensor(&self) -> Option<&TensorData> {
        match self {
            Value::Tensor(v) => Some(v),
            _ => None,
        }
    }

    /// Attempt to get the value as a structured map.
    pub fn as_map(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Map(v) => Some(v),
            _ => None,
        }
    }

    /// Get a component by name (x, y, z, w).

    pub fn component(&self, name: &str) -> Option<f64> {
        match (self, name) {
            (Value::Scalar(v), _) => Some(*v),
            (Value::Integer(v), _) => Some(*v as f64),
            (Value::Vec2(v), "x") => Some(v[0]),
            (Value::Vec2(v), "y") => Some(v[1]),
            (Value::Vec3(v), "x") => Some(v[0]),
            (Value::Vec3(v), "y") => Some(v[1]),
            (Value::Vec3(v), "z") => Some(v[2]),
            (Value::Vec4(v), "x") => Some(v[0]),
            (Value::Vec4(v), "y") => Some(v[1]),
            (Value::Vec4(v), "z") => Some(v[2]),
            (Value::Vec4(v), "w") => Some(v[3]),
            (Value::Quat(v), "w") => Some(v[0]),
            (Value::Quat(v), "x") => Some(v[1]),
            (Value::Quat(v), "y") => Some(v[2]),
            (Value::Quat(v), "z") => Some(v[3]),
            // Matrix component access (mXY where X=row, Y=col)
            // Storage is column-major: [col0_row0, col0_row1, ..., col1_row0, col1_row1, ...]
            (Value::Mat2(v), "m00") => Some(v[0]),
            (Value::Mat2(v), "m10") => Some(v[1]),
            (Value::Mat2(v), "m01") => Some(v[2]),
            (Value::Mat2(v), "m11") => Some(v[3]),
            (Value::Mat3(v), component) if component.starts_with('m') && component.len() == 3 => {
                let row = component.as_bytes()[1] - b'0';
                let col = component.as_bytes()[2] - b'0';
                if row < 3 && col < 3 {
                    Some(v[(col * 3 + row) as usize])
                } else {
                    None
                }
            }
            (Value::Mat4(v), component) if component.starts_with('m') && component.len() == 3 => {
                let row = component.as_bytes()[1] - b'0';
                let col = component.as_bytes()[2] - b'0';
                if row < 4 && col < 4 {
                    Some(v[(col * 4 + row) as usize])
                } else {
                    None
                }
            }
            (Value::Map(fields), _) => fields
                .iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, v)| v.as_scalar()),
            _ => None,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Scalar(0.0)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Scalar(v) => write!(f, "{:.4}", v),
            Value::Boolean(v) => write!(f, "{}", v),
            Value::Integer(v) => write!(f, "{}", v),
            Value::Vec2(v) => write!(f, "[{:.4}, {:.4}]", v[0], v[1]),
            Value::Vec3(v) => write!(f, "[{:.4}, {:.4}, {:.4}]", v[0], v[1], v[2]),
            Value::Vec4(v) => write!(f, "[{:.4}, {:.4}, {:.4}, {:.4}]", v[0], v[1], v[2], v[3]),
            Value::Quat(v) => write!(f, "[{:.4}, {:.4}, {:.4}, {:.4}]", v[0], v[1], v[2], v[3]),
            Value::Mat2(v) => {
                // Display as 2x2 matrix (column-major storage, display row-major for readability)
                write!(
                    f,
                    "[[{:.3}, {:.3}], [{:.3}, {:.3}]]",
                    v[0],
                    v[2], // row 0: m00, m01
                    v[1],
                    v[3]
                ) // row 1: m10, m11
            }
            Value::Mat3(v) => {
                // Display as 3x3 matrix (column-major storage, display row-major for readability)
                write!(
                    f,
                    "[[{:.3}, {:.3}, {:.3}], [{:.3}, {:.3}, {:.3}], [{:.3}, {:.3}, {:.3}]]",
                    v[0],
                    v[3],
                    v[6], // row 0: m00, m01, m02
                    v[1],
                    v[4],
                    v[7], // row 1: m10, m11, m12
                    v[2],
                    v[5],
                    v[8]
                ) // row 2: m20, m21, m22
            }
            Value::Mat4(v) => {
                // Display as 4x4 matrix (column-major storage, display row-major for readability)
                write!(
                    f,
                    "[[{:.2}, {:.2}, {:.2}, {:.2}], [{:.2}, {:.2}, {:.2}, {:.2}], [{:.2}, {:.2}, {:.2}, {:.2}], [{:.2}, {:.2}, {:.2}, {:.2}]]",
                    v[0],
                    v[4],
                    v[8],
                    v[12], // row 0
                    v[1],
                    v[5],
                    v[9],
                    v[13], // row 1
                    v[2],
                    v[6],
                    v[10],
                    v[14], // row 2
                    v[3],
                    v[7],
                    v[11],
                    v[15]
                ) // row 3
            }
            Value::Tensor(t) => write!(f, "{}", t),
            Value::Map(v) => {
                let items: Vec<_> = v.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
                write!(f, "{{{}}}", items.join(", "))
            }
        }
    }
}

/// Trait for converting a reference to a Value into a concrete type.
/// Used by kernel macros to unpack arguments.
pub trait FromValue: Sized {
    fn from_value(value: &Value) -> Option<Self>;
}

/// Trait for converting a concrete type into a Value.
/// Used by kernel macros to pack return values.
pub trait IntoValue {
    fn into_value(self) -> Value;
}

impl FromValue for f64 {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_scalar()
    }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value {
        Value::Scalar(self)
    }
}

impl FromValue for bool {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_bool()
    }
}

impl IntoValue for bool {
    fn into_value(self) -> Value {
        Value::Boolean(self)
    }
}

impl FromValue for i64 {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_int()
    }
}

impl IntoValue for i64 {
    fn into_value(self) -> Value {
        Value::Integer(self)
    }
}

impl FromValue for [f64; 2] {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_vec2()
    }
}

impl IntoValue for [f64; 2] {
    fn into_value(self) -> Value {
        Value::Vec2(self)
    }
}

impl FromValue for [f64; 3] {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_vec3()
    }
}

impl IntoValue for [f64; 3] {
    fn into_value(self) -> Value {
        Value::Vec3(self)
    }
}

impl FromValue for [f64; 4] {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_vec4()
    }
}

impl IntoValue for [f64; 4] {
    fn into_value(self) -> Value {
        Value::Vec4(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat(pub [f64; 4]);

impl FromValue for Quat {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_quat().map(Quat)
    }
}

impl IntoValue for Quat {
    fn into_value(self) -> Value {
        Value::Quat(self.0)
    }
}

pub struct Mat2(pub [f64; 4]);

impl FromValue for Mat2 {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_mat2().map(Mat2)
    }
}

impl IntoValue for Mat2 {
    fn into_value(self) -> Value {
        Value::Mat2(self.0)
    }
}

pub struct Mat3(pub [f64; 9]);

impl FromValue for Mat3 {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_mat3().map(Mat3)
    }
}

impl IntoValue for Mat3 {
    fn into_value(self) -> Value {
        Value::Mat3(self.0)
    }
}

pub struct Mat4(pub [f64; 16]);

impl FromValue for Mat4 {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_mat4().map(Mat4)
    }
}

impl IntoValue for Mat4 {
    fn into_value(self) -> Value {
        Value::Mat4(self.0)
    }
}

impl FromValue for TensorData {
    fn from_value(value: &Value) -> Option<Self> {
        value.as_tensor().cloned()
    }
}

impl IntoValue for TensorData {
    fn into_value(self) -> Value {
        Value::Tensor(self)
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_conversions() {
        let v = 42.0;
        let value = v.into_value();
        assert!(matches!(value, Value::Scalar(42.0)));
        assert_eq!(f64::from_value(&value), Some(42.0));
    }

    #[test]
    fn test_bool_conversions() {
        let v = true;
        let value = v.into_value();
        assert!(matches!(value, Value::Boolean(true)));
        assert_eq!(bool::from_value(&value), Some(true));
    }

    #[test]
    fn test_int_conversions() {
        let v = 123i64;
        let value = v.into_value();
        assert!(matches!(value, Value::Integer(123)));
        assert_eq!(i64::from_value(&value), Some(123));
        // Implicit casting
        assert_eq!(f64::from_value(&value), Some(123.0));
    }

    #[test]
    fn test_vec3_conversions() {
        let v = [1.0, 2.0, 3.0];
        let value = v.into_value();
        match &value {
            Value::Vec3(arr) => assert_eq!(arr, &[1.0, 2.0, 3.0]),
            _ => panic!("Expected Vec3"),
        }
        assert_eq!(<[f64; 3]>::from_value(&value), Some([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_quat_conversions() {
        let q = Quat([1.0, 0.0, 0.0, 0.0]);
        let value = q.into_value();
        match &value {
            Value::Quat(arr) => assert_eq!(arr, &[1.0, 0.0, 0.0, 0.0]),
            _ => panic!("Expected Quat"),
        }
        assert_eq!(
            Quat::from_value(&value).map(|q| q.0),
            Some([1.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn test_mat3_conversions() {
        // Identity matrix (column-major storage)
        let m = Mat3([
            1.0, 0.0, 0.0, // column 0
            0.0, 1.0, 0.0, // column 1
            0.0, 0.0, 1.0, // column 2
        ]);
        let value = m.into_value();
        match &value {
            Value::Mat3(arr) => assert_eq!(arr, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            _ => panic!("Expected Mat3"),
        }
        assert_eq!(
            Mat3::from_value(&value).map(|m| m.0),
            Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn test_mat3_component_access() {
        let m = Value::Mat3([
            1.0, 2.0, 3.0, // column 0: m00=1, m10=2, m20=3
            4.0, 5.0, 6.0, // column 1: m01=4, m11=5, m21=6
            7.0, 8.0, 9.0, // column 2: m02=7, m12=8, m22=9
        ]);

        // Test component access
        assert_eq!(m.component("m00"), Some(1.0));
        assert_eq!(m.component("m10"), Some(2.0));
        assert_eq!(m.component("m20"), Some(3.0));
        assert_eq!(m.component("m01"), Some(4.0));
        assert_eq!(m.component("m11"), Some(5.0));
        assert_eq!(m.component("m21"), Some(6.0));
        assert_eq!(m.component("m02"), Some(7.0));
        assert_eq!(m.component("m12"), Some(8.0));
        assert_eq!(m.component("m22"), Some(9.0));
    }

    #[test]
    fn test_mat2_mat4_conversions() {
        let m2 = Mat2([1.0, 2.0, 3.0, 4.0]);
        let v2 = m2.into_value();
        assert!(matches!(v2, Value::Mat2(_)));
        assert_eq!(
            Mat2::from_value(&v2).map(|m| m.0),
            Some([1.0, 2.0, 3.0, 4.0])
        );

        let m4 = Mat4([1.0; 16]);
        let v4 = m4.into_value();
        assert!(matches!(v4, Value::Mat4(_)));
        assert_eq!(Mat4::from_value(&v4).map(|m| m.0), Some([1.0; 16]));
    }

    #[test]
    fn test_tensor_conversions() {
        let t = TensorData::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = t.clone().into_value();
        assert!(matches!(v, Value::Tensor(_)));

        let t2 = TensorData::from_value(&v).unwrap();
        assert_eq!(t2.shape(), (2, 3));
        assert_eq!(t2.get(0, 0), 1.0);
        assert_eq!(t2.get(1, 2), 6.0);
    }

    #[test]
    fn test_tensor_display() {
        let t = TensorData::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let v = Value::Tensor(t);
        let display = format!("{}", v);
        assert!(display.contains("Tensor(2×2)"));
    }
}
