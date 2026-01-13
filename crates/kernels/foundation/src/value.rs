use serde::{Deserialize, Serialize};

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
    /// 4D vector (e.g., quaternion, RGBA color).
    Vec4([f64; 4]),
    // TODO: Mat3, Mat4, Tensor, Grid, Seq
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
            _ => None,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Scalar(0.0)
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
}
