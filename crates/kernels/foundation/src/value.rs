use serde::{Deserialize, Serialize};

/// Runtime value for signals and fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Single scalar value (e.g., temperature, pressure, density).
    Scalar(f64),
    /// 2D vector (e.g., UV coordinates, 2D position).
    Vec2([f64; 2]),
    /// 3D vector (e.g., position, velocity, force).
    Vec3([f64; 3]),
    /// 4D vector (e.g., quaternion, RGBA color).
    Vec4([f64; 4]),
    // TODO: Mat4, Tensor, Grid, Seq
}

impl Value {
    /// Attempt to get the value as a scalar.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
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

    /// Get a component by name (x, y, z, w).
    pub fn component(&self, name: &str) -> Option<f64> {
        match (self, name) {
            (Value::Scalar(v), _) => Some(*v),
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
