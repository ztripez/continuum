use std::collections::BTreeMap;

use continuum_foundation::Value;
use serde::{Deserialize, Serialize};

/// Transport-agnostic value representation for world APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiValue {
    Scalar(f64),
    Boolean(bool),
    Integer(i64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Quat([f64; 4]),
    Mat2([f64; 4]),
    Mat3([f64; 9]),
    Mat4([f64; 16]),
    Struct(BTreeMap<String, ApiValue>),
}

impl ApiValue {
    /// Convert to the canonical runtime value.
    pub fn to_value(&self) -> Value {
        match self {
            ApiValue::Scalar(value) => Value::Scalar(*value),
            ApiValue::Boolean(value) => Value::Boolean(*value),
            ApiValue::Integer(value) => Value::Integer(*value),
            ApiValue::Vec2(value) => Value::Vec2(*value),
            ApiValue::Vec3(value) => Value::Vec3(*value),
            ApiValue::Vec4(value) => Value::Vec4(*value),
            ApiValue::Quat(value) => Value::Quat(continuum_foundation::Quat(*value)),
            ApiValue::Mat2(value) => Value::Mat2(continuum_foundation::Mat2(*value)),
            ApiValue::Mat3(value) => Value::Mat3(continuum_foundation::Mat3(*value)),
            ApiValue::Mat4(value) => Value::Mat4(continuum_foundation::Mat4(*value)),
            ApiValue::Struct(fields) => {
                let items = fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect();
                Value::map(items)
            }
        }
    }

    /// Convert from the canonical runtime value.
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Scalar(v) => ApiValue::Scalar(*v),
            Value::Boolean(v) => ApiValue::Boolean(*v),
            Value::Integer(v) => ApiValue::Integer(*v),
            Value::Vec2(v) => ApiValue::Vec2(*v),
            Value::Vec3(v) => ApiValue::Vec3(*v),
            Value::Vec4(v) => ApiValue::Vec4(*v),
            Value::Quat(v) => ApiValue::Quat(v.0),
            Value::Mat2(v) => ApiValue::Mat2(v.0),
            Value::Mat3(v) => ApiValue::Mat3(v.0),
            Value::Mat4(v) => ApiValue::Mat4(v.0),
            Value::Map(map) => {
                let fields = map
                    .as_ref()
                    .iter()
                    .map(|(key, value)| (key.to_string(), Self::from_value(value)))
                    .collect();
                ApiValue::Struct(fields)
            }
            Value::Tensor(_) => {
                panic!("Fail Loud: ApiValue cannot represent Value::Tensor yet. Use a field or observer for large data.");
            }
        }
    }
}

/// A request to the world execution API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldRequest {
    pub id: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// A response from the world execution API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldResponse {
    pub id: u64,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// An event emitted by a world execution API server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub payload: serde_json::Value,
}

/// API message envelope (transport-agnostic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldMessage {
    Request(WorldRequest),
    Response(WorldResponse),
    Event(WorldEvent),
}

/// Framing helper for async streams.
pub mod framing {
    use super::*;
    use tokio::io::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};

    /// Write a message to an async stream with a 4-byte length prefix.
    pub async fn write_message<W: AsyncWrite + Unpin>(
        writer: &mut W,
        message: &WorldMessage,
    ) -> Result<(), std::io::Error> {
        let data = serde_json::to_vec(message).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writer.write_u32(data.len() as u32).await?;
        writer.write_all(&data).await?;
        Ok(())
    }

    /// Read a message from an async stream with a 4-byte length prefix.
    pub async fn read_message<R: AsyncRead + Unpin>(
        reader: &mut R,
    ) -> Result<WorldMessage, std::io::Error> {
        let len = reader.read_u32().await?;
        let mut buffer = vec![0u8; len as usize];
        reader.read_exact(&mut buffer).await?;
        serde_json::from_slice(&buffer).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
