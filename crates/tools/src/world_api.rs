use serde::{Deserialize, Serialize};

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
    use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

    /// Write a message to an async stream with a 4-byte length prefix.
    pub async fn write_message<W: AsyncWrite + Unpin>(
        writer: &mut W,
        message: &WorldMessage,
    ) -> Result<(), std::io::Error> {
        let data = serde_json::to_vec(message)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
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
        serde_json::from_slice(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
