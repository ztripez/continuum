use crate::adapter::ContinuumDebugAdapter;
use dap::prelude::*;
use serde::Serialize;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tracing::error;

/// Wrapper that adds seq and type fields required by DAP protocol
#[derive(Serialize)]
struct ProtocolMessage<T> {
    seq: i64,
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(flatten)]
    body: T,
}

pub struct DapServer {
    adapter: Arc<ContinuumDebugAdapter>,
    seq_number: Arc<Mutex<i64>>,
}

impl DapServer {
    pub fn new() -> Self {
        let (tx, _rx) = mpsc::channel(100);
        let adapter = Arc::new(ContinuumDebugAdapter::new(tx));

        Self {
            adapter,
            seq_number: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn run<R, W>(&self, reader: R, writer: W) -> Result<(), Box<dyn std::error::Error>>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        let mut reader = BufReader::new(reader);
        let writer = Arc::new(Mutex::new(BufWriter::new(writer)));

        let (event_tx, mut event_rx) = mpsc::channel::<dap::events::Event>(100);

        // Update adapter to use this event_tx
        self.adapter.set_event_sender(event_tx).await;

        let writer_clone = writer.clone();
        let seq_clone = self.seq_number.clone();
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                tracing::info!("Sending event: {:?}", event);
                let mut seq = seq_clone.lock().await;
                *seq += 1;
                let wrapped = ProtocolMessage {
                    seq: *seq,
                    msg_type: "event".to_string(),
                    body: event,
                };
                drop(seq);

                if let Ok(event_json) = serde_json::to_string(&wrapped) {
                    tracing::debug!("Event JSON: {}", event_json);
                    let output =
                        format!("Content-Length: {}\r\n\r\n{}", event_json.len(), event_json);
                    let mut w = writer_clone.lock().await;
                    if let Err(e) = w.write_all(output.as_bytes()).await {
                        error!(error = %e, "Failed to write DAP event");
                    }
                    if let Err(e) = w.flush().await {
                        error!(error = %e, "Failed to flush DAP event");
                    }
                    tracing::debug!("Event sent and flushed");
                }
            }
        });

        loop {
            let mut line = String::new();
            tracing::trace!("Waiting for next message...");
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    tracing::info!("EOF received, shutting down");
                    break;
                }
                Ok(_) => {
                    tracing::trace!("Read line: {}", line.trim());
                    if !line.starts_with("Content-Length:") {
                        continue;
                    }
                    let length: usize =
                        line.trim_start_matches("Content-Length:").trim().parse()?;

                    // Skip empty line
                    line.clear();
                    reader.read_line(&mut line).await?;

                    // Read body
                    let mut body = vec![0u8; length];
                    reader.read_exact(&mut body).await?;

                    let request_str = String::from_utf8(body)?;
                    tracing::debug!("Received request: {}", request_str);
                    let request: Request = serde_json::from_str(&request_str)?;

                    let response = self.adapter.handle_request(request).await;

                    // Wrap response with seq and type fields required by DAP protocol
                    let mut seq = self.seq_number.lock().await;
                    *seq += 1;
                    let wrapped = ProtocolMessage {
                        seq: *seq,
                        msg_type: "response".to_string(),
                        body: response,
                    };
                    drop(seq);

                    let response_json = serde_json::to_string(&wrapped)?;
                    tracing::debug!("Sending response: {}", response_json);

                    let output = format!(
                        "Content-Length: {}\r\n\r\n{}",
                        response_json.len(),
                        response_json
                    );

                    let mut w = writer.lock().await;
                    if let Err(e) = w.write_all(output.as_bytes()).await {
                        error!(error = %e, "Failed to write DAP response");
                        break;
                    }
                    if let Err(e) = w.flush().await {
                        error!(error = %e, "Failed to flush DAP response");
                        break;
                    }
                    tracing::debug!("Response sent and flushed");
                }
                Err(e) => {
                    error!(error = %e, "DAP Server Error");
                    break;
                }
            }
        }

        Ok(())
    }
}
