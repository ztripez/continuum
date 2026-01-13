use crate::adapter::ContinuumDebugAdapter;
use dap::prelude::*;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tracing::error;

pub struct DapServer {
    adapter: Arc<ContinuumDebugAdapter>,
}

impl DapServer {
    pub fn new() -> Self {
        let (tx, _rx) = mpsc::channel(100);
        let adapter = Arc::new(ContinuumDebugAdapter::new(tx));

        Self { adapter }
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
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                if let Ok(event_json) = serde_json::to_string(&event) {
                    let output =
                        format!("Content-Length: {}\r\n\r\n{}", event_json.len(), event_json);
                    let mut w = writer_clone.lock().await;
                    if let Err(e) = w.write_all(output.as_bytes()).await {
                        error!(error = %e, "Failed to write DAP event");
                    }
                    let _ = w.flush().await;
                }
            }
        });

        loop {
            let mut line = String::new();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
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
                    let request: Request = serde_json::from_str(&request_str)?;

                    let response = self.adapter.handle_request(request).await;
                    let response_json = serde_json::to_string(&response)?;

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
                    let _ = w.flush().await;
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
