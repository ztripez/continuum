//! File-based lens sink (JSON format)
//!
//! Writes field data to JSON files for analysis and debugging.
//!
//! # Architecture
//!
//! ```text
//! Simulation Thread              Background Writer Thread
//! ─────────────────              ────────────────────────
//! emit_tick()
//!   ├─ serialize JSON
//!   ├─ send(path, json) ────────→ receive msg
//!   └─ return immediately             └─ fs::write to disk
//!
//! flush()
//!   ├─ send Flush + ack ────────→ (drain queue)
//!   └─ block on ack                   └─ send ack
//!
//! close()
//!   ├─ send Close + manifest ───→ write manifest
//!   └─ join thread                    └─ exit loop
//! ```
//!
//! **Philosophy**: Serialization happens on the calling thread (CPU-bound, fast),
//! while file I/O is offloaded to a background thread to avoid blocking the
//! simulation.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use tracing::{debug, error, warn};

use super::{LensData, LensSink, LensSinkError, Result};
use crate::storage::FieldSample;
use crate::types::FieldId;

/// Queue depth for the background writer (bounded channel capacity).
const WRITER_QUEUE_DEPTH: usize = 16;

/// Manifest describing a lens snapshot run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensManifest {
    /// Unique run identifier (timestamp-based)
    pub run_id: String,

    /// Creation timestamp
    pub created_at: String,

    /// Simulation seed
    pub seed: u64,

    /// Total steps executed
    pub steps: u64,

    /// Stride (ticks between emissions)
    pub stride: u64,

    /// List of field IDs captured
    pub fields: Vec<String>,
}

/// Single tick of field data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    /// Simulation tick
    pub tick: u64,

    /// Simulation time in seconds
    pub time_seconds: f64,

    /// Field samples (field_id → samples)
    pub fields: std::collections::HashMap<String, Vec<FieldSample>>,
}

/// File-based sink configuration
#[derive(Debug, Clone)]
pub struct FileSinkConfig {
    /// Base output directory
    pub output_dir: PathBuf,

    /// Simulation seed (for manifest)
    pub seed: u64,

    /// Total steps (for manifest)
    pub steps: u64,

    /// Stride (for manifest)
    pub stride: u64,

    /// Field filter (empty = all fields)
    pub field_filter: Vec<FieldId>,
}

// ============================================================================
// Background Writer
// ============================================================================

/// Message sent to the background writer thread.
enum WriterMsg {
    /// Write a tick's JSON data to the given path.
    Write { path: PathBuf, json: String },

    /// Flush: drain pending writes, then ack.
    Flush { ack: SyncSender<()> },

    /// Close: write manifest, then ack and exit.
    Close {
        manifest_path: PathBuf,
        manifest_json: String,
        ack: SyncSender<Result<()>>,
    },
}

/// Spawn the background file writer thread.
fn spawn_writer_thread(rx: Receiver<WriterMsg>) -> JoinHandle<()> {
    thread::Builder::new()
        .name("file-sink-writer".to_string())
        .spawn(move || {
            debug!("File sink writer thread started");

            while let Ok(msg) = rx.recv() {
                match msg {
                    WriterMsg::Write { path, json } => {
                        if let Err(e) = fs::write(&path, &json) {
                            error!(path = %path.display(), error = %e, "Failed to write tick file");
                        }
                    }
                    WriterMsg::Flush { ack } => {
                        // All pending writes have been processed (channel is FIFO).
                        // Ack back. Ignore send error (caller may have timed out).
                        let _ = ack.send(());
                    }
                    WriterMsg::Close {
                        manifest_path,
                        manifest_json,
                        ack,
                    } => {
                        let result =
                            fs::write(&manifest_path, &manifest_json).map_err(LensSinkError::from);
                        let _ = ack.send(result);
                        debug!("File sink writer thread shutting down");
                        return;
                    }
                }
            }

            debug!("File sink writer thread exiting (channel closed)");
        })
        .expect("failed to spawn file-sink-writer thread")
}

// ============================================================================
// FileSink
// ============================================================================

/// File-based lens sink (JSON format) with background I/O.
///
/// Serialization (`serde_json::to_string_pretty`) runs on the calling thread.
/// File writes are offloaded to a dedicated background thread via a bounded
/// channel, keeping the simulation thread non-blocking for I/O.
pub struct FileSink {
    config: FileSinkConfig,
    run_dir: PathBuf,
    manifest_path: PathBuf,
    captured_fields: Vec<String>,
    is_closed: bool,
    writer_tx: Option<SyncSender<WriterMsg>>,
    writer_handle: Option<JoinHandle<()>>,
}

impl FileSink {
    /// Create a new file sink with a background writer thread.
    pub fn new(config: FileSinkConfig) -> Result<Self> {
        // Create run directory with timestamp
        let run_id = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let run_dir = config.output_dir.join(&run_id);

        fs::create_dir_all(&run_dir).map_err(|e| {
            LensSinkError::Config(format!("Failed to create output directory: {}", e))
        })?;

        let manifest_path = run_dir.join("manifest.json");

        let (tx, rx) = mpsc::sync_channel(WRITER_QUEUE_DEPTH);
        let handle = spawn_writer_thread(rx);

        Ok(Self {
            config,
            run_dir,
            manifest_path,
            captured_fields: Vec::new(),
            is_closed: false,
            writer_tx: Some(tx),
            writer_handle: Some(handle),
        })
    }

    fn check_not_closed(&self) -> Result<()> {
        if self.is_closed {
            Err(LensSinkError::AlreadyClosed)
        } else {
            Ok(())
        }
    }

    /// Send a message to the writer, handling disconnection.
    fn send_to_writer(&self, msg: WriterMsg) -> Result<()> {
        let tx = self
            .writer_tx
            .as_ref()
            .ok_or(LensSinkError::Config("writer thread not running".into()))?;

        match tx.try_send(msg) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(msg)) => {
                // Queue full — block rather than drop data (unlike checkpoints,
                // lens data loss is unacceptable for analysis correctness).
                warn!("File sink writer queue full, blocking until space available");
                tx.send(msg)
                    .map_err(|_| LensSinkError::Config("file sink writer thread died".into()))?;
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(LensSinkError::Config("file sink writer thread died".into()))
            }
        }
    }
}

impl LensSink for FileSink {
    fn emit_tick(&mut self, tick: u64, time_seconds: f64, data: LensData) -> Result<()> {
        self.check_not_closed()?;

        // Track captured fields
        for field_id in data.fields.keys() {
            let field_str = field_id.to_string();
            if !self.captured_fields.contains(&field_str) {
                self.captured_fields.push(field_str);
            }
        }

        // Convert to serializable format
        let tick_data = TickData {
            tick,
            time_seconds,
            fields: data
                .fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        };

        // Serialize on the calling thread (CPU-bound, fast)
        let json = serde_json::to_string_pretty(&tick_data)
            .map_err(|e| LensSinkError::Serialization(e.to_string()))?;

        // Offload file write to background thread
        let filename = format!("tick_{:06}.json", tick);
        let file_path = self.run_dir.join(filename);
        self.send_to_writer(WriterMsg::Write {
            path: file_path,
            json,
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.check_not_closed()?;

        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        self.send_to_writer(WriterMsg::Flush { ack: ack_tx })?;

        // Block until the writer has drained all pending writes
        ack_rx
            .recv()
            .map_err(|_| LensSinkError::Config("file sink writer thread died during flush".into()))
    }

    fn close(&mut self) -> Result<()> {
        if self.is_closed {
            return Ok(());
        }

        // Build manifest
        let manifest = LensManifest {
            run_id: self
                .run_dir
                .file_name()
                .expect("run_dir must have a filename component")
                .to_string_lossy()
                .to_string(),
            created_at: chrono::Local::now().to_rfc3339(),
            seed: self.config.seed,
            steps: self.config.steps,
            stride: self.config.stride,
            fields: self.captured_fields.clone(),
        };

        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| LensSinkError::Serialization(e.to_string()))?;

        // Send close message with manifest data, wait for ack
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        self.send_to_writer(WriterMsg::Close {
            manifest_path: self.manifest_path.clone(),
            manifest_json,
            ack: ack_tx,
        })?;

        let write_result = ack_rx.recv().map_err(|_| {
            LensSinkError::Config("file sink writer thread died during close".into())
        })?;

        // Join the writer thread (it exits after processing Close)
        if let Some(handle) = self.writer_handle.take()
            && let Err(e) = handle.join()
        {
            error!("File sink writer thread panicked: {:?}", e);
        }

        // Drop the sender so it's clear the thread is gone
        self.writer_tx.take();
        self.is_closed = true;

        write_result
    }

    fn output_path(&self) -> Option<PathBuf> {
        Some(self.run_dir.clone())
    }
}

impl Drop for FileSink {
    fn drop(&mut self) {
        if !self.is_closed {
            // Best-effort: drop the sender to signal shutdown, then join
            self.writer_tx.take();
            if let Some(handle) = self.writer_handle.take()
                && let Err(e) = handle.join()
            {
                error!("File sink writer thread panicked during drop: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn test_file_sink_create() {
        let temp_dir = std::env::temp_dir().join("continuum_test_file_sink");
        let _ = fs::remove_dir_all(&temp_dir); // Clean up from previous runs

        let config = FileSinkConfig {
            output_dir: temp_dir.clone(),
            seed: 42,
            steps: 100,
            stride: 10,
            field_filter: Vec::new(),
        };

        let sink = FileSink::new(config);
        assert!(sink.is_ok());

        // Check run directory was created
        assert!(temp_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_file_sink_emit_and_close() {
        let temp_dir = std::env::temp_dir().join("continuum_test_file_sink_emit");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = FileSinkConfig {
            output_dir: temp_dir.clone(),
            seed: 42,
            steps: 10,
            stride: 1,
            field_filter: Vec::new(),
        };

        let mut sink = FileSink::new(config).unwrap();

        // Emit a tick
        let field_id = FieldId::new("test.field");
        let mut fields = IndexMap::new();
        fields.insert(
            field_id,
            vec![FieldSample {
                position: [0.0, 0.0, 0.0],
                value: crate::types::Value::Scalar(42.0),
            }],
        );

        let data = LensData { fields };

        sink.emit_tick(0, 0.0, data).unwrap();
        sink.close().unwrap();

        // Check manifest was written
        let manifest_path = sink.manifest_path.clone();
        assert!(manifest_path.exists());

        let manifest_content = fs::read_to_string(&manifest_path).unwrap();
        let manifest: LensManifest = serde_json::from_str(&manifest_content).unwrap();

        assert_eq!(manifest.seed, 42);
        assert_eq!(manifest.steps, 10);
        assert_eq!(manifest.stride, 1);
        assert_eq!(manifest.fields, vec!["test.field"]);

        // Check tick file was written
        let tick_files: Vec<_> = fs::read_dir(&sink.run_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|s| s == "json").unwrap_or(false))
            .filter(|e| e.file_name().to_string_lossy().starts_with("tick_"))
            .collect();

        assert_eq!(tick_files.len(), 1);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
