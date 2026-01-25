//! Checkpoint persistence for simulation state.
//!
//! Checkpoints enable crash recovery and long-running simulations by serializing
//! complete causal state to disk. The checkpoint system is designed with these principles:
//!
//! - **Non-blocking I/O**: Simulation never blocks on checkpoint writes
//! - **Deterministic resume**: Same world + checkpoint → identical continuation
//! - **Fail loudly**: World IR mismatch fails resume (unless --force)
//! - **Portable**: Bincode + zstd compression for cross-platform compatibility
//!
//! # Architecture
//!
//! ```text
//! Runtime Thread                Background Writer Thread
//! ──────────────                ─────────────────────────
//! request_checkpoint()          
//!   ├─ clone state              
//!   ├─ try_send to queue ───────→ receive job
//!   └─ return immediately           ├─ serialize to bincode
//!                                   ├─ compress with zstd
//!                                   └─ write to disk
//! ```
//!
//! **Philosophy**: Lost checkpoint > blocked simulation
//!
//! If the queue is full, checkpoint requests are dropped with a warning.
//! The simulation continues without interruption.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::SystemTime;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::soa_storage::MemberSignalBuffer;
use crate::storage::{EntityStorage, SignalStorage};
use crate::types::{EraId, StratumId, Value};

/// Checkpoint format version (increment on breaking changes).
pub const CHECKPOINT_VERSION: u32 = 1;

/// Default queue depth for checkpoint writer (bounded channel capacity).
pub const DEFAULT_QUEUE_DEPTH: usize = 3;

/// Default zstd compression level (3 = good balance of speed/size).
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

// ============================================================================
// Checkpoint Format
// ============================================================================

/// Checkpoint file header with metadata and validation info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHeader {
    /// Checkpoint format version
    pub version: u32,

    /// Blake3 hash of CompiledWorld IR (for validation on resume)
    pub world_ir_hash: [u8; 32],

    /// Simulation tick at checkpoint
    pub tick: u64,

    /// Simulation time in seconds at checkpoint
    pub sim_time: f64,

    /// World seed (for determinism)
    pub seed: u64,

    /// Current era at checkpoint
    pub current_era: EraId,

    /// Timestamp when checkpoint was created
    pub created_at: SystemTime,

    /// Git commit hash of world directory (if available)
    pub world_git_hash: Option<String>,
}

/// Era configuration state (subset needed for validation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EraConfigSnapshot {
    pub dt: f64,
    pub strata_count: usize,
}

/// Stratum execution state (for gated execution).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StratumState {
    /// Cadence counter for gated execution
    pub cadence_counter: u64,

    /// Whether stratum is currently gated (not executing)
    pub is_gated: bool,
}

/// Serializable member signal data (extracted from SoA buffers).
///
/// The SoA buffers use custom allocators and cannot be directly serialized.
/// This struct provides a portable representation that can be reconstructed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberSignalData {
    /// Signal name → (type_id, instance values)
    /// Values stored as generic Value enum for portability
    pub signals: IndexMap<String, Vec<(usize, Value)>>,

    /// Entity type → instance count
    pub entity_instance_counts: IndexMap<String, usize>,

    /// Total instance count across all entities
    pub total_instance_count: usize,
}

impl MemberSignalData {
    /// Extract member signal data from SoA buffer.
    ///
    /// This converts the SoA representation (with custom allocators) into a
    /// portable format that can be serialized.
    pub fn from_buffer(buffer: &MemberSignalBuffer) -> Result<Self, CheckpointError> {
        let mut signals = IndexMap::new();
        let registry = buffer.registry();

        // Extract all registered signals
        for (signal_name, _meta) in registry.iter() {
            let instance_count = buffer.instance_count_for_signal(signal_name);
            let mut values = Vec::with_capacity(instance_count);

            // Extract each instance value
            for instance_idx in 0..instance_count {
                if let Some(value) = buffer.get_current(signal_name, instance_idx) {
                    values.push((instance_idx, value));
                }
            }

            signals.insert(signal_name.clone(), values);
        }

        Ok(Self {
            signals,
            entity_instance_counts: buffer.entity_instance_counts().clone(),
            total_instance_count: buffer.instance_count(),
        })
    }

    /// Restore member signal data into SoA buffer.
    ///
    /// This reconstructs the SoA representation from the portable serialized format.
    pub fn restore_into_buffer(
        &self,
        buffer: &mut MemberSignalBuffer,
    ) -> Result<(), CheckpointError> {
        // Restore all signal values
        for (signal_name, values) in &self.signals {
            for (instance_idx, value) in values {
                buffer
                    .set_current(signal_name, *instance_idx, value.clone())
                    .map_err(|e| CheckpointError::Serialization(e))?;
            }
        }

        Ok(())
    }
}

/// Complete checkpoint state (all causal state for resume).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    /// Global signal values (current + previous tick)
    pub signals: SignalStorage,

    /// Entity instances
    pub entities: EntityStorage,

    /// Member signal data (extracted from SoA buffers)
    pub member_signals: MemberSignalData,

    /// Era configurations (for validation)
    pub era_configs: IndexMap<EraId, EraConfigSnapshot>,

    /// Stratum states (cadence counters, gate status)
    pub stratum_states: IndexMap<StratumId, StratumState>,
}

/// Complete checkpoint (header + state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub header: CheckpointHeader,
    pub state: CheckpointState,
}

// ============================================================================
// Checkpoint Writer (Non-Blocking Background Thread)
// ============================================================================

/// Job submitted to background checkpoint writer.
struct CheckpointJob {
    path: PathBuf,
    checkpoint: Checkpoint,
    compression_level: i32,
}

/// Non-blocking checkpoint writer with bounded queue.
///
/// Uses a background thread to handle serialization and I/O,
/// ensuring the main simulation thread never blocks on checkpoint writes.
pub struct CheckpointWriter {
    tx: Option<SyncSender<CheckpointJob>>,
    handle: Option<JoinHandle<()>>,
}

impl CheckpointWriter {
    /// Create a new checkpoint writer with specified queue depth.
    pub fn new(queue_depth: usize) -> Self {
        let (tx, rx) = sync_channel(queue_depth);
        let handle = Some(spawn_writer_thread(rx));
        Self {
            tx: Some(tx),
            handle,
        }
    }

    /// Request a checkpoint write (non-blocking).
    ///
    /// If the queue is full, returns an error and drops the checkpoint.
    /// The simulation should continue without interruption.
    pub fn request_checkpoint(
        &self,
        path: PathBuf,
        checkpoint: Checkpoint,
        compression_level: i32,
    ) -> Result<(), CheckpointError> {
        let tx = self.tx.as_ref().ok_or(CheckpointError::WriterDied)?;

        let job = CheckpointJob {
            path,
            checkpoint,
            compression_level,
        };

        match tx.try_send(job) {
            Ok(()) => {
                debug!("Checkpoint job queued successfully");
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                warn!("Checkpoint queue full, dropping checkpoint request");
                Err(CheckpointError::QueueFull)
            }
            Err(TrySendError::Disconnected(_)) => {
                error!("Checkpoint writer thread died");
                Err(CheckpointError::WriterDied)
            }
        }
    }
}

impl Drop for CheckpointWriter {
    fn drop(&mut self) {
        // Drop sender to signal shutdown (consuming the Option)
        self.tx.take();

        // Wait for writer thread to finish pending jobs
        if let Some(handle) = self.handle.take() {
            debug!("Waiting for checkpoint writer thread to finish");
            if let Err(e) = handle.join() {
                error!("Checkpoint writer thread panicked: {:?}", e);
            } else {
                debug!("Checkpoint writer thread joined successfully");
            }
        }
    }
}

/// Spawn the background writer thread.
fn spawn_writer_thread(rx: Receiver<CheckpointJob>) -> JoinHandle<()> {
    thread::Builder::new()
        .name("checkpoint-writer".to_string())
        .spawn(move || {
            info!("Checkpoint writer thread started");

            while let Ok(job) = rx.recv() {
                if let Err(e) = write_checkpoint(&job.path, &job.checkpoint, job.compression_level)
                {
                    error!(
                        path = %job.path.display(),
                        tick = job.checkpoint.header.tick,
                        error = %e,
                        "Failed to write checkpoint"
                    );
                } else {
                    info!(
                        path = %job.path.display(),
                        tick = job.checkpoint.header.tick,
                        "Checkpoint written successfully"
                    );
                }
            }

            info!("Checkpoint writer thread shutting down");
        })
        .expect("failed to spawn checkpoint writer thread")
}

// ============================================================================
// Serialization and Compression
// ============================================================================

/// Write a checkpoint to disk (bincode + zstd compression).
fn write_checkpoint(
    path: &Path,
    checkpoint: &Checkpoint,
    compression_level: i32,
) -> Result<(), CheckpointError> {
    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| CheckpointError::Io(e.to_string()))?;
    }

    // Serialize to bincode
    let serialized = bincode::serialize(checkpoint)
        .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

    debug!(bytes = serialized.len(), "Checkpoint serialized to bincode");

    // Compress with zstd
    let compressed = zstd::encode_all(&serialized[..], compression_level)
        .map_err(|e| CheckpointError::Compression(e.to_string()))?;

    let compression_ratio = serialized.len() as f64 / compressed.len() as f64;
    debug!(
        compressed_bytes = compressed.len(),
        ratio = format!("{:.2}x", compression_ratio),
        "Checkpoint compressed with zstd"
    );

    // Write to disk
    std::fs::write(path, compressed).map_err(|e| CheckpointError::Io(e.to_string()))?;

    Ok(())
}

/// Load a checkpoint from disk (decompress + deserialize).
pub fn load_checkpoint(path: &Path) -> Result<Checkpoint, CheckpointError> {
    // Read compressed data
    let compressed = std::fs::read(path).map_err(|e| CheckpointError::Io(e.to_string()))?;

    debug!(
        path = %path.display(),
        compressed_bytes = compressed.len(),
        "Checkpoint file read"
    );

    // Decompress with zstd
    let serialized = zstd::decode_all(&compressed[..])
        .map_err(|e| CheckpointError::Decompression(e.to_string()))?;

    debug!(
        decompressed_bytes = serialized.len(),
        "Checkpoint decompressed"
    );

    // Deserialize from bincode
    let checkpoint: Checkpoint = bincode::deserialize(&serialized)
        .map_err(|e| CheckpointError::Deserialization(e.to_string()))?;

    info!(
        path = %path.display(),
        tick = checkpoint.header.tick,
        sim_time = checkpoint.header.sim_time,
        "Checkpoint loaded successfully"
    );

    Ok(checkpoint)
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("Checkpoint queue is full (dropped to avoid blocking simulation)")]
    QueueFull,

    #[error("Checkpoint writer thread has died")]
    WriterDied,

    #[error("I/O error: {0}")]
    Io(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error(
        "World IR mismatch: checkpoint hash = {checkpoint_hash}, current hash = {current_hash}"
    )]
    WorldIrMismatch {
        checkpoint_hash: String,
        current_hash: String,
    },

    #[error("Era config mismatch: {0}")]
    EraConfigMismatch(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_checkpoint() -> Checkpoint {
        Checkpoint {
            header: CheckpointHeader {
                version: CHECKPOINT_VERSION,
                world_ir_hash: [0u8; 32],
                tick: 1000,
                sim_time: 1000.5,
                seed: 42,
                current_era: "main".into(),
                created_at: SystemTime::now(),
                world_git_hash: None,
            },
            state: CheckpointState {
                signals: SignalStorage::default(),
                entities: EntityStorage::default(),
                member_signals: MemberSignalData {
                    signals: IndexMap::new(),
                    entity_instance_counts: IndexMap::new(),
                    total_instance_count: 0,
                },
                era_configs: IndexMap::new(),
                stratum_states: IndexMap::new(),
            },
        }
    }

    #[test]
    fn test_checkpoint_serialization_roundtrip() {
        let checkpoint = create_test_checkpoint();

        // Serialize
        let serialized = bincode::serialize(&checkpoint).expect("serialize failed");
        assert!(!serialized.is_empty());

        // Deserialize
        let deserialized: Checkpoint =
            bincode::deserialize(&serialized).expect("deserialize failed");

        // Verify header matches
        assert_eq!(deserialized.header.version, checkpoint.header.version);
        assert_eq!(deserialized.header.tick, checkpoint.header.tick);
        assert_eq!(deserialized.header.sim_time, checkpoint.header.sim_time);
        assert_eq!(deserialized.header.seed, checkpoint.header.seed);
    }

    #[test]
    fn test_checkpoint_compression() {
        let checkpoint = create_test_checkpoint();

        // Serialize
        let serialized = bincode::serialize(&checkpoint).expect("serialize failed");

        // Compress
        let compressed = zstd::encode_all(&serialized[..], DEFAULT_COMPRESSION_LEVEL)
            .expect("compression failed");

        // Should be compressed (or at least not larger)
        assert!(compressed.len() <= serialized.len());

        // Decompress
        let decompressed = zstd::decode_all(&compressed[..]).expect("decompression failed");

        // Should match original
        assert_eq!(decompressed, serialized);
    }

    #[test]
    fn test_checkpoint_write_and_load() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint.ckpt");

        let checkpoint = create_test_checkpoint();

        // Write
        write_checkpoint(&path, &checkpoint, DEFAULT_COMPRESSION_LEVEL).expect("write failed");

        // Load
        let loaded = load_checkpoint(&path).expect("load failed");

        // Verify
        assert_eq!(loaded.header.tick, checkpoint.header.tick);
        assert_eq!(loaded.header.sim_time, checkpoint.header.sim_time);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_member_signal_extraction_and_restoration() {
        use crate::soa_storage::{MemberSignalBuffer, ValueType};
        use crate::types::Value;

        // Create a buffer and register some signals
        let mut buffer = MemberSignalBuffer::new();
        buffer.register_signal("terra.plate.age".to_string(), ValueType::scalar());
        buffer.register_signal("terra.plate.velocity".to_string(), ValueType::vec3());
        buffer.register_entity_count("terra.plate", 3);
        buffer.init_instances(3);

        // Set some values
        buffer
            .set_current("terra.plate.age", 0, Value::Scalar(100.0))
            .unwrap();
        buffer
            .set_current("terra.plate.age", 1, Value::Scalar(200.0))
            .unwrap();
        buffer
            .set_current("terra.plate.age", 2, Value::Scalar(300.0))
            .unwrap();
        buffer
            .set_current("terra.plate.velocity", 0, Value::Vec3([1.0, 2.0, 3.0]))
            .unwrap();
        buffer
            .set_current("terra.plate.velocity", 1, Value::Vec3([4.0, 5.0, 6.0]))
            .unwrap();
        buffer
            .set_current("terra.plate.velocity", 2, Value::Vec3([7.0, 8.0, 9.0]))
            .unwrap();

        // Extract
        let data = MemberSignalData::from_buffer(&buffer).expect("extraction failed");

        // Verify extraction
        assert_eq!(data.total_instance_count, 3);
        assert_eq!(data.entity_instance_counts.get("terra.plate"), Some(&3));
        assert_eq!(data.signals.len(), 2);

        // Verify scalar signal
        let age_values = data.signals.get("terra.plate.age").unwrap();
        assert_eq!(age_values.len(), 3);
        assert_eq!(age_values[0], (0, Value::Scalar(100.0)));
        assert_eq!(age_values[1], (1, Value::Scalar(200.0)));
        assert_eq!(age_values[2], (2, Value::Scalar(300.0)));

        // Verify vec3 signal
        let velocity_values = data.signals.get("terra.plate.velocity").unwrap();
        assert_eq!(velocity_values.len(), 3);
        assert_eq!(velocity_values[0], (0, Value::Vec3([1.0, 2.0, 3.0])));
        assert_eq!(velocity_values[1], (1, Value::Vec3([4.0, 5.0, 6.0])));
        assert_eq!(velocity_values[2], (2, Value::Vec3([7.0, 8.0, 9.0])));

        // Create a new buffer and restore
        let mut restored_buffer = MemberSignalBuffer::new();
        restored_buffer.register_signal("terra.plate.age".to_string(), ValueType::scalar());
        restored_buffer.register_signal("terra.plate.velocity".to_string(), ValueType::vec3());
        restored_buffer.register_entity_count("terra.plate", 3);
        restored_buffer.init_instances(3);

        data.restore_into_buffer(&mut restored_buffer)
            .expect("restoration failed");

        // Verify restored values match original
        assert_eq!(
            restored_buffer.get_current("terra.plate.age", 0),
            Some(Value::Scalar(100.0))
        );
        assert_eq!(
            restored_buffer.get_current("terra.plate.age", 1),
            Some(Value::Scalar(200.0))
        );
        assert_eq!(
            restored_buffer.get_current("terra.plate.age", 2),
            Some(Value::Scalar(300.0))
        );
        assert_eq!(
            restored_buffer.get_current("terra.plate.velocity", 0),
            Some(Value::Vec3([1.0, 2.0, 3.0]))
        );
        assert_eq!(
            restored_buffer.get_current("terra.plate.velocity", 1),
            Some(Value::Vec3([4.0, 5.0, 6.0]))
        );
        assert_eq!(
            restored_buffer.get_current("terra.plate.velocity", 2),
            Some(Value::Vec3([7.0, 8.0, 9.0]))
        );
    }
}
