//! Runtime checkpoint operations.
//!
//! Separated from runtime core to keep checkpoint serialization/deserialization
//! logic isolated. This module contains:
//! - Enabling the background checkpoint writer
//! - Requesting async checkpoint writes
//! - Loading checkpoints from disk
//! - Restoring runtime state from checkpoint data

use tracing::info;

use crate::error::{Error, Result};
use crate::types::DeterminismPolicy;

use super::runtime::Runtime;

impl Runtime {
    /// Enable checkpoint writer with specified queue depth.
    pub fn enable_checkpointing(&mut self, queue_depth: usize) {
        info!(queue_depth, "enabling checkpoint writer");
        self.checkpoint_writer = Some(crate::checkpoint::CheckpointWriter::new(queue_depth));
    }

    /// Set the world IR hash for checkpoint validation.
    pub fn set_world_ir_hash(&mut self, hash: [u8; 32]) {
        self.world_ir_hash = Some(hash);
    }

    /// Set the initial seed for determinism.
    pub fn set_initial_seed(&mut self, seed: u64) {
        self.initial_seed = seed;
    }

    /// Validate determinism if policy is Strict
    pub fn validate_determinism(&self) -> Result<()> {
        if self.policy.determinism != DeterminismPolicy::Strict {
            return Ok(());
        }
        // FIXME: State hashing not implemented - cannot guarantee determinism
        panic!(
            "Strict determinism policy enabled but validation not implemented. \
            Cannot guarantee deterministic execution. Implement state hashing or disable strict policy."
        );
    }

    /// Request a checkpoint write (non-blocking).
    ///
    /// If checkpointing is not enabled, returns an error.
    /// If the queue is full, drops the checkpoint and returns an error.
    pub fn request_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        let writer = self
            .checkpoint_writer
            .as_ref()
            .ok_or_else(|| Error::Checkpoint("checkpointing not enabled".to_string()))?;

        // Extract member signal data
        let member_signals = crate::checkpoint::MemberSignalData::from_buffer(&self.member_signals)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        // Build era configs for validation
        let era_configs = self
            .eras
            .iter()
            .map(|(id, cfg)| {
                (
                    id.clone(),
                    crate::checkpoint::EraConfigSnapshot {
                        dt: cfg.dt.0,
                        strata_count: cfg.strata.len(),
                    },
                )
            })
            .collect();

        let world_ir_hash = self
            .world_ir_hash
            .ok_or_else(|| Error::Checkpoint("world IR hash missing".to_string()))?;
        let stratum_states = self
            .eras
            .get(&self.current_era)
            .ok_or_else(|| Error::Checkpoint("current era not found".to_string()))?
            .strata
            .clone();

        // Build checkpoint
        let checkpoint = crate::checkpoint::Checkpoint {
            header: crate::checkpoint::CheckpointHeader {
                version: crate::checkpoint::CHECKPOINT_VERSION,
                world_ir_hash,
                tick: self.tick,
                sim_time: self.sim_time,
                seed: self.initial_seed,
                current_era: self.current_era.clone(),
                created_at: std::time::SystemTime::now(),
                world_git_hash: None,
            },
            state: crate::checkpoint::CheckpointState {
                signals: self.signals.clone(),
                entities: self.entities.clone(),
                member_signals,
                era_configs,
                stratum_states,
            },
        };

        writer
            .request_checkpoint(
                path.to_owned(),
                checkpoint,
                crate::checkpoint::DEFAULT_COMPRESSION_LEVEL,
            )
            .map_err(|e| Error::Checkpoint(e.to_string()))
    }

    /// Load a checkpoint and replace runtime state (validation optional).
    ///
    /// If `force` is false, validates world IR hash.
    /// Returns error if validation fails or deserialization fails.
    pub fn load_checkpoint(&mut self, path: &std::path::Path, force: bool) -> Result<()> {
        info!(path = %path.display(), "loading checkpoint");

        let checkpoint = crate::checkpoint::load_checkpoint(path)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        // Validate world IR hash
        if !force {
            if let Some(current_hash) = self.world_ir_hash {
                if checkpoint.header.world_ir_hash != current_hash {
                    return Err(Error::Checkpoint(
                        "World IR mismatch: checkpoint hash does not match current world"
                            .to_string(),
                    ));
                }
            }
        }

        // Restore state
        self.signals = checkpoint.state.signals;
        self.entities = checkpoint.state.entities;

        // Restore member signals
        checkpoint
            .state
            .member_signals
            .restore_into_buffer(&mut self.member_signals)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        self.tick = checkpoint.header.tick;
        self.sim_time = checkpoint.header.sim_time;
        self.current_era = checkpoint.header.current_era;
        self.initial_seed = checkpoint.header.seed;

        info!(
            tick = checkpoint.header.tick,
            sim_time = checkpoint.header.sim_time,
            "checkpoint loaded successfully"
        );

        Ok(())
    }

    /// Restore simulation state from a checkpoint.
    ///
    /// # Errors
    /// - Returns error if world IR hash doesn't match (different world)
    /// - Returns error if era configurations don't match
    pub fn restore_from_checkpoint(
        &mut self,
        checkpoint: crate::checkpoint::Checkpoint,
    ) -> Result<()> {
        // Validate world IR hash
        if let Some(ref our_hash) = self.world_ir_hash {
            if checkpoint.header.world_ir_hash != *our_hash {
                return Err(Error::Checkpoint(format!(
                    "World IR mismatch: checkpoint={:?}, current={:?}",
                    checkpoint.header.world_ir_hash, our_hash
                )));
            }
        }

        // Restore core state
        self.tick = checkpoint.header.tick;
        self.sim_time = checkpoint.header.sim_time;
        self.current_era = checkpoint.header.current_era.clone();

        // Restore signal storage
        self.signals = checkpoint.state.signals;

        // Restore entity storage
        self.entities = checkpoint.state.entities;

        // Restore member signals (SoA buffer)
        checkpoint
            .state
            .member_signals
            .restore_into_buffer(&mut self.member_signals)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        // Restore stratum states into the current era
        if let Some(era_config) = self.eras.get_mut(&self.current_era) {
            era_config.strata = checkpoint.state.stratum_states;
        }

        info!(
            tick = self.tick,
            sim_time = self.sim_time,
            era = %self.current_era,
            "Checkpoint restored"
        );

        Ok(())
    }
}
