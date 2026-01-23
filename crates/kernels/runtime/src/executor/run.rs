//! Simulation run orchestration.
//!
//! This module provides the high-level `run_simulation` function that drives
//! a Runtime through multiple ticks with support for:
//! - Observer lens sinks (field reconstruction)
//! - Automated checkpointing
//! - Signal printing for debugging
//!
//! The run logic is separate from the Runtime to keep execution concerns
//! (tick/phase stepping) distinct from orchestration concerns (loops, I/O,
//! checkpoint scheduling).

use crate::error::{Error, Result};
use crate::lens_sink::{LensData, LensSink};
use crate::types::SignalId;
use std::path::PathBuf;
use tracing::debug;

use super::Runtime;

/// Configuration for periodic state persistence (checkpoints).
///
/// Checkpoints enable deterministic replay and recovery from saved simulation state.
#[derive(Debug, Clone)]
pub struct CheckpointOptions {
    /// Directory where checkpoint files (`.ckpt`) will be stored.
    pub checkpoint_dir: PathBuf,
    /// Number of ticks between scheduled checkpoints.
    pub stride: u64,
    /// Optional real-time interval between checkpoints (e.g., every 5 minutes).
    pub wall_clock_interval: Option<std::time::Duration>,
    /// Number of historical checkpoints to retain before pruning.
    pub keep_last_n: Option<usize>,
}

/// Options for configuring a simulation run.
pub struct RunOptions {
    /// Total number of simulation steps to execute.
    pub steps: u64,
    /// Whether to print resolved signal values to stdout after each tick.
    pub print_signals: bool,
    /// List of specific signal IDs to print (if `print_signals` is true).
    pub signals: Vec<SignalId>,
    /// Optional observer sink for field reconstruction and snapshot output.
    pub lens_sink: Option<Box<dyn LensSink>>,
    /// Optional configuration for automated checkpointing.
    pub checkpoint: Option<CheckpointOptions>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            steps: 1,
            print_signals: false,
            signals: Vec::new(),
            lens_sink: None,
            checkpoint: None,
        }
    }
}

/// Summary report returned after a successful simulation run.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// The output directory for lens snapshots, if a sink was provided.
    pub run_dir: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("runtime execution failed: {0}")]
    Execution(String),
    #[error("snapshot write failed: {0}")]
    Snapshot(String),
}

pub fn run_simulation(
    runtime: &mut Runtime,
    mut options: RunOptions,
) -> std::result::Result<RunReport, RunError> {
    let run_dir = options.lens_sink.as_ref().and_then(|s| s.output_path());

    // Lens sink is handled separately now - no manifest created here
    // The sink handles its own manifest/metadata

    debug!("Checking warmup status...");
    if !runtime.is_warmup_complete() {
        debug!("Executing warmup...");
        runtime
            .execute_warmup()
            .map_err(|e| RunError::Execution(e.to_string()))?;
        debug!("Warmup complete");
    } else {
        debug!("Warmup already complete");
    }

    let mut last_checkpoint_time = std::time::Instant::now();
    // Setup checkpoint directory if enabled
    if let Some(checkpoint) = &options.checkpoint {
        debug!(
            "Creating checkpoint directory: {:?}",
            checkpoint.checkpoint_dir
        );
        std::fs::create_dir_all(&checkpoint.checkpoint_dir)
            .map_err(|e| RunError::Execution(e.to_string()))?;
    }

    debug!("Starting main simulation loop ({} steps)...", options.steps);
    for i in 0..options.steps {
        debug!("Executing tick {} / {}...", i + 1, options.steps);
        runtime
            .execute_tick()
            .map_err(|e| RunError::Execution(e.to_string()))?;
        debug!("Tick {} complete", i + 1);

        if options.print_signals {
            let mut line = format!("Tick {:04}: ", runtime.tick());
            for id in &options.signals {
                let val = runtime
                    .get_signal(id)
                    .ok_or_else(|| RunError::Execution(format!("Signal '{}' not found", id)))?;
                line.push_str(&format!("{}={} ", id, val));
            }
            tracing::info!("{}", line);
        }

        // Emit field data to lens sink
        if let Some(ref mut sink) = options.lens_sink {
            let fields = runtime.drain_fields();
            let lens_data = LensData { fields };
            sink.emit_tick(runtime.tick(), runtime.sim_time(), lens_data)
                .map_err(|e| RunError::Snapshot(e.to_string()))?;
        }

        // Checkpoint logic
        if let Some(checkpoint) = &options.checkpoint {
            let should_checkpoint = {
                let stride_met = runtime.tick() % checkpoint.stride == 0;
                let wall_clock_met = checkpoint
                    .wall_clock_interval
                    .map(|interval| last_checkpoint_time.elapsed() >= interval)
                    .unwrap_or(false);
                stride_met || wall_clock_met
            };

            if should_checkpoint {
                let checkpoint_path = checkpoint
                    .checkpoint_dir
                    .join(format!("checkpoint_{:010}.ckpt", runtime.tick()));

                runtime
                    .request_checkpoint(&checkpoint_path)
                    .map_err(|e| RunError::Execution(e.to_string()))?;
                debug!("Checkpoint requested for tick {}", runtime.tick());
                last_checkpoint_time = std::time::Instant::now();

                // Update 'latest' symlink
                let latest_link = checkpoint.checkpoint_dir.join("latest");
                if let Err(err) = std::fs::remove_file(&latest_link) {
                    if err.kind() != std::io::ErrorKind::NotFound {
                        return Err(RunError::Execution(err.to_string()));
                    }
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::symlink;
                    symlink(
                        format!("checkpoint_{:010}.ckpt", runtime.tick()),
                        &latest_link,
                    )
                    .map_err(|e| RunError::Execution(e.to_string()))?;
                }

                // Prune old checkpoints if configured
                if let Some(keep_n) = checkpoint.keep_last_n {
                    prune_old_checkpoints(&checkpoint.checkpoint_dir, keep_n)
                        .map_err(|e| RunError::Execution(e.to_string()))?;
                }
            }
        }
    }

    // Close lens sink (writes manifest, finalizes output)
    if let Some(ref mut sink) = options.lens_sink {
        sink.close()
            .map_err(|e| RunError::Snapshot(e.to_string()))?;
    }

    Ok(RunReport { run_dir })
}

/// Prune old checkpoints, keeping only the last N.
fn prune_old_checkpoints(checkpoint_dir: &std::path::Path, keep_n: usize) -> Result<()> {
    let entries =
        std::fs::read_dir(checkpoint_dir).map_err(|e| Error::Checkpoint(e.to_string()))?;

    let mut checkpoints = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| Error::Checkpoint(e.to_string()))?;
        let path = entry.path();
        let extension = match path.extension() {
            Some(ext) => ext,
            None => continue,
        };
        let extension_str = extension.to_str().ok_or_else(|| {
            Error::Checkpoint(format!(
                "Checkpoint entry '{}' has non-UTF8 extension",
                path.display()
            ))
        })?;
        if extension_str == "ckpt" {
            checkpoints.push(entry);
        }
    }

    if checkpoints.len() <= keep_n {
        return Ok(());
    }

    // Sort by filename (which includes tick number)
    checkpoints.sort_by_key(|entry| entry.file_name());

    // Remove oldest checkpoints
    let to_remove = checkpoints.len() - keep_n;
    for entry in checkpoints.iter().take(to_remove) {
        std::fs::remove_file(entry.path()).map_err(|e| Error::Checkpoint(e.to_string()))?;
        debug!("Pruned old checkpoint: {}", entry.path().display());
    }
    Ok(())
}
