//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: `run <world-dir> [--steps N] [--dt SECONDS] [--scenario NAME]`

use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use tracing::{error, info, warn};

use continuum_cdsl::{compile, deserialize_world};
use continuum_runtime::executor::{run_simulation, CheckpointOptions, RunOptions};
use continuum_lens::ReconstructedSink;
use continuum_runtime::lens_sink::{
    FileSink, FileSinkConfig, FilteredSink, LensSink, LensSinkConfig,
};
use continuum_runtime::{build_runtime, Runtime};

#[derive(Parser, Debug)]
#[command(name = "run")]
#[command(about = "Execute a Continuum world from a directory or binary bundle (.cvm)")]
struct Args {
    /// Path to the World root directory or binary bundle (.cvm)
    path: PathBuf,

    /// Number of simulation steps
    #[arg(long = "steps", default_value = "10")]
    steps: u64,

    /// Override dt (seconds per tick)
    #[arg(long)]
    dt: Option<f64>,

    /// Scenario name or path to scenario YAML file.
    /// If a name is provided, looks for `scenarios/<name>.yaml` in the world directory.
    /// If a path is provided, loads the scenario from that file.
    #[arg(long)]
    scenario: Option<String>,

    /// Seed for deterministic lens output (required when --save is used)
    #[arg(long)]
    seed: Option<u64>,

    /// List available scenarios and exit
    #[arg(long)]
    list_scenarios: bool,

    /// Directory for snapshot outputs
    #[arg(long = "save", alias = "snapshot-dir")]
    save_dir: Option<PathBuf>,

    /// Snapshot stride
    #[arg(long = "stride", alias = "snapshot-stride", default_value = "10")]
    stride: u64,

    // ========================================================================
    // Checkpoint Options
    // ========================================================================
    /// Enable checkpointing and set checkpoint directory
    #[arg(long = "checkpoint-dir")]
    checkpoint_dir: Option<PathBuf>,

    /// Checkpoint every N ticks
    #[arg(long = "checkpoint-stride", default_value = "1000")]
    checkpoint_stride: u64,

    /// Checkpoint at most once per N seconds (wall-clock throttling)
    #[arg(long = "checkpoint-interval")]
    checkpoint_interval: Option<u64>,

    /// Keep only the last N checkpoints (prune older ones)
    #[arg(long = "keep-checkpoints")]
    keep_checkpoints: Option<usize>,

    /// Resume from latest checkpoint in checkpoint directory
    #[arg(long = "resume")]
    resume: bool,

    /// Skip world IR validation when resuming (dangerous!)
    #[arg(long = "force-resume")]
    force_resume: bool,
}


fn main() {
    continuum_tools::init_logging();

    let args = Args::parse();

    // Handle --list-scenarios
    if args.list_scenarios {
        let scenarios = find_scenarios(&args.path);
        if scenarios.is_empty() {
            info!("No scenarios found in {}/scenarios/", args.path.display());
        } else {
            info!("Available scenarios:");
            for path in scenarios {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    info!("  {}", name);
                }
            }
        }
        return;
    }

    // Load scenario if specified
    let scenario = if let Some(ref scenario_arg) = args.scenario {
        let scenario_path = if scenario_arg.ends_with(".yaml") || scenario_arg.ends_with(".yml") {
            PathBuf::from(scenario_arg)
        } else {
            // Look for scenario in world's scenarios directory
            args.path
                .join("scenarios")
                .join(format!("{}.yaml", scenario_arg))
        };

        match Scenario::load(&scenario_path) {
            Ok(s) => {
                info!(
                    "Loaded scenario: {} ({})",
                    s.metadata.name,
                    scenario_path.display()
                );
                Some(s)
            }
            Err(e) => {
                error!(
                    "Failed to load scenario '{}': {}",
                    scenario_path.display(),
                    e
                );
                process::exit(1);
            }
        }
    } else {
        None
    };

    let compiled = if args.path.is_file() && args.path.extension().map_or(false, |ext| ext == "cvm")
    {
        info!("Loading binary bundle from: {}", args.path.display());
        let data = match fs::read(&args.path) {
            Ok(d) => d,
            Err(e) => {
                error!("Error reading file '{}': {}", args.path.display(), e);
                process::exit(1);
            }
        };
        match deserialize_world(&data) {
            Ok(w) => w,
            Err(e) => {
                error!(
                    "Error decoding binary bundle '{}': {}",
                    args.path.display(),
                    e
                );
                process::exit(1);
            }
        }
    } else {
        // Load and compile world from directory
        info!("Compiling world from: {}", args.path.display());

        match compile(&args.path) {
            Ok(w) => {
                info!("Successfully compiled world source");
                w
            }
            Err(errors) => {
                for err in errors {
                    error!("{}", err);
                }
                process::exit(1);
            }
        }
    };

    info!("  Strata: {}", compiled.world.strata.len());
    info!("  Eras: {}", compiled.world.eras.len());
    info!("  Signals: {}", compiled.world.globals.len());
    info!("  Fields: {}", field_ids.len());

    info!("Building runtime...");
    let mut runtime = build_runtime(compiled);

    // ========================================================================
    // Checkpoint Setup
    // ========================================================================

    // Enable checkpointing if checkpoint-dir is specified
    if args.checkpoint_dir.is_some() {
        info!("Enabling checkpoint writer (queue depth: 3)");
        runtime.enable_checkpointing(3);
    }

    // Resume from checkpoint if requested
    if args.resume {
        let checkpoint_dir = args
            .checkpoint_dir
            .as_ref()
            .map(|p| p.clone())
            .unwrap_or_else(|| PathBuf::from("./checkpoints"));

        info!("Attempting to resume from: {}", checkpoint_dir.display());

        // Find latest checkpoint
        let checkpoint_path = match find_latest_checkpoint(&checkpoint_dir) {
            Ok(path) => path,
            Err(err) => {
                error!("Failed to scan checkpoint directory: {}", err);
                process::exit(1);
            }
        };

        match checkpoint_path {
            Some(path) => {
                info!("Found checkpoint: {}", path.display());
                if let Err(e) = runtime.load_checkpoint(&path, args.force_resume) {
                    error!("Failed to load checkpoint: {}", e);
                    if !args.force_resume {
                        error!("Tip: Use --force-resume to skip world IR validation");
                    }
                    process::exit(1);
                }
                info!("Resumed from tick {}", runtime.tick());
            }
            None => {
                error!("No checkpoint found in: {}", checkpoint_dir.display());
                error!("Cannot resume - no checkpoint exists");
                process::exit(1);
            }
        }
    }

    let signals = &compiled.world.globals;
    let mut field_ids: Vec<continuum_runtime::types::FieldId> = compiled
        .world
        .globals
        .iter()
        .filter_map(|(path, node)| {
            if node.role_id() == continuum_cdsl::ast::RoleId::Field {
                Some(continuum_runtime::types::FieldId::from(path.to_string()))
            } else {
                None
            }
        })
        .collect();
    field_ids.extend(
        compiled
            .world
            .members
            .iter()
            .filter_map(|(path, node)| {
                if node.role_id() == continuum_cdsl::ast::RoleId::Field {
                    Some(continuum_runtime::types::FieldId::from(path.to_string()))
                } else {
                    None
                }
            }),
    );

    // Prepare lens sink if save directory requested
    let lens_sink = args.save_dir.as_ref().map(|dir| {
        let seed = args.seed.unwrap_or_else(|| {
            error!("--seed is required when using --save");
            process::exit(1);
        });
        let config = FileSinkConfig {
            output_dir: dir.clone(),
            seed,
            steps: args.steps,
            stride: args.stride,
            field_filter: field_ids.clone(),
        };

        let file_sink = FileSink::new(config).expect("Failed to create file sink");

        // Wrap with FilteredSink to apply stride
        let sink_config = LensSinkConfig {
            stride: args.stride,
            field_filter: field_ids.clone(),
        };

        let filtered = FilteredSink::new(file_sink, sink_config);
        let reconstructed = ReconstructedSink::new(Box::new(filtered)).unwrap_or_else(|e| {
            error!("Failed to initialize lens: {}", e);
            process::exit(1);
        });
        Box::new(reconstructed) as Box<dyn LensSink>
    });

    info!("Running {} steps...", args.steps);

    // Build checkpoint options if checkpoint-dir is specified
    let checkpoint = args.checkpoint_dir.as_ref().map(|dir| CheckpointOptions {
        checkpoint_dir: dir.clone(),
        stride: args.checkpoint_stride,
        wall_clock_interval: args.checkpoint_interval.map(std::time::Duration::from_secs),
        keep_last_n: args.keep_checkpoints,
    });

    let report = match run_simulation(
        &mut runtime,
        RunOptions {
            steps: args.steps,
            print_signals: true,
            signals: signals
                .keys()
                .map(|p| continuum_runtime::types::SignalId::from(p.to_string()))
                .collect(),
            lens_sink,
            checkpoint,
        },
    ) {
        Ok(report) => report,
        Err(e) => {
            error!("Run failed: {}", e);
            process::exit(1);
        }
    };

    if let Some(dir) = report.run_dir {
        info!("Snapshot output enabled: {}", dir.display());
    }

    info!("Simulation complete!");
}

/// Find the latest checkpoint in a directory.
///
/// Looks for 'latest' symlink first, then finds the newest checkpoint by filename.
fn find_latest_checkpoint(checkpoint_dir: &PathBuf) -> Result<Option<PathBuf>, String> {
    // Try 'latest' symlink first
    let latest_link = checkpoint_dir.join("latest");
    if latest_link.exists() {
        return Ok(Some(latest_link));
    }

    // Find newest checkpoint by filename
    let mut checkpoints = Vec::new();
    let entries = std::fs::read_dir(checkpoint_dir)
        .map_err(|e| format!("Failed to read {}: {}", checkpoint_dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Dir entry error: {}", e))?;
        let path = entry.path();
        let extension = path
            .extension()
            .ok_or_else(|| format!("Checkpoint entry '{}' missing extension", path.display()))?;
        let extension_str = extension.to_str().ok_or_else(|| {
            format!("Checkpoint entry '{}' has non-UTF8 extension", path.display())
        })?;
        if extension_str == "ckpt" {
            checkpoints.push(entry);
        }
    }

    if checkpoints.is_empty() {
        return Ok(None);
    }

    // Sort by filename (descending) to get latest
    checkpoints.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    Ok(checkpoints.first().map(|entry| entry.path()))
}
