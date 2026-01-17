//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: `run <world-dir> [--steps N] [--dt SECONDS] [--scenario NAME]`

use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use tracing::{error, info, warn};

use continuum_compiler::ir::{
    BinaryBundle, RuntimeBuildOptions, Scenario, build_runtime, compile, find_scenarios,
};
use continuum_runtime::executor::{CheckpointOptions, RunOptions, run_simulation};
use continuum_runtime::lens_sink::{FileSink, FileSinkConfig, FilteredSink, LensSinkConfig};

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

    let (world, compilation) =
        if args.path.is_file() && args.path.extension().map_or(false, |ext| ext == "cvm") {
            info!("Loading binary bundle from: {}", args.path.display());
            let data = match fs::read(&args.path) {
                Ok(d) => d,
                Err(e) => {
                    error!("Error reading file '{}': {}", args.path.display(), e);
                    process::exit(1);
                }
            };
            let bundle: BinaryBundle = match bincode::deserialize(&data) {
                Ok(b) => b,
                Err(e) => {
                    error!(
                        "Error decoding binary bundle '{}': {}",
                        args.path.display(),
                        e
                    );
                    process::exit(1);
                }
            };
            info!("Loaded world: {}", bundle.world_name);
            (bundle.world, bundle.compilation)
        } else {
            // Load and compile world from directory
            info!("Loading world source from: {}", args.path.display());

            let compile_result = continuum_compiler::compile_from_dir_result(&args.path);

            // Log diagnostics using proper logging
            compile_result.log_diagnostics();

            if compile_result.has_errors() {
                process::exit(1);
            }

            let world = compile_result.world.expect("no world despite no errors");
            info!("Successfully compiled world source");

            // Compile to DAGs
            info!("Compiling to DAGs...");
            let compilation = match compile(&world) {
                Ok(c) => c,
                Err(e) => {
                    error!("Compilation error: {}", e);
                    process::exit(1);
                }
            };
            (world, compilation)
        };

    let signals = world.signals();
    let fields = world.fields();

    info!("  Strata: {}", world.strata().len());
    info!("  Eras: {}", world.eras().len());
    info!("  Signals: {}", signals.len());
    info!("  Fields: {}", fields.len());
    info!("  Constants: {}", world.constants.len());
    info!("  Config: {}", world.config.len());

    // Validate scenario against world if present
    if let Some(ref s) = scenario {
        if let Err(e) = s.validate_against_world(&world) {
            warn!("Scenario validation warning: {}", e);
        }
    }

    info!("Building runtime...");
    let (mut runtime, report) = match build_runtime(
        &world,
        compilation,
        RuntimeBuildOptions {
            dt_override: args.dt,
            scenario,
        },
    ) {
        Ok(result) => result,
        Err(e) => {
            error!("Runtime build error: {}", e);
            process::exit(1);
        }
    };

    if report.resolver_count > 0 || report.aggregate_count > 0 {
        info!(
            "  Total: {} resolvers, {} aggregate resolvers registered",
            report.resolver_count, report.aggregate_count
        );
    }
    if report.assertion_count > 0 {
        info!("  Registered {} assertions", report.assertion_count);
    }
    if report.field_count > 0 {
        info!("  Registered {} field measures", report.field_count);
    }
    if report.fracture_count > 0 {
        info!("  Registered {} fractures", report.fracture_count);
    }
    if report.member_signal_count > 0 {
        info!(
            "  Initialized {} member signals (max {} instances)",
            report.member_signal_count, report.max_member_instances
        );
    }

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
        let checkpoint_path = find_latest_checkpoint(&checkpoint_dir);

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

    // Prepare lens sink if save directory requested
    let lens_sink = args.save_dir.as_ref().map(|dir| {
        let config = FileSinkConfig {
            output_dir: dir.clone(),
            seed: 0, // TODO: Get actual seed from scenario or args
            steps: args.steps,
            stride: args.stride,
            field_filter: fields.keys().cloned().collect(),
        };

        let file_sink = FileSink::new(config).expect("Failed to create file sink");

        // Wrap with FilteredSink to apply stride
        let sink_config = LensSinkConfig {
            stride: args.stride,
            field_filter: fields.keys().cloned().collect(),
        };

        Box::new(FilteredSink::new(file_sink, sink_config))
            as Box<dyn continuum_runtime::lens_sink::LensSink>
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
            signals: signals.keys().cloned().collect(),
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
fn find_latest_checkpoint(checkpoint_dir: &PathBuf) -> Option<PathBuf> {
    // Try 'latest' symlink first
    let latest_link = checkpoint_dir.join("latest");
    if latest_link.exists() {
        return Some(latest_link);
    }

    // Find newest checkpoint by filename
    let mut checkpoints: Vec<_> = std::fs::read_dir(checkpoint_dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "ckpt")
                .unwrap_or(false)
        })
        .collect();

    if checkpoints.is_empty() {
        return None;
    }

    // Sort by filename (descending) to get latest
    checkpoints.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    Some(checkpoints.first()?.path())
}
