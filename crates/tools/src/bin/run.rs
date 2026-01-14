//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: `run <world-dir> [--steps N] [--dt SECONDS]`

use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use tracing::{error, info, warn};

use continuum_compiler::ir::{BinaryBundle, RuntimeBuildOptions, build_runtime, compile};
use continuum_runtime::executor::{RunOptions, SnapshotOptions, run_simulation};

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

    /// Directory for snapshot outputs
    #[arg(long = "save", alias = "snapshot-dir")]
    save_dir: Option<PathBuf>,

    /// Snapshot stride
    #[arg(long = "stride", alias = "snapshot-stride", default_value = "10")]
    stride: u64,
}

fn main() {
    continuum_tools::init_logging();

    let args = Args::parse();

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

            if compile_result.has_errors() {
                error!("{}", compile_result.format_diagnostics().trim_end());
                process::exit(1);
            }

            if !compile_result.diagnostics.is_empty() {
                warn!("{}", compile_result.format_diagnostics().trim_end());
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

    info!("Building runtime...");
    let (mut runtime, report) = match build_runtime(
        &world,
        compilation,
        RuntimeBuildOptions {
            dt_override: args.dt,
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

    // Prepare snapshot directory if requested
    let snapshot = args.save_dir.as_ref().map(|dir| SnapshotOptions {
        output_dir: dir.clone(),
        stride: args.stride,
        signals: signals.keys().cloned().collect(),
        fields: fields.keys().cloned().collect(),
        seed: 0,
    });

    info!("Running {} steps...", args.steps);

    let report = match run_simulation(
        &mut runtime,
        RunOptions {
            steps: args.steps,
            print_signals: true,
            signals: signals.keys().cloned().collect(),
            snapshot,
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
