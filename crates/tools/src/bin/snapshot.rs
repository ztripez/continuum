//! Simulation Snapshot Tool
//!
//! Capture simulation signal and field state at regular intervals.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin snapshot -- --steps 100 --stride 10 --fields "atmosphere.temperature" ./world
//! ```

// Link against functions crate to pull in kernel function registrations
extern crate continuum_functions;

use std::path::PathBuf;
use std::process;

use clap::Parser;
use tracing::{error, info};

use continuum_compiler::ir::{RuntimeBuildOptions, build_runtime, compile};
use continuum_runtime::executor::{RunOptions, SnapshotOptions, run_simulation};

/// Simulation snapshot capture tool
#[derive(Parser, Debug)]
#[command(name = "snapshot")]
#[command(about = "Capture simulation signal and field state at regular intervals")]
struct Args {
    /// Path to the World root directory
    world_dir: PathBuf,

    /// Number of simulation steps to run
    #[arg(short, long, default_value = "100")]
    steps: u64,

    /// Time per step (simulation time) in seconds
    #[arg(long)]
    dt: Option<f64>,

    /// Capture every N ticks
    #[arg(short, long, default_value = "10")]
    stride: u64,

    /// Fields to capture (comma-separated, empty = all fields)
    #[arg(short, long, default_value = "")]
    fields: String,

    /// Output directory for snapshots
    #[arg(short, long, default_value = "./snapshots")]
    output: PathBuf,

    /// World seed for deterministic simulation
    #[arg(long, default_value = "1")]
    seed: u64,
}

fn main() {
    continuum_tools::init_logging();

    let args = Args::parse();

    // Load and compile world using unified compiler
    info!("Loading world from: {}", args.world_dir.display());
    let world = match continuum_compiler::compile_from_dir(&args.world_dir) {
        Ok(w) => {
            info!("Successfully compiled world");
            w
        }
        Err(diagnostics) => {
            for diag in diagnostics {
                let file_str = diag
                    .file
                    .as_ref()
                    .map(|f| format!("{}: ", f.display()))
                    .unwrap_or_default();
                let span_str = diag
                    .span
                    .as_ref()
                    .map(|s| format!("at {:?}: ", s))
                    .unwrap_or_default();
                error!("{}{}{}", file_str, span_str, diag.message);
            }
            process::exit(1);
        }
    };

    // Compile to DAGs
    let compilation = match compile(&world) {
        Ok(c) => c,
        Err(e) => {
            error!("Compilation error: {}", e);
            process::exit(1);
        }
    };

    let (mut runtime, _report) = match build_runtime(
        &world,
        compilation,
        RuntimeBuildOptions {
            dt_override: args.dt,
            scenario: None,
        },
    ) {
        Ok(result) => result,
        Err(e) => {
            error!("Runtime build error: {}", e);
            process::exit(1);
        }
    };

    let signals = world.signals();
    let fields = world.fields();

    let requested_fields: Vec<continuum_foundation::FieldId> = if args.fields.is_empty() {
        fields.keys().cloned().collect()
    } else {
        args.fields
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(continuum_foundation::FieldId::from)
            .collect()
    };

    let snapshot = SnapshotOptions {
        output_dir: args.output.clone(),
        stride: args.stride,
        signals: signals.keys().cloned().collect(),
        fields: requested_fields,
        seed: args.seed,
    };

    info!("Starting snapshot run: steps={}", args.steps);

    let report = match run_simulation(
        &mut runtime,
        RunOptions {
            steps: args.steps,
            print_signals: false,
            signals: signals.keys().cloned().collect(),
            snapshot: Some(snapshot),
        },
    ) {
        Ok(report) => report,
        Err(e) => {
            error!("Run failed: {}", e);
            process::exit(1);
        }
    };

    if let Some(dir) = report.run_dir {
        info!("Snapshot run complete. Saved to: {}", dir.display());
    }
}
