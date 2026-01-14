//! World Runner.
//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: `world-run <world-dir> [--steps N] [--dt SECONDS]`

use std::path::PathBuf;
use std::process;

use clap::Parser;
use tracing::{error, info, warn};

use continuum_compiler::ir::{RuntimeBuildOptions, build_runtime, compile};
use continuum_runtime::executor::{RunOptions, SnapshotOptions, run_simulation};

#[derive(Parser, Debug)]
#[command(name = "world-run")]
#[command(about = "Compile and execute a Continuum world")]
struct Args {
    /// Path to the World root directory
    world_dir: PathBuf,

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

    // Load and compile world using unified compiler
    info!("Loading world from: {}", args.world_dir.display());

    let compile_result = continuum_compiler::compile_from_dir_result(&args.world_dir);

    if compile_result.has_errors() {
        error!("{}", compile_result.format_diagnostics().trim_end());
        process::exit(1);
    }

    if !compile_result.diagnostics.is_empty() {
        warn!("{}", compile_result.format_diagnostics().trim_end());
    }

    let world = compile_result.world.expect("no world despite no errors");
    info!("Successfully compiled world");

    let strata = world.strata();
    let eras = world.eras();
    let signals = world.signals();
    let fields = world.fields();

    info!("  Strata: {}", strata.len());
    info!("  Eras: {}", eras.len());
    info!("  Signals: {}", signals.len());
    info!("  Fields: {}", fields.len());
    info!("  Constants: {}", world.constants.len());
    info!("  Config: {}", world.config.len());

    // Compile to DAGs
    info!("Compiling to DAGs...");
    let compilation = match compile(&world) {
        Ok(c) => c,
        Err(e) => {
            error!("Compilation error: {}", e);
            process::exit(1);
        }
    };

    info!("  Resolver indices: {}", compilation.resolver_indices.len());
    info!("  Field indices: {}", compilation.field_indices.len());
    info!("  Fracture indices: {}", compilation.fracture_indices.len());
    info!("  Eras in DAG: {}", compilation.dags.era_count());

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
            "  Total: {} resolvers, {} aggregate resolvers",
            report.resolver_count, report.aggregate_count
        );
    }
    if report.assertion_count > 0 {
        info!("  Registered {} assertions", report.assertion_count);
    }
    if report.field_count > 0 {
        info!("  Registered {} field measures", report.field_count);
    }
    if report.skipped_fields > 0 {
        info!(
            "  Skipped {} fields with entity expressions (EntityExecutor not yet implemented)",
            report.skipped_fields
        );
    }
    if report.fracture_count > 0 {
        info!("  Registered {} fractures", report.fracture_count);
    }
    if report.member_signal_count > 0 {
        info!(
            "  Initialized {} member signals (max {} instances)",
            report.member_signal_count, report.max_member_instances
        );
        if report.member_resolvers.total() > 0 {
            info!(
                "  Total: {} scalar + {} Vec3 = {} member resolvers registered",
                report.member_resolvers.scalar_count,
                report.member_resolvers.vec3_count,
                report.member_resolvers.total()
            );
        }
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
