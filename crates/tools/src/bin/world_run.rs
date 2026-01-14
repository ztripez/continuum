//! World Runner.
//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: `world-run <world-dir> [--steps N] [--dt SECONDS]`

use std::fs;
use std::path::PathBuf;
use std::process;

use chrono::Local;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use continuum_compiler::ir::{RuntimeBuildOptions, build_runtime, compile};
use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::Value;

#[derive(Serialize, Deserialize)]
struct RunManifest {
    run_id: String,
    created_at: String,
    seed: u64, // Not currently passed in args, defaulting to 0 for now or extracting if added
    steps: u64,
    stride: u64,
    signals: Vec<String>,
    fields: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct TickSnapshot {
    tick: u64,
    time_seconds: f64,
    signals: std::collections::HashMap<String, Value>,
    fields: std::collections::HashMap<String, Vec<FieldSample>>,
}

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
    let _fractures = world.fractures();
    let _entities = world.entities();
    let _members = world.members();

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
    let run_dir = if let Some(base_dir) = args.save_dir {
        let run_id = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let dir = base_dir.join(&run_id);
        fs::create_dir_all(&dir).expect("failed to create run directory");

        // Write manifest
        let manifest = RunManifest {
            run_id: run_id.clone(),
            created_at: Local::now().to_rfc3339(),
            seed: 0, // TODO: threaded seed support
            steps: args.steps,
            stride: args.stride,
            signals: signals.keys().map(|id| id.to_string()).collect(),
            fields: fields.keys().map(|id| id.to_string()).collect(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialization failed");
        fs::write(dir.join("run.json"), manifest_json).expect("failed to write run.json");

        info!("Snapshot output enabled: {}", dir.display());
        Some(dir)
    } else {
        None
    };

    // Run warmup if any functions registered
    if !runtime.is_warmup_complete() {
        info!("Executing warmup phase...");
        match runtime.execute_warmup() {
            Ok(result) => {
                info!(
                    "Warmup complete: {} iterations, converged: {}",
                    result.iterations, result.converged
                );
            }
            Err(e) => {
                error!("Warmup failed: {}", e);
                process::exit(1);
            }
        }
    }

    // Run simulation
    info!("Running {} steps...", args.steps);

    for step in 0..args.steps {
        match runtime.execute_tick() {
            Ok(ctx) => {
                print!("Step {}: ", step);

                // Print signal values
                let mut first = true;
                for (signal_id, _) in &signals {
                    let runtime_id = signal_id.clone();
                    if let Some(value) = runtime.get_signal(&runtime_id) {
                        if !first {
                            print!(", ");
                        }
                        first = false;
                        match value {
                            Value::Scalar(v) => print!("{} = {:.4}", signal_id, v),
                            Value::Vec3(v) => {
                                print!("{} = [{:.2}, {:.2}, {:.2}]", signal_id, v[0], v[1], v[2])
                            }
                            _ => print!("{} = {:?}", signal_id, value),
                        }
                    }
                }
                println!(" (dt={:.2e}s)", ctx.dt.seconds());

                // Capture snapshot if enabled
                if let Some(ref dir) = run_dir {
                    if step % args.stride == 0 {
                        let mut signal_values = std::collections::HashMap::new();
                        for id in signals.keys() {
                            if let Some(val) = runtime.get_signal(id) {
                                signal_values.insert(id.to_string(), val.clone());
                            }
                        }

                        let mut field_values = std::collections::HashMap::new();
                        // Drain fields so they are captured (and cleared for next tick)
                        let all_fields = runtime.drain_fields();
                        for (id, samples) in &all_fields {
                            field_values.insert(id.to_string(), samples.clone());
                        }

                        let snapshot = TickSnapshot {
                            tick: ctx.tick,
                            time_seconds: ctx.tick as f64 * ctx.dt.0, // Approximation
                            signals: signal_values,
                            fields: field_values,
                        };

                        let snap_json =
                            serde_json::to_string_pretty(&snapshot).expect("serialization failed");
                        let snap_path = dir.join(format!("tick_{:06}.json", step));
                        if let Err(e) = fs::write(&snap_path, snap_json) {
                            error!("Failed to write snapshot: {}", e);
                        }
                    } else {
                        // Ensure fields are drained even if not snapshotting, to prevent buffer growth
                        runtime.drain_fields();
                    }
                } else {
                    // Print field values if any were emitted (legacy behavior)
                    let fields = runtime.drain_fields();
                    if !fields.is_empty() {
                        for (field_id, samples) in &fields {
                            for sample in samples {
                                match &sample.value {
                                    Value::Scalar(v) => {
                                        println!("  [field] {} = {:.4}", field_id, v)
                                    }
                                    Value::Vec3(v) => {
                                        println!(
                                            "  [field] {} = [{:.2}, {:.2}, {:.2}]",
                                            field_id, v[0], v[1], v[2]
                                        )
                                    }
                                    _ => println!("  [field] {} = {:?}", field_id, sample.value),
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Error at step {}: {}", step, e);
                process::exit(1);
            }
        }
    }

    info!("Simulation complete!");
}
