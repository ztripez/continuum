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

use std::fs;
use std::path::PathBuf;
use std::process;

use chrono::Local;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use continuum_compiler::ir::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_signal_resolver,
    compile, convert_assertion_severity, get_initial_signal_value,
};
// use continuum_foundation::EraId;
use continuum_runtime::executor::Runtime;
use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::{Dt, Value};

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

#[derive(Serialize, Deserialize)]
struct RunManifest {
    run_id: String,
    created_at: String,
    seed: u64,
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

    // Prepare runtime
    let initial_era = world
        .eras
        .iter()
        .find(|(_, era)| era.is_initial)
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| {
            world
                .eras
                .keys()
                .next()
                .cloned()
                .unwrap_or_else(|| continuum_foundation::EraId::from("default"))
        });

    let mut era_configs = build_era_configs(&world);
    if let Some(dt) = args.dt {
        for config in era_configs.values_mut() {
            config.dt = Dt(dt);
        }
    }

    let mut runtime = Runtime::new(initial_era, era_configs, compilation.dags);

    // Register all functions (resolvers, assertions, fields, fractures)
    for (signal_id, signal) in &world.signals {
        if let Some(resolver) = build_signal_resolver(signal, &world) {
            runtime.register_resolver(resolver);
        }
        for assertion in &signal.assertions {
            let assertion_fn = build_assertion(&assertion.condition, &world);
            let severity = convert_assertion_severity(assertion.severity);
            runtime.register_assertion(
                signal_id.clone(),
                assertion_fn,
                severity,
                assertion.message.clone(),
            );
        }
    }

    for (field_id, field) in &world.fields {
        if let Some(ref expr) = field.measure {
            let runtime_id = field_id.clone();
            // Skip fields with entity expressions (aggregates, etc.)
            if let Some(measure_fn) = build_field_measure(&runtime_id, expr, &world) {
                runtime.register_measure_op(measure_fn);
            }
        }
    }

    for (_, fracture) in &world.fractures {
        runtime.register_fracture(build_fracture(fracture, &world));
    }

    // Initialize signals
    for (signal_id, _) in &world.signals {
        runtime.init_signal(
            signal_id.clone(),
            get_initial_signal_value(&world, signal_id),
        );
    }

    // Create run directory
    let run_id = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let run_dir = args.output.join(&run_id);
    fs::create_dir_all(&run_dir).expect("failed to create run directory");

    // Write manifest
    let manifest = RunManifest {
        run_id: run_id.clone(),
        created_at: Local::now().to_rfc3339(),
        seed: args.seed,
        steps: args.steps,
        stride: args.stride,
        signals: world.signals.keys().map(|id| id.to_string()).collect(),
        fields: world.fields.keys().map(|id| id.to_string()).collect(),
    };
    let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialization failed");
    fs::write(run_dir.join("run.json"), manifest_json).expect("failed to write run.json");

    info!("Starting snapshot run: {}, steps={}", run_id, args.steps);

    // Filter requested fields
    let requested_fields: Vec<String> = if args.fields.is_empty() {
        world.fields.keys().map(|id| id.to_string()).collect()
    } else {
        args.fields
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    // Run loop
    for step in 0..=args.steps {
        if step > 0 {
            if let Err(e) = runtime.execute_tick() {
                error!("Error at tick {}: {}", step - 1, e);
                process::exit(1);
            }
        }

        // Capture snapshot at stride
        if step % args.stride == 0 {
            let mut signal_values = std::collections::HashMap::new();
            for id in world.signals.keys() {
                if let Some(val) = runtime.get_signal(id) {
                    signal_values.insert(id.to_string(), val.clone());
                }
            }

            let mut field_values = std::collections::HashMap::new();
            let all_fields = runtime.drain_fields();
            for field_name in &requested_fields {
                let id = continuum_foundation::FieldId::from(field_name.as_str());
                if let Some(samples) = all_fields.get(&id) {
                    field_values.insert(field_name.clone(), samples.clone());
                }
            }

            let snapshot = TickSnapshot {
                tick: step,
                time_seconds: runtime.tick_context().tick as f64 * runtime.tick_context().dt.0,
                signals: signal_values,
                fields: field_values,
            };

            let snap_json = serde_json::to_string_pretty(&snapshot).expect("serialization failed");
            let snap_path = run_dir.join(format!("tick_{:06}.json", step));
            fs::write(snap_path, snap_json).expect("failed to write snapshot");

            info!("Captured snapshot at tick {}", step);
        }
    }

    info!("Snapshot run complete. Saved to: {}", run_dir.display());
}
