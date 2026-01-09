//! World Runner.
//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: `world-run <world-dir> [--steps N] [--dt SECONDS]`

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use chrono::Local;
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use continuum_dsl::load_world;
use continuum_foundation::{FieldId, SignalId};
use continuum_ir::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_resolver,
    compile, convert_assertion_severity, get_initial_signal_value, lower, validate,
};
use continuum_runtime::executor::Runtime;
use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::{Dt, Value};

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

fn main() {
    continuum_tools::init_logging();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <world-dir> [--steps N] [--dt SECONDS] [--save <DIR>] [--stride N]",
            args[0]
        );
        process::exit(1);
    }

    let world_dir = Path::new(&args[1]);

    // Parse optional arguments
    let mut num_steps: u64 = 10;
    let mut dt_override: Option<f64> = None;
    let mut save_dir: Option<PathBuf> = None;
    let mut save_stride: u64 = 10;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" | "--ticks" if i + 1 < args.len() => {
                num_steps = args[i + 1].parse().unwrap_or_else(|_| {
                    error!("Invalid step count");
                    process::exit(1);
                });
                i += 2;
            }
            "--dt" if i + 1 < args.len() => {
                dt_override = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    error!("Invalid dt value");
                    process::exit(1);
                }));
                i += 2;
            }
            "--save" | "--snapshot-dir" if i + 1 < args.len() => {
                save_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--stride" | "--snapshot-stride" if i + 1 < args.len() => {
                save_stride = args[i + 1].parse().unwrap_or_else(|_| {
                    error!("Invalid stride value");
                    process::exit(1);
                });
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Load world
    info!("Loading world from: {}", world_dir.display());

    let load_result = match load_world(world_dir) {
        Ok(r) => r,
        Err(e) => {
            error!("Error loading world: {}", e);
            process::exit(1);
        }
    };

    info!("Found {} .cdsl file(s)", load_result.files.len());
    info!("Parsed {} total items", load_result.unit.items.len());

    // Lower to IR
    info!("Lowering to IR...");
    let world = match lower(&load_result.unit) {
        Ok(w) => w,
        Err(e) => {
            error!("Lowering error: {}", e);
            process::exit(1);
        }
    };

    info!("  Strata: {}", world.strata.len());
    info!("  Eras: {}", world.eras.len());
    info!("  Signals: {}", world.signals.len());
    info!("  Fields: {}", world.fields.len());
    info!("  Constants: {}", world.constants.len());
    info!("  Config: {}", world.config.len());

    // Validate IR
    info!("Validating...");
    let warnings = validate(&world);
    if warnings.is_empty() {
        info!("  No warnings");
    } else {
        warn!("{} warning(s):", warnings.len());
        for warning in &warnings {
            warn!("  - {} (in {})", warning.message, warning.entity);
        }
    }

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

    // Build runtime
    info!("Building runtime...");

    // Find initial era
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

    info!("  Initial era: {}", initial_era);

    // Build era configs (with optional dt override)
    let mut era_configs = build_era_configs(&world);

    if let Some(dt) = dt_override {
        info!("  Overriding dt = {:.2e} seconds", dt);
        for config in era_configs.values_mut() {
            config.dt = Dt(dt);
        }
    }

    for (era_id, era) in &world.eras {
        let effective_dt = era_configs
            .get(era_id)
            .map(|c| c.dt.0)
            .unwrap_or(era.dt_seconds);
        info!("  Era {}: dt = {:.2e} seconds", era_id, effective_dt);
    }

    // Create runtime
    let mut runtime = Runtime::new(initial_era.clone(), era_configs, compilation.dags);

    // Register resolvers
    for (signal_id, signal) in &world.signals {
        if let Some(ref expr) = signal.resolve {
            let resolver = build_resolver(expr, &world, signal.uses_dt_raw);
            let idx = runtime.register_resolver(resolver);
            info!("  Registered resolver for {} (idx={})", signal_id, idx);
        }
    }

    // Register assertions
    let mut assertion_count = 0;
    for (signal_id, signal) in &world.signals {
        for assertion in &signal.assertions {
            let assertion_fn = build_assertion(&assertion.condition, &world);
            let severity = convert_assertion_severity(assertion.severity);
            runtime.register_assertion(
                SignalId(signal_id.0.clone()),
                assertion_fn,
                severity,
                assertion.message.clone(),
            );
            assertion_count += 1;
        }
    }
    if assertion_count > 0 {
        info!("  Registered {} assertions", assertion_count);
    }

    // Register field measure functions
    let mut field_count = 0;
    for (field_id, field) in &world.fields {
        if let Some(ref expr) = field.measure {
            let runtime_id = FieldId(field_id.0.clone());
            let measure_fn = build_field_measure(&runtime_id, expr, &world);
            let idx = runtime.register_measure_op(measure_fn);
            info!("  Registered field measure for {} (idx={})", field_id, idx);
            field_count += 1;
        }
    }
    if field_count > 0 {
        info!("  Registered {} field measures", field_count);
    }

    // Register fracture detectors
    let mut fracture_count = 0;
    for (fracture_id, fracture) in &world.fractures {
        let fracture_fn = build_fracture(fracture, &world);
        let idx = runtime.register_fracture(fracture_fn);
        info!("  Registered fracture {} (idx={})", fracture_id, idx);
        fracture_count += 1;
    }
    if fracture_count > 0 {
        info!("  Registered {} fractures", fracture_count);
    }

    // Initialize signals
    for (signal_id, _signal) in &world.signals {
        let value = get_initial_signal_value(&world, signal_id);
        runtime.init_signal(SignalId(signal_id.0.clone()), value.clone());
        match value {
            Value::Scalar(v) => info!("  Initialized signal {} = {}", signal_id, v),
            _ => info!("  Initialized signal {} = {:?}", signal_id, value),
        }
    }

    // Prepare snapshot directory if requested
    let run_dir = if let Some(base_dir) = save_dir {
        let run_id = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let dir = base_dir.join(&run_id);
        fs::create_dir_all(&dir).expect("failed to create run directory");

        // Write manifest
        let manifest = RunManifest {
            run_id: run_id.clone(),
            created_at: Local::now().to_rfc3339(),
            seed: 0, // TODO: threaded seed support
            steps: num_steps,
            stride: save_stride,
            signals: world.signals.keys().map(|id| id.0.clone()).collect(),
            fields: world.fields.keys().map(|id| id.0.clone()).collect(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialization failed");
        fs::write(dir.join("run.json"), manifest_json).expect("failed to write run.json");

        info!("Snapshot output enabled: {}", dir.display());
        Some(dir)
    } else {
        None
    };

    // Run simulation
    info!("Running {} steps...", num_steps);

    for step in 0..num_steps {
        match runtime.execute_tick() {
            Ok(ctx) => {
                print!("Step {}: ", step);

                // Print signal values
                let mut first = true;
                for (signal_id, _) in &world.signals {
                    let runtime_id = SignalId(signal_id.0.clone());
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
                    if step % save_stride == 0 {
                        let mut signal_values = std::collections::HashMap::new();
                        for id in world.signals.keys() {
                            if let Some(val) = runtime.get_signal(&SignalId(id.0.clone())) {
                                signal_values.insert(id.0.clone(), val.clone());
                            }
                        }

                        let mut field_values = std::collections::HashMap::new();
                        // Drain fields so they are captured (and cleared for next tick)
                        let all_fields = runtime.drain_fields();
                        for (id, samples) in &all_fields {
                            field_values.insert(id.0.clone(), samples.clone());
                        }

                        let snapshot = TickSnapshot {
                            tick: ctx.tick,
                            time_seconds: ctx.tick as f64 * ctx.dt.0, // Approximation
                            signals: signal_values,
                            fields: field_values,
                        };

                        let snap_json = serde_json::to_string_pretty(&snapshot)
                            .expect("serialization failed");
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
