//! World Runner
//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: world-run <world-dir> [--steps N] [--dt SECONDS]

use std::env;
use std::path::Path;
use std::process;

use continuum_dsl::load_world;
use continuum_foundation::{FieldId, SignalId};
use continuum_ir::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_resolver,
    compile, convert_assertion_severity, get_initial_signal_value, lower, validate,
};
use continuum_runtime::executor::Runtime;
use continuum_runtime::types::{Dt, Value};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <world-dir> [--steps N] [--dt SECONDS]", args[0]);
        process::exit(1);
    }

    let world_dir = Path::new(&args[1]);

    // Parse optional arguments
    let mut num_steps: u64 = 10;
    let mut dt_override: Option<f64> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" | "--ticks" if i + 1 < args.len() => {
                num_steps = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid step count");
                    process::exit(1);
                });
                i += 2;
            }
            "--dt" if i + 1 < args.len() => {
                dt_override = Some(args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid dt value");
                    process::exit(1);
                }));
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Load world
    println!("Loading world from: {}", world_dir.display());

    let load_result = match load_world(world_dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    println!("Found {} .cdsl file(s)", load_result.files.len());
    println!("Parsed {} total items", load_result.unit.items.len());

    // Lower to IR
    println!("\nLowering to IR...");
    let world = match lower(&load_result.unit) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Lowering error: {}", e);
            process::exit(1);
        }
    };

    println!("  Strata: {}", world.strata.len());
    println!("  Eras: {}", world.eras.len());
    println!("  Signals: {}", world.signals.len());
    println!("  Fields: {}", world.fields.len());
    println!("  Constants: {}", world.constants.len());
    println!("  Config: {}", world.config.len());

    // Validate IR
    println!("\nValidating...");
    let warnings = validate(&world);
    if warnings.is_empty() {
        println!("  No warnings");
    } else {
        eprintln!("\n⚠ {} warning(s):", warnings.len());
        for warning in &warnings {
            eprintln!("  - {} (in {})", warning.message, warning.entity);
        }
        eprintln!();
    }

    // Compile to DAGs
    println!("\nCompiling to DAGs...");
    let compilation = match compile(&world) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Compilation error: {}", e);
            process::exit(1);
        }
    };

    println!("  Resolver indices: {}", compilation.resolver_indices.len());
    println!("  Field indices: {}", compilation.field_indices.len());
    println!("  Fracture indices: {}", compilation.fracture_indices.len());
    println!("  Eras in DAG: {}", compilation.dags.era_count());

    // Build runtime
    println!("\nBuilding runtime...");

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

    println!("  Initial era: {}", initial_era);

    // Build era configs (with optional dt override)
    let mut era_configs = build_era_configs(&world);

    if let Some(dt) = dt_override {
        println!("  Overriding dt = {:.2e} seconds", dt);
        for config in era_configs.values_mut() {
            config.dt = Dt(dt);
        }
    }

    for (era_id, era) in &world.eras {
        let effective_dt = era_configs
            .get(era_id)
            .map(|c| c.dt.0)
            .unwrap_or(era.dt_seconds);
        println!("  Era {}: dt = {:.2e} seconds", era_id, effective_dt);
    }

    // Create runtime
    let mut runtime = Runtime::new(initial_era.clone(), era_configs, compilation.dags);

    // Register resolvers
    for (signal_id, signal) in &world.signals {
        if let Some(ref expr) = signal.resolve {
            let resolver = build_resolver(expr, &world, signal.uses_dt_raw);
            let idx = runtime.register_resolver(resolver);
            println!("  Registered resolver for {} (idx={})", signal_id, idx);
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
        println!("  Registered {} assertions", assertion_count);
    }

    // Register field measure functions
    let mut field_count = 0;
    for (field_id, field) in &world.fields {
        if let Some(ref expr) = field.measure {
            let runtime_id = FieldId(field_id.0.clone());
            let measure_fn = build_field_measure(&runtime_id, expr, &world);
            let idx = runtime.register_measure_op(measure_fn);
            println!("  Registered field measure for {} (idx={})", field_id, idx);
            field_count += 1;
        }
    }
    if field_count > 0 {
        println!("  Registered {} field measures", field_count);
    }

    // Register fracture detectors
    let mut fracture_count = 0;
    for (fracture_id, fracture) in &world.fractures {
        let fracture_fn = build_fracture(fracture, &world);
        let idx = runtime.register_fracture(fracture_fn);
        println!("  Registered fracture {} (idx={})", fracture_id, idx);
        fracture_count += 1;
    }
    if fracture_count > 0 {
        println!("  Registered {} fractures", fracture_count);
    }

    // Initialize signals
    for (signal_id, _signal) in &world.signals {
        let value = get_initial_signal_value(&world, signal_id);
        runtime.init_signal(SignalId(signal_id.0.clone()), value.clone());
        match value {
            Value::Scalar(v) => println!("  Initialized signal {} = {}", signal_id, v),
            _ => println!("  Initialized signal {} = {:?}", signal_id, value),
        }
    }

    // Run simulation
    println!("\n{}", "=".repeat(50));
    println!("Running {} steps...\n", num_steps);

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

                // Print field values if any were emitted
                let fields = runtime.drain_fields();
                if !fields.is_empty() {
                    for (field_id, samples) in &fields {
                        for sample in samples {
                            match &sample.value {
                                Value::Scalar(v) => println!("  [field] {} = {:.4}", field_id, v),
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
            Err(e) => {
                eprintln!("\nError at step {}: {}", step, e);
                process::exit(1);
            }
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("Simulation complete!");
}
