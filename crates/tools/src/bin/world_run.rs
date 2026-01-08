//! World Runner
//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: world-run <world-dir> [--ticks N]

use std::env;
use std::path::Path;
use std::process;

use continuum_dsl::load_world;
use continuum_foundation::SignalId;
use continuum_ir::{
    build_assertion, build_era_configs, build_resolver, compile, convert_assertion_severity,
    get_initial_signal_value, lower,
};
use continuum_runtime::executor::Runtime;
use continuum_runtime::types::Value;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <world-dir> [--ticks N]", args[0]);
        process::exit(1);
    }

    let world_dir = Path::new(&args[1]);

    // Parse optional --ticks argument
    let mut num_ticks: u64 = 10;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--ticks" && i + 1 < args.len() {
            num_ticks = args[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("Error: invalid tick count");
                process::exit(1);
            });
            i += 2;
        } else {
            i += 1;
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

    // Build era configs
    let era_configs = build_era_configs(&world);
    for (era_id, era) in &world.eras {
        println!("  Era {}: dt = {:.2e} seconds", era_id, era.dt_seconds);
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
    println!("Running {} ticks...\n", num_ticks);

    for tick in 0..num_ticks {
        match runtime.execute_tick() {
            Ok(ctx) => {
                print!("Tick {}: ", tick);

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
            }
            Err(e) => {
                eprintln!("\nError at tick {}: {}", tick, e);
                process::exit(1);
            }
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("Simulation complete!");
}
