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
use tracing::{error, info};

use continuum_compiler::ir::{
    build_aggregate_resolver, build_assertion, build_era_configs, build_field_measure,
    build_fracture, build_member_resolver, build_signal_resolver, build_vec3_member_resolver,
    build_warmup_fn, compile, convert_assertion_severity, eval_initial_expr,
    get_initial_signal_value,
};
use continuum_foundation::{InstanceId, PrimitiveStorageClass};
use continuum_runtime::executor::{ResolverFn, Runtime};
use continuum_runtime::soa_storage::ValueType as MemberValueType;
use continuum_runtime::storage::{EntityInstances, FieldSample, InstanceData};
use continuum_runtime::types::{Dt, Value, WarmupConfig};

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

fn offset_to_line_col(text: &str, offset: usize) -> (u32, u32) {
    let mut line = 0;
    let mut col = 0;
    let mut current_byte = 0;
    for c in text.chars() {
        if current_byte >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
        current_byte += c.len_utf8();
    }
    (line, col)
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

    // Load and compile world using unified compiler
    info!("Loading world from: {}", world_dir.display());

    let compile_result = continuum_compiler::compile_from_dir_result(world_dir);

    if compile_result.has_errors() {
        for diag in compile_result.diagnostics {
            if diag.severity == continuum_compiler::Severity::Error {
                let loc = if let (Some(file), Some(span)) = (&diag.file, &diag.span) {
                    if let Some(source) = compile_result.sources.get(file) {
                        let (line, col) = offset_to_line_col(source, span.start);
                        format!("{}:{}:{} ", file.display(), line + 1, col + 1)
                    } else {
                        format!("{}: ", file.display())
                    }
                } else {
                    String::new()
                };
                error!("{}{}", loc, diag.message);
            }
        }
        process::exit(1);
    }

    // Print warnings
    for diag in &compile_result.diagnostics {
        if diag.severity == continuum_compiler::Severity::Warning {
            let loc = if let (Some(file), Some(span)) = (&diag.file, &diag.span) {
                if let Some(source) = compile_result.sources.get(file) {
                    let (line, col) = offset_to_line_col(source, span.start);
                    format!("{}:{}:{} ", file.display(), line + 1, col + 1)
                } else {
                    format!("{}:at {:?} ", file.display(), span)
                }
            } else {
                String::new()
            };
            tracing::warn!("{}{}", loc, diag.message);
        }
    }

    let world = compile_result.world.expect("no world despite no errors");
    info!("Successfully compiled world");

    let strata = world.strata();
    let eras = world.eras();
    let signals = world.signals();
    let fields = world.fields();
    let fractures = world.fractures();
    let entities = world.entities();
    let members = world.members();

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

    // Build runtime
    info!("Building runtime...");

    // Find initial era
    let initial_era = eras
        .iter()
        .find(|(_, era)| era.is_initial)
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| {
            eras.keys()
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

    for (era_id, era) in &eras {
        let effective_dt = era_configs
            .get(era_id)
            .map(|c| c.dt.0)
            .unwrap_or(era.dt_seconds);
        info!("  Era {}: dt = {:.2e} seconds", era_id, effective_dt);
    }

    // Create runtime
    let mut runtime = Runtime::new(initial_era.clone(), era_configs, compilation.dags);

    // Register resolvers
    // IMPORTANT: Must register in the same order as DAG builder expects (all signals)
    // Signals with entity expressions get:
    //   1. A placeholder resolver (maintains index ordering for DAG)
    //   2. An aggregate resolver (runs in Phase 3c after member resolution)
    let mut resolver_count = 0;
    let mut aggregate_count = 0;
    for (signal_id, signal) in &signals {
        if let Some(resolver) = build_signal_resolver(signal, &world) {
            let idx = runtime.register_resolver(resolver);
            info!("  Registered resolver for {} (idx={})", signal_id, idx);
            resolver_count += 1;
        } else if let Some(ref resolve_expr) = signal.resolve {
            // Signal has entity expressions - register placeholder for DAG ordering
            // and a separate aggregate resolver for Phase 3c
            let signal_name = signal_id.to_string();
            let placeholder: ResolverFn = Box::new(move |_ctx| {
                // This placeholder should never be called - aggregates run in Phase 3c
                panic!(
                    "Signal '{}' placeholder called - aggregate signals run in Phase 3c",
                    signal_name
                );
            });
            let idx = runtime.register_resolver(placeholder);

            // Build and register the actual aggregate resolver
            let aggregate_resolver = build_aggregate_resolver(resolve_expr, &world);
            runtime.register_aggregate_resolver(signal_id.clone(), aggregate_resolver);
            info!(
                "  Registered aggregate resolver for {} (placeholder idx={})",
                signal_id, idx
            );
            aggregate_count += 1;
        } else {
            // Signal has no resolve expression - just register a no-op placeholder
            let signal_name = signal_id.to_string();
            let placeholder: ResolverFn = Box::new(move |_ctx| {
                panic!("Signal '{}' has no resolve expression", signal_name);
            });
            let idx = runtime.register_resolver(placeholder);
            info!(
                "  Registered placeholder for {} (idx={}) - no resolve expr",
                signal_id, idx
            );
        }

        // Register warmup if present
        if let Some(ref warmup) = signal.warmup {
            let warmup_fn = build_warmup_fn(&warmup.iterate, &world.constants, &world.config);
            let config = WarmupConfig {
                max_iterations: warmup.iterations,
                convergence_epsilon: warmup.convergence,
            };
            runtime.register_warmup(signal_id.clone(), warmup_fn, config);
            info!("  Registered warmup for {}", signal_id);
        }
    }
    info!(
        "  Total: {} resolvers, {} aggregate resolvers",
        resolver_count, aggregate_count
    );

    // Register assertions
    let mut assertion_count = 0;
    for (signal_id, signal) in &signals {
        for assertion in &signal.assertions {
            let assertion_fn = build_assertion(&assertion.condition, &world);
            let severity = convert_assertion_severity(assertion.severity);
            runtime.register_assertion(
                signal_id.clone(),
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
    // Fields with entity expressions (aggregates, etc.) are skipped - they require EntityExecutor
    let mut field_count = 0;
    let mut skipped_fields = 0;
    for (field_id, field) in &fields {
        if let Some(ref expr) = field.measure {
            let runtime_id = field_id.clone();
            if let Some(measure_fn) = build_field_measure(&runtime_id, expr, &world) {
                let idx = runtime.register_measure_op(measure_fn);
                info!("  Registered field measure for {} (idx={})", field_id, idx);
                field_count += 1;
            } else {
                // Field contains entity expressions - requires EntityExecutor (not implemented yet)
                skipped_fields += 1;
            }
        }
    }
    if field_count > 0 {
        info!("  Registered {} field measures", field_count);
    }
    if skipped_fields > 0 {
        info!(
            "  Skipped {} fields with entity expressions (EntityExecutor not yet implemented)",
            skipped_fields
        );
    }

    // Register fracture detectors
    let mut fracture_count = 0;
    for (fracture_id, fracture) in &fractures {
        let fracture_fn = build_fracture(fracture, &world);
        let idx = runtime.register_fracture(fracture_fn);
        info!("  Registered fracture {} (idx={})", fracture_id, idx);
        fracture_count += 1;
    }
    if fracture_count > 0 {
        info!("  Registered {} fractures", fracture_count);
    }

    // Initialize signals
    for (signal_id, _signal) in &signals {
        let value = get_initial_signal_value(&world, signal_id);
        runtime.init_signal(signal_id.clone(), value.clone());
        match value {
            Value::Scalar(v) => info!("  Initialized signal {} = {}", signal_id, v),
            _ => info!("  Initialized signal {} = {:?}", signal_id, value),
        }
    }

    // Initialize entities
    for (entity_id, entity) in &entities {
        // Determine instance count from config, bounds, or default
        let count = if let Some(ref count_source) = entity.count_source {
            world
                .config
                .get(count_source)
                .map(|(v, _)| *v as usize)
                .unwrap_or(1)
        } else if let Some((min, max)) = entity.count_bounds {
            // If bounds are fixed (min == max), use that; otherwise use min
            if min == max {
                min as usize
            } else {
                min as usize
            }
        } else {
            1
        };

        // Create instances with unique IDs
        let mut instances = EntityInstances::new();
        for i in 0..count {
            let instance_id = InstanceId::from(format!("{}_{}", entity_id, i));

            // Initialize member fields for this instance
            let mut fields = indexmap::IndexMap::new();
            for (_member_id, member) in &members {
                if &member.entity_id == entity_id {
                    // Use default value based on member's value type
                    let initial_value = member.value_type.default_value();
                    fields.insert(member.signal_name.clone(), initial_value);
                }
            }

            instances.insert(instance_id, InstanceData::new(fields));
        }

        runtime.init_entity(entity_id.clone(), instances);
        info!(
            "  Initialized entity {} with {} instances",
            entity_id, count
        );
    }

    // Initialize member signals for SoA execution
    if !members.is_empty() {
        info!("Initializing member signals...");

        // Build per-entity instance counts and find max for storage allocation
        let mut entity_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (entity_id, entity) in &entities {
            let count = if let Some(ref count_source) = entity.count_source {
                world
                    .config
                    .get(count_source)
                    .map(|(v, _)| *v as usize)
                    .unwrap_or(1)
            } else if let Some((min, max)) = entity.count_bounds {
                if min == max {
                    min as usize
                } else {
                    min as usize
                }
            } else {
                1
            };
            entity_counts.insert(entity_id.to_string(), count);
        }

        // Use max instance count for storage allocation (signals need enough slots for largest entity)
        let max_instance_count = entity_counts.values().copied().max().unwrap_or(1);

        // Register member signals (use full member ID to avoid name collisions)
        for (member_id, member) in &members {
            let value_type = match member.value_type.storage_class() {
                PrimitiveStorageClass::Scalar => MemberValueType::scalar(),
                PrimitiveStorageClass::Vec2 => MemberValueType::vec2(),
                PrimitiveStorageClass::Vec3 => MemberValueType::vec3(),
                PrimitiveStorageClass::Vec4 => {
                    if member.value_type.primitive_id().name() == "Quat" {
                        MemberValueType::quat()
                    } else {
                        MemberValueType::vec4()
                    }
                }
                _ => MemberValueType::scalar(),
            };
            // Use full member ID (e.g., "stellar.star.mass") instead of just signal_name ("mass")
            // to avoid collisions between entities with same-named members
            runtime.register_member_signal(&member_id.to_string(), value_type);
        }

        // Initialize member instances with max count for storage
        runtime.init_member_instances(max_instance_count);

        // Register per-entity instance counts for aggregate operations
        for (entity_id, count) in &entity_counts {
            runtime.register_entity_count(entity_id, *count);
            info!("  Registered entity {} with {} instances", entity_id, count);
        }

        info!(
            "  Initialized {} member signals (max {} instances)",
            members.len(),
            max_instance_count
        );

        // Set initial values for members with initial expressions
        let mut initialized_count = 0;
        for (member_id, member) in &members {
            if let Some(ref initial_expr) = member.initial {
                let initial_value =
                    eval_initial_expr(initial_expr, &world.constants, &world.config);

                // Get the correct instance count for this member's entity
                let instance_count = entity_counts
                    .get(&member.entity_id.to_string())
                    .copied()
                    .unwrap_or(1);

                // Set initial value for all instances of this member
                for instance_idx in 0..instance_count {
                    if let Err(e) = runtime.set_member_signal(
                        &member_id.to_string(),
                        instance_idx,
                        initial_value.clone(),
                    ) {
                        error!("  Failed to set initial value for {}: {}", member_id, e);
                        process::exit(1);
                    }
                }

                info!(
                    "  Initialized member {} with value {:?} ({} instances)",
                    member_id, initial_value, instance_count
                );
                initialized_count += 1;
            }
        }
        if initialized_count > 0 {
            info!(
                "  Set initial values for {} member signals",
                initialized_count
            );
            // Commit initial values so they become "previous" values for resolvers
            runtime.commit_member_initials();
        }

        // Build and register member resolvers
        let mut scalar_resolver_count = 0;
        let mut vec3_resolver_count = 0;
        for (member_id, member) in &members {
            if let Some(ref resolve_expr) = member.resolve {
                // Entity prefix is the entity ID (e.g., "terra.plate" for "terra.plate.age")
                let entity_prefix = &member.entity_id.to_string();

                // Use the appropriate builder based on value type
                match member.value_type.storage_class() {
                    PrimitiveStorageClass::Vec3 => {
                        let resolver = build_vec3_member_resolver(
                            resolve_expr,
                            &world.constants,
                            &world.config,
                            entity_prefix,
                        );
                        runtime.register_vec3_member_resolver(member_id.to_string(), resolver);
                        info!(
                            "  Registered Vec3 member resolver for {} (entity={})",
                            member_id, entity_prefix
                        );
                        vec3_resolver_count += 1;
                    }
                    _ => {
                        // Scalar (and other types for now)
                        let resolver = build_member_resolver(
                            resolve_expr,
                            &world.constants,
                            &world.config,
                            entity_prefix,
                        );
                        runtime.register_member_resolver(member_id.to_string(), resolver);
                        info!(
                            "  Registered scalar member resolver for {} (entity={})",
                            member_id, entity_prefix
                        );
                        scalar_resolver_count += 1;
                    }
                }
            } else {
                info!("  Skipped member {} - no resolve expression", member_id);
            }
        }
        info!(
            "  Total: {} scalar + {} Vec3 = {} member resolvers registered",
            scalar_resolver_count,
            vec3_resolver_count,
            scalar_resolver_count + vec3_resolver_count
        );
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
    info!("Running {} steps...", num_steps);

    for step in 0..num_steps {
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
                    if step % save_stride == 0 {
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
