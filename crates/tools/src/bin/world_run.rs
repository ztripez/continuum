//! World Runner
//!
//! Loads, compiles, and executes a Continuum world.
//!
//! Usage: world-run <world-dir> [--ticks N]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process;

use indexmap::IndexMap;

use continuum_dsl::ast::CompilationUnit;
use continuum_foundation::{EraId, SignalId, StratumId};
use continuum_ir::{
    compile, lower, BinaryOpIr, CompiledExpr, CompiledWorld, UnaryOpIr, ValueType,
};
use continuum_runtime::executor::{EraConfig, ResolverFn, Runtime};
use continuum_runtime::types::{Dt, StratumState, Value};
use continuum_runtime::operators;

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

    if !world_dir.exists() || !world_dir.is_dir() {
        eprintln!("Error: '{}' is not a valid directory", world_dir.display());
        process::exit(1);
    }

    // Check for world.yaml
    let world_yaml = world_dir.join("world.yaml");
    if !world_yaml.exists() {
        eprintln!(
            "Error: no world.yaml found in '{}'",
            world_dir.display()
        );
        process::exit(1);
    }

    println!("Loading world from: {}", world_dir.display());

    // Find and parse all .cdsl files
    let mut cdsl_files = Vec::new();
    collect_cdsl_files(world_dir, &mut cdsl_files);
    cdsl_files.sort();

    println!("Found {} .cdsl file(s)", cdsl_files.len());

    // Merge all compilation units
    let mut merged_unit = CompilationUnit::default();

    for file in &cdsl_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", file.display(), e);
                process::exit(1);
            }
        };

        let (result, parse_errors) = continuum_dsl::parse(&source);

        if !parse_errors.is_empty() {
            eprintln!("Parse errors in {}:", file.display());
            for err in &parse_errors {
                eprintln!("  - {}", err);
            }
            process::exit(1);
        }

        let unit = match result {
            Some(u) => u,
            None => {
                eprintln!("Failed to parse {}", file.display());
                process::exit(1);
            }
        };

        // Validate
        let validation_errors = continuum_dsl::validate(&unit);
        if !validation_errors.is_empty() {
            eprintln!("Validation errors in {}:", file.display());
            for err in &validation_errors {
                eprintln!("  - {}", err);
            }
            process::exit(1);
        }

        // Merge items
        merged_unit.items.extend(unit.items);
    }

    println!("Parsed {} total items", merged_unit.items.len());

    // Lower to IR
    println!("\nLowering to IR...");
    let world = match lower(&merged_unit) {
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
                .unwrap_or_else(|| EraId::from("default"))
        });

    println!("  Initial era: {}", initial_era);

    // Build era configs
    let era_configs = build_era_configs(&world);

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

    // Initialize signals
    for (signal_id, signal) in &world.signals {
        let initial_value = get_initial_value(&world, signal_id);
        let value = match signal.value_type {
            ValueType::Scalar => Value::Scalar(initial_value),
            ValueType::Vec2 => Value::Vec2([initial_value; 2]),
            ValueType::Vec3 => Value::Vec3([initial_value; 3]),
            ValueType::Vec4 => Value::Vec4([initial_value; 4]),
        };
        runtime.init_signal(
            SignalId(signal_id.0.clone()),
            value,
        );
        println!("  Initialized signal {} = {}", signal_id, initial_value);
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
                            Value::Vec3(v) => print!("{} = [{:.2}, {:.2}, {:.2}]", signal_id, v[0], v[1], v[2]),
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

fn collect_cdsl_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_cdsl_files(&path, files);
            } else if path.extension().is_some_and(|e| e == "cdsl") {
                files.push(path);
            }
        }
    }
}

fn build_era_configs(world: &CompiledWorld) -> HashMap<EraId, EraConfig> {
    let mut configs = HashMap::new();

    for (era_id, era) in &world.eras {
        let mut strata = HashMap::new();
        for (stratum_id, state) in &era.strata_states {
            let runtime_state = match state {
                continuum_ir::StratumStateIr::Active => StratumState::Active,
                continuum_ir::StratumStateIr::ActiveWithStride(s) => {
                    StratumState::ActiveWithStride(*s)
                }
                continuum_ir::StratumStateIr::Gated => StratumState::Gated,
            };
            strata.insert(StratumId(stratum_id.0.clone()), runtime_state);
        }

        println!("  Era {}: dt = {:.2e} seconds", era_id, era.dt_seconds);

        configs.insert(
            EraId(era_id.0.clone()),
            EraConfig {
                dt: Dt(era.dt_seconds),
                strata,
                transition: None, // TODO: implement transition conditions
            },
        );
    }

    configs
}

fn get_initial_value(world: &CompiledWorld, signal_id: &continuum_foundation::SignalId) -> f64 {
    // Check for config.*.initial_* pattern
    let signal_name = &signal_id.0;
    let parts: Vec<&str> = signal_name.split('.').collect();

    // Try various config key patterns
    if parts.len() >= 2 {
        let last = parts.last().unwrap();

        // Try config.<domain>.initial_<signal>
        for (key, value) in &world.config {
            if key.ends_with(&format!("initial_{}", last)) {
                return *value;
            }
        }
    }

    // Default to 0
    0.0
}

fn build_resolver(
    expr: &CompiledExpr,
    world: &CompiledWorld,
    uses_dt_raw: bool,
) -> ResolverFn {
    // Clone what we need for the closure
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let dt = if uses_dt_raw {
            ctx.dt.seconds()
        } else {
            ctx.dt.seconds()
        };

        let result = eval_expr(&expr, ctx.prev, ctx.inputs, dt, &constants, &config, ctx.signals);
        Value::Scalar(result)
    })
}

fn eval_expr(
    expr: &CompiledExpr,
    prev: &Value,
    inputs: f64,
    dt: f64,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    signals: &continuum_runtime::storage::SignalStorage,
) -> f64 {
    match expr {
        CompiledExpr::Literal(v) => *v,
        CompiledExpr::Prev => prev.as_scalar().unwrap_or(0.0),
        CompiledExpr::DtRaw => dt,
        CompiledExpr::SumInputs => inputs,
        CompiledExpr::Signal(id) => {
            let runtime_id = SignalId(id.0.clone());
            signals
                .get(&runtime_id)
                .and_then(|v| v.as_scalar())
                .unwrap_or(0.0)
        }
        CompiledExpr::Const(name) => constants.get(name).copied().unwrap_or(0.0),
        CompiledExpr::Config(name) => config.get(name).copied().unwrap_or(0.0),
        CompiledExpr::Binary { op, left, right } => {
            let l = eval_expr(left, prev, inputs, dt, constants, config, signals);
            let r = eval_expr(right, prev, inputs, dt, constants, config, signals);
            match op {
                BinaryOpIr::Add => l + r,
                BinaryOpIr::Sub => l - r,
                BinaryOpIr::Mul => l * r,
                BinaryOpIr::Div => l / r,
                BinaryOpIr::Pow => l.powf(r),
                BinaryOpIr::Eq => if (l - r).abs() < f64::EPSILON { 1.0 } else { 0.0 },
                BinaryOpIr::Ne => if (l - r).abs() >= f64::EPSILON { 1.0 } else { 0.0 },
                BinaryOpIr::Lt => if l < r { 1.0 } else { 0.0 },
                BinaryOpIr::Le => if l <= r { 1.0 } else { 0.0 },
                BinaryOpIr::Gt => if l > r { 1.0 } else { 0.0 },
                BinaryOpIr::Ge => if l >= r { 1.0 } else { 0.0 },
                BinaryOpIr::And => if l != 0.0 && r != 0.0 { 1.0 } else { 0.0 },
                BinaryOpIr::Or => if l != 0.0 || r != 0.0 { 1.0 } else { 0.0 },
            }
        }
        CompiledExpr::Unary { op, operand } => {
            let v = eval_expr(operand, prev, inputs, dt, constants, config, signals);
            match op {
                UnaryOpIr::Neg => -v,
                UnaryOpIr::Not => if v == 0.0 { 1.0 } else { 0.0 },
            }
        }
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<f64> = args
                .iter()
                .map(|a| eval_expr(a, prev, inputs, dt, constants, config, signals))
                .collect();

            match function.as_str() {
                "decay" => {
                    // decay(value, halflife)
                    if arg_values.len() >= 2 {
                        let result = operators::decay(arg_values[0], arg_values[1], dt);
                        eprintln!("  decay({}, {}, {}) = {}", arg_values[0], arg_values[1], dt, result);
                        result
                    } else {
                        0.0
                    }
                }
                "relax" => {
                    // relax(current, target, tau)
                    if arg_values.len() >= 3 {
                        operators::relax(arg_values[0], arg_values[1], arg_values[2], dt)
                    } else {
                        0.0
                    }
                }
                "integrate" => {
                    // integrate(current, rate)
                    if arg_values.len() >= 2 {
                        operators::integrate(arg_values[0], arg_values[1], dt)
                    } else {
                        0.0
                    }
                }
                "clamp" => {
                    // clamp(value, min, max)
                    if arg_values.len() >= 3 {
                        arg_values[0].clamp(arg_values[1], arg_values[2])
                    } else {
                        0.0
                    }
                }
                "min" => arg_values.iter().cloned().fold(f64::INFINITY, f64::min),
                "max" => arg_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "abs" => arg_values.first().map(|v| v.abs()).unwrap_or(0.0),
                "sqrt" => arg_values.first().map(|v| v.sqrt()).unwrap_or(0.0),
                "sin" => arg_values.first().map(|v| v.sin()).unwrap_or(0.0),
                "cos" => arg_values.first().map(|v| v.cos()).unwrap_or(0.0),
                "exp" => arg_values.first().map(|v| v.exp()).unwrap_or(0.0),
                "ln" => arg_values.first().map(|v| v.ln()).unwrap_or(0.0),
                "log10" => arg_values.first().map(|v| v.log10()).unwrap_or(0.0),
                "pow" => {
                    if arg_values.len() >= 2 {
                        arg_values[0].powf(arg_values[1])
                    } else {
                        0.0
                    }
                }
                _ => {
                    eprintln!("Warning: unknown function '{}'", function);
                    0.0
                }
            }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            let cond = eval_expr(condition, prev, inputs, dt, constants, config, signals);
            if cond != 0.0 {
                eval_expr(then_branch, prev, inputs, dt, constants, config, signals)
            } else {
                eval_expr(else_branch, prev, inputs, dt, constants, config, signals)
            }
        }
        CompiledExpr::Let { name: _, value, body } => {
            // For now, just evaluate body (let bindings need local environment)
            let _val = eval_expr(value, prev, inputs, dt, constants, config, signals);
            eval_expr(body, prev, inputs, dt, constants, config, signals)
        }
        CompiledExpr::FieldAccess { .. } => {
            // Field access on expressions - not yet supported
            0.0
        }
    }
}
