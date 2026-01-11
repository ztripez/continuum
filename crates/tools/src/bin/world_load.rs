//! World Loader.
//!
//! Loads and parses a Continuum world from a directory.
//!
//! Usage: `world-load <world-dir>`

// Link against functions crate to pull in kernel function registrations
extern crate continuum_functions;

use std::env;
use std::path::Path;
use std::process;
use tracing::{error, info};

use continuum_compiler::dsl::ast::Item;

fn main() {
    continuum_tools::init_logging();

    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <world-dir>", args[0]);
        process::exit(1);
    }

    let world_dir = Path::new(&args[1]);

    if !world_dir.exists() {
        error!("Directory '{}' does not exist", world_dir.display());
        process::exit(1);
    }

    if !world_dir.is_dir() {
        error!("'{}' is not a directory", world_dir.display());
        process::exit(1);
    }

    info!("Loading world from: {}", world_dir.display());

    // Use unified compiler to load and compile (lowering happens too)
    match continuum_compiler::compile_from_dir(world_dir) {
        Ok(world) => {
            info!("Successfully compiled world");
            info!("  - Signals: {}", world.signals.len());
            info!("  - Fields: {}", world.fields.len());
            info!("  - Operators: {}", world.operators.len());
            info!("  - Entities: {}", world.entities.len());
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
    }
}

#[allow(dead_code)]
fn describe_item(item: &Item) -> String {
    match item {
        Item::ConstBlock(consts) => {
            format!("const block ({} constants)", consts.entries.len())
        }
        Item::ConfigBlock(config) => {
            format!("config block ({} entries)", config.entries.len())
        }
        Item::TypeDef(t) => format!("type {}", t.name.node),
        Item::FnDef(f) => format!("fn {}", f.path.node),
        Item::StrataDef(s) => format!("strata {}", s.path.node),
        Item::EraDef(e) => format!("era {}", e.name.node),
        Item::SignalDef(s) => format!("signal {}", s.path.node),
        Item::FieldDef(f) => format!("field {}", f.path.node),
        Item::OperatorDef(o) => format!("operator {}", o.path.node),
        Item::ImpulseDef(i) => format!("impulse {}", i.path.node),
        Item::FractureDef(f) => format!("fracture {}", f.path.node),
        Item::ChronicleDef(c) => format!("chronicle {}", c.path.node),
        Item::EntityDef(e) => format!("entity {}", e.path.node),
        Item::MemberDef(m) => format!("member {}", m.path.node),
        Item::WorldDef(w) => format!("world {}", w.path.node),
    }
}
