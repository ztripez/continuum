//! World Loader.
//!
//! Loads and parses a Continuum world from a directory.
//!
//! Usage: `world-load <world-dir>`

// Link against functions crate to pull in kernel function registrations
extern crate continuum_functions;

use std::env;
use std::fs;
use std::path::Path;
use std::process;

use tracing::{error, info};

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

    // Check for world.yaml
    let world_yaml = world_dir.join("world.yaml");
    if !world_yaml.exists() {
        error!("No world.yaml found in '{}'", world_dir.display());
        process::exit(1);
    }

    info!("Loading world from: {}", world_dir.display());

    // Find all .cdsl files recursively
    let mut cdsl_files = Vec::new();
    collect_cdsl_files(world_dir, &mut cdsl_files);

    // Sort lexicographically for deterministic ordering
    cdsl_files.sort();

    info!("Found {} .cdsl file(s)", cdsl_files.len());
    for file in &cdsl_files {
        info!("  - {}", file.display());
    }

    // Parse each file
    let mut total_items = 0;
    let mut has_errors = false;

    for file in &cdsl_files {
        let rel_path = file.strip_prefix(world_dir).unwrap_or(file);
        info!("Parsing: {}", rel_path.display());

        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                error!("Error reading file: {}", e);
                has_errors = true;
                continue;
            }
        };

        let (result, parse_errors) = continuum_dsl::parse(&source);

        if !parse_errors.is_empty() {
            error!("Parse errors in {}:", rel_path.display());
            for err in &parse_errors {
                error!("  - {}", err);
            }
            has_errors = true;
            continue;
        }

        let unit = match result {
            Some(u) => u,
            None => {
                error!("Failed to produce AST for {}", rel_path.display());
                has_errors = true;
                continue;
            }
        };

        // Run validation
        let validation_errors = continuum_dsl::validate(&unit);
        if !validation_errors.is_empty() {
            error!("Validation errors in {}:", rel_path.display());
            for err in &validation_errors {
                error!("  - {}", err);
            }
            has_errors = true;
            continue;
        }

        info!("Parsed {} item(s) in {}", unit.items.len(), rel_path.display());
        for item in &unit.items {
            let desc = describe_item(&item.node);
            info!("    - {}", desc);
        }

        total_items += unit.items.len();
    }

    if has_errors {
        error!("Completed with errors");
        process::exit(1);
    } else {
        info!("Successfully parsed {} total items", total_items);
    }
}

fn collect_cdsl_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_cdsl_files(&path, files);
            } else if path.extension().map(|e| e == "cdsl").unwrap_or(false) {
                files.push(path);
            }
        }
    }
}

fn describe_item(item: &continuum_dsl::Item) -> String {
    match item {
        continuum_dsl::Item::ConstBlock(consts) => {
            format!("const block ({} constants)", consts.entries.len())
        }
        continuum_dsl::Item::ConfigBlock(config) => {
            format!("config block ({} entries)", config.entries.len())
        }
        continuum_dsl::Item::TypeDef(t) => format!("type {}", t.name.node),
        continuum_dsl::Item::FnDef(f) => format!("fn {}", f.path.node),
        continuum_dsl::Item::StrataDef(s) => format!("strata {}", s.path.node),
        continuum_dsl::Item::EraDef(e) => format!("era {}", e.name.node),
        continuum_dsl::Item::SignalDef(s) => format!("signal {}", s.path.node),
        continuum_dsl::Item::FieldDef(f) => format!("field {}", f.path.node),
        continuum_dsl::Item::OperatorDef(o) => format!("operator {}", o.path.node),
        continuum_dsl::Item::ImpulseDef(i) => format!("impulse {}", i.path.node),
        continuum_dsl::Item::FractureDef(f) => format!("fracture {}", f.path.node),
        continuum_dsl::Item::ChronicleDef(c) => format!("chronicle {}", c.path.node),
        continuum_dsl::Item::EntityDef(e) => format!("entity {}", e.path.node),
        continuum_dsl::Item::MemberDef(m) => format!("member {}", m.path.node),
    }
}
