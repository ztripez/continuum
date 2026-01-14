//!
//! Loads and compiles a Continuum world from a directory.
//!
//! Usage: `check <world-dir>`

// Link against functions crate to pull in kernel function registrations
extern crate continuum_functions;

use clap::Parser;
use std::path::PathBuf;
use std::process;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "check")]
#[command(about = "Compile a Continuum world and report diagnostics")]
struct Args {
    /// Path to the World root directory
    world_dir: PathBuf,
}

fn main() {
    continuum_tools::init_logging();

    let args = Args::parse();

    if !args.world_dir.exists() {
        error!("Directory '{}' does not exist", args.world_dir.display());
        process::exit(1);
    }

    if !args.world_dir.is_dir() {
        error!("'{}' is not a directory", args.world_dir.display());
        process::exit(1);
    }

    info!("Loading world from: {}", args.world_dir.display());

    let compile_result = continuum_compiler::compile_from_dir_result(&args.world_dir);
    let diagnostics = compile_result.format_diagnostics();

    if compile_result.has_errors() {
        error!("Errors found:\n{}", diagnostics);
        process::exit(1);
    }

    if !compile_result.diagnostics.is_empty() {
        warn!("Warnings found:\n{}", diagnostics);
    }

    if let Some(world) = compile_result.world.as_ref() {
        info!("Successfully compiled world");
        info!("  - Signals: {}", world.signals().len());
        info!("  - Fields: {}", world.fields().len());
        info!("  - Operators: {}", world.operators().len());
        info!("  - Entities: {}", world.entities().len());
    } else {
        info!("No errors found.");
    }
}
