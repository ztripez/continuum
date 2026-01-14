//!
//! Compile a Continuum world into a bytecode bundle.
//!
//! Usage: `compile <world-dir> [--out-dir DIR] [--output FILE]`

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process;
use tracing::{error, info, warn};

use continuum_compiler::ir::{BinaryBundle, CompilationResult, CompiledWorld, compile};

#[derive(Parser, Debug)]
#[command(name = "compile")]
#[command(about = "Compile a Continuum world into a binary bundle (.cvm)")]
struct Args {
    /// Path to the World root directory
    world_dir: PathBuf,

    /// Output directory for the binary bundle
    #[arg(long = "out-dir", default_value = "build")]
    out_dir: PathBuf,

    /// Explicit output file path
    #[arg(long = "output")]
    output: Option<PathBuf>,
}

fn main() {
    continuum_tools::init_logging();

    let args = Args::parse();

    let compile_result = continuum_compiler::compile_from_dir_result(&args.world_dir);
    if compile_result.has_errors() {
        error!("{}", compile_result.format_diagnostics().trim_end());
        process::exit(1);
    }

    if !compile_result.diagnostics.is_empty() {
        warn!("{}", compile_result.format_diagnostics().trim_end());
    }

    let world = compile_result.world.expect("no world despite no errors");
    let world_name = args
        .world_dir
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "world".to_string());

    let output_path = args.output.clone().unwrap_or_else(|| {
        let file_name = format!("{}.cvm", world_name);
        args.out_dir.join(file_name)
    });

    if let Some(parent) = output_path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            error!("Failed to create {}: {}", parent.display(), error);
            process::exit(1);
        }
    }

    info!("Compiling to DAGs...");
    let compilation = match compile(&world) {
        Ok(c) => c,
        Err(e) => {
            error!("Compilation error: {}", e);
            process::exit(1);
        }
    };

    let bundle = BinaryBundle {
        version: 1,
        world_name,
        world,
        compilation,
    };

    let encoded = match bincode::serialize(&bundle) {
        Ok(data) => data,
        Err(error) => {
            error!("Failed to encode binary bundle: {}", error);
            process::exit(1);
        }
    };

    if let Err(error) = fs::write(&output_path, encoded) {
        error!("Failed to write {}: {}", output_path.display(), error);
        process::exit(1);
    }

    info!("Wrote binary bundle to {}", output_path.display());
}
