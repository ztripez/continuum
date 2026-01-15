//! DSL Lint Tool
//!
//! Parses a single .cdsl file or a directory of .cdsl files and reports detailed diagnostics.
//!
//! Usage: dsl-lint <FILE_OR_DIR>

use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "dsl-lint")]
struct Cli {
    path: PathBuf,
}

fn main() {
    continuum_tools::init_logging();

    let cli = Cli::parse();
    let target_path = cli.path;

    if !target_path.exists() {
        error!("Path '{}' does not exist", target_path.display());
        process::exit(1);
    }

    let compile_result = match compile_target(&target_path) {
        Ok(result) => result,
        Err(message) => {
            error!("{}", message);
            process::exit(1);
        }
    };

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

fn compile_target(path: &Path) -> Result<continuum_compiler::CompileResult, String> {
    if path.is_dir() {
        return Ok(continuum_compiler::compile_from_dir_result(path));
    }

    let source = fs::read_to_string(path)
        .map_err(|error| format!("Error reading file '{}': {}", path.display(), error))?;
    let mut source_map = HashMap::new();
    source_map.insert(path.to_path_buf(), source.as_str());

    Ok(continuum_compiler::compile(&source_map))
}
