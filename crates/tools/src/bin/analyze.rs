//! Analysis Tool
//!
//! Analyze simulation snapshots.

use clap::{Parser, Subcommand};
use std::process;

use continuum_tools::analyze::commands::baseline::{run as run_baseline, BaselineCommand};

#[derive(Parser, Debug)]
#[command(name = "analyze")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(subcommand)]
    Baseline(BaselineCommand),
}

fn main() {
    continuum_tools::init_logging();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Baseline(cmd) => run_baseline(cmd),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
