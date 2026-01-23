//! Continuum Run - Executes a world with IPC server for inspector
//!
//! This binary loads a world, starts an IPC server on a Unix socket,
//! and keeps it running for the inspector to connect to.

use clap::Parser;
use continuum_tools::ipc_server::SimulationServer;
use continuum_tools::run_world_intent::{RunWorldIntent, WorldSource};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "continuum-run")]
#[command(about = "Run a Continuum world with IPC server for inspector")]
struct Cli {
    /// Path to a world directory or .cvm bundle
    world: PathBuf,

    /// Path to Unix socket for IPC
    #[arg(long, default_value = "/tmp/continuum-inspector.sock")]
    socket: PathBuf,

    /// Number of simulation steps to run (0 = run indefinitely via IPC)
    #[arg(long, default_value = "0")]
    steps: u64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "continuum_run=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    info!("Loading world from: {}", cli.world.display());

    let source = match WorldSource::from_path(cli.world) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to determine world source: {:?}", e);
            std::process::exit(1);
        }
    };

    let intent = RunWorldIntent::new(source, cli.steps);

    info!("Creating simulation server...");
    let server = match SimulationServer::new(intent) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to load world:\n{}", e);
            std::process::exit(1);
        }
    };

    info!("World loaded successfully");
    info!("Starting IPC server on: {}", cli.socket.display());
    info!("");
    info!("Connect with inspector:");
    info!("  cargo run --bin continuum_inspector");
    info!("");

    if let Err(e) = server.run(&cli.socket).await {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}
