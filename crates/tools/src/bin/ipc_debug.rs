#!/usr/bin/env -S cargo +nightly -Zscript
//! Combined IPC server + web proxy for easy debugging
//!
//! This tool starts both the IPC server and web proxy in a single process,
//! making it easy to debug and inspect worlds through the browser.
//!
//! Usage:
//!   cargo run --bin ipc-debug -- <world-path>
//!   cargo run --bin ipc-debug -- examples/terra
//!
//! Then open: http://localhost:8080

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::signal;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "ipc-debug", about = "Start IPC server + web UI for debugging")]
struct Args {
    /// Path to the world directory
    world: PathBuf,

    /// Unix socket path for IPC communication
    #[arg(long, default_value = "/tmp/continuum-debug.sock")]
    socket: PathBuf,

    /// Web server bind address
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    info!("Starting Continuum IPC Debug Server");
    info!("World: {}", args.world.display());
    info!("Socket: {}", args.socket.display());
    info!("Web UI: http://{}", args.bind);
    info!("");
    info!("Press Ctrl+C to stop");
    info!("");

    // Clean up old socket if it exists
    if args.socket.exists() {
        std::fs::remove_file(&args.socket)?;
    }

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // Handle Ctrl+C
    tokio::spawn(async move {
        signal::ctrl_c().await.ok();
        info!("Shutting down...");
        r.store(false, Ordering::SeqCst);
    });

    // Try to find the binary in the same directory as this executable
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().unwrap();

    let world_ipc_bin = exe_dir.join("world-ipc");
    let inspector_bin = exe_dir.join("continuum_inspector");

    // Check if binaries exist
    if !world_ipc_bin.exists() {
        anyhow::bail!(
            "world-ipc binary not found at {}\nPlease build it first: cargo build --bin world-ipc",
            world_ipc_bin.display()
        );
    }

    if !inspector_bin.exists() {
        anyhow::bail!(
            "continuum_inspector binary not found at {}\nPlease build it first: cargo build --bin continuum_inspector",
            inspector_bin.display()
        );
    }

    // Start IPC server
    info!("Starting IPC server...");
    let mut ipc_server = Command::new(&world_ipc_bin)
        .arg("--socket")
        .arg(&args.socket)
        .arg(&args.world)
        .spawn()?;

    // Wait a moment for IPC server to start
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Check if IPC server is still running
    match ipc_server.try_wait()? {
        Some(status) => {
            anyhow::bail!("IPC server exited immediately with status: {}", status);
        }
        None => {
            info!("IPC server started successfully");
        }
    }

    // Start web inspector
    info!("Starting web inspector...");
    let mut web_server = Command::new(&inspector_bin)
        .arg("--socket")
        .arg(&args.socket)
        .arg("--bind")
        .arg(&args.bind)
        .spawn()?;

    // Wait a moment for web server to start
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    info!("");
    info!("✓ IPC Debug Server Ready");
    info!("✓ Continuum Inspector: http://{}", args.bind);
    info!("");

    // Monitor both processes
    while running.load(Ordering::SeqCst) {
        // Check if either process has exited
        if let Some(status) = ipc_server.try_wait()? {
            warn!("IPC server exited with status: {}", status);
            break;
        }

        if let Some(status) = web_server.try_wait()? {
            warn!("Web server exited with status: {}", status);
            break;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    // Clean shutdown
    info!("Stopping IPC server...");
    ipc_server.kill().ok();
    ipc_server.wait().ok();

    info!("Stopping web inspector...");
    web_server.kill().ok();
    web_server.wait().ok();

    // Clean up socket
    if args.socket.exists() {
        std::fs::remove_file(&args.socket).ok();
    }

    info!("Shutdown complete");

    Ok(())
}
