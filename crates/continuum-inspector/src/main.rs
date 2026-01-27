//! Continuum Inspector - Web-based debugger and visualization tool.
//!
//! Provides a development server that spawns and manages `continuum-run` processes,
//! proxies IPC communication via WebSocket, and serves a browser-based UI for
//! inspecting simulation state, fields, and execution metrics.

mod handlers;
mod helpers;
mod process;
mod spawner;
mod state;
mod websocket;

use axum::{routing::{get, post}, Router};
use clap::Parser;
use handlers::{
    load_simulation_handler, restart_simulation_handler, simulation_status_handler,
    stop_simulation_handler,
};
use process::{kill_simulation, spawn_simulation};
use spawner::RealProcessSpawner;
use state::AppState;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use websocket::ws_handler;

#[derive(Parser, Debug)]
#[command(name = "continuum-inspector")]
#[command(about = "Web inspector for Continuum worlds")]
struct Cli {
    /// Path to a world directory or .cvm bundle
    world: Option<PathBuf>,

    /// TCP address to bind the web server
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind: SocketAddr,

    /// Path to Unix socket (for simulation IPC)
    #[arg(long, default_value = "/tmp/continuum-inspector.sock")]
    socket: PathBuf,

    /// Directory containing frontend assets
    #[arg(long)]
    static_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "continuum_inspector=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    // If a world path is provided, we expect an external IPC server or we could launch one in-process
    // For this refactor, we focus on the transition to WorldApi.
    // The inspector acts as a bridge between WebSocket (JSON) and Unix Socket (WorldApi Framing).

    let state = AppState {
        socket: cli.socket.clone(),
        child: Arc::new(Mutex::new(None)),
        current_world: Arc::new(Mutex::new(cli.world.clone())),
    };

    // If a world was provided at startup, launch it
    if let Some(world_path) = cli.world {
        info!("Auto-launching world: {}", world_path.display());
        let spawner = RealProcessSpawner;
        if let Err(err) = spawn_simulation(&spawner, &state, world_path, None).await {
            error!("Failed to auto-launch world: {err}");
        }
    }

    let static_dir = cli
        .static_dir
        .unwrap_or_else(|| PathBuf::from("crates/continuum-inspector/static"));

    info!("Serving static files from: {}", static_dir.display());

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/sim/load", post(load_simulation_handler))
        .route("/api/sim/stop", post(stop_simulation_handler))
        .route("/api/sim/restart", post(restart_simulation_handler))
        .route("/api/sim/status", get(simulation_status_handler))
        .fallback_service(ServeDir::new(static_dir))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    // Setup graceful shutdown
    let shutdown_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            error!("Failed to register shutdown signal: {e}");
        }
        info!("Received shutdown signal, cleaning up...");
        if let Err(e) = kill_simulation(&shutdown_state).await {
            error!("Failed to kill simulation during shutdown: {e}");
        }
        if let Err(e) = std::fs::remove_file(&shutdown_state.socket) {
            warn!("Failed to remove socket during shutdown: {e}");
        }
        info!("Shutdown complete");
        std::process::exit(0);
    });

    let listener = match TcpListener::bind(cli.bind).await {
        Ok(listener) => listener,
        Err(err) => {
            error!("Failed to bind {}: {err}", cli.bind);
            std::process::exit(1);
        }
    };

    info!("");
    info!("Continuum Inspector: http://{}", cli.bind);
    info!("");

    if let Err(err) = axum::serve(listener, app).await {
        error!("Server error: {err}");
    }
}
