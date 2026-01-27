use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json,
};
use clap::Parser;
use continuum_tools::world_api::framing::{read_message, write_message};
use continuum_tools::world_api::{WorldMessage, WorldRequest};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

struct AppState {
    socket: PathBuf,
    child: Arc<Mutex<Option<Child>>>,
    current_world: Arc<Mutex<Option<PathBuf>>>,
}

impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            socket: self.socket.clone(),
            child: Arc::clone(&self.child),
            current_world: Arc::clone(&self.current_world),
        }
    }
}

#[derive(Debug, Deserialize)]
struct LoadSimulationRequest {
    world_path: PathBuf,
    scenario: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiResponse {
    success: bool,
    message: String,
}

#[derive(Debug, Serialize)]
struct SimulationStatusResponse {
    running: bool,
    world_path: Option<PathBuf>,
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
        if let Err(err) = spawn_simulation(&state, world_path, None).await {
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

async fn ws_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| proxy_socket(socket, state))
}

async fn proxy_socket(mut websocket: WebSocket, state: AppState) {
    info!("New WebSocket connection");

    let mut stream = None;
    for i in 0..10 {
        match UnixStream::connect(&state.socket).await {
            Ok(s) => {
                info!("Connected to Unix socket {}", state.socket.display());
                stream = Some(s);
                break;
            }
            Err(err) => {
                if i == 9 {
                    warn!(
                        "Failed to connect to socket {} after 10 attempts: {err}",
                        state.socket.display()
                    );
                    let _ = websocket
                        .send(Message::Text(
                            serde_json::json!({
                                "id": 0,
                                "error": format!("socket connect failed: {}", err)
                            })
                            .to_string(),
                        ))
                        .await;
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }

    let stream = stream.expect("Unix socket connection failed");
    let (read_half, write_half) = stream.into_split();
    let (mut ws_sender, mut ws_receiver) = websocket.split();

    let socket_to_ws = async move {
        let mut read_half = read_half;
        loop {
            match read_message(&mut read_half).await {
                Ok(msg) => {
                    let json_text = match msg {
                        WorldMessage::Response(resp) => serde_json::to_string(&resp),
                        WorldMessage::Event(event) => serde_json::to_string(&event),
                        WorldMessage::Request(_) => {
                            warn!("Received request from simulation server, ignoring");
                            continue;
                        }
                    };

                    match json_text {
                        Ok(text) => {
                            debug!("IPC -> WS: {}", text);
                            if ws_sender.send(Message::Text(text)).await.is_err() {
                                info!("WebSocket closed (send failed)");
                                break;
                            }
                        }
                        Err(err) => {
                            warn!("Failed to serialize JSON: {err}");
                        }
                    }
                }
                Err(err) => {
                    error!("Failed to read from Unix socket: {err}");
                    break;
                }
            }
        }
    };

    let ws_to_socket = async move {
        let mut write_half = write_half;
        while let Some(msg) = ws_receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("WS -> IPC: {}", text);

                    let req: WorldRequest = match serde_json::from_str(&text) {
                        Ok(req) => req,
                        Err(err) => {
                            warn!("Failed to parse JSON request: {err}");
                            continue;
                        }
                    };

                    if let Err(err) =
                        write_message(&mut write_half, &WorldMessage::Request(req)).await
                    {
                        error!("Failed to write to Unix socket: {err}");
                        break;
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket closed by client");
                    break;
                }
                Ok(_) => {}
                Err(err) => {
                    error!("WebSocket error: {err}");
                    break;
                }
            }
        }
    };

    tokio::select! {
        _ = socket_to_ws => {},
        _ = ws_to_socket => {},
    }

    info!("WebSocket connection closed");
}

// === Process Management Handlers ===

async fn load_simulation_handler(
    State(state): State<AppState>,
    Json(payload): Json<LoadSimulationRequest>,
) -> impl IntoResponse {
    info!("Load simulation request: {:?}", payload.world_path);

    // Kill existing simulation if running
    if let Err(err) = kill_simulation(&state).await {
        error!("Failed to kill existing simulation: {err}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Failed to stop existing simulation: {err}"),
            }),
        );
    }

    // Wait for socket to be removed
    for _ in 0..10 {
        if !state.socket.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Spawn new simulation
    match spawn_simulation(&state, payload.world_path.clone(), payload.scenario).await {
        Ok(()) => {
            *state.current_world.lock().await = Some(payload.world_path.clone());
            (
                StatusCode::OK,
                Json(ApiResponse {
                    success: true,
                    message: format!("Loaded world: {}", payload.world_path.display()),
                }),
            )
        }
        Err(err) => {
            error!("Failed to spawn simulation: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("Failed to start simulation: {err}"),
                }),
            )
        }
    }
}

async fn stop_simulation_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("Stop simulation request");

    match kill_simulation(&state).await {
        Ok(()) => {
            *state.current_world.lock().await = None;
            (
                StatusCode::OK,
                Json(ApiResponse {
                    success: true,
                    message: "Simulation stopped".to_string(),
                }),
            )
        }
        Err(err) => {
            error!("Failed to stop simulation: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("Failed to stop simulation: {err}"),
                }),
            )
        }
    }
}

async fn restart_simulation_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("Restart simulation request");

    let world_path = match state.current_world.lock().await.clone() {
        Some(path) => path,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse {
                    success: false,
                    message: "No world currently loaded".to_string(),
                }),
            )
        }
    };

    // Kill existing
    if let Err(err) = kill_simulation(&state).await {
        error!("Failed to kill simulation during restart: {err}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Failed to stop simulation: {err}"),
            }),
        );
    }

    // Wait for socket cleanup
    for _ in 0..10 {
        if !state.socket.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Respawn
    match spawn_simulation(&state, world_path, None).await {
        Ok(()) => (
            StatusCode::OK,
            Json(ApiResponse {
                success: true,
                message: "Simulation restarted".to_string(),
            }),
        ),
        Err(err) => {
            error!("Failed to restart simulation: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("Failed to restart simulation: {err}"),
                }),
            )
        }
    }
}

async fn simulation_status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let child_guard = state.child.lock().await;
    let running = child_guard.is_some();
    let world_path = state.current_world.lock().await.clone();

    (
        StatusCode::OK,
        Json(SimulationStatusResponse {
            running,
            world_path,
        }),
    )
}

// === Helper Functions ===

async fn spawn_simulation(
    state: &AppState,
    world_path: PathBuf,
    scenario: Option<String>,
) -> Result<(), String> {
    info!("Spawning continuum-run for world: {}", world_path.display());

    // Remove old socket file if it exists
    if state.socket.exists() {
        std::fs::remove_file(&state.socket)
            .map_err(|e| format!("Failed to remove old socket: {e}"))?;
    }

    // Build command
    let mut cmd = Command::new("continuum-run");
    cmd.arg(&world_path).arg("--socket").arg(&state.socket);

    if let Some(scenario_name) = scenario {
        cmd.arg("--scenario").arg(scenario_name);
    }

    // Spawn process
    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn continuum-run: {e}"))?;

    *state.child.lock().await = Some(child);

    // Wait for socket to exist (5 second timeout)
    for i in 0..50 {
        if state.socket.exists() {
            info!("Socket ready: {}", state.socket.display());
            return Ok(());
        }
        if i == 49 {
            // Kill the process since it failed to create socket
            if let Err(e) = kill_simulation(state).await {
                error!("Failed to kill unresponsive process: {e}");
            }
            return Err(format!(
                "Timeout waiting for continuum-run to create socket at {}. \
                 Process may have failed to start - check process logs.",
                state.socket.display()
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    unreachable!("Socket wait loop exited without return")
}

async fn kill_simulation(state: &AppState) -> Result<(), String> {
    let mut child_guard = state.child.lock().await;

    if let Some(mut child) = child_guard.take() {
        info!("Killing continuum-run process");

        // Try graceful shutdown first (SIGTERM)
        child
            .kill()
            .await
            .map_err(|e| format!("Failed to kill process: {e}"))?;

        // Wait for process to exit (with timeout)
        match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
            Ok(Ok(status)) => {
                info!("Process exited with status: {status}");
                if !status.success() {
                    warn!("Process exited with non-zero status: {status}");
                }
            }
            Ok(Err(err)) => {
                return Err(format!("Error waiting for process to exit: {err}"));
            }
            Err(_) => {
                error!("Process did not exit within 5 seconds after SIGTERM - may be zombie");
                return Err("Timeout waiting for process to exit - process may be stuck".to_string());
            }
        }

        // Remove socket file
        if state.socket.exists() {
            std::fs::remove_file(&state.socket)
                .map_err(|e| format!("Failed to remove socket: {e}"))?;
        }

        Ok(())
    } else {
        Ok(()) // No process to kill
    }
}
