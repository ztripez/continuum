use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use clap::Parser;
use continuum_tools::ipc_protocol::{
    IpcFrame, JsonRequest, ipc_event_to_json, ipc_response_to_json, json_request_to_ipc,
    read_frame, write_frame,
};
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command};
use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, UnixStream};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "continuum-inspector")]
#[command(about = "Web inspector for Continuum worlds")]
struct Cli {
    /// Path to a world directory (launches IPC server automatically)
    world: Option<PathBuf>,

    /// TCP address to bind the web server
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind: SocketAddr,

    /// Path to Unix socket (only needed if connecting to existing IPC server)
    #[arg(long, default_value = "/tmp/continuum-inspector.sock")]
    socket: PathBuf,

    /// Directory containing frontend assets
    #[arg(long)]
    static_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    socket: PathBuf,
}

struct IpcServerGuard {
    child: Child,
    socket: PathBuf,
}

impl Drop for IpcServerGuard {
    fn drop(&mut self) {
        info!("Stopping IPC server...");
        let _ = self.child.kill();
        let _ = self.child.wait();
        if self.socket.exists() {
            let _ = std::fs::remove_file(&self.socket);
        }
    }
}

fn find_world_ipc_binary() -> Option<PathBuf> {
    // Try same directory as current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("world-ipc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // Try cargo target directory
    let candidate = PathBuf::from("target/debug/world-ipc");
    if candidate.exists() {
        return Some(candidate);
    }
    let candidate = PathBuf::from("target/release/world-ipc");
    if candidate.exists() {
        return Some(candidate);
    }
    None
}

fn launch_ipc_server(world: &PathBuf, socket: &PathBuf) -> Result<IpcServerGuard, String> {
    let binary = find_world_ipc_binary().ok_or_else(|| {
        "world-ipc binary not found. Run: cargo build --bin world-ipc".to_string()
    })?;

    // Clean up old socket if exists
    if socket.exists() {
        std::fs::remove_file(socket).ok();
    }

    info!(
        "Launching IPC server: {} --socket {} {}",
        binary.display(),
        socket.display(),
        world.display()
    );

    let child = Command::new(&binary)
        .arg("--socket")
        .arg(socket)
        .arg(world)
        .spawn()
        .map_err(|e| format!("Failed to spawn world-ipc: {e}"))?;

    Ok(IpcServerGuard {
        child,
        socket: socket.clone(),
    })
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

    // Launch IPC server if world path provided
    let _ipc_guard = if let Some(ref world) = cli.world {
        match launch_ipc_server(world, &cli.socket) {
            Ok(guard) => {
                // Wait for socket to be ready
                for i in 0..20 {
                    if cli.socket.exists() {
                        info!("IPC server ready");
                        break;
                    }
                    if i == 19 {
                        error!("IPC server failed to create socket");
                        std::process::exit(1);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
                Some(guard)
            }
            Err(e) => {
                error!("{e}");
                std::process::exit(1);
            }
        }
    } else {
        if !cli.socket.exists() {
            error!("Socket {} does not exist. Either:", cli.socket.display());
            error!("  1. Provide a world path: continuum_inspector <world>");
            error!("  2. Start an IPC server manually and specify --socket");
            std::process::exit(1);
        }
        None
    };

    let state = AppState {
        socket: cli.socket.clone(),
    };

    let static_dir = cli
        .static_dir
        .unwrap_or_else(|| PathBuf::from("crates/continuum-inspector/static"));

    info!("Serving static files from: {}", static_dir.display());

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new(static_dir))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

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

    let stream = stream.unwrap();
    let (read_half, write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    let (mut ws_sender, mut ws_receiver) = websocket.split();

    let socket_to_ws = async {
        loop {
            match read_frame(&mut reader).await {
                Ok(frame) => {
                    let json_text = match frame {
                        IpcFrame::Response(resp) => {
                            serde_json::to_string(&ipc_response_to_json(&resp))
                        }
                        IpcFrame::Event(event) => serde_json::to_string(&ipc_event_to_json(&event)),
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

    let ws_to_socket = async {
        while let Some(msg) = ws_receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("WS -> IPC: {}", text);

                    let json_req: JsonRequest = match serde_json::from_str(&text) {
                        Ok(req) => req,
                        Err(err) => {
                            warn!("Failed to parse JSON request: {err}");
                            continue;
                        }
                    };

                    let ipc_req = match json_request_to_ipc(json_req) {
                        Ok(req) => req,
                        Err(err) => {
                            warn!("Failed to convert JSON to IPC: {err}");
                            continue;
                        }
                    };

                    if let Err(err) = write_frame(&mut writer, &ipc_req).await {
                        error!("Failed to write to Unix socket: {err}");
                        break;
                    }

                    if let Err(err) = writer.flush().await {
                        error!("Failed to flush Unix socket: {err}");
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
