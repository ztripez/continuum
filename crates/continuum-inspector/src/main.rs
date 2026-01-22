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
use continuum_tools::world_api::framing::{read_message, write_message};
use continuum_tools::world_api::{WorldMessage, WorldRequest};
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::{TcpListener, UnixStream};
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

#[derive(Clone)]
struct AppState {
    socket: PathBuf,
    // simulation is now managed via the IPC socket
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
