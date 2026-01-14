//! Web proxy for Continuum IPC sockets.
//!
//! Serves a small frontend and forwards WebSocket frames to a Unix socket.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, UnixStream};
use tower_http::services::ServeDir;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(name = "world-ipc-web")]
struct Cli {
    /// TCP address to bind the web server.
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind: SocketAddr,

    /// Path to the Unix socket exposed by the IPC server.
    #[arg(long)]
    socket: PathBuf,

    /// Directory containing frontend assets.
    #[arg(long, default_value = "crates/tools/assets/ipc-web")]
    static_dir: PathBuf,
}

#[derive(Clone)]
struct AppState {
    socket: PathBuf,
}

#[tokio::main]
async fn main() {
    continuum_tools::init_logging();

    let cli = Cli::parse();

    let state = AppState { socket: cli.socket };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .nest_service("/", ServeDir::new(cli.static_dir))
        .with_state(state);

    let listener = match TcpListener::bind(cli.bind).await {
        Ok(listener) => listener,
        Err(err) => {
            error!("Failed to bind {}: {err}", cli.bind);
            std::process::exit(1);
        }
    };

    info!("Listening on http://{}", cli.bind);

    if let Err(err) = axum::serve(listener, app).await {
        error!("Server error: {err}");
    }
}

async fn ws_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| proxy_socket(socket, state))
}

async fn proxy_socket(mut websocket: WebSocket, state: AppState) {
    let stream = match UnixStream::connect(&state.socket).await {
        Ok(stream) => stream,
        Err(err) => {
            warn!(
                "Failed to connect to socket {}: {err}",
                state.socket.display()
            );
            let _ = websocket
                .send(Message::Text(format!("socket connect failed: {}", err)))
                .await;
            let _ = websocket.close().await;
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = websocket.split();
    let (mut socket_reader, mut socket_writer) = stream.into_split();

    let ws_to_socket = async {
        while let Some(result) = ws_receiver.next().await {
            match result {
                Ok(Message::Text(text)) => {
                    if socket_writer.write_all(text.as_bytes()).await.is_err() {
                        break;
                    }
                }
                Ok(Message::Binary(data)) => {
                    if socket_writer.write_all(&data).await.is_err() {
                        break;
                    }
                }
                Ok(Message::Close(_)) => {
                    break;
                }
                Ok(_) => {}
                Err(err) => {
                    warn!("WebSocket receive error: {err}");
                    break;
                }
            }
        }

        let _ = socket_writer.shutdown().await;
    };

    let socket_to_ws = async {
        let mut buffer = vec![0u8; 4096];
        loop {
            match socket_reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(count) => {
                    if ws_sender
                        .send(Message::Binary(buffer[..count].to_vec()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(err) => {
                    warn!("Socket read error: {err}");
                    let _ = ws_sender
                        .send(Message::Text(format!("socket read failed: {err}")))
                        .await;
                    break;
                }
            }
        }
    };

    tokio::select! {
        _ = ws_to_socket => {},
        _ = socket_to_ws => {},
    }
}
