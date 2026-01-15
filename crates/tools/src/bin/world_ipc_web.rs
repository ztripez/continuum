//! Web proxy for Continuum IPC sockets.
//!
//! Serves a small frontend and forwards WebSocket frames to a Unix socket.
//! Translates JSON frames (WebSocket) to Bincode (Unix Socket).

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use clap::Parser;
use continuum_tools::ipc_protocol::{
    IpcFrame, JsonRequest, ipc_event_to_json, ipc_response_to_json, json_request_to_ipc,
    read_frame, write_frame,
};
use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, UnixStream};
use tower_http::services::ServeDir;
use tracing::{debug, error, info, warn};

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
    info!("New WebSocket connection");

    // Retry connection to Unix socket for up to 10 seconds
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
                    let _ = websocket.close().await;
                    return;
                }
                debug!("Waiting for Unix socket (attempt {})...", i + 1);
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }
    }

    let stream = stream.unwrap();
    let (mut ws_sender, mut ws_receiver) = websocket.split();
    let (reader, writer) = stream.into_split();
    let mut socket_reader = BufReader::new(reader);
    let mut socket_writer = BufWriter::new(writer);

    let (ws_tx, mut ws_rx) = tokio::sync::mpsc::channel::<Message>(1000);

    let ws_send_loop = async {
        while let Some(msg) = ws_rx.recv().await {
            if ws_sender.send(msg).await.is_err() {
                debug!("WebSocket sender closed");
                break;
            }
        }
    };

    let ws_tx_clone = ws_tx.clone();
    let ws_to_socket = async move {
        while let Some(result) = ws_receiver.next().await {
            match result {
                Ok(Message::Text(text)) => {
                    debug!("Received WebSocket message: {}", text);
                    // 1. Decode JSON from WebSocket
                    let json_req: serde_json::Result<JsonRequest> = serde_json::from_str(&text);
                    match json_req {
                        Ok(req) => {
                            // 2. Translate JsonRequest to IpcRequest
                            match json_request_to_ipc(req) {
                                Ok(ipc_req) => {
                                    debug!("Translated to IPC request: {:?}", ipc_req);
                                    // 3. Write IpcRequest as Bincode to socket
                                    if let Err(err) =
                                        write_frame(&mut socket_writer, &ipc_req).await
                                    {
                                        warn!("Failed to write to socket: {err}");
                                        break;
                                    }
                                    if let Err(err) = socket_writer.flush().await {
                                        warn!("Failed to flush socket: {err}");
                                        break;
                                    }
                                }
                                Err(err) => {
                                    warn!("Failed to translate request: {err}");
                                }
                            }
                        }
                        Err(err) => {
                            warn!("Invalid JSON request: {err}");
                            let _ = ws_tx_clone
                                .send(Message::Text(
                                    serde_json::json!({
                                        "id": 0,
                                        "error": format!("invalid json: {}", err)
                                    })
                                    .to_string(),
                                ))
                                .await;
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket client closed connection");
                    break;
                }
                Ok(_) => {} // Ignore binary frames from client
                Err(err) => {
                    warn!("WebSocket receive error: {err}");
                    break;
                }
            }
        }
        // Connection closed
    };

    let socket_to_ws = async move {
        loop {
            // 3. Read IpcFrame (Bincode) from socket
            match read_frame::<_, IpcFrame>(&mut socket_reader).await {
                Ok(frame) => {
                    debug!("Received IPC frame from socket");
                    // 4. Convert to JSON and send to WebSocket
                    let json_msg = match frame {
                        IpcFrame::Response(resp) => {
                            serde_json::to_string(&ipc_response_to_json(&resp))
                        }
                        IpcFrame::Event(event) => serde_json::to_string(&ipc_event_to_json(&event)),
                    };

                    match json_msg {
                        Ok(text) => {
                            if ws_tx.send(Message::Text(text)).await.is_err() {
                                break;
                            }
                        }
                        Err(err) => {
                            warn!("Failed to serialize JSON response: {err}");
                        }
                    }
                }
                Err(err) => {
                    warn!("Socket read error: {err}");
                    break;
                }
            }
        }
    };

    tokio::select! {
        _ = ws_send_loop => {
            debug!("ws_send_loop finished");
        },
        _ = ws_to_socket => {
            debug!("ws_to_socket finished");
        },
        _ = socket_to_ws => {
            debug!("socket_to_ws finished");
        },
    }
    info!("WebSocket connection closed");
}
