//! WebSocket proxy for IPC communication between browser and `continuum-run`.

use crate::state::AppState;
use axum::extract::{
    ws::{Message, WebSocket, WebSocketUpgrade},
    State,
};
use axum::response::IntoResponse;
use continuum_tools::world_api::framing::{read_message, write_message};
use continuum_tools::world_api::{WorldMessage, WorldRequest};
use futures::{SinkExt, StreamExt};
use tokio::net::UnixStream;
use tracing::{debug, error, info, warn};

/// Axum handler for WebSocket upgrade at `/ws` endpoint.
///
/// Upgrades the HTTP connection to WebSocket and spawns `proxy_socket` task
/// to bidirectionally proxy messages between browser and Unix socket IPC.
pub async fn ws_handler(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| proxy_socket(socket, state))
}

/// Proxies messages bidirectionally between WebSocket (browser) and Unix socket (continuum-run).
///
/// # Parameters
/// - `websocket`: Upgraded WebSocket connection from browser client
/// - `state`: Shared application state containing Unix socket path
///
/// # Process
/// 1. Attempts to connect to Unix socket (10 retries with 1s backoff)
/// 2. Spawns two concurrent tasks:
///    - **socket_to_ws**: Reads IPC messages (Response/Event) and forwards as WebSocket text
///    - **ws_to_socket**: Reads WebSocket text, parses as Request, writes to IPC
/// 3. Runs both tasks until either closes (socket error, WebSocket close, etc.)
///
/// # Error Handling
/// - Socket connection failure after 10 retries: sends error to browser, closes WebSocket
/// - IPC read/write errors: logs error, closes both directions
/// - WebSocket errors or client close: logs, closes IPC connection
/// - JSON parse errors: logs warning, continues processing other messages
///
/// # Notes
/// - Only forwards Response/Event from IPC to browser (ignores Request from simulation)
/// - WebSocket text messages must be valid JSON `WorldRequest` objects
/// - Non-text WebSocket messages (binary, ping, pong) are ignored
/// - Uses `tokio::select!` to race both directions; first to finish terminates both
pub async fn proxy_socket(mut websocket: WebSocket, state: AppState) {
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
                    if let Err(send_err) = websocket
                        .send(Message::Text(
                            serde_json::json!({
                                "id": 0,
                                "error": format!("socket connect failed: {}", err)
                            })
                            .to_string(),
                        ))
                        .await
                    {
                        warn!("Failed to send socket connection error to browser: {send_err}");
                    }
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
