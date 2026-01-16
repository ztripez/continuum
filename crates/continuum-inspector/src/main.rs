use axum::{
    Router,
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
};
use std::net::SocketAddr;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "continuum_inspector=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Build our application with routes
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new("static"))
        .layer(TraceLayer::new_for_http());

    // Run server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Continuum Inspector starting on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// WebSocket handler
async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

// Handle WebSocket connection
async fn handle_socket(mut socket: WebSocket) {
    tracing::info!("WebSocket connection established");

    // TODO: Integrate with IPC protocol
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(msg) => {
                tracing::debug!("Received WebSocket message: {:?}", msg);
                // Echo for now
                if socket.send(msg).await.is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::error!("WebSocket error: {}", e);
                break;
            }
        }
    }

    tracing::info!("WebSocket connection closed");
}
