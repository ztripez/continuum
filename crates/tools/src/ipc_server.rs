//! Unix-socket IPC server for the Continuum inspector.
//!
//! Accepts connections over a Unix domain socket, dispatches JSON-framed
//! requests to [`RequestRouter`], and broadcasts world events to connected
//! clients.
//!
//! The simulation runtime lives on a dedicated thread ([`SimThread`]),
//! communicated with via channel-based [`SimProxy`]. No locks are needed
//! for runtime access — all queries and mutations go through message passing.

use crate::request_handlers::{RequestRouter, ServerState};
use crate::run_world_intent::RunWorldIntent;
use crate::sim_thread::SimThread;
use crate::world_api::framing::{read_message, write_message};
use crate::world_api::{WorldEvent, WorldMessage, WorldResponse};
use continuum_runtime::build_runtime;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::UnixListener;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

/// A simulation controller that executes a world intent and provides IPC access.
///
/// Note: This is a *controller*, not an observer. It can mutate simulation state
/// via step commands. See AGENTS.md for observer boundary principles.
pub struct SimulationController {
    state: Arc<ServerState>,
    router: Arc<RequestRouter>,
    event_tx: broadcast::Sender<WorldEvent>,
    /// Handle to the simulation thread — dropped on controller shutdown.
    _sim_thread: SimThread,
}

impl SimulationController {
    /// Create a new simulation controller for the given intent.
    ///
    /// Compiles the world, builds the runtime, spawns the simulation thread,
    /// and returns a controller ready to accept IPC connections.
    pub fn new(intent: RunWorldIntent) -> Result<Self, crate::run_world_intent::RunWorldError> {
        let compiled = intent.load()?;
        let runtime = build_runtime(compiled.clone(), None);

        // Spawn simulation thread — it takes exclusive ownership of Runtime
        let sim_thread = SimThread::spawn(runtime);
        let sim = sim_thread.proxy().clone();

        let (event_tx, _) = broadcast::channel(1000);

        Ok(Self {
            state: Arc::new(ServerState {
                compiled,
                sim,
                execution_state: parking_lot::RwLock::new(
                    crate::request_handlers::ExecutionState::Stopped,
                ),
                tick_rate: std::sync::atomic::AtomicU32::new(60),
                last_error: parking_lot::RwLock::new(None),
            }),
            router: Arc::new(RequestRouter::new()),
            event_tx,
            _sim_thread: sim_thread,
        })
    }

    /// Run the server, listening on the given Unix socket path.
    pub async fn run(self, socket_path: &Path) -> Result<(), std::io::Error> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!(
            "Simulation controller listening on {}",
            socket_path.display()
        );

        let mut error_count = 0;
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    error_count = 0;
                    info!("Client connected to Simulation IPC");
                    let state = self.state.clone();
                    let router = self.router.clone();
                    let event_tx = self.event_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, state, router, event_tx).await {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept IPC connection: {}", e);
                    error_count += 1;
                    if error_count > 10 {
                        error!("Too many consecutive IO errors, shutting down simulation controller");
                        return Err(e);
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
        }
    }
}

async fn handle_connection<S>(
    stream: S,
    state: Arc<ServerState>,
    router: Arc<RequestRouter>,
    event_tx: broadcast::Sender<WorldEvent>,
) -> Result<(), std::io::Error>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (read_half, write_half) = tokio::io::split(stream);

    // Channel for sending messages to write task
    let (write_tx, mut write_rx) = tokio::sync::mpsc::channel::<WorldMessage>(100);

    // Subscribe to broadcast events
    let mut event_rx = event_tx.subscribe();

    // Task 1: Read requests and send responses via channel
    let write_tx_for_requests = write_tx.clone();
    let request_task = tokio::spawn(async move {
        let mut read_half = read_half;
        loop {
            let msg = match read_message(&mut read_half).await {
                Ok(msg) => msg,
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::UnexpectedEof {
                        info!("Client disconnected (read)");
                        return Ok::<(), std::io::Error>(());
                    }
                    return Err(e);
                }
            };

            match msg {
                WorldMessage::Request(req) => {
                    debug!("Received request: {} ({})", req.kind, req.id);
                    // Offload request handling to blocking thread because
                    // SimProxy methods block on crossbeam channel replies.
                    let router_clone = router.clone();
                    let state_clone = state.clone();
                    let response = tokio::task::spawn_blocking(move || {
                        router_clone.handle(req, &state_clone)
                    })
                    .await
                    .unwrap_or_else(|join_err| {
                        error!("Request handler panicked: {}", join_err);
                        WorldResponse {
                            id: 0,
                            ok: false,
                            payload: None,
                            error: Some(format!("handler panicked: {}", join_err)),
                        }
                    });
                    let _ = write_tx_for_requests
                        .send(WorldMessage::Response(response))
                        .await;
                }
                WorldMessage::Response(_) => {
                    let err = "Received response from client (unexpected message type)";
                    warn!("{}", err);
                    let response = WorldResponse {
                        id: 0,
                        ok: false,
                        payload: None,
                        error: Some(err.to_string()),
                    };
                    let _ = write_tx_for_requests
                        .send(WorldMessage::Response(response))
                        .await;
                }
                WorldMessage::Event(_) => {
                    let err = "Received event from client (unexpected message type)";
                    warn!("{}", err);
                    let response = WorldResponse {
                        id: 0,
                        ok: false,
                        payload: None,
                        error: Some(err.to_string()),
                    };
                    let _ = write_tx_for_requests
                        .send(WorldMessage::Response(response))
                        .await;
                }
            }
        }
    });

    // Task 2: Forward broadcast events to write channel
    let event_forward_task = tokio::spawn(async move {
        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    if write_tx.send(WorldMessage::Event(event)).await.is_err() {
                        info!("Write channel closed, stopping event forwarding");
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("Event channel closed");
                    return;
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("Client lagged behind by {} events", n);
                }
            }
        }
    });

    // Task 3: Write messages from channel to stream
    let write_task = tokio::spawn(async move {
        let mut write_half = write_half;
        while let Some(msg) = write_rx.recv().await {
            if let Err(e) = write_message(&mut write_half, &msg).await {
                error!("Failed to write message to client: {}", e);
                return Err(e);
            }
        }
        Ok::<(), std::io::Error>(())
    });

    // Wait for any task to complete
    tokio::select! {
        result = request_task => result.map_err(std::io::Error::other)?,
        _ = event_forward_task => Ok(()),
        result = write_task => result.map_err(std::io::Error::other)?,
    }
}
