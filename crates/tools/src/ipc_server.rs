use crate::request_handlers::{RequestRouter, ServerState, ExecutionState};
use crate::run_world_intent::RunWorldIntent;
use crate::world_api::framing::{read_message, write_message};
use crate::world_api::{WorldEvent, WorldMessage, WorldResponse};
use continuum_runtime::build_runtime;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::UnixListener;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Continuous run loop that executes ticks while state is Running.
async fn run_loop(state: Arc<ServerState>, event_tx: broadcast::Sender<WorldEvent>) {
    info!("Run loop started");
    
    loop {
        // Check execution state
        let exec_state = *state.execution_state.read().expect("rwlock poisoned");
        
        match exec_state {
            ExecutionState::Stopped => {
                info!("Run loop: Stopped state detected, exiting");
                break;
            }
            ExecutionState::Paused | ExecutionState::Error => {
                // Sleep and check again
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
            ExecutionState::Running => {}
        }
        
        // Execute tick
        let result = {
            let mut rt = state.runtime.lock().expect("runtime mutex poisoned");
            rt.execute_tick()
        };
        
        match result {
            Ok(tick_ctx) => {
                // Broadcast tick event
                let _ = event_tx.send(WorldEvent {
                    kind: "tick".to_string(),
                    payload: serde_json::json!({
                        "tick": tick_ctx.tick,
                        "era": tick_ctx.era.to_string(),
                        "sim_time": tick_ctx.sim_time,
                        "phase": "complete",
                    }),
                });
            }
            Err(e) => {
                // PAUSE ON ERROR
                error!("Tick execution error: {}", e);
                {
                    let mut exec_state_mut = state.execution_state.write().expect("rwlock poisoned");
                    *exec_state_mut = ExecutionState::Error;
                }
                {
                    let mut last_error_mut = state.last_error.write().expect("rwlock poisoned");
                    *last_error_mut = Some(e.to_string());
                }
                
                // Broadcast error event
                let _ = event_tx.send(WorldEvent {
                    kind: "error".to_string(),
                    payload: serde_json::json!({ "message": e.to_string() }),
                });
            }
        }
        
        // Throttle based on tick_rate
        let tick_rate = state.tick_rate.load(std::sync::atomic::Ordering::Relaxed);
        if tick_rate > 0 {
            let delay = Duration::from_secs_f64(1.0 / tick_rate as f64);
            tokio::time::sleep(delay).await;
        } else {
            // Yield to allow other tasks
            tokio::task::yield_now().await;
        }
    }
    
    info!("Run loop exited");
}

/// A simulation controller that executes a world intent and provides IPC access.
///
/// Note: This is a *controller*, not an observer. It can mutate simulation state
/// via step commands. See AGENTS.md for observer boundary principles.
pub struct SimulationController {
    state: Arc<ServerState>,
    router: Arc<RequestRouter>,
    event_tx: broadcast::Sender<WorldEvent>,
    run_handle: Arc<tokio::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl SimulationController {
    /// Create a new simulation controller for the given intent.
    pub fn new(intent: RunWorldIntent) -> Result<Self, crate::run_world_intent::RunWorldError> {
        let compiled = intent.load()?;
        let runtime = build_runtime(compiled.clone(), None);

        let (event_tx, _) = broadcast::channel(1000); // Buffer up to 1000 events

        Ok(Self {
            state: Arc::new(ServerState {
                compiled,
                runtime: std::sync::Mutex::new(runtime),
                execution_state: std::sync::RwLock::new(crate::request_handlers::ExecutionState::Stopped),
                tick_rate: std::sync::atomic::AtomicU32::new(60), // Default 60 ticks/sec
                last_error: std::sync::RwLock::new(None),
            }),
            router: Arc::new(RequestRouter::new()),
            event_tx,
            run_handle: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    /// Run the server, listening on the given Unix socket path.
    pub async fn run(self, socket_path: &Path) -> Result<(), std::io::Error> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!("Simulation controller listening on {}", socket_path.display());

        // Spawn run loop task
        {
            let state = self.state.clone();
            let event_tx = self.event_tx.clone();
            let handle = tokio::spawn(run_loop(state, event_tx));
            let mut run_handle_guard = self.run_handle.lock().await;
            *run_handle_guard = Some(handle);
        }

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
                    let response = router.handle(req, &state);
                    let _ = write_tx_for_requests.send(WorldMessage::Response(response)).await;
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
                    let _ = write_tx_for_requests.send(WorldMessage::Response(response)).await;
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
                    let _ = write_tx_for_requests.send(WorldMessage::Response(response)).await;
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
        result = request_task => result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
        _ = event_forward_task => Ok(()),
        result = write_task => result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
    }
}

