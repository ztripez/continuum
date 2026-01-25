use crate::request_handlers::{RequestRouter, ServerState};
use crate::run_world_intent::RunWorldIntent;
use crate::world_api::framing::{read_message, write_message};
use crate::world_api::{WorldMessage, WorldResponse};
use continuum_runtime::build_runtime;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::UnixListener;
use tracing::{debug, error, info, warn};

/// A simulation controller that executes a world intent and provides IPC access.
///
/// Note: This is a *controller*, not an observer. It can mutate simulation state
/// via step commands. See AGENTS.md for observer boundary principles.
pub struct SimulationController {
    state: Arc<ServerState>,
    router: Arc<RequestRouter>,
}

impl SimulationController {
    /// Create a new simulation controller for the given intent.
    pub fn new(intent: RunWorldIntent) -> Result<Self, crate::run_world_intent::RunWorldError> {
        let compiled = intent.load()?;
        let runtime = build_runtime(compiled.clone(), None);

        Ok(Self {
            state: Arc::new(ServerState {
                compiled,
                runtime: std::sync::Mutex::new(runtime),
            }),
            router: Arc::new(RequestRouter::new()),
        })
    }

    /// Run the server, listening on the given Unix socket path.
    pub async fn run(self, socket_path: &Path) -> Result<(), std::io::Error> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!("Simulation controller listening on {}", socket_path.display());

        let mut error_count = 0;
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    error_count = 0;
                    info!("Client connected to Simulation IPC");
                    let state = self.state.clone();
                    let router = self.router.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, state, router).await {
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
    mut stream: S,
    state: Arc<ServerState>,
    router: Arc<RequestRouter>,
) -> Result<(), std::io::Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        let msg = match read_message(&mut stream).await {
            Ok(msg) => msg,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    info!("Client disconnected");
                    return Ok(());
                }
                return Err(e);
            }
        };

        match msg {
            WorldMessage::Request(req) => {
                debug!("Received request: {} ({})", req.kind, req.id);
                let response = router.handle(req, &state);
                write_message(&mut stream, &WorldMessage::Response(response)).await?;
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
                write_message(&mut stream, &WorldMessage::Response(response)).await?;
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
                write_message(&mut stream, &WorldMessage::Response(response)).await?;
            }
        }
    }
}

