use std::path::Path;
use tokio::net::UnixListener;
use tokio::io::{AsyncRead, AsyncWrite};
use crate::world_api::{WorldMessage, WorldRequest, WorldResponse};
use crate::world_api::framing::{read_message, write_message};
use crate::run_world_intent::RunWorldIntent;
use continuum_runtime::{Runtime, build_runtime};
use continuum_cdsl::ast::{CompiledWorld, RoleId};
use tracing::{info, error, debug, warn};
use std::sync::{Arc, Mutex};

/// Shared state for the simulation server.
struct ServerState {
    compiled: CompiledWorld,
    runtime: Mutex<Runtime>,
}

/// A simulation server that executes a world intent and provides IPC access.
pub struct SimulationServer {
    state: Arc<ServerState>,
}

impl SimulationServer {
    /// Create a new simulation server for the given intent.
    pub fn new(intent: RunWorldIntent) -> Result<Self, crate::run_world_intent::RunWorldError> {
        let compiled = intent.source.load()?;
        let runtime = build_runtime(compiled.clone());
        
        Ok(Self {
            state: Arc::new(ServerState {
                compiled,
                runtime: Mutex::new(runtime),
            }),
        })
    }

    /// Run the server, listening on the given Unix socket path.
    pub async fn run(self, socket_path: &Path) -> Result<(), std::io::Error> {
        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        let listener = UnixListener::bind(socket_path)?;
        info!("Simulation server listening on {}", socket_path.display());

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    info!("Client connected to Simulation IPC");
                    let state = self.state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, state).await {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => error!("Failed to accept IPC connection: {}", e),
            }
        }
    }
}

async fn handle_connection<S>(mut stream: S, state: Arc<ServerState>) -> Result<(), std::io::Error> 
where S: AsyncRead + AsyncWrite + Unpin {
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
                let response = handle_request(req, &state);
                write_message(&mut stream, &WorldMessage::Response(response)).await?;
            }
            _ => {
                warn!("Received unexpected message type from client, ignoring");
            }
        }
    }
}

fn handle_request(req: WorldRequest, state: &ServerState) -> WorldResponse {
    let mut rt = state.runtime.lock().unwrap();
    
    match req.kind.as_str() {
        "run.step" => {
            let steps = req.payload.get("steps").and_then(|v| v.as_u64()).unwrap_or(1);
            for _ in 0..steps {
                if let Err(e) = rt.execute_tick() {
                    return WorldResponse {
                        id: req.id,
                        ok: false,
                        payload: None,
                        error: Some(format!("Execution error: {}", e)),
                    };
                }
            }
            WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::json!({
                    "tick": rt.tick(),
                    "sim_time": rt.sim_time(),
                })),
                error: None,
            }
        }
        "signal.get" => {
            let path = match req.payload.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'path' parameter".to_string()),
                },
            };
            
            let signal_id = continuum_foundation::SignalId::from(path);
            match rt.get_signal(&signal_id) {
                Some(val) => WorldResponse {
                    id: req.id,
                    ok: true,
                    payload: Some(serde_json::to_value(val).unwrap()),
                    error: None,
                },
                None => WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Signal '{}' not found", path)),
                },
            }
        }
        "signal.list" => {
            let signals: Vec<_> = state.compiled.world.globals.iter()
                .filter(|(_, node)| node.role_id() == RoleId::Signal)
                .map(|(path, node)| {
                    serde_json::json!({
                        "path": path.to_string(),
                        "doc": node.doc,
                        "type": node.output.as_ref().map(|t| t.to_string()),
                    })
                })
                .collect();
            
            WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::to_value(signals).unwrap()),
                error: None,
            }
        }
        "field.list" => {
            let fields: Vec<_> = state.compiled.world.globals.iter()
                .filter(|(_, node)| node.role_id() == RoleId::Field)
                .map(|(path, node)| {
                    serde_json::json!({
                        "path": path.to_string(),
                        "doc": node.doc,
                        "type": node.output.as_ref().map(|t| t.to_string()),
                    })
                })
                .collect();
            
            WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::to_value(fields).unwrap()),
                error: None,
            }
        }
        "impulse.list" => {
            let impulses: Vec<_> = state.compiled.world.globals.iter()
                .filter(|(_, node)| node.role_id() == RoleId::Impulse)
                .map(|(path, node)| {
                    serde_json::json!({
                        "path": path.to_string(),
                        "doc": node.doc,
                    })
                })
                .collect();
            
            WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::to_value(impulses).unwrap()),
                error: None,
            }
        }
        "impulse.emit" => {
            let path = match req.payload.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'path' parameter".to_string()),
                },
            };
            let payload = match req.payload.get("payload") {
                Some(p) => match serde_json::from_value(p.clone()) {
                    Ok(v) => v,
                    Err(e) => return WorldResponse {
                        id: req.id,
                        ok: false,
                        payload: None,
                        error: Some(format!("Invalid payload: {}", e)),
                    },
                },
                None => continuum_foundation::Value::Scalar(0.0), // Default payload
            };
            
            let impulse_id = continuum_foundation::ImpulseId::from(path);
            match rt.inject_impulse_by_id(&impulse_id, payload) {
                Ok(_) => WorldResponse {
                    id: req.id,
                    ok: true,
                    payload: None,
                    error: None,
                },
                Err(e) => WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Failed to inject impulse: {}", e)),
                },
            }
        }
        "assertion.list" => {
            let mut assertions = Vec::new();
            for node in state.compiled.world.globals.values() {
                for assertion in &node.assertions {
                    assertions.push(serde_json::json!({
                        "path": node.path.to_string(),
                        "message": assertion.message,
                        "severity": format!("{:?}", assertion.severity),
                    }));
                }
            }
            WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::to_value(assertions).unwrap()),
                error: None,
            }
        }
        "assertion.failures" => {
            let failures = rt.assertion_checker().failures();
            WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::to_value(failures).unwrap()),
                error: None,
            }
        }
        _ => WorldResponse {
            id: req.id,
            ok: false,
            payload: None,
            error: Some(format!("Unknown request kind: {}", req.kind)),
        },
    }
}
