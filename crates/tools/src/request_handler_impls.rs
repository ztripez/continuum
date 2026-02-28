//! IPC request handler implementations.
//!
//! Each handler implements the `RequestHandler` trait, mapping an IPC request
//! kind string to a function that queries engine state and returns a response.
//!
//! Handlers that need runtime data use [`SimProxy`] to send commands to
//! the simulation thread. The proxy methods block on crossbeam channel replies,
//! which is fine because handlers run on `spawn_blocking` threads.

use crate::ipc_types::{AssertionInfo, FieldInfo, ImpulseInfo, SignalInfo};
use crate::world_api::{WorldRequest, WorldResponse};
use continuum_cdsl::ast::RoleId;

use crate::request_handlers::{ExecutionState, RequestHandler, ServerState};

// ============================================================================
// World / Status Handlers
// ============================================================================

pub(crate) struct WorldGetHandler;

impl RequestHandler for WorldGetHandler {
    fn kind(&self) -> &'static str {
        "world.get"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        match serde_json::to_value(&state.compiled.world.metadata) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Failed to serialize world metadata: {}", e)),
            },
        }
    }
}

pub(crate) struct StatusHandler;

impl RequestHandler for StatusHandler {
    fn kind(&self) -> &'static str {
        "status"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        // Query runtime state via sim proxy
        let sim_status = match state.sim.status() {
            Ok(s) => s,
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        let execution_state = *state.execution_state.read();
        let tick_rate = state
            .tick_rate
            .load(std::sync::atomic::Ordering::Relaxed);
        let last_error = state.last_error.read().clone();

        let exec_state_str = match execution_state {
            ExecutionState::Stopped => "stopped",
            ExecutionState::Running => "running",
            ExecutionState::Paused => "paused",
            ExecutionState::Error => "error",
        };

        match serde_json::to_value(serde_json::json!({
            "tick": sim_status.tick,
            "sim_time": sim_status.sim_time,
            "era": sim_status.era.to_string(),
            "execution_state": exec_state_str,
            "tick_rate": tick_rate,
            "last_error": last_error,
        })) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

// ============================================================================
// Execution Control Handlers
// ============================================================================

pub(crate) struct RunStepHandler;

impl RequestHandler for RunStepHandler {
    fn kind(&self) -> &'static str {
        "run.step"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let steps = match req.payload.get("steps").and_then(|v| v.as_u64()) {
            Some(s) => s,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing or invalid 'steps' parameter (requires u64)".to_string()),
                };
            }
        };

        let result = match state.sim.execute_steps(steps) {
            Ok(r) => r,
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        match result {
            Ok(tick_ctx) => {
                match serde_json::to_value(serde_json::json!({
                    "tick": tick_ctx.tick + 1, // Report post-increment tick to match old API
                    "sim_time": tick_ctx.sim_time,
                })) {
                    Ok(payload) => WorldResponse {
                        id: req.id,
                        ok: true,
                        payload: Some(payload),
                        error: None,
                    },
                    Err(e) => WorldResponse {
                        id: req.id,
                        ok: false,
                        payload: None,
                        error: Some(format!("Serialization error: {}", e)),
                    },
                }
            }
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Execution error: {}", e)),
            },
        }
    }
}

pub(crate) struct RunHandler;

impl RequestHandler for RunHandler {
    fn kind(&self) -> &'static str {
        "run"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        // Read tick_rate from request or use current value
        let tick_rate = req
            .payload
            .get("tick_rate")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or_else(|| {
                state
                    .tick_rate
                    .load(std::sync::atomic::Ordering::Relaxed)
            });

        // Store tick_rate for status queries
        state
            .tick_rate
            .store(tick_rate, std::sync::atomic::Ordering::Relaxed);

        // Send run command to simulation thread
        if state.sim.run(tick_rate).is_err() {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Simulation thread disconnected".to_string()),
            };
        }

        // Update observed state
        *state.execution_state.write() = ExecutionState::Running;
        *state.last_error.write() = None;

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({"message": "Simulation started"})),
            error: None,
        }
    }
}

pub(crate) struct StopHandler;

impl RequestHandler for StopHandler {
    fn kind(&self) -> &'static str {
        "stop"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        if state.sim.stop().is_err() {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Simulation thread disconnected".to_string()),
            };
        }

        *state.execution_state.write() = ExecutionState::Stopped;

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({"message": "Simulation stopped"})),
            error: None,
        }
    }
}

pub(crate) struct PauseHandler;

impl RequestHandler for PauseHandler {
    fn kind(&self) -> &'static str {
        "pause"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let current = *state.execution_state.read();
        if current != ExecutionState::Running {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Cannot pause: simulation not running".to_string()),
            };
        }

        if state.sim.pause().is_err() {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Simulation thread disconnected".to_string()),
            };
        }

        *state.execution_state.write() = ExecutionState::Paused;

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({"message": "Simulation paused"})),
            error: None,
        }
    }
}

pub(crate) struct ResumeHandler;

impl RequestHandler for ResumeHandler {
    fn kind(&self) -> &'static str {
        "resume"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let current = *state.execution_state.read();
        match current {
            ExecutionState::Paused | ExecutionState::Error => {}
            ExecutionState::Running => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Cannot resume: already running".to_string()),
                };
            }
            ExecutionState::Stopped => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(
                        "Cannot resume: simulation stopped (use 'run' instead)".to_string(),
                    ),
                };
            }
        }

        if state.sim.resume().is_err() {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Simulation thread disconnected".to_string()),
            };
        }

        *state.execution_state.write() = ExecutionState::Running;
        *state.last_error.write() = None;

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({"message": "Simulation resumed"})),
            error: None,
        }
    }
}

// ============================================================================
// Signal Handlers
// ============================================================================

pub(crate) struct SignalListHandler;

impl RequestHandler for SignalListHandler {
    fn kind(&self) -> &'static str {
        "signal.list"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let signals: Vec<SignalInfo> = state
            .compiled
            .world
            .globals
            .iter()
            .filter(|(_, node)| node.role_id() == RoleId::Signal)
            .map(|(path, node)| SignalInfo {
                id: path.to_string(),
                title: node.title.clone(),
                symbol: None,
                doc: node.doc.clone(),
                value_type: node.output.as_ref().map(|t| t.to_string()),
                unit: None,
                range: None,
                stratum: node.stratum.as_ref().map(|s| s.to_string()),
            })
            .collect();

        match serde_json::to_value(signals) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

pub(crate) struct SignalDescribeHandler;

impl RequestHandler for SignalDescribeHandler {
    fn kind(&self) -> &'static str {
        "signal.describe"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let signal_id = match req.payload.get("signal_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'signal_id' parameter".to_string()),
                };
            }
        };

        let signal_path = continuum_foundation::Path::from(signal_id.to_string());
        let node = match state.compiled.world.globals.get(&signal_path) {
            Some(node) if node.role_id() == RoleId::Signal => node,
            _ => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Signal not found: {}", signal_id)),
                };
            }
        };

        let info = SignalInfo {
            id: signal_id.to_string(),
            title: node.title.clone(),
            symbol: None,
            doc: node.doc.clone(),
            value_type: node.output.as_ref().map(|t| t.to_string()),
            unit: None,
            range: None,
            stratum: node.stratum.as_ref().map(|s| s.to_string()),
        };

        match serde_json::to_value(info) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

pub(crate) struct SignalGetHandler;

impl RequestHandler for SignalGetHandler {
    fn kind(&self) -> &'static str {
        "signal.get"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let signal_id = match req.payload.get("signal_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'signal_id' parameter".to_string()),
                };
            }
        };

        let signal_path = continuum_runtime::types::SignalId::from(signal_id.to_string());
        let value = match state.sim.get_signal(signal_path) {
            Ok(Some(v)) => v,
            Ok(None) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!(
                        "Signal not found or not resolved: {}",
                        signal_id
                    )),
                };
            }
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        match serde_json::to_value(serde_json::json!({
            "signal_id": signal_id,
            "value": value,
        })) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

// ============================================================================
// Field Handlers
// ============================================================================

pub(crate) struct FieldListHandler;

impl RequestHandler for FieldListHandler {
    fn kind(&self) -> &'static str {
        "field.list"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let fields: Vec<FieldInfo> = state
            .compiled
            .world
            .globals
            .iter()
            .filter(|(_, node)| node.role_id() == RoleId::Field)
            .map(|(path, node)| FieldInfo {
                id: path.to_string(),
                title: node.title.clone(),
                symbol: None,
                doc: node.doc.clone(),
                topology: None,
                value_type: node.output.as_ref().map(|t| t.to_string()),
                unit: None,
                range: None,
            })
            .collect();

        match serde_json::to_value(fields) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

pub(crate) struct FieldDescribeHandler;

impl RequestHandler for FieldDescribeHandler {
    fn kind(&self) -> &'static str {
        "field.describe"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let field_id = match req.payload.get("field_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'field_id' parameter".to_string()),
                };
            }
        };

        let field_path = continuum_foundation::Path::from(field_id.to_string());
        let node = match state.compiled.world.globals.get(&field_path) {
            Some(node) if node.role_id() == RoleId::Field => node,
            _ => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Field not found: {}", field_id)),
                };
            }
        };

        let info = FieldInfo {
            id: field_id.to_string(),
            title: node.title.clone(),
            symbol: None,
            doc: node.doc.clone(),
            topology: None,
            value_type: node.output.as_ref().map(|t| t.to_string()),
            unit: None,
            range: None,
        };

        match serde_json::to_value(info) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

// ============================================================================
// Entity Handlers
// ============================================================================

pub(crate) struct EntityListHandler;

impl RequestHandler for EntityListHandler {
    fn kind(&self) -> &'static str {
        "entity.list"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let entities: Vec<String> = state
            .compiled
            .world
            .entities
            .keys()
            .map(|path| path.to_string())
            .collect();

        match serde_json::to_value(entities) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

pub(crate) struct EntityDescribeHandler;

impl RequestHandler for EntityDescribeHandler {
    fn kind(&self) -> &'static str {
        "entity.describe"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let entity_id = match req.payload.get("entity_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'entity_id' parameter".to_string()),
                };
            }
        };

        let entity_path = continuum_foundation::Path::from(entity_id.to_string());
        let entity = match state.compiled.world.entities.get(&entity_path) {
            Some(e) => e,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Entity not found: {}", entity_id)),
                };
            }
        };

        match serde_json::to_value(serde_json::json!({
            "id": entity_id,
            "doc": entity.doc,
        })) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

// ============================================================================
// Impulse Handlers
// ============================================================================

pub(crate) struct ImpulseListHandler;

impl RequestHandler for ImpulseListHandler {
    fn kind(&self) -> &'static str {
        "impulse.list"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let impulses: Vec<ImpulseInfo> = state
            .compiled
            .world
            .globals
            .iter()
            .filter(|(_, node)| node.role_id() == RoleId::Impulse)
            .map(|(path, node)| ImpulseInfo {
                path: path.to_string(),
                doc: node.doc.clone(),
                payload_type: node.output.as_ref().map(|t| t.to_string()),
            })
            .collect();

        match serde_json::to_value(impulses) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

pub(crate) struct ImpulseEmitHandler;

impl RequestHandler for ImpulseEmitHandler {
    fn kind(&self) -> &'static str {
        "impulse.emit"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let impulse_id = match req.payload.get("impulse_id").and_then(|v| v.as_str()) {
            Some(id) => id,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'impulse_id' parameter".to_string()),
                };
            }
        };

        let payload_json = match req.payload.get("payload") {
            Some(p) => p.clone(),
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'payload' parameter".to_string()),
                };
            }
        };

        // Convert serde_json::Value to continuum_runtime::Value
        let payload_value: continuum_runtime::Value = match serde_json::from_value(payload_json) {
            Ok(v) => v,
            Err(e) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Invalid payload format: {}", e)),
                };
            }
        };

        let impulse_path = continuum_runtime::types::ImpulseId::from(impulse_id.to_string());
        let result = match state.sim.inject_impulse(impulse_path, payload_value) {
            Ok(r) => r,
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        match result {
            Ok(()) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::json!({ "injected": true })),
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
}

// ============================================================================
// Assertion Handlers
// ============================================================================

pub(crate) struct AssertionListHandler;

impl RequestHandler for AssertionListHandler {
    fn kind(&self) -> &'static str {
        "assertion.list"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let assertions: Vec<AssertionInfo> = state
            .compiled
            .world
            .globals
            .iter()
            .filter_map(|(path, node)| {
                if node.role_id() == RoleId::Signal && !node.assertions.is_empty() {
                    // Return first assertion for now (API limitation)
                    return node.assertions.first().map(|a| AssertionInfo {
                        signal_id: path.to_string(),
                        severity: format!("{:?}", a.severity),
                        message: a.message.clone(),
                    });
                }
                None
            })
            .collect();

        match serde_json::to_value(assertions) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

pub(crate) struct AssertionFailuresHandler;

impl RequestHandler for AssertionFailuresHandler {
    fn kind(&self) -> &'static str {
        "assertion.failures"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        let failures = match state.sim.assertion_failures() {
            Ok(f) => f,
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        match serde_json::to_value(failures) {
            Ok(payload) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(payload),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Serialization error: {}", e)),
            },
        }
    }
}

// ============================================================================
// Checkpoint Handlers
// ============================================================================

/// Default checkpoint directory.
fn default_checkpoint_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".continuum")
        .join("checkpoints")
}

/// Generate default checkpoint filename.
fn default_checkpoint_path(world_name: &str, tick: u64) -> std::path::PathBuf {
    default_checkpoint_dir().join(format!("{}-tick-{}.ckpt", world_name, tick))
}

pub(crate) struct CheckpointSaveHandler;

impl RequestHandler for CheckpointSaveHandler {
    fn kind(&self) -> &'static str {
        "checkpoint.save"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        // Must be stopped or paused to save
        let exec_state = *state.execution_state.read();
        if exec_state == ExecutionState::Running {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Cannot save checkpoint while running (pause first)".to_string()),
            };
        }

        let path = match req.payload.get("path").and_then(|v| v.as_str()) {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                // Need tick for default filename — query via proxy
                let tick = state
                    .sim
                    .status()
                    .map(|s| s.tick)
                    .unwrap_or(0);
                let world_name = &state.compiled.world.metadata.path;
                default_checkpoint_path(&world_name.to_string(), tick)
            }
        };

        // Ensure directory exists
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Failed to create checkpoint directory: {}", e)),
            };
        }

        // Request checkpoint via sim proxy
        let result = match state.sim.request_checkpoint(path.clone()) {
            Ok(r) => r,
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        match result {
            Ok(()) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::json!({
                    "path": path.display().to_string(),
                    "message": "Checkpoint requested"
                })),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Failed to request checkpoint: {}", e)),
            },
        }
    }
}

pub(crate) struct CheckpointLoadHandler;

impl RequestHandler for CheckpointLoadHandler {
    fn kind(&self) -> &'static str {
        "checkpoint.load"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        // Must be stopped to load
        let exec_state = *state.execution_state.read();
        if exec_state != ExecutionState::Stopped {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Must stop simulation before loading checkpoint".to_string()),
            };
        }

        let path = match req.payload.get("path").and_then(|v| v.as_str()) {
            Some(p) => std::path::Path::new(p),
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Missing 'path' parameter".to_string()),
                };
            }
        };

        // Load checkpoint from disk
        let checkpoint = match continuum_runtime::checkpoint::load_checkpoint(path) {
            Ok(ckpt) => ckpt,
            Err(e) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Failed to load checkpoint: {}", e)),
                };
            }
        };

        // Restore via sim proxy
        let result = match state.sim.restore_checkpoint(checkpoint) {
            Ok(r) => r,
            Err(_) => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Simulation thread disconnected".to_string()),
                };
            }
        };

        match result {
            Ok(restore) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::json!({
                    "tick": restore.tick,
                    "sim_time": restore.sim_time,
                    "era": restore.era.to_string(),
                    "message": "Checkpoint loaded successfully"
                })),
                error: None,
            },
            Err(e) => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Failed to restore checkpoint: {}", e)),
            },
        }
    }
}

pub(crate) struct CheckpointListHandler;

impl RequestHandler for CheckpointListHandler {
    fn kind(&self) -> &'static str {
        "checkpoint.list"
    }

    fn handle(&self, req: &WorldRequest, _state: &ServerState) -> WorldResponse {
        let checkpoint_dir = default_checkpoint_dir();

        if !checkpoint_dir.exists() {
            return WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::json!({"checkpoints": []})),
                error: None,
            };
        }

        let mut checkpoints = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "ckpt")
                    && let Ok(metadata) = entry.metadata()
                {
                    checkpoints.push(serde_json::json!({
                        "path": path.display().to_string(),
                        "size": metadata.len(),
                        "modified": metadata.modified().ok().and_then(|t| {
                            t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs())
                        }),
                    }));
                }
            }
        }

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({"checkpoints": checkpoints})),
            error: None,
        }
    }
}
