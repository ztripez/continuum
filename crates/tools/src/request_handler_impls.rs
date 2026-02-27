//! IPC request handler implementations.
//!
//! Each handler implements the `RequestHandler` trait, mapping an IPC request
//! kind string to a function that queries engine state and returns a response.

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
        let rt = state
            .runtime
            .lock()
            .expect("runtime mutex poisoned - fatal error");
        let tick = rt.tick();
        let sim_time = rt.sim_time();
        let era = rt.era();

        let execution_state = *state.execution_state.read().expect("rwlock poisoned");
        let tick_rate = state.tick_rate.load(std::sync::atomic::Ordering::Relaxed);
        let last_error = state.last_error.read().expect("rwlock poisoned").clone();

        let exec_state_str = match execution_state {
            ExecutionState::Stopped => "stopped",
            ExecutionState::Running => "running",
            ExecutionState::Paused => "paused",
            ExecutionState::Error => "error",
        };

        match serde_json::to_value(serde_json::json!({
            "tick": tick,
            "sim_time": sim_time,
            "era": era,
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

        let mut rt = state
            .runtime
            .lock()
            .expect("runtime mutex poisoned - fatal error");
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

        let tick = rt.tick();
        let sim_time = rt.sim_time();

        match serde_json::to_value(serde_json::json!({
            "tick": tick,
            "sim_time": sim_time,
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

pub(crate) struct RunHandler;

impl RequestHandler for RunHandler {
    fn kind(&self) -> &'static str {
        "run"
    }

    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse {
        // Set tick_rate if provided
        if let Some(tick_rate) = req.payload.get("tick_rate").and_then(|v| v.as_u64()) {
            state
                .tick_rate
                .store(tick_rate as u32, std::sync::atomic::Ordering::Relaxed);
        }

        // Transition state to Running
        {
            let mut exec_state = state.execution_state.write().expect("rwlock poisoned");
            *exec_state = ExecutionState::Running;
        }

        // Clear any previous error
        {
            let mut last_error = state.last_error.write().expect("rwlock poisoned");
            *last_error = None;
        }

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
        // Transition state to Stopped
        {
            let mut exec_state = state.execution_state.write().expect("rwlock poisoned");
            *exec_state = ExecutionState::Stopped;
        }

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
        // Transition state to Paused
        {
            let mut exec_state = state.execution_state.write().expect("rwlock poisoned");
            // Only allow pause from Running state
            if *exec_state == ExecutionState::Running {
                *exec_state = ExecutionState::Paused;
            } else {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some("Cannot pause: simulation not running".to_string()),
                };
            }
        }

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
        // Transition state from Paused/Error back to Running
        {
            let mut exec_state = state.execution_state.write().expect("rwlock poisoned");
            match *exec_state {
                ExecutionState::Paused | ExecutionState::Error => {
                    *exec_state = ExecutionState::Running;
                }
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
        }

        // Clear error if resuming from error state
        {
            let mut last_error = state.last_error.write().expect("rwlock poisoned");
            *last_error = None;
        }

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

        let rt = state
            .runtime
            .lock()
            .expect("runtime mutex poisoned - fatal error");
        let signal_path = continuum_runtime::types::SignalId::from(signal_id.to_string());
        let value = match rt.get_signal(&signal_path) {
            Some(v) => v,
            None => {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Signal not found or not resolved: {}", signal_id)),
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

        let mut rt = state
            .runtime
            .lock()
            .expect("runtime mutex poisoned - fatal error");
        let impulse_path = continuum_runtime::types::ImpulseId::from(impulse_id.to_string());
        if let Err(e) = rt.inject_impulse_by_id(&impulse_path, payload_value) {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Failed to inject impulse: {}", e)),
            };
        }

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({ "injected": true })),
            error: None,
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
        let rt = state
            .runtime
            .lock()
            .expect("runtime mutex poisoned - fatal error");
        let failures = rt.assertion_checker().failures();

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
        let exec_state = *state.execution_state.read().expect("rwlock poisoned");
        if exec_state == ExecutionState::Running {
            return WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some("Cannot save checkpoint while running (pause first)".to_string()),
            };
        }

        let path = req
            .payload
            .get("path")
            .and_then(|v| v.as_str())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let rt = state.runtime.lock().expect("runtime mutex poisoned");
                let world_name = &state.compiled.world.metadata.path;
                default_checkpoint_path(&world_name.to_string(), rt.tick())
            });

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return WorldResponse {
                    id: req.id,
                    ok: false,
                    payload: None,
                    error: Some(format!("Failed to create checkpoint directory: {}", e)),
                };
            }
        }

        // Request checkpoint (uses runtime's checkpoint writer)
        let rt = state.runtime.lock().expect("runtime mutex poisoned");
        match rt.request_checkpoint(&path) {
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
        let exec_state = *state.execution_state.read().expect("rwlock poisoned");
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

        // Restore into runtime
        let mut rt = state.runtime.lock().expect("runtime mutex poisoned");
        match rt.restore_from_checkpoint(checkpoint) {
            Ok(()) => WorldResponse {
                id: req.id,
                ok: true,
                payload: Some(serde_json::json!({
                    "tick": rt.tick(),
                    "sim_time": rt.sim_time(),
                    "era": rt.era().to_string(),
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
                if path.extension().map_or(false, |ext| ext == "ckpt") {
                    if let Ok(metadata) = entry.metadata() {
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
        }

        WorldResponse {
            id: req.id,
            ok: true,
            payload: Some(serde_json::json!({"checkpoints": checkpoints})),
            error: None,
        }
    }
}
