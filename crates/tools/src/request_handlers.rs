use crate::ipc_types::{AssertionInfo, FieldInfo, ImpulseInfo, SignalInfo};
use crate::world_api::{WorldRequest, WorldResponse};
use continuum_cdsl::ast::{CompiledWorld, RoleId};
use continuum_runtime::Runtime;
use indexmap::IndexMap;
use std::sync::Mutex;

/// Shared state for request handlers.
pub struct ServerState {
    pub compiled: CompiledWorld,
    pub runtime: Mutex<Runtime>,
}

/// Trait for handling specific IPC request types.
pub trait RequestHandler: Send + Sync {
    /// The request kind this handler responds to (e.g., "world.get").
    fn kind(&self) -> &'static str;

    /// Handle the request and produce a response.
    fn handle(&self, req: &WorldRequest, state: &ServerState) -> WorldResponse;
}

/// Router for dispatching requests to handlers.
pub struct RequestRouter {
    handlers: IndexMap<&'static str, Box<dyn RequestHandler>>,
}

impl RequestRouter {
    /// Create a new router with all built-in handlers.
    pub fn new() -> Self {
        let mut router = Self {
            handlers: IndexMap::new(),
        };

        // Register all handlers
        router.register(Box::new(WorldGetHandler));
        router.register(Box::new(RunStepHandler));
        router.register(Box::new(SignalListHandler));
        router.register(Box::new(SignalDescribeHandler));
        router.register(Box::new(SignalGetHandler));
        router.register(Box::new(FieldListHandler));
        router.register(Box::new(FieldDescribeHandler));
        router.register(Box::new(EntityListHandler));
        router.register(Box::new(EntityDescribeHandler));
        router.register(Box::new(ImpulseListHandler));
        router.register(Box::new(ImpulseEmitHandler));
        router.register(Box::new(AssertionListHandler));
        router.register(Box::new(AssertionFailuresHandler));
        router.register(Box::new(StatusHandler));

        router
    }

    fn register(&mut self, handler: Box<dyn RequestHandler>) {
        self.handlers.insert(handler.kind(), handler);
    }

    pub fn handle(&self, req: WorldRequest, state: &ServerState) -> WorldResponse {
        match self.handlers.get(req.kind.as_str()) {
            Some(handler) => handler.handle(&req, state),
            None => WorldResponse {
                id: req.id,
                ok: false,
                payload: None,
                error: Some(format!("Unknown request kind: {}", req.kind)),
            },
        }
    }
}

impl Default for RequestRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Handler Implementations
// ============================================================================

struct WorldGetHandler;

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

struct RunStepHandler;

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

struct SignalListHandler;

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

struct SignalDescribeHandler;

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

struct SignalGetHandler;

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

struct FieldListHandler;

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

struct FieldDescribeHandler;

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

struct EntityListHandler;

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

struct EntityDescribeHandler;

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

struct ImpulseListHandler;

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

struct ImpulseEmitHandler;

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

struct AssertionListHandler;

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

struct AssertionFailuresHandler;

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

struct StatusHandler;

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

        match serde_json::to_value(serde_json::json!({
            "tick": tick,
            "sim_time": sim_time,
            "era": era,
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
