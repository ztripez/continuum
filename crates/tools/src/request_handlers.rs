use crate::world_api::{WorldRequest, WorldResponse};
use continuum_cdsl::ast::CompiledWorld;
use continuum_runtime::Runtime;
use indexmap::IndexMap;
use std::sync::{atomic::AtomicU32, Mutex, RwLock};

use crate::request_handler_impls::*;

/// Execution state of the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionState {
    Stopped,
    Running,
    Paused,
    Error, // Paused due to error
}

/// Shared state for request handlers.
pub struct ServerState {
    pub compiled: CompiledWorld,
    pub runtime: Mutex<Runtime>,
    pub execution_state: RwLock<ExecutionState>,
    pub tick_rate: AtomicU32, // ticks per second (0 = unlimited)
    pub last_error: RwLock<Option<String>>,
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
        router.register(Box::new(RunHandler));
        router.register(Box::new(StopHandler));
        router.register(Box::new(PauseHandler));
        router.register(Box::new(ResumeHandler));
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
        router.register(Box::new(CheckpointSaveHandler));
        router.register(Box::new(CheckpointLoadHandler));
        router.register(Box::new(CheckpointListHandler));
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
