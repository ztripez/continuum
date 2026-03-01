//! Request handler dispatch for the IPC server.
//!
//! Defines [`RequestHandler`] trait, [`ServerState`] shared across handlers,
//! and [`RequestRouter`] which maps IPC request kind strings to their
//! corresponding handler implementations.

use crate::sim_proxy::SimProxy;
use crate::world_api::{WorldEvent, WorldRequest, WorldResponse};
use continuum_cdsl::ast::CompiledWorld;
use indexmap::IndexMap;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use tokio::sync::broadcast;

use crate::request_handler_impls::*;

/// Execution state of the simulation, as observed by the IPC layer.
///
/// This is the IPC server's view of state — the authoritative state lives
/// inside the simulation thread. This mirror is updated by control handlers.
///
/// Represented as `u8` for lock-free atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecutionState {
    Stopped = 0,
    Running = 1,
    Paused = 2,
    Error = 3, // Paused due to error
}

impl ExecutionState {
    /// Convert from raw `u8`. Returns `None` for invalid values.
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Stopped),
            1 => Some(Self::Running),
            2 => Some(Self::Paused),
            3 => Some(Self::Error),
            _ => None,
        }
    }
}

/// Atomic wrapper for [`ExecutionState`] stored as [`AtomicU8`].
///
/// Provides lock-free read/write/compare-exchange operations for
/// the execution state, avoiding `RwLock` overhead on a hot path
/// that every status query and control handler touches.
pub struct AtomicExecutionState(AtomicU8);

impl AtomicExecutionState {
    /// Create a new atomic execution state.
    pub fn new(state: ExecutionState) -> Self {
        Self(AtomicU8::new(state as u8))
    }

    /// Load the current execution state.
    pub fn load(&self) -> ExecutionState {
        let v = self.0.load(Ordering::Acquire);
        ExecutionState::from_u8(v).unwrap_or_else(|| {
            panic!("AtomicExecutionState contains invalid value: {v}");
        })
    }

    /// Store a new execution state.
    pub fn store(&self, state: ExecutionState) {
        self.0.store(state as u8, Ordering::Release);
    }

    /// Compare-and-exchange: if current == `expected`, set to `new` and return `Ok(expected)`.
    /// Otherwise return `Err(actual)`.
    pub fn compare_exchange(
        &self,
        expected: ExecutionState,
        new: ExecutionState,
    ) -> Result<ExecutionState, ExecutionState> {
        self.0
            .compare_exchange(
                expected as u8,
                new as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|v| {
                ExecutionState::from_u8(v).unwrap_or_else(|| {
                    panic!("AtomicExecutionState compare_exchange returned invalid value: {v}");
                })
            })
            .map_err(|v| {
                ExecutionState::from_u8(v).unwrap_or_else(|| {
                    panic!("AtomicExecutionState compare_exchange returned invalid value: {v}");
                })
            })
    }
}

/// Shared state for request handlers.
///
/// The `compiled` world provides read-only access to the world schema.
/// The `sim` proxy sends commands to the simulation thread which owns
/// the `Runtime` exclusively — no locks needed for runtime access.
pub struct ServerState {
    /// Compiled world schema (signals, fields, entities, etc.).
    pub compiled: CompiledWorld,

    /// Proxy to the simulation thread that owns Runtime exclusively.
    pub sim: SimProxy,

    /// Observed execution state (mirrors the sim thread's internal state).
    /// Lock-free via `AtomicU8`.
    pub execution_state: AtomicExecutionState,

    /// Tick rate in ticks per second (0 = unlimited).
    pub tick_rate: AtomicU32,

    /// Last error message from the simulation thread.
    pub last_error: parking_lot::RwLock<Option<String>>,

    /// Broadcast channel for pushing events to connected inspector clients.
    ///
    /// Handlers can use this to emit events (e.g., tick updates after step execution).
    /// The sim thread also holds a clone for emitting events during continuous execution.
    pub event_tx: broadcast::Sender<WorldEvent>,
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
        router.register(Box::new(SignalHistoryHandler));
        router.register(Box::new(FieldListHandler));
        router.register(Box::new(FieldDescribeHandler));
        router.register(Box::new(FieldSamplesHandler));
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

    /// Dispatch a world request to the appropriate handler and return a response.
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
