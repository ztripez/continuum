//! Channel-based simulation proxy for the IPC server.
//!
//! Provides [`SimProxy`] — a typed interface for communicating with a dedicated
//! simulation thread via crossbeam channels. The simulation thread owns `Runtime`
//! exclusively; all queries and mutations go through message passing, eliminating
//! lock contention.
//!
//! Two channels separate concerns:
//! - **Control channel** (`ControlCommand`): high-priority state transitions (run/pause/stop)
//! - **Command channel** (`SimCommand`): queries and mutations with oneshot reply channels

use continuum_runtime::checkpoint::Checkpoint;
use continuum_runtime::executor::AssertionFailure;
use continuum_runtime::{EraId, ImpulseId, SignalId, TickContext, Value};
use crossbeam_channel as cbc;
use std::path::PathBuf;

// ============================================================================
// Control Commands (high-priority state transitions)
// ============================================================================

/// High-priority state transition commands for the simulation thread.
///
/// These are checked before every tick and during idle waits, ensuring
/// responsive pause/stop behavior regardless of command queue depth.
pub enum ControlCommand {
    /// Start or resume continuous tick execution at the given rate.
    ///
    /// `tick_rate` is ticks per second (0 = unlimited).
    Run { tick_rate: u32 },

    /// Pause continuous execution. The simulation thread stops ticking
    /// but remains alive and responsive to commands.
    Pause,

    /// Resume continuous execution after a pause, keeping the previously
    /// set tick rate.
    Resume,

    /// Stop the simulation thread entirely. The thread exits its loop
    /// and the join handle becomes joinable.
    Stop,
}

// ============================================================================
// Simulation Commands (queries and mutations with replies)
// ============================================================================

/// Query or mutation command sent to the simulation thread.
///
/// Each variant carries a `reply` channel sender. The simulation thread
/// processes the command and sends exactly one response back through it.
/// The caller blocks on the corresponding receiver.
pub enum SimCommand {
    /// Query current simulation status (tick, time, era).
    GetStatus { reply: cbc::Sender<SimStatus> },

    /// Read a signal's current resolved value.
    GetSignal {
        id: SignalId,
        reply: cbc::Sender<Option<Value>>,
    },

    /// Execute N ticks synchronously and return the final tick context.
    ExecuteSteps {
        count: u64,
        reply: cbc::Sender<Result<TickContext, String>>,
    },

    /// Inject an impulse into the simulation.
    InjectImpulse {
        id: ImpulseId,
        value: Value,
        reply: cbc::Sender<Result<(), String>>,
    },

    /// Request a checkpoint write to the given path.
    RequestCheckpoint {
        path: PathBuf,
        reply: cbc::Sender<Result<(), String>>,
    },

    /// Restore simulation state from a checkpoint.
    RestoreCheckpoint {
        checkpoint: Box<Checkpoint>,
        reply: cbc::Sender<Result<RestoreResult, String>>,
    },

    /// Get all assertion failures accumulated so far.
    GetAssertionFailures {
        reply: cbc::Sender<Vec<AssertionFailure>>,
    },
}

// ============================================================================
// Response Types
// ============================================================================

/// Snapshot of simulation status returned by `GetStatus`.
#[derive(Debug, Clone)]
pub struct SimStatus {
    /// Current tick number.
    pub tick: u64,
    /// Accumulated simulation time in seconds.
    pub sim_time: f64,
    /// Current era identifier.
    pub era: EraId,
}

/// Result of a checkpoint restore operation.
#[derive(Debug, Clone)]
pub struct RestoreResult {
    /// Tick number after restore.
    pub tick: u64,
    /// Simulation time after restore.
    pub sim_time: f64,
    /// Era after restore.
    pub era: EraId,
}

// ============================================================================
// SimProxy
// ============================================================================

/// Proxy for communicating with the simulation thread.
///
/// All methods are blocking — they send a command and wait for the reply.
/// Intended to be called from `spawn_blocking` threads in the tokio runtime,
/// matching the existing `RequestHandler` trait pattern.
///
/// The proxy is cheaply cloneable (channel senders are `Clone`).
#[derive(Clone)]
pub struct SimProxy {
    control_tx: cbc::Sender<ControlCommand>,
    cmd_tx: cbc::Sender<SimCommand>,
}

/// Error returned when the simulation thread has disconnected.
#[derive(Debug)]
pub struct SimDisconnected;

impl std::fmt::Display for SimDisconnected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("simulation thread disconnected")
    }
}

impl std::error::Error for SimDisconnected {}

impl SimProxy {
    /// Create a new proxy from channel senders.
    ///
    /// Typically called by `SimulationController::new()` after creating
    /// the channels and spawning the simulation thread.
    pub fn new(control_tx: cbc::Sender<ControlCommand>, cmd_tx: cbc::Sender<SimCommand>) -> Self {
        Self { control_tx, cmd_tx }
    }

    // -- Control commands (fire-and-forget, no reply) --

    /// Start continuous tick execution at the given rate.
    pub fn run(&self, tick_rate: u32) -> Result<(), SimDisconnected> {
        self.control_tx
            .send(ControlCommand::Run { tick_rate })
            .map_err(|_| SimDisconnected)
    }

    /// Pause continuous execution.
    pub fn pause(&self) -> Result<(), SimDisconnected> {
        self.control_tx
            .send(ControlCommand::Pause)
            .map_err(|_| SimDisconnected)
    }

    /// Resume continuous execution.
    pub fn resume(&self) -> Result<(), SimDisconnected> {
        self.control_tx
            .send(ControlCommand::Resume)
            .map_err(|_| SimDisconnected)
    }

    /// Stop the simulation thread.
    pub fn stop(&self) -> Result<(), SimDisconnected> {
        self.control_tx
            .send(ControlCommand::Stop)
            .map_err(|_| SimDisconnected)
    }

    // -- Query/mutation commands (blocking, wait for reply) --

    /// Get current simulation status.
    pub fn status(&self) -> Result<SimStatus, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::GetStatus { reply: tx })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }

    /// Read a signal's current value.
    pub fn get_signal(&self, id: SignalId) -> Result<Option<Value>, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::GetSignal { id, reply: tx })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }

    /// Execute N ticks and return the final tick context.
    pub fn execute_steps(
        &self,
        count: u64,
    ) -> Result<Result<TickContext, String>, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::ExecuteSteps { count, reply: tx })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }

    /// Inject an impulse.
    pub fn inject_impulse(
        &self,
        id: ImpulseId,
        value: Value,
    ) -> Result<Result<(), String>, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::InjectImpulse {
                id,
                value,
                reply: tx,
            })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }

    /// Request a checkpoint write.
    pub fn request_checkpoint(&self, path: PathBuf) -> Result<Result<(), String>, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::RequestCheckpoint { path, reply: tx })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }

    /// Restore from a checkpoint.
    pub fn restore_checkpoint(
        &self,
        checkpoint: Checkpoint,
    ) -> Result<Result<RestoreResult, String>, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::RestoreCheckpoint {
                checkpoint: Box::new(checkpoint),
                reply: tx,
            })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }

    /// Get all assertion failures.
    pub fn assertion_failures(&self) -> Result<Vec<AssertionFailure>, SimDisconnected> {
        let (tx, rx) = cbc::bounded(1);
        self.cmd_tx
            .send(SimCommand::GetAssertionFailures { reply: tx })
            .map_err(|_| SimDisconnected)?;
        rx.recv().map_err(|_| SimDisconnected)
    }
}
