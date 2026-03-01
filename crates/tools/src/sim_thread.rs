//! Dedicated simulation thread that owns `Runtime` exclusively.
//!
//! The simulation thread receives commands via crossbeam channels and executes
//! them with sole ownership of the `Runtime`. This eliminates all lock contention:
//! no `Mutex<Runtime>`, no reader/writer races, no priority inversion.
//!
//! # Architecture
//!
//! ```text
//! IPC Handlers (tokio tasks)     Simulation Thread (std::thread)
//! ─────────────────────────      ──────────────────────────────
//! SimProxy::status()             
//!   ├─ send GetStatus ──────────→ crossbeam select! {
//!   └─ block on reply              control_rx => handle state transition
//!                                  cmd_rx     => handle query/mutation
//! SimProxy::run(rate)              timeout    => execute_tick() if Running
//!   └─ send Run ────────────────→ }
//!                                  └─ reply via oneshot channel
//! ```
//!
//! # Tick Rate
//!
//! When running, the thread sleeps between ticks based on the configured rate:
//! - `tick_rate > 0`: `sleep(1.0 / tick_rate)` between ticks
//! - `tick_rate == 0`: unlimited (no sleep, yield only)
//!
//! # Shutdown
//!
//! Sending `ControlCommand::Stop` causes the thread to exit its main loop.
//! Dropping all `SimProxy` clones (and thus all senders) also causes shutdown
//! via channel disconnection.

use crate::sim_proxy::{
    ControlCommand, FieldSnapshot, RestoreResult, SimCommand, SimProxy, SimStatus,
};
use crate::world_api::WorldEvent;
use continuum_runtime::Runtime;
use crossbeam_channel as cbc;
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{error, info, trace, warn};

/// Capacity of the control channel (small, high-priority).
const CONTROL_CHANNEL_CAP: usize = 8;

/// Capacity of the command channel (larger, queued work).
const COMMAND_CHANNEL_CAP: usize = 64;

/// Internal execution state tracked by the simulation thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunState {
    /// Thread is alive but not ticking. Responds to commands only.
    Idle,
    /// Continuously executing ticks at the configured rate.
    Running,
    /// Paused — like Idle but distinguishes intent for resume logic.
    Paused,
}

/// Handle to a running simulation thread.
///
/// Owns the join handle and provides the [`SimProxy`] for communication.
/// Dropping this handle sends a stop signal and joins the thread.
pub struct SimThread {
    proxy: SimProxy,
    handle: Option<JoinHandle<Runtime>>,
}

impl SimThread {
    /// Spawn a new simulation thread that takes exclusive ownership of `runtime`.
    ///
    /// Returns a `SimThread` handle containing a cloneable [`SimProxy`] for
    /// sending commands to the thread.
    ///
    /// If `event_tx` is provided, the thread emits [`WorldEvent`] messages
    /// after each tick execution. This powers real-time UI updates in the
    /// inspector without polling.
    pub fn spawn(runtime: Runtime, event_tx: Option<broadcast::Sender<WorldEvent>>) -> Self {
        let (control_tx, control_rx) = cbc::bounded(CONTROL_CHANNEL_CAP);
        let (cmd_tx, cmd_rx) = cbc::bounded(COMMAND_CHANNEL_CAP);

        // Shared field snapshot — written by sim thread after each tick,
        // read by IPC handlers for field.samples requests. Using Arc<RwLock>
        // avoids channel overhead for potentially large field data.
        let field_snapshot: Arc<RwLock<FieldSnapshot>> =
            Arc::new(RwLock::new(FieldSnapshot::default()));
        let field_snapshot_writer = field_snapshot.clone();

        let proxy = SimProxy::new(control_tx, cmd_tx, field_snapshot);

        let handle = thread::Builder::new()
            .name("sim-thread".to_string())
            .spawn(move || {
                sim_thread_main(runtime, control_rx, cmd_rx, event_tx, field_snapshot_writer)
            })
            .expect("failed to spawn simulation thread");

        Self {
            proxy,
            handle: Some(handle),
        }
    }

    /// Get a cloneable proxy for communicating with the simulation thread.
    pub fn proxy(&self) -> &SimProxy {
        &self.proxy
    }

    /// Stop the simulation thread and reclaim the `Runtime`.
    ///
    /// Sends a `Stop` command, then joins the thread. Returns the `Runtime`
    /// so the caller can inspect final state or hand it off elsewhere.
    ///
    /// If the thread has already exited (e.g., due to channel disconnect),
    /// this still joins and returns the runtime.
    pub fn stop_and_join(mut self) -> Result<Runtime, String> {
        // Best-effort stop signal — thread may already be gone
        let _ = self.proxy.stop();
        self.join_inner()
    }

    /// Join the thread, consuming the handle.
    fn join_inner(&mut self) -> Result<Runtime, String> {
        match self.handle.take() {
            Some(handle) => handle
                .join()
                .map_err(|e| format!("simulation thread panicked: {:?}", e)),
            None => Err("simulation thread already joined".to_string()),
        }
    }
}

impl Drop for SimThread {
    fn drop(&mut self) {
        if self.handle.is_some() {
            // Best-effort stop
            let _ = self.proxy.stop();
            if let Err(e) = self.join_inner() {
                error!("SimThread drop: {}", e);
            }
        }
    }
}

// ============================================================================
// Simulation Thread Main Loop
// ============================================================================

/// Main loop for the simulation thread. Returns the `Runtime` on exit.
fn sim_thread_main(
    mut runtime: Runtime,
    control_rx: cbc::Receiver<ControlCommand>,
    cmd_rx: cbc::Receiver<SimCommand>,
    event_tx: Option<broadcast::Sender<WorldEvent>>,
    field_snapshot: Arc<RwLock<FieldSnapshot>>,
) -> Runtime {
    info!("Simulation thread started");

    // Run warmup if needed
    if !runtime.is_warmup_complete() {
        info!("Executing warmup...");
        match runtime.execute_warmup() {
            Ok(_result) => info!("Warmup complete"),
            Err(e) => error!("Warmup failed: {}", e),
        }
    }

    let mut state = RunState::Idle;
    let mut tick_rate: u32 = 0;
    let mut last_tick_time = Instant::now();

    loop {
        match state {
            RunState::Running => {
                // When running: check for control commands (non-blocking), then
                // drain pending sim commands, then execute one tick.
                if let Some(should_exit) = drain_control(&control_rx, &mut state, &mut tick_rate) {
                    if should_exit {
                        break;
                    }
                    // State may have changed — re-enter loop
                    continue;
                }

                // Drain all pending commands before ticking
                drain_commands(&cmd_rx, &mut runtime, &event_tx, &field_snapshot);

                // Execute tick
                match runtime.execute_tick() {
                    Ok(tick_ctx) => {
                        trace!(tick = runtime.tick(), "Tick executed");
                        drain_fields_to_snapshot(&mut runtime, &field_snapshot);
                        emit_tick_event(&event_tx, &tick_ctx, &runtime, "running");
                    }
                    Err(e) => {
                        error!(tick = runtime.tick(), error = %e, "Tick execution failed");
                        // Pause on error — don't keep ticking a broken state
                        state = RunState::Paused;
                        continue;
                    }
                }

                // Throttle based on tick rate
                if tick_rate > 0 {
                    let target_duration = Duration::from_secs_f64(1.0 / f64::from(tick_rate));
                    let elapsed = last_tick_time.elapsed();
                    if elapsed < target_duration {
                        // Sleep for remainder, but check control channel periodically
                        // so we can respond to pause/stop within ~10ms
                        let remaining = target_duration - elapsed;
                        let should_exit = sleep_with_control_check(
                            remaining,
                            &control_rx,
                            &mut state,
                            &mut tick_rate,
                        );
                        if should_exit {
                            break;
                        }
                        if state != RunState::Running {
                            continue;
                        }
                    }
                }

                last_tick_time = Instant::now();
            }

            RunState::Idle | RunState::Paused => {
                // When idle/paused: block on either channel with a timeout
                // so we don't spin. The timeout lets us periodically check
                // for channel disconnection.
                cbc::select! {
                    recv(control_rx) -> msg => {
                        match msg {
                            Ok(cmd) => {
                                if handle_control_cmd(cmd, &mut state, &mut tick_rate) {
                                    break;
                                }
                            }
                            Err(cbc::RecvError) => {
                                info!("Control channel disconnected, shutting down");
                                break;
                            }
                        }
                    }
                    recv(cmd_rx) -> msg => {
                        match msg {
                            Ok(cmd) => handle_sim_cmd(cmd, &mut runtime, &event_tx, &field_snapshot),
                            Err(cbc::RecvError) => {
                                info!("Command channel disconnected, shutting down");
                                break;
                            }
                        }
                    }
                    default(Duration::from_millis(100)) => {
                        // Periodic wake — allows detecting channel closure
                    }
                }
            }
        }
    }

    info!(tick = runtime.tick(), "Simulation thread exiting");
    runtime
}

// ============================================================================
// Control Command Handling
// ============================================================================

/// Drain all pending control commands (non-blocking).
///
/// Returns `Some(true)` if the thread should exit, `Some(false)` if state
/// changed (caller should re-evaluate), `None` if no control commands.
fn drain_control(
    control_rx: &cbc::Receiver<ControlCommand>,
    state: &mut RunState,
    tick_rate: &mut u32,
) -> Option<bool> {
    let mut any_received = false;

    loop {
        match control_rx.try_recv() {
            Ok(cmd) => {
                any_received = true;
                if handle_control_cmd(cmd, state, tick_rate) {
                    return Some(true); // exit
                }
            }
            Err(cbc::TryRecvError::Empty) => break,
            Err(cbc::TryRecvError::Disconnected) => {
                info!("Control channel disconnected during drain");
                return Some(true);
            }
        }
    }

    if any_received {
        Some(false)
    } else {
        None
    }
}

/// Handle a single control command. Returns `true` if the thread should exit.
fn handle_control_cmd(cmd: ControlCommand, state: &mut RunState, tick_rate: &mut u32) -> bool {
    match cmd {
        ControlCommand::Run { tick_rate: rate } => {
            *tick_rate = rate;
            *state = RunState::Running;
            info!(tick_rate = rate, "Simulation running");
            false
        }
        ControlCommand::Pause => {
            *state = RunState::Paused;
            info!("Simulation paused");
            false
        }
        ControlCommand::Resume => {
            if *state == RunState::Paused {
                *state = RunState::Running;
                info!("Simulation resumed");
            } else {
                warn!(state = ?state, "Resume received but not paused");
            }
            false
        }
        ControlCommand::Stop => {
            info!("Stop command received");
            true
        }
    }
}

// ============================================================================
// Simulation Command Handling
// ============================================================================

/// Drain all pending simulation commands (non-blocking).
fn drain_commands(
    cmd_rx: &cbc::Receiver<SimCommand>,
    runtime: &mut Runtime,
    event_tx: &Option<broadcast::Sender<WorldEvent>>,
    field_snapshot: &Arc<RwLock<FieldSnapshot>>,
) {
    while let Ok(cmd) = cmd_rx.try_recv() {
        handle_sim_cmd(cmd, runtime, event_tx, field_snapshot);
    }
}

/// Handle a single simulation command by dispatching to the runtime
/// and sending the reply.
fn handle_sim_cmd(
    cmd: SimCommand,
    runtime: &mut Runtime,
    event_tx: &Option<broadcast::Sender<WorldEvent>>,
    field_snapshot: &Arc<RwLock<FieldSnapshot>>,
) {
    match cmd {
        SimCommand::GetStatus { reply } => {
            let status = SimStatus {
                tick: runtime.tick(),
                sim_time: runtime.sim_time(),
                era: runtime.era().clone(),
            };
            let _ = reply.send(status);
        }

        SimCommand::GetSignal { id, reply } => {
            let value = runtime.get_signal(&id);
            let _ = reply.send(value);
        }

        SimCommand::ExecuteSteps { count, reply } => {
            let mut last_ctx = None;
            let mut error = None;

            for _ in 0..count {
                match runtime.execute_tick() {
                    Ok(ctx) => {
                        drain_fields_to_snapshot(runtime, field_snapshot);
                        emit_tick_event(event_tx, &ctx, runtime, "stopped");
                        last_ctx = Some(ctx);
                    }
                    Err(e) => {
                        error = Some(e.to_string());
                        break;
                    }
                }
            }

            let result = match error {
                Some(e) => Err(e),
                None => last_ctx.ok_or_else(|| "no ticks executed".to_string()),
            };
            let _ = reply.send(result);
        }

        SimCommand::InjectImpulse { id, value, reply } => {
            let result = runtime
                .inject_impulse_by_id(&id, value)
                .map_err(|e| e.to_string());
            let _ = reply.send(result);
        }

        SimCommand::RequestCheckpoint { path, reply } => {
            let result = runtime.request_checkpoint(&path).map_err(|e| e.to_string());
            let _ = reply.send(result);
        }

        SimCommand::RestoreCheckpoint { checkpoint, reply } => {
            let result = runtime
                .restore_from_checkpoint(*checkpoint)
                .map(|()| RestoreResult {
                    tick: runtime.tick(),
                    sim_time: runtime.sim_time(),
                    era: runtime.era().clone(),
                })
                .map_err(|e| e.to_string());
            let _ = reply.send(result);
        }

        SimCommand::GetAssertionFailures { reply } => {
            let failures = runtime.assertion_checker().failures().to_vec();
            let _ = reply.send(failures);
        }
    }
}

// ============================================================================
// Field Sample Drain
// ============================================================================

/// Drain field samples from the runtime and store them in the shared snapshot.
///
/// Called after every successful `execute_tick()`. The snapshot is overwritten
/// each tick — it holds only the latest sample data. This is intentional:
/// field samples are per-tick measurements, not accumulated history.
fn drain_fields_to_snapshot(runtime: &mut Runtime, field_snapshot: &Arc<RwLock<FieldSnapshot>>) {
    let samples = runtime.drain_fields();
    if samples.is_empty() {
        return;
    }

    let snapshot = FieldSnapshot {
        tick: runtime.tick(),
        sim_time: runtime.sim_time(),
        samples,
    };

    match field_snapshot.write() {
        Ok(mut guard) => *guard = snapshot,
        Err(poisoned) => {
            // Recover from poisoned lock — snapshot data is non-critical
            warn!("Field snapshot lock was poisoned, recovering");
            *poisoned.into_inner() = snapshot;
        }
    }
}

// ============================================================================
// Event Emission
// ============================================================================

/// Emit a tick event to connected inspector clients.
///
/// Constructs a `WorldEvent { kind: "tick" }` from the tick context and
/// runtime state, and sends it via the broadcast channel. Receivers that
/// have lagged or disconnected are silently ignored — tick events are
/// best-effort delivery for UI updates.
fn emit_tick_event(
    event_tx: &Option<broadcast::Sender<WorldEvent>>,
    tick_ctx: &continuum_runtime::TickContext,
    runtime: &Runtime,
    execution_state: &str,
) {
    let Some(tx) = event_tx else { return };

    let event = WorldEvent {
        kind: "tick".to_string(),
        payload: serde_json::json!({
            "tick": runtime.tick(),
            "sim_time": tick_ctx.sim_time,
            "era": tick_ctx.era.to_string(),
            "execution_state": execution_state,
            "warmup_complete": runtime.is_warmup_complete(),
        }),
    };
    // Note: field_count is not included here because drain_fields() already
    // cleared the buffer before this call. The frontend can query field.samples
    // for actual data.

    // Best-effort: ignore send errors (no receivers, lagged, etc.)
    let _ = tx.send(event);
}

// ============================================================================
// Helpers
// ============================================================================

/// Sleep for `duration` but wake up every ~10ms to check the control channel.
///
/// This ensures we respond to pause/stop within 10ms even during tick throttling.
///
/// Returns `true` if a `Stop` command was received and the thread should exit.
fn sleep_with_control_check(
    duration: Duration,
    control_rx: &cbc::Receiver<ControlCommand>,
    state: &mut RunState,
    tick_rate: &mut u32,
) -> bool {
    let deadline = Instant::now() + duration;

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }

        let sleep_chunk = remaining.min(Duration::from_millis(10));

        match control_rx.recv_timeout(sleep_chunk) {
            Ok(cmd) => {
                if handle_control_cmd(cmd, state, tick_rate) {
                    // Stop received — caller must break out of main loop
                    return true;
                }
                if *state != RunState::Running {
                    return false;
                }
            }
            Err(cbc::RecvTimeoutError::Timeout) => {}
            Err(cbc::RecvTimeoutError::Disconnected) => {
                info!("Control channel disconnected during throttle sleep");
                *state = RunState::Idle;
                return false;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_runtime::build_runtime;
    use indexmap::IndexMap;
    use std::path::PathBuf;

    /// Create a minimal runtime for testing via CDSL compilation.
    fn test_runtime() -> Runtime {
        let mut sources = IndexMap::new();
        sources.insert(
            PathBuf::from("world.cdsl"),
            r#"
            world test {
            }

            strata sim {
                : stride(1)
            }

            era main {
                : initial
                : dt(1.0<s>)
                strata { sim: active }
            }

            signal counter {
                : Scalar
                : strata(sim)
                resolve { 1.0 }
            }
            "#
            .to_string(),
        );

        let compiled =
            continuum_cdsl::compile_from_memory(sources).expect("test world compilation failed");
        build_runtime(compiled, None)
    }

    #[test]
    fn test_spawn_and_stop() {
        let runtime = test_runtime();
        let initial_tick = runtime.tick();

        let sim = SimThread::spawn(runtime, None);
        let proxy = sim.proxy().clone();

        // Query status while idle
        let status = proxy.status().expect("status query failed");
        assert_eq!(status.tick, initial_tick);

        // Stop and reclaim runtime
        let runtime = sim.stop_and_join().expect("stop failed");
        assert_eq!(runtime.tick(), initial_tick);
    }

    #[test]
    fn test_execute_steps() {
        let runtime = test_runtime();
        let sim = SimThread::spawn(runtime, None);
        let proxy = sim.proxy().clone();

        // Execute 5 ticks via command
        let result = proxy.execute_steps(5).expect("execute_steps failed");
        let tick_ctx = result.expect("tick execution failed");
        // TickContext captures the tick value at Configure time (pre-increment),
        // so after 5 ticks the last context has tick=4 while runtime.tick()=5.
        assert_eq!(tick_ctx.tick, 4);

        // Verify runtime state via status (post-increment tick)
        let status = proxy.status().expect("status query failed");
        assert_eq!(status.tick, 5);

        let runtime = sim.stop_and_join().expect("stop failed");
        assert_eq!(runtime.tick(), 5);
    }

    #[test]
    fn test_run_pause_resume_stop() {
        let runtime = test_runtime();
        let sim = SimThread::spawn(runtime, None);
        let proxy = sim.proxy().clone();

        // Start running at 1000 ticks/sec
        proxy.run(1000).expect("run failed");

        // Let it tick for a bit
        std::thread::sleep(Duration::from_millis(50));

        // Pause
        proxy.pause().expect("pause failed");
        std::thread::sleep(Duration::from_millis(20));

        // Check that ticks advanced
        let status1 = proxy.status().expect("status failed");
        assert!(status1.tick > 0, "expected some ticks after running");

        // Let it sit paused and verify tick doesn't advance
        std::thread::sleep(Duration::from_millis(50));
        let status2 = proxy.status().expect("status failed");
        assert_eq!(
            status1.tick, status2.tick,
            "tick should not advance while paused"
        );

        // Resume
        proxy.resume().expect("resume failed");
        std::thread::sleep(Duration::from_millis(50));

        let status3 = proxy.status().expect("status failed");
        assert!(
            status3.tick > status2.tick,
            "tick should advance after resume"
        );

        // Stop
        let _runtime = sim.stop_and_join().expect("stop failed");
    }

    #[test]
    fn test_tick_events_emitted_on_steps() {
        let runtime = test_runtime();
        let (event_tx, mut event_rx) = broadcast::channel(100);

        let sim = SimThread::spawn(runtime, Some(event_tx));
        let proxy = sim.proxy().clone();

        // Execute 3 ticks — should emit 3 tick events
        let result = proxy.execute_steps(3).expect("execute_steps failed");
        assert!(result.is_ok(), "tick execution failed");

        // Drain events — we should have received 3
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        assert_eq!(
            events.len(),
            3,
            "expected 3 tick events, got {}",
            events.len()
        );

        // Verify event structure
        for event in &events {
            assert_eq!(event.kind, "tick");
            assert!(event.payload.get("tick").is_some());
            assert!(event.payload.get("sim_time").is_some());
            assert!(event.payload.get("era").is_some());
            assert_eq!(
                event
                    .payload
                    .get("execution_state")
                    .and_then(|v| v.as_str()),
                Some("stopped")
            );
        }

        let _runtime = sim.stop_and_join().expect("stop failed");
    }

    #[test]
    fn test_tick_events_emitted_on_run() {
        let runtime = test_runtime();
        let (event_tx, mut event_rx) = broadcast::channel(1000);

        let sim = SimThread::spawn(runtime, Some(event_tx));
        let proxy = sim.proxy().clone();

        // Start running at high rate
        proxy.run(1000).expect("run failed");
        std::thread::sleep(Duration::from_millis(50));

        // Pause to stop ticking
        proxy.pause().expect("pause failed");
        std::thread::sleep(Duration::from_millis(20));

        // Should have received tick events during the run
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        assert!(
            !events.is_empty(),
            "expected tick events during continuous run"
        );

        // All events should report "running" execution state
        for event in &events {
            assert_eq!(event.kind, "tick");
            assert_eq!(
                event
                    .payload
                    .get("execution_state")
                    .and_then(|v| v.as_str()),
                Some("running")
            );
        }

        let _runtime = sim.stop_and_join().expect("stop failed");
    }

    #[test]
    fn test_drop_stops_thread() {
        let runtime = test_runtime();
        let sim = SimThread::spawn(runtime, None);
        let proxy = sim.proxy().clone();

        proxy.run(100).expect("run failed");
        std::thread::sleep(Duration::from_millis(50));

        let status = proxy.status().expect("status should work");
        assert!(status.tick > 0, "expected ticks after running");

        // Stop and join — thread should exit cleanly
        let runtime = sim.stop_and_join().expect("stop failed");
        assert!(runtime.tick() > 0);

        // Proxy should now report disconnected
        assert!(proxy.status().is_err());
    }
}
