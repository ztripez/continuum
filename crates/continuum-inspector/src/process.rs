//! Simulation process management: spawning and terminating `continuum-run`.

use crate::helpers::wait_for_condition;
use crate::spawner::ProcessSpawner;
use crate::state::AppState;
use std::path::PathBuf;
use tracing::{error, info, warn};

/// Spawns a new `continuum-run` process for the given world and waits for socket readiness.
///
/// # Parameters
/// - `spawner`: Process spawning strategy (real or mock)
/// - `state`: Shared application state (holds child process handle and socket path)
/// - `world_path`: Path to world directory or `.cvm` bundle to load
/// - `scenario`: Optional scenario name to pass via `--scenario` flag
///
/// # Returns
/// - `Ok(())` if process spawned successfully and socket created within 5 seconds
/// - `Err(String)` with error message if spawn fails or socket timeout occurs
///
/// # Process
/// 1. Removes old socket file if present
/// 2. Uses `spawner` trait to spawn process
/// 3. Stores `Child` handle in `state.child`
/// 4. Waits up to 5 seconds for socket file to appear
/// 5. Kills process and returns error if socket doesn't appear (startup failure)
///
/// # Errors
/// Returns error string if:
/// - Old socket removal fails (filesystem error)
/// - Process spawn fails (`continuum-run` not in PATH or execution error)
/// - Socket doesn't appear within 5 seconds (process crashed or misconfigured)
///
/// # Notes
/// - Socket timeout indicates process startup failure; check `continuum-run` logs
/// - Process is killed automatically if socket timeout occurs (cleanup safety)
/// - Use `RealProcessSpawner` for production, `MockProcessSpawner` for tests
pub async fn spawn_simulation<S: ProcessSpawner>(
    spawner: &S,
    state: &AppState,
    world_path: PathBuf,
    scenario: Option<String>,
) -> Result<(), String> {
    info!("Spawning continuum-run for world: {}", world_path.display());

    // Remove old socket file if it exists
    if state.socket.exists() {
        std::fs::remove_file(&state.socket)
            .map_err(|e| format!("Failed to remove old socket: {e}"))?;
    }

    // Spawn process using injected spawner
    let child = spawner.spawn_continuum_run(
        &world_path,
        &state.socket,
        scenario.as_deref(),
    )?;

    *state.child.lock().await = Some(child);

    // Wait for socket to exist (5 second timeout)
    if let Err(_) = wait_for_condition(|| state.socket.exists(), 5000, "timeout").await {
        // Kill the process since it failed to create socket
        if let Err(e) = kill_simulation(state).await {
            error!("Failed to kill unresponsive process: {e}");
        }
        return Err(format!(
            "Timeout waiting for continuum-run to create socket at {}. \
             Process may have failed to start - check process logs.",
            state.socket.display()
        ));
    }

    info!("Socket ready: {}", state.socket.display());
    Ok(())
}

/// Gracefully terminates the running simulation process and cleans up IPC socket.
///
/// # Parameters
/// - `state`: Shared application state containing child process handle and socket path
///
/// # Returns
/// - `Ok(())` if process killed and socket removed, or no process was running
/// - `Err(String)` with error message if kill or socket removal fails
///
/// # Process
/// 1. Takes `Child` handle from `state.child` (leaves `None` in its place)
/// 2. Sends SIGTERM via `child.kill().await` (despite name, this is graceful on Unix)
/// 3. Waits up to 5 seconds for process to exit via `tokio::time::timeout`
/// 4. Returns error if process doesn't exit or wait fails
/// 5. Removes Unix socket file if present
///
/// # Errors
/// Returns error string if:
/// - `child.kill()` fails (process already dead is NOT an error)
/// - Process doesn't exit within 5 seconds (may be zombie)
/// - Process wait fails (system error)
/// - Socket removal fails (filesystem permission error)
///
/// # Notes
/// - If no process is running (`child` is `None`), returns `Ok(())` immediately
/// - Process kill timeout returns error (zombie process warning)
/// - Always attempts socket cleanup even if kill fails
pub async fn kill_simulation(state: &AppState) -> Result<(), String> {
    let mut child_guard = state.child.lock().await;

    if let Some(mut child) = child_guard.take() {
        info!("Killing continuum-run process");

        // Try graceful shutdown first (SIGTERM)
        child
            .kill()
            .await
            .map_err(|e| format!("Failed to kill process: {e}"))?;

        // Wait for process to exit (with timeout)
        match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
            Ok(Ok(status)) => {
                info!("Process exited with status: {status}");
                if !status.success() {
                    warn!("Process exited with non-zero status: {status}");
                }
            }
            Ok(Err(err)) => {
                return Err(format!("Error waiting for process to exit: {err}"));
            }
            Err(_) => {
                error!("Process did not exit within 5 seconds after SIGTERM - may be zombie");
                return Err("Timeout waiting for process to exit - process may be stuck".to_string());
            }
        }

        // Remove socket file
        if state.socket.exists() {
            std::fs::remove_file(&state.socket)
                .map_err(|e| format!("Failed to remove socket: {e}"))?;
        }

        Ok(())
    } else {
        Ok(()) // No process to kill
    }
}
