//! Shared application state and request/response types for the Inspector API.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::Mutex;

/// Shared application state for the Inspector server.
///
/// Holds the Unix socket path for IPC, a reference to the spawned `continuum-run`
/// child process, and the currently loaded world path. All mutable state is
/// protected by `Arc<Mutex<...>>` for safe concurrent access across async tasks.
pub struct AppState {
    /// Path to Unix socket used for IPC with the simulation process
    pub socket: PathBuf,
    /// Handle to the spawned `continuum-run` child process, if running
    pub child: Arc<Mutex<Option<Child>>>,
    /// Path to the currently loaded world directory or .cvm bundle
    pub current_world: Arc<Mutex<Option<PathBuf>>>,
}

/// Clones `AppState` by cloning `Arc` pointers, not the underlying data.
///
/// This allows multiple Axum handlers to share the same mutex-guarded state
/// without deep-copying the `Child` or `current_world`. The `socket` path
/// is cloned as an owned `PathBuf`.
impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            socket: self.socket.clone(),
            child: Arc::clone(&self.child),
            current_world: Arc::clone(&self.current_world),
        }
    }
}

/// HTTP request payload for `/api/sim/load` endpoint.
///
/// Specifies the world to load and optionally a scenario name. The inspector will
/// kill any existing simulation, spawn `continuum-run` with the provided world,
/// and establish IPC via Unix socket.
#[derive(Debug, Deserialize)]
pub struct LoadSimulationRequest {
    /// Absolute or relative path to a world directory or `.cvm` bundle
    pub world_path: PathBuf,
    /// Optional scenario name to pass to `continuum-run --scenario`
    pub scenario: Option<String>,
}

/// Standard JSON response structure for simulation control API endpoints.
///
/// Returned by `/api/sim/load`, `/api/sim/stop`, and `/api/sim/restart`.
/// Indicates success/failure and provides a human-readable message for logging
/// or display in the frontend.
#[derive(Debug, Serialize)]
pub struct ApiResponse {
    /// Whether the requested operation succeeded
    pub success: bool,
    /// Human-readable status or error message
    pub message: String,
}

/// HTTP response payload for `/api/sim/status` endpoint.
///
/// Reports whether a simulation is currently running and the path to the loaded world.
/// Used by the frontend to synchronize UI state (e.g., enable/disable Load/Stop buttons).
#[derive(Debug, Serialize)]
pub struct SimulationStatusResponse {
    /// True if `continuum-run` process is spawned and the socket exists
    pub running: bool,
    /// Path to the currently loaded world, if any
    pub world_path: Option<PathBuf>,
}
