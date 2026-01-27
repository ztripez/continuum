//! REST API handlers for simulation control endpoints.

use crate::helpers::{into_api_response, wait_for_condition};
use crate::process::{kill_simulation, spawn_simulation};
use crate::spawner::RealProcessSpawner;
use crate::state::{ApiResponse, AppState, LoadSimulationRequest, SimulationStatusResponse};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use tracing::{error, info};

/// POST `/api/sim/load` - Loads a world and spawns `continuum-run` process.
///
/// # Request Body
/// ```json
/// {
///   "world_path": "/path/to/world",
///   "scenario": "optional-scenario-name"
/// }
/// ```
///
/// # Response
/// - `200 OK` with success message if world loaded successfully
/// - `500 INTERNAL_SERVER_ERROR` with error message if load fails
///
/// # Process
/// 1. Kills any existing simulation process
/// 2. Waits for socket cleanup (non-fatal if timeout)
/// 3. Spawns new `continuum-run` process for the specified world
/// 4. Updates `current_world` state on success
pub async fn load_simulation_handler(
    State(state): State<AppState>,
    Json(payload): Json<LoadSimulationRequest>,
) -> impl IntoResponse {
    info!("Load simulation request: {:?}", payload.world_path);

    // Kill existing simulation if running
    if let Err(err) = kill_simulation(&state).await {
        error!("Failed to kill existing simulation: {err}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Failed to stop existing simulation: {err}"),
            }),
        );
    }

    // Wait for socket to be removed
    wait_for_condition(
        || !state.socket.exists(),
        1000,
        "Socket file not removed after kill (1s timeout)",
    )
    .await
    .ok(); // Non-fatal if socket lingers

    // Spawn new simulation
    let world_path = payload.world_path.clone();
    let spawner = RealProcessSpawner;
    let result = spawn_simulation(&spawner, &state, world_path.clone(), payload.scenario).await;
    
    if result.is_ok() {
        *state.current_world.lock().await = Some(world_path.clone());
    }
    
    into_api_response(result, |_| format!("Loaded world: {}", world_path.display()))
}

/// POST `/api/sim/stop` - Stops the running simulation process.
///
/// # Response
/// - `200 OK` with success message if simulation stopped
/// - `500 INTERNAL_SERVER_ERROR` with error message if stop fails
///
/// # Process
/// 1. Sends SIGTERM to `continuum-run` process
/// 2. Waits for process exit (5 second timeout)
/// 3. Removes Unix socket file
/// 4. Clears `current_world` state on success
pub async fn stop_simulation_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("Stop simulation request");

    let result = kill_simulation(&state).await;
    if result.is_ok() {
        *state.current_world.lock().await = None;
    }

    into_api_response(result, |_| "Simulation stopped".to_string())
}

/// POST `/api/sim/restart` - Restarts the currently loaded simulation.
///
/// # Response
/// - `200 OK` with success message if simulation restarted
/// - `400 BAD_REQUEST` if no world is currently loaded
/// - `500 INTERNAL_SERVER_ERROR` with error message if restart fails
///
/// # Process
/// 1. Checks if a world is currently loaded (returns 400 if not)
/// 2. Kills existing simulation process
/// 3. Waits for socket cleanup (non-fatal if timeout)
/// 4. Respawns `continuum-run` with the same world path (no scenario)
pub async fn restart_simulation_handler(State(state): State<AppState>) -> impl IntoResponse {
    info!("Restart simulation request");

    let world_path = match state.current_world.lock().await.clone() {
        Some(path) => path,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse {
                    success: false,
                    message: "No world currently loaded".to_string(),
                }),
            )
        }
    };

    // Kill existing
    if let Err(err) = kill_simulation(&state).await {
        error!("Failed to kill simulation during restart: {err}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Failed to stop simulation: {err}"),
            }),
        );
    }

    // Wait for socket cleanup
    wait_for_condition(
        || !state.socket.exists(),
        1000,
        "Socket file not removed after kill (1s timeout)",
    )
    .await
    .ok(); // Non-fatal if socket lingers

    // Respawn
    let spawner = RealProcessSpawner;
    let result = spawn_simulation(&spawner, &state, world_path, None).await;
    into_api_response(result, |_| "Simulation restarted".to_string())
}

/// GET `/api/sim/status` - Returns current simulation status.
///
/// # Response
/// ```json
/// {
///   "running": true,
///   "world_path": "/path/to/world"
/// }
/// ```
///
/// Reports whether a simulation is currently running and the path to the loaded world.
/// Used by the frontend to synchronize UI state (enable/disable buttons, show status).
pub async fn simulation_status_handler(State(state): State<AppState>) -> impl IntoResponse {
    let child_guard = state.child.lock().await;
    let running = child_guard.is_some();
    let world_path = state.current_world.lock().await.clone();

    (
        StatusCode::OK,
        Json(SimulationStatusResponse {
            running,
            world_path,
        }),
    )
}
