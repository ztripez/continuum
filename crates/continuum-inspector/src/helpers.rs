//! Shared utility functions for the Inspector server.

use axum::http::StatusCode;
use axum::Json;
use crate::state::ApiResponse;
use tracing::error;

/// Waits for a condition to become true, polling at 100ms intervals.
///
/// Returns `Ok(())` if the condition becomes true within the timeout period.
/// Returns `Err` with the provided error message if the timeout is exceeded.
///
/// # Parameters
/// - `condition`: Closure that returns `true` when the wait condition is satisfied
/// - `timeout_ms`: Maximum time to wait in milliseconds
/// - `error_msg`: Error message to return on timeout
pub async fn wait_for_condition<F>(
    condition: F,
    timeout_ms: u64,
    error_msg: &str,
) -> Result<(), String>
where
    F: Fn() -> bool,
{
    let iterations = timeout_ms / 100;
    for _ in 0..iterations {
        if condition() {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    Err(error_msg.to_string())
}

/// Converts a `Result` into an HTTP API response with consistent error handling.
///
/// # Parameters
/// - `result`: Operation result to convert
/// - `success_msg`: Message to return on success (supports closures for dynamic messages)
///
/// # Returns
/// - `200 OK` with `ApiResponse { success: true, message }` on `Ok`
/// - `500 INTERNAL_SERVER_ERROR` with `ApiResponse { success: false, message }` on `Err`
///
/// Errors are logged via `tracing::error!` before being returned.
pub fn into_api_response<T, F>(
    result: Result<T, String>,
    success_msg: F,
) -> (StatusCode, Json<ApiResponse>)
where
    F: FnOnce(&T) -> String,
{
    match result {
        Ok(val) => (
            StatusCode::OK,
            Json(ApiResponse {
                success: true,
                message: success_msg(&val),
            }),
        ),
        Err(err) => {
            error!("API error: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: err,
                }),
            )
        }
    }
}
