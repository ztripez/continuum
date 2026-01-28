//! Process management tests for Inspector.
//!
//! Tests critical process lifecycle: spawn/kill, socket creation, timeout handling.
//! Uses `MockProcessSpawner` for unit tests, avoiding dependency on real `continuum-run`.

use continuum_inspector::helpers::wait_for_condition;
use continuum_inspector::process::{kill_simulation, spawn_simulation};
use continuum_inspector::spawner::mock::{MockBehavior, MockProcessSpawner};
use continuum_inspector::state::AppState;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::Mutex;

fn create_state(socket_path: PathBuf) -> AppState {
    AppState {
        socket: socket_path,
        child: Arc::new(Mutex::new(None)),
        current_world: Arc::new(Mutex::new(None)),
    }
}

#[tokio::test]
async fn test_wait_for_condition_success() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = flag.clone();

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        flag_clone.store(true, Ordering::SeqCst);
    });

    let result = wait_for_condition(move || flag.load(Ordering::SeqCst), 1000, "Flag not set")
        .await;

    assert!(result.is_ok(), "Condition should be met");
}

#[tokio::test]
async fn test_wait_for_condition_timeout() {
    let result = wait_for_condition(|| false, 300, "Expected timeout").await;

    assert!(result.is_err(), "Should timeout");
    assert_eq!(result.unwrap_err(), "Expected timeout");
}

#[tokio::test]
async fn test_double_kill_safety() {
    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let state = create_state(socket_path);

    let result = kill_simulation(&state).await;
    assert!(result.is_ok(), "Kill with no child should succeed");
}

#[tokio::test]
async fn test_spawn_success_creates_socket() {
    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let world_path = PathBuf::from("/fake/world");

    let spawner = MockProcessSpawner::new(MockBehavior::Success {
        socket_delay_ms: 100,
    });
    let state = create_state(socket_path.clone());

    let result = spawn_simulation(&spawner, &state, world_path.clone(), None).await;

    assert!(result.is_ok(), "Spawn should succeed");
    assert!(socket_path.exists(), "Socket should be created");

    let spawned = spawner.spawned_processes();
    assert_eq!(spawned.len(), 1);
    assert_eq!(spawned[0].0, world_path);
    assert_eq!(spawned[0].1, socket_path);
    assert_eq!(spawned[0].2, None);

    kill_simulation(&state)
        .await
        .expect("Cleanup kill should succeed after spawn");
}

#[tokio::test]
async fn test_spawn_timeout_kills_process() {
    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let world_path = PathBuf::from("/fake/world");

    let spawner = MockProcessSpawner::new(MockBehavior::NeverCreatesSocket);
    let state = create_state(socket_path.clone());

    let result = spawn_simulation(&spawner, &state, world_path, None).await;

    assert!(result.is_err(), "Should timeout");
    assert!(
        result.unwrap_err().contains("Timeout"),
        "Error should mention timeout"
    );

    let child_guard = state.child.lock().await;
    assert!(child_guard.is_none(), "Process should be killed after timeout");
}

#[tokio::test]
async fn test_spawn_with_scenario() {
    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let world_path = PathBuf::from("/fake/world");
    let scenario = Some("test-scenario".to_string());

    let spawner = MockProcessSpawner::new(MockBehavior::Success {
        socket_delay_ms: 100,
    });
    let state = create_state(socket_path.clone());

    let result = spawn_simulation(&spawner, &state, world_path.clone(), scenario.clone()).await;

    assert!(result.is_ok(), "Spawn with scenario should succeed");

    let spawned = spawner.spawned_processes();
    assert_eq!(spawned[0].2, scenario);

    kill_simulation(&state)
        .await
        .expect("Cleanup kill should succeed after spawn");
}

#[tokio::test]
async fn test_kill_removes_socket() {
    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let world_path = PathBuf::from("/fake/world");

    let spawner = MockProcessSpawner::new(MockBehavior::Success {
        socket_delay_ms: 100,
    });
    let state = create_state(socket_path.clone());

    spawn_simulation(&spawner, &state, world_path, None)
        .await
        .expect("spawn_simulation should succeed");
    assert!(socket_path.exists(), "Socket should exist after spawn");

    let result = kill_simulation(&state).await;
    assert!(result.is_ok(), "Kill should succeed");
    assert!(
        !socket_path.exists(),
        "Socket should be removed after kill"
    );
}

#[tokio::test]
async fn test_spawn_fails_propagates_error() {
    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let world_path = PathBuf::from("/fake/world");

    let spawner = MockProcessSpawner::new(MockBehavior::SpawnFails {
        error: "Mock spawn failure".to_string(),
    });
    let state = create_state(socket_path);

    let result = spawn_simulation(&spawner, &state, world_path, None).await;

    assert!(result.is_err(), "Should fail");
    assert_eq!(result.unwrap_err(), "Mock spawn failure");
}

#[tokio::test]
#[cfg(unix)]
#[ignore] // Requires root or specific filesystem for permission tests
async fn test_socket_cleanup_failure_propagates() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new().expect("TempDir should create test directory");
    let socket_path = temp_dir.path().join("test.sock");
    let world_path = PathBuf::from("/fake/world");

    let spawner = MockProcessSpawner::new(MockBehavior::Success {
        socket_delay_ms: 100,
    });
    let state = create_state(socket_path.clone());

    spawn_simulation(&spawner, &state, world_path, None)
        .await
        .expect("spawn_simulation should succeed");

    let parent = socket_path
        .parent()
        .expect("Socket path should have parent directory");
    let original_perms = std::fs::metadata(parent)
        .expect("Parent directory metadata should be readable")
        .permissions();
    let mut perms = original_perms.clone();
    perms.set_mode(0o444);
    std::fs::set_permissions(parent, perms)
        .expect("set_permissions should succeed for test setup");

    let result = kill_simulation(&state).await;

    std::fs::set_permissions(parent, original_perms)
        .expect("set_permissions should restore permissions");

    assert!(result.is_err(), "Should fail on socket removal");
    assert!(
        result.unwrap_err().contains("Failed to remove socket"),
        "Error should mention socket removal"
    );
}
