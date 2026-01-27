//! Process spawning abstraction for testability.
//!
//! Provides a trait-based interface for spawning child processes, enabling
//! dependency injection and mocking in tests without modifying production code.

use std::path::PathBuf;
use tokio::process::{Child, Command};

/// Trait for spawning child processes.
///
/// Abstracts `tokio::process::Command` to enable mocking in tests.
/// Production code uses `RealProcessSpawner`, tests use `MockProcessSpawner`.
pub trait ProcessSpawner: Send + Sync {
    /// Spawns `continuum-run` with the given world path and socket.
    ///
    /// # Parameters
    /// - `world_path`: Path to world directory or `.cvm` bundle
    /// - `socket_path`: Path to Unix socket for IPC
    /// - `scenario`: Optional scenario name to pass via `--scenario`
    ///
    /// # Returns
    /// - `Ok(Child)` with spawned process handle
    /// - `Err(String)` with error message if spawn fails
    fn spawn_continuum_run(
        &self,
        world_path: &PathBuf,
        socket_path: &PathBuf,
        scenario: Option<&str>,
    ) -> Result<Child, String>;
}

/// Production implementation that spawns real `continuum-run` processes.
///
/// Wraps `tokio::process::Command` to execute the actual binary.
/// Used in production and integration tests with real processes.
pub struct RealProcessSpawner;

impl ProcessSpawner for RealProcessSpawner {
    fn spawn_continuum_run(
        &self,
        world_path: &PathBuf,
        socket_path: &PathBuf,
        scenario: Option<&str>,
    ) -> Result<Child, String> {
        let mut cmd = Command::new("continuum-run");
        cmd.arg(world_path).arg("--socket").arg(socket_path);

        if let Some(scenario_name) = scenario {
            cmd.arg("--scenario").arg(scenario_name);
        }

        cmd.spawn()
            .map_err(|e| format!("Failed to spawn continuum-run: {e}"))
    }
}

pub mod mock {
    //! Mock process spawner for testing.
    //!
    //! Available for integration tests and external test crates.

    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{self, AsyncWriteExt};

    /// Behavior specification for mock spawned processes.
    #[derive(Debug, Clone)]
    pub enum MockBehavior {
        /// Process spawns successfully and creates socket after delay
        Success { socket_delay_ms: u64 },
        /// Process spawns but never creates socket (timeout scenario)
        NeverCreatesSocket,
        /// Process spawn fails immediately
        SpawnFails { error: String },
        /// Process spawns but ignores SIGTERM (zombie scenario)
        IgnoresSigterm,
    }

    /// Mock process spawner for testing without real processes.
    ///
    /// Allows tests to control spawn behavior, socket creation timing,
    /// and process lifecycle without executing `continuum-run`.
    pub struct MockProcessSpawner {
        behavior: Arc<Mutex<MockBehavior>>,
        spawned_processes: Arc<Mutex<Vec<SpawnedProcess>>>,
    }

    #[derive(Debug, Clone)]
    struct SpawnedProcess {
        world_path: PathBuf,
        socket_path: PathBuf,
        scenario: Option<String>,
    }

    impl MockProcessSpawner {
        /// Creates a new mock spawner with the specified behavior.
        pub fn new(behavior: MockBehavior) -> Self {
            Self {
                behavior: Arc::new(Mutex::new(behavior)),
                spawned_processes: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// Returns list of processes that were spawned (for verification).
        pub fn spawned_processes(&self) -> Vec<(PathBuf, PathBuf, Option<String>)> {
            self.spawned_processes
                .lock()
                .expect("MockProcessSpawner spawned_processes mutex poisoned")
                .iter()
                .map(|p| {
                    (
                        p.world_path.clone(),
                        p.socket_path.clone(),
                        p.scenario.clone(),
                    )
                })
                .collect()
        }

        /// Changes the mock behavior (for multi-stage tests).
        pub fn set_behavior(&self, behavior: MockBehavior) {
            *self.behavior.lock().expect("MockProcessSpawner behavior mutex poisoned") = behavior;
        }
    }

    impl ProcessSpawner for MockProcessSpawner {
        fn spawn_continuum_run(
            &self,
            world_path: &PathBuf,
            socket_path: &PathBuf,
            scenario: Option<&str>,
        ) -> Result<Child, String> {
            // Record spawn attempt
            self.spawned_processes
                .lock()
                .expect("MockProcessSpawner spawned_processes mutex poisoned")
                .push(SpawnedProcess {
                    world_path: world_path.clone(),
                    socket_path: socket_path.clone(),
                    scenario: scenario.map(|s| s.to_string()),
                });

            let behavior = self
                .behavior
                .lock()
                .expect("MockProcessSpawner behavior mutex poisoned")
                .clone();

            match behavior {
                MockBehavior::SpawnFails { error } => Err(error),
                MockBehavior::Success { socket_delay_ms } => {
                    // Spawn mock process that creates socket after delay
                    let socket_path = socket_path.clone();
                    Ok(spawn_mock_process(move || async move {
                        tokio::time::sleep(tokio::time::Duration::from_millis(socket_delay_ms))
                            .await;
                        // Create socket file
                        tokio::fs::File::create(&socket_path).await.expect(
                            "MockProcessSpawner failed to create socket file - check test temp dir permissions"
                        );
                    }))
                }
                MockBehavior::NeverCreatesSocket => {
                    // Spawn process that does nothing (simulates hang)
                    Ok(spawn_mock_process(|| async {
                        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                    }))
                }
                MockBehavior::IgnoresSigterm => {
                    // Spawn process that ignores signals (zombie simulation)
                    Ok(spawn_mock_process(|| async {
                        tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                    }))
                }
            }
        }
    }

    /// Spawns a mock child process that runs the given async closure.
    ///
    /// Uses a simple echo process as a placeholder - the actual behavior
    /// is controlled by the closure which runs in a background task.
    fn spawn_mock_process<F, Fut>(behavior: F) -> Child
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        // Spawn behavior in background
        tokio::spawn(behavior());

        // Return a dummy child process (sleep command)
        // This gives us a real PID that can be killed
        Command::new("sleep")
            .arg("3600")
            .spawn()
            .unwrap_or_else(|e| {
                panic!(
                    "MockProcessSpawner: Failed to spawn sleep process for mock behavior. \
                     Error: {e}. Check system resources and PATH."
                )
            })
    }
}
