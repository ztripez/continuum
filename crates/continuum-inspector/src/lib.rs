//! Continuum Inspector library - exports modules for testing.

pub mod helpers;
pub mod process;
pub mod spawner;
pub mod state;
pub mod websocket;

// Re-export for convenience
pub use spawner::ProcessSpawner;
pub use state::AppState;
