// Allow unwrap in tests
#![cfg_attr(test, allow(clippy::unwrap_used))]

//! Continuum tooling support library.
//!
//! Provides the IPC server, request routing, and world execution intent used
//! by the CLI runner and the inspector to drive and observe simulations.

pub mod ipc_server;
pub mod ipc_types;
pub(crate) mod request_handler_impls;
pub mod request_handlers;
pub mod run_world_intent;
pub mod sim_proxy;
pub mod sim_thread;
pub mod world_api;

// Re-export renamed type
pub use ipc_server::SimulationController;
