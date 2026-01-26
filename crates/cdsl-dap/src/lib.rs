//! CDSL Debug Adapter Protocol (DAP) Implementation.
//!
//! Provides interactive debugging support for the Continuum DSL.

pub mod adapter;
pub mod server;

use continuum_cdsl::ast::World;
use continuum_runtime::Runtime;

/// High-level debug session state.
pub struct DebugSession {
    pub world: World,
    pub runtime: Runtime,
    pub breakpoints: std::collections::HashSet<usize>, // Byte offsets or line numbers?
}
