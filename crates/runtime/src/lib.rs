//! Continuum Runtime
//!
//! Executes DAGs and advances simulation time.

pub mod types;
pub mod storage;
pub mod dag;
pub mod executor;
pub mod error;

pub use error::{Error, Result};
pub use types::*;
