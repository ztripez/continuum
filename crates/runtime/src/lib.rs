//! Continuum Runtime
//!
//! Executes DAGs and advances simulation time.

pub mod dag;
pub mod error;
pub mod executor;
pub mod operators;
pub mod storage;
pub mod types;

pub use error::{Error, Result};
pub use types::*;
