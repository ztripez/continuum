//! GPU compute offload for Continuum field emission.
//!
//! This crate provides GPU-accelerated field emission for the Measure phase.
//! Fields are ideal candidates for GPU acceleration because they are:
//!
//! - **Non-causal**: No determinism constraints (relaxed mode OK)
//! - **Embarrassingly parallel**: Each field sample is independent
//! - **Large data**: Spatial fields can have 1M+ samples
//! - **Read-only inputs**: Only reads resolved signals
//!
//! # Architecture
//!
//! The GPU pipeline consists of:
//!
//! 1. **Signal Upload**: Resolved signal values (f64) are converted to f32
//!    and uploaded to GPU storage buffers.
//!
//! 2. **Compute Dispatch**: WGSL compute shaders process field samples in
//!    parallel. Shaders are generated from DSL field definitions via Naga.
//!
//! 3. **Result Download**: Field samples are read back to CPU and converted
//!    to [`FieldBuffer`] format for observer consumption.
//!
//! # Precision
//!
//! GPU compute uses f32 precision, which is sufficient for observation and
//! visualization. This is acceptable because fields are non-causal and don't
//! affect simulation determinism.
//!
//! # CPU Fallback
//!
//! When GPU is unavailable (no compatible adapter or feature disabled),
//! the system falls back to CPU-based field emission transparently.

#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod shaders;

mod emitter;
mod error;

#[cfg(feature = "gpu")]
pub use context::GpuContext;
#[cfg(feature = "gpu")]
pub use pipeline::GpuFieldPipeline;

pub use emitter::{FieldEmitter, FieldEmitterConfig};
pub use error::GpuError;
