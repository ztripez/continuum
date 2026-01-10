//! Continuum Runtime.
//!
//! This crate provides the execution engine for Continuum simulations.
//! It takes compiled IR from `continuum_ir` and executes simulations
//! tick by tick according to the phase model.
//!
//! # Architecture
//!
//! The runtime is organized into several modules:
//!
//! - [`types`] - Core types: [`Phase`], [`Value`], [`StratumState`], [`Dt`]
//! - [`storage`] - Signal and entity storage with tick management
//! - [`soa_storage`] - SoA (Struct-of-Arrays) storage for vectorized execution
//! - [`reductions`] - Deterministic reduction operations for entity aggregates
//! - [`vectorized`] - Unified vectorized primitive abstraction
//! - [`executor`] - Phase executors and the main [`Runtime`] type
//! - [`dag`] - Execution graph construction and scheduling
//! - [`error`] - Error types for runtime failures
//!
//! # Execution Model
//!
//! Each simulation tick proceeds through five phases in order:
//!
//! 1. **Configure** - Freeze execution context for the tick
//! 2. **Collect** - Accumulate inputs and impulse payloads
//! 3. **Resolve** - Compute new signal values from expressions
//! 4. **Fracture** - Detect tension conditions and emit responses
//! 5. **Measure** - Emit field values for observation
//!
//! # Example
//!
//! ```ignore
//! use continuum_runtime::{Runtime, EraConfig, Dt};
//!
//! let mut runtime = Runtime::new(era_configs, initial_era);
//! runtime.init_signals(world, |world, id| get_initial_value(world, id));
//!
//! for _ in 0..1000 {
//!     runtime.tick();
//! }
//! ```

pub mod dag;
pub mod error;
pub mod executor;
pub mod reductions;
pub mod soa_storage;
pub mod storage;
pub mod types;
pub mod vectorized;

pub use error::{Error, Result};
pub use executor::{
    AssertContext, AssertionChecker, AssertionFn, AssertionSeverity, CollectContext, CollectFn,
    EraConfig, FractureContext, FractureFn, ImpulseContext, ImpulseFn, MeasureContext, MeasureFn,
    PhaseExecutor, ResolveContext, ResolverFn, Runtime, TransitionFn, WarmupContext,
    WarmupExecutor, WarmupFn,
};
pub use soa_storage::{
    AlignedBuffer, MemberSignalBuffer, MemberSignalMeta, MemberSignalRegistry, PopulationStorage,
    TypedBuffer, ValueType, SIMD_ALIGNMENT,
};
pub use types::*;
pub use vectorized::{
    Cardinality, EntityIndex, FieldPrimitive, FieldSampleIdentity, FractureIdentity,
    FracturePrimitive, GlobalSignal, IndexSpace, MemberSignal, MemberSignalId,
    MemberSignalIdentity, SampleIndex, VectorizedPrimitive,
};
