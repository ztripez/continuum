//! Effect Operations (Bare Names)
//!
//! Side-effecting operations that mutate simulation state or produce artifacts.
//! These use BARE NAMES (empty namespace) in the DSL.
//!
//! # DISABLED: Waiting for Runtime Context
//!
//! These functions are currently commented out because they require runtime
//! execution context that doesn't exist yet in the compiler rewrite.
//! All functions call unimplemented!() and use Value type directly, which
//! breaks macro expectations (expects FromValue trait).
//!
//! Will be re-enabled when:
//! 1. Runtime execution context infrastructure exists
//! 2. Macro supports variadic signatures or Value wrapper types
//!
//! # TODO: Macro Extension Required
//!
//! Once re-enabled, these should use new Rust-syntax type constraints:
//!
//! ```rust,ignore
//! use continuum_kernel_types::prelude::*;
//!
//! #[kernel_fn(
//!     namespace = "",       // BARE NAME: called as "emit(...)" not "effect.emit(...)"
//!     purity = Effect,      // NOT Pure - mutates state
//!     shape_in = [Any, Any],
//!     unit_in = [Any, Any],
//!     shape_out = Scalar,
//!     unit_out = Dimensionless
//! )]
//! pub fn emit(target: Value, value: Value) -> Value { ... }
//! ```
//!
//! # Bare Name Convention
//!
//! Effect operations are called without a namespace prefix:
//! - `emit(signal.x, computed_value)` NOT `effect.emit(...)`
//! - `spawn(EntityType, { field: value })` NOT `effect.spawn(...)`
//!
//! This emphasizes their special status as causal operations.

// Disabled pending runtime context - see file header
// use continuum_kernel_macros::kernel_fn;
// use continuum_kernel_registry::Value;

/*
DISABLED PENDING RUNTIME CONTEXT

/// Emit: `emit(target, value)`
///
/// Emits a value to a signal's input accumulator.
/// Only valid in Collect phase (impulse apply blocks, operators).
///
/// # Phase Restrictions
///
/// - **Allowed**: Collect, Fracture (Apply)
/// - **Forbidden**: Resolve, Measure (pure phases)
///
/// # Parameters
///
/// - `target`: Signal path to emit to
/// - `value`: Value to accumulate
///
/// # Returns
///
/// Unit (side effect only)
///
/// # TODO
///
/// Current implementation is a stub. Requires execution context support.
#[kernel_fn(namespace = "effect")]
pub fn emit(_target: Value, _value: Value) -> Value {
    // TODO: Requires runtime execution context
    // This is an effect operation that accumulates values into signal inputs
    // Real implementation needs access to the signal resolution context
    unimplemented!("emit requires execution context (not yet implemented in compiler rewrite)")
}

/// Spawn: `spawn(entity_type, initial_state)`
///
/// Creates a new entity instance during fracture resolution.
/// Only valid in Fracture phase.
///
/// # TODO
///
/// Entity lifecycle (spawn/destroy) not yet implemented in compiler rewrite.
#[kernel_fn(namespace = "effect")]
pub fn spawn(_entity_type: Value, _initial_state: Value) -> Value {
    unimplemented!("spawn requires entity lifecycle support (not yet implemented)")
}

/// Destroy: `destroy(entity_id)`
///
/// Marks an entity for removal at end of tick.
/// Only valid in Fracture phase.
///
/// # TODO
///
/// Entity lifecycle (spawn/destroy) not yet implemented in compiler rewrite.
#[kernel_fn(namespace = "effect")]
pub fn destroy(_entity_id: Value) -> Value {
    unimplemented!("destroy requires entity lifecycle support (not yet implemented)")
}

/// Log: `log(message, value)`
///
/// Emits a diagnostic log message (observer artifact).
/// Does not affect simulation state.
///
/// # TODO
///
/// Logging/tracing infrastructure not yet implemented in compiler rewrite.
#[kernel_fn(namespace = "effect")]
pub fn log(_message: Value, _value: Value) -> Value {
    unimplemented!("log requires observer infrastructure (not yet implemented)")
}
*/
