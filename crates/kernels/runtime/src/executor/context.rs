//! Execution contexts for different phases
//!
//! Each phase has a specific context type that provides access to
//! the appropriate data and operations.

use crate::soa_storage::MemberSignalBuffer;
use crate::storage::EntityStorage;
use crate::types::{Dt, Value};

/// Context available to warmup functions
pub struct WarmupContext<'a> {
    /// Current warmup value for this signal
    pub prev: &'a Value,
    /// Access to other signals via member signal buffer
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Current warmup iteration (0-indexed)
    pub iteration: u32,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context available to resolver functions
pub struct ResolveContext<'a> {
    /// Previous tick's value for this signal
    pub prev: &'a Value,
    /// Access to other signals via member signal buffer
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Accumulated inputs for this signal
    pub inputs: f64,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context available to collect operators
pub struct CollectContext<'a> {
    /// Access to signals via member signal buffer (previous tick values)
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Channel to write inputs
    pub channels: &'a mut crate::storage::InputChannels,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context available to fracture evaluation
pub struct FractureContext<'a> {
    /// Access to signals via member signal buffer (current tick values)
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context available to measure operators
pub struct MeasureContext<'a> {
    /// Access to signals via member signal buffer (current tick values, post-resolve)
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Field buffer for emission
    pub fields: &'a mut crate::storage::FieldBuffer,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context available to impulse application
pub struct ImpulseContext<'a> {
    /// Access to signals via member signal buffer (previous tick values)
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Channel to write inputs
    pub channels: &'a mut crate::storage::InputChannels,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context for assertion evaluation
pub struct AssertContext<'a> {
    /// Current value of the signal being asserted
    pub current: &'a Value,
    /// Previous tick's value
    pub prev: &'a Value,
    /// Access to all signals via member signal buffer
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}

/// Context available to chronicle handlers (Measure phase)
///
/// Chronicles are observer-only constructs that read resolved signals
/// and emit events. They cannot affect causality.
pub struct ChronicleContext<'a> {
    /// Access to signals via member signal buffer (current tick values, post-resolve)
    pub signals: &'a MemberSignalBuffer,
    /// Access to entity instances
    pub entities: &'a EntityStorage,
    /// Time step
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
}
