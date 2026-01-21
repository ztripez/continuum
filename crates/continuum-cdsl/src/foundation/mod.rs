//! Compiler foundation types
//!
//! These types are used throughout the compiler but are distinct from
//! runtime foundation types. They represent compile-time constructs.

pub mod span;
pub mod types;

pub use continuum_foundation::Path;
pub use span::{SourceFile, SourceMap, Span};
pub use types::{Bounds, KernelType, Type, UserType, UserTypeId};

// Re-export Shape and Unit from kernel-types (single source of truth)
pub use continuum_kernel_types::{Shape, Unit, UnitDimensions, UnitKind};

// Re-export from runtime foundation
pub use continuum_foundation::{
    // Typed IDs
    AnalyzerId,
    // Execution phases and capabilities
    AssertionSeverity,
    Capability,
    CapabilitySet,
    ChronicleId,
    EntityId,
    EraId,
    FieldId,
    FnId,
    FractureId,
    ImpulseId,
    InstanceId,
    MemberId,
    OperatorId,
    Phase,
    PhaseSet,
    SignalId,
    StratumId,
    TypeId,
};
