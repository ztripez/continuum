//! Compiler foundation types
//!
//! These types are used throughout the compiler but are distinct from
//! runtime foundation types. They represent compile-time constructs.

pub mod path;
pub mod shape;
pub mod span;
pub mod types;
pub mod unit;

pub use path::Path;
pub use shape::Shape;
pub use span::{SourceFile, SourceMap, Span};
pub use types::{Bounds, KernelType, Type, UserType, UserTypeId};
pub use unit::{Unit, UnitDimensions, UnitKind};

// Re-export from runtime foundation
pub use continuum_foundation::{
    // Typed IDs
    AnalyzerId,
    // Execution phases and capabilities
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
