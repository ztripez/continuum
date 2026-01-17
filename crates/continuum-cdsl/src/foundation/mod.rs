//! Compiler foundation types
//!
//! These types are used throughout the compiler but are distinct from
//! runtime foundation types. They represent compile-time constructs.

pub mod path;
pub mod phase;
pub mod shape;
pub mod span;
pub mod types;
pub mod unit;

pub use path::Path;
pub use phase::{Capability, CapabilitySet, Phase, PhaseSet};
pub use shape::Shape;
pub use span::{SourceFile, SourceMap, Span};
pub use types::{Bounds, KernelType, Type, UserType, UserTypeId};
pub use unit::{Unit, UnitDimensions, UnitKind};

// Re-export typed IDs
pub use continuum_foundation::{
    AnalyzerId, ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId,
    MemberId, OperatorId, SignalId, StratumId, TypeId,
};
