//! Vectorized primitive abstractions for simulation execution.
//!
//! This module defines the unified [`VectorizedPrimitive`] trait that enables
//! treating all simulation primitives as vectorized data with stable identity.
//! This abstraction enables uniform lowering strategies (L1/L2/L3) across
//! different primitive types.
//!
//! # Core Abstraction
//!
//! All simulation primitives share common properties:
//!
//! - **Identity** - Stable addressing that persists across ticks
//! - **Cardinality** - Number of values (1 for globals, N for members)
//! - **Value type** - The data stored at each index
//!
//! # Primitive Types
//!
//! | Primitive | Identity | Cardinality |
//! |-----------|----------|-------------|
//! | GlobalSignal | `SignalId` | 1 |
//! | MemberSignal | `(MemberSignalId, EntityIndex)` | N (population) |
//! | Field | `(FieldId, SampleIndex)` | M (samples) |
//! | Fracture | `FractureId` | 1 or N |
//!
//! # Addressing Modes
//!
//! Primitives support two addressing modes:
//!
//! ```cdsl
//! // By index (fast, requires stable ordering)
//! member.human.person.age[42]
//!
//! // By key (semantic, requires lookup)
//! member.human.person.age["alice"]
//! ```
//!
//! # Example
//!
//! ```ignore
//! use continuum_runtime::vectorized::{GlobalSignal, MemberSignal, VectorizedPrimitive};
//!
//! // Global signal has cardinality 1
//! let temp = GlobalSignal::new("terra.temperature".into());
//! assert_eq!(temp.cardinality(), 1);
//!
//! // Member signal has cardinality = entity population
//! let age = MemberSignal::new("human.person.age".into(), 100); // 100 people
//! assert_eq!(age.cardinality(), 100);
//! ```

use std::fmt::Debug;
use std::hash::Hash;

use indexmap::IndexMap;

use crate::soa_storage::{MemberSignalBuffer, ValueType};
use crate::storage::{FieldBuffer, FieldSample, SignalStorage};
use crate::types::{EntityId, FieldId, FractureId, SignalId, Value};

// Re-export MemberSignalId from foundation
pub use continuum_foundation::MemberSignalId;

// ============================================================================
// Identity Types
// ============================================================================

/// A stable index into a vectorized primitive's data.
///
/// This newtype provides type safety and documents the contract that indices
/// are stable across ticks within a simulation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityIndex(pub usize);

/// A stable sample index for field data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SampleIndex(pub usize);

/// Identity for a member signal value.
///
/// Combines the signal identity with the entity instance to uniquely
/// identify a single value in the member signal family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberSignalIdentity {
    /// The member signal's unique identifier
    pub signal_id: MemberSignalId,
    /// The entity instance index
    pub entity_index: EntityIndex,
}

/// Identity for a field sample.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldSampleIdentity {
    /// The field's unique identifier
    pub field_id: FieldId,
    /// The sample index
    pub sample_index: SampleIndex,
}

// ============================================================================
// Vectorized Primitive Trait
// ============================================================================

/// Unified trait for all simulation primitives that can be vectorized.
///
/// This trait abstracts over the different cardinalities and storage patterns
/// of simulation primitives, enabling uniform treatment in lowering strategies.
///
/// # Type Parameters
///
/// The associated types define the primitive's shape:
///
/// - `Identity` - How to address a single value (e.g., `SignalId`, `(MemberSignalId, EntityIndex)`)
/// - `ValueType` - The data type stored at each position
///
/// # Guarantees
///
/// Implementations must ensure:
///
/// 1. **Stable identity** - Same identity addresses same logical value across ticks
/// 2. **Deterministic ordering** - Iteration order is stable and reproducible
/// 3. **Bounded cardinality** - `cardinality()` returns accurate count
pub trait VectorizedPrimitive: Debug {
    /// The type used to uniquely identify a single value.
    type Identity: Clone + Eq + Hash + Debug;

    /// The value type stored at each position.
    type Value: Clone + Debug;

    /// Number of values in this primitive.
    ///
    /// - Global signals: 1
    /// - Member signals: entity population count
    /// - Fields: sample count
    fn cardinality(&self) -> usize;

    /// Check if this primitive is scalar (cardinality = 1).
    fn is_scalar(&self) -> bool {
        self.cardinality() == 1
    }

    /// Check if this primitive is vectorized (cardinality > 1).
    fn is_vectorized(&self) -> bool {
        self.cardinality() > 1
    }

    /// Get the identity for a value at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= cardinality()`.
    fn identity_at(&self, index: usize) -> Self::Identity;

    /// Iterate over all identities in deterministic order.
    fn identities(&self) -> Box<dyn Iterator<Item = Self::Identity> + '_>;
}

// ============================================================================
// Global Signal
// ============================================================================

/// A global (non-entity-indexed) signal.
///
/// Global signals have cardinality 1 - they represent a single value
/// that applies simulation-wide, like `signal.terra.temperature`.
///
/// # Identity
///
/// The identity is simply the `SignalId` path.
///
/// # Storage
///
/// Global signals use the standard double-buffered `SignalStorage`.
#[derive(Debug, Clone)]
pub struct GlobalSignal {
    /// The signal's unique identifier
    id: SignalId,
    /// The signal's value type
    value_type: ValueType,
}

impl GlobalSignal {
    /// Create a new global signal descriptor.
    pub fn new(id: SignalId, value_type: ValueType) -> Self {
        Self { id, value_type }
    }

    /// Get the signal ID.
    pub fn id(&self) -> &SignalId {
        &self.id
    }

    /// Get the value type.
    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    /// Get the current value from storage.
    pub fn get_current<'a>(&self, storage: &'a SignalStorage) -> Option<&'a Value> {
        storage.get(&self.id)
    }

    /// Get the previous tick's value from storage.
    pub fn get_previous<'a>(&self, storage: &'a SignalStorage) -> Option<&'a Value> {
        storage.get_prev(&self.id)
    }

    /// Set the current value in storage.
    pub fn set_current(&self, storage: &mut SignalStorage, value: Value) {
        storage.set_current(self.id.clone(), value);
    }
}

impl VectorizedPrimitive for GlobalSignal {
    type Identity = SignalId;
    type Value = Value;

    fn cardinality(&self) -> usize {
        1
    }

    fn identity_at(&self, index: usize) -> Self::Identity {
        assert!(
            index == 0,
            "GlobalSignal has cardinality 1, index must be 0"
        );
        self.id.clone()
    }

    fn identities(&self) -> Box<dyn Iterator<Item = Self::Identity> + '_> {
        Box::new(std::iter::once(self.id.clone()))
    }
}

// ============================================================================
// Member Signal
// ============================================================================

/// A per-entity member signal family.
///
/// Member signals have cardinality N where N is the entity population.
/// Each entity instance has its own value for this signal.
///
/// # Identity
///
/// Each value is identified by `(MemberSignalId, EntityIndex)`.
///
/// # Storage
///
/// Member signals use SoA storage via [`MemberSignalBuffer`] for efficient
/// vectorized access.
///
/// # Example
///
/// ```ignore
/// // The signal "person.age" for all people
/// let age = MemberSignal::new(
///     MemberSignalId::new("human.person".into(), "age"),
///     ValueType::Scalar,
///     100, // 100 people
/// );
///
/// // Cardinality matches population
/// assert_eq!(age.cardinality(), 100);
///
/// // Each value has a unique identity
/// let identity = age.identity_at(42);
/// assert_eq!(identity.entity_index.0, 42);
/// ```
#[derive(Debug, Clone)]
pub struct MemberSignal {
    /// The member signal family identifier
    id: MemberSignalId,
    /// The value type for all instances
    value_type: ValueType,
    /// Number of entity instances (cardinality)
    population: usize,
}

impl MemberSignal {
    /// Create a new member signal descriptor.
    pub fn new(id: MemberSignalId, value_type: ValueType, population: usize) -> Self {
        Self {
            id,
            value_type,
            population,
        }
    }

    /// Get the member signal ID.
    pub fn id(&self) -> &MemberSignalId {
        &self.id
    }

    /// Get the entity ID this signal belongs to.
    pub fn entity_id(&self) -> &EntityId {
        &self.id.entity_id
    }

    /// Get the signal name within the entity.
    pub fn signal_name(&self) -> &str {
        &self.id.signal_name
    }

    /// Get the value type.
    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    /// Get the current value for an entity index from storage.
    pub fn get_current(&self, storage: &MemberSignalBuffer, index: EntityIndex) -> Option<Value> {
        storage.get_current(&self.id.signal_name, index.0)
    }

    /// Get the previous tick's value for an entity index.
    pub fn get_previous(&self, storage: &MemberSignalBuffer, index: EntityIndex) -> Option<Value> {
        storage.get_previous(&self.id.signal_name, index.0)
    }

    /// Set the current value for an entity index.
    pub fn set_current(&self, storage: &mut MemberSignalBuffer, index: EntityIndex, value: Value) {
        storage.set_current(&self.id.signal_name, index.0, value);
    }

    /// Get all current values as a slice (for scalar signals).
    pub fn scalar_slice<'a>(&self, storage: &'a MemberSignalBuffer) -> Option<&'a [f64]> {
        storage.scalar_slice(&self.id.signal_name)
    }

    /// Get all current values as a mutable slice (for scalar signals).
    pub fn scalar_slice_mut<'a>(
        &self,
        storage: &'a mut MemberSignalBuffer,
    ) -> Option<&'a mut [f64]> {
        storage.scalar_slice_mut(&self.id.signal_name)
    }

    /// Get all current values as a Vec3 slice.
    pub fn vec3_slice<'a>(&self, storage: &'a MemberSignalBuffer) -> Option<&'a [[f64; 3]]> {
        storage.vec3_slice(&self.id.signal_name)
    }

    /// Get all current values as a mutable Vec3 slice.
    pub fn vec3_slice_mut<'a>(
        &self,
        storage: &'a mut MemberSignalBuffer,
    ) -> Option<&'a mut [[f64; 3]]> {
        storage.vec3_slice_mut(&self.id.signal_name)
    }
}

impl VectorizedPrimitive for MemberSignal {
    type Identity = MemberSignalIdentity;
    type Value = Value;

    fn cardinality(&self) -> usize {
        self.population
    }

    fn identity_at(&self, index: usize) -> Self::Identity {
        assert!(
            index < self.population,
            "Index {} out of bounds for population {}",
            index,
            self.population
        );
        MemberSignalIdentity {
            signal_id: self.id.clone(),
            entity_index: EntityIndex(index),
        }
    }

    fn identities(&self) -> Box<dyn Iterator<Item = Self::Identity> + '_> {
        let id = self.id.clone();
        Box::new((0..self.population).map(move |i| MemberSignalIdentity {
            signal_id: id.clone(),
            entity_index: EntityIndex(i),
        }))
    }
}

// ============================================================================
// Field Primitive
// ============================================================================

/// A spatial field for observer output.
///
/// Fields have cardinality M where M is the number of spatial samples.
/// They are emitted during the Measure phase and consumed by observers.
///
/// # Identity
///
/// Each sample is identified by `(FieldId, SampleIndex)`.
///
/// # Storage
///
/// Fields use the [`FieldBuffer`] for accumulating samples during Measure.
#[derive(Debug, Clone)]
pub struct FieldPrimitive {
    /// The field's unique identifier
    id: FieldId,
    /// The value type for samples
    value_type: ValueType,
    /// Expected sample count (may be dynamic)
    sample_count: usize,
}

impl FieldPrimitive {
    /// Create a new field descriptor.
    pub fn new(id: FieldId, value_type: ValueType, sample_count: usize) -> Self {
        Self {
            id,
            value_type,
            sample_count,
        }
    }

    /// Get the field ID.
    pub fn id(&self) -> &FieldId {
        &self.id
    }

    /// Get the value type.
    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    /// Get samples from the field buffer.
    pub fn get_samples<'a>(&self, buffer: &'a FieldBuffer) -> Option<&'a [FieldSample]> {
        buffer.get_samples(&self.id)
    }

    /// Emit a sample to the field buffer.
    pub fn emit(&self, buffer: &mut FieldBuffer, position: [f64; 3], value: Value) {
        buffer.emit(self.id.clone(), position, value);
    }

    /// Emit a scalar sample at origin (convenience for point values).
    pub fn emit_scalar(&self, buffer: &mut FieldBuffer, value: f64) {
        buffer.emit_scalar(self.id.clone(), value);
    }
}

impl VectorizedPrimitive for FieldPrimitive {
    type Identity = FieldSampleIdentity;
    type Value = FieldSample;

    fn cardinality(&self) -> usize {
        self.sample_count
    }

    fn identity_at(&self, index: usize) -> Self::Identity {
        assert!(
            index < self.sample_count,
            "Index {} out of bounds for sample count {}",
            index,
            self.sample_count
        );
        FieldSampleIdentity {
            field_id: self.id.clone(),
            sample_index: SampleIndex(index),
        }
    }

    fn identities(&self) -> Box<dyn Iterator<Item = Self::Identity> + '_> {
        let id = self.id.clone();
        Box::new((0..self.sample_count).map(move |i| FieldSampleIdentity {
            field_id: id.clone(),
            sample_index: SampleIndex(i),
        }))
    }
}

// ============================================================================
// Fracture Primitive
// ============================================================================

/// A fracture detector for tension conditions.
///
/// Fractures can have cardinality 1 (global condition) or N (per-entity).
///
/// # Identity
///
/// Global fractures are identified by `FractureId` alone.
/// Per-entity fractures are identified by `(FractureId, EntityIndex)`.
#[derive(Debug, Clone)]
pub struct FracturePrimitive {
    /// The fracture's unique identifier
    id: FractureId,
    /// Optional entity type for per-entity fractures
    entity_id: Option<EntityId>,
    /// Cardinality (1 for global, N for per-entity)
    cardinality: usize,
}

impl FracturePrimitive {
    /// Create a global (scalar) fracture.
    pub fn global(id: FractureId) -> Self {
        Self {
            id,
            entity_id: None,
            cardinality: 1,
        }
    }

    /// Create a per-entity fracture.
    pub fn per_entity(id: FractureId, entity_id: EntityId, population: usize) -> Self {
        Self {
            id,
            entity_id: Some(entity_id),
            cardinality: population,
        }
    }

    /// Get the fracture ID.
    pub fn id(&self) -> &FractureId {
        &self.id
    }

    /// Get the entity ID if this is a per-entity fracture.
    pub fn entity_id(&self) -> Option<&EntityId> {
        self.entity_id.as_ref()
    }

    /// Check if this is a global fracture.
    pub fn is_global(&self) -> bool {
        self.entity_id.is_none()
    }

    /// Check if this is a per-entity fracture.
    pub fn is_per_entity(&self) -> bool {
        self.entity_id.is_some()
    }
}

/// Identity for a fracture condition check.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FractureIdentity {
    /// The fracture's unique identifier
    pub fracture_id: FractureId,
    /// Entity index if per-entity, None if global
    pub entity_index: Option<EntityIndex>,
}

impl VectorizedPrimitive for FracturePrimitive {
    type Identity = FractureIdentity;
    type Value = bool; // Fracture condition result

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn identity_at(&self, index: usize) -> Self::Identity {
        assert!(
            index < self.cardinality,
            "Index {} out of bounds for cardinality {}",
            index,
            self.cardinality
        );
        FractureIdentity {
            fracture_id: self.id.clone(),
            entity_index: if self.is_per_entity() {
                Some(EntityIndex(index))
            } else {
                None
            },
        }
    }

    fn identities(&self) -> Box<dyn Iterator<Item = Self::Identity> + '_> {
        let id = self.id.clone();
        let is_per_entity = self.is_per_entity();
        Box::new((0..self.cardinality).map(move |i| FractureIdentity {
            fracture_id: id.clone(),
            entity_index: if is_per_entity {
                Some(EntityIndex(i))
            } else {
                None
            },
        }))
    }
}

// ============================================================================
// Index Space
// ============================================================================

/// Stable, deterministic mapping from string keys to contiguous indices.
///
/// This type provides the foundation for entity index spaces, ensuring:
///
/// - **Stable ordering** - Same key always maps to same index
/// - **Deterministic iteration** - Keys iterated in insertion order
/// - **O(1) lookup** - Fast key-to-index translation
///
/// # Example
///
/// ```ignore
/// let mut space = IndexSpace::new();
/// space.insert("alice".to_string());
/// space.insert("bob".to_string());
/// space.insert("charlie".to_string());
///
/// assert_eq!(space.get_index("alice"), Some(EntityIndex(0)));
/// assert_eq!(space.get_index("bob"), Some(EntityIndex(1)));
/// assert_eq!(space.get_key(EntityIndex(2)), Some(&"charlie".to_string()));
/// ```
#[derive(Debug, Clone, Default)]
pub struct IndexSpace {
    /// Key → index mapping (maintains insertion order)
    key_to_index: IndexMap<String, usize>,
}

impl IndexSpace {
    /// Create an empty index space.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an index space with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            key_to_index: IndexMap::with_capacity(capacity),
        }
    }

    /// Insert a key and return its assigned index.
    ///
    /// If the key already exists, returns the existing index.
    pub fn insert(&mut self, key: String) -> EntityIndex {
        let len = self.key_to_index.len();
        let index = *self.key_to_index.entry(key).or_insert(len);
        EntityIndex(index)
    }

    /// Get the index for a key.
    pub fn get_index(&self, key: &str) -> Option<EntityIndex> {
        self.key_to_index.get(key).map(|&i| EntityIndex(i))
    }

    /// Get the key for an index.
    pub fn get_key(&self, index: EntityIndex) -> Option<&String> {
        self.key_to_index.get_index(index.0).map(|(k, _)| k)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.key_to_index.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.key_to_index.is_empty()
    }

    /// Iterate over keys in deterministic (insertion) order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.key_to_index.keys()
    }

    /// Iterate over (key, index) pairs in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, EntityIndex)> {
        self.key_to_index.iter().map(|(k, &i)| (k, EntityIndex(i)))
    }
}

// ============================================================================
// Cardinality Classification
// ============================================================================

/// Classification of primitives by cardinality.
///
/// This enum enables lowering strategies to select optimal execution paths
/// based on the primitive's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// Single value (cardinality = 1)
    Scalar,
    /// Small vector (cardinality <= 16)
    Small(usize),
    /// Medium vector (16 < cardinality <= 1024)
    Medium(usize),
    /// Large vector (cardinality > 1024)
    Large(usize),
}

impl Cardinality {
    /// Classify a cardinality value.
    pub fn classify(n: usize) -> Self {
        match n {
            0 | 1 => Cardinality::Scalar,
            2..=16 => Cardinality::Small(n),
            17..=1024 => Cardinality::Medium(n),
            _ => Cardinality::Large(n),
        }
    }

    /// Get the raw count.
    pub fn count(&self) -> usize {
        match self {
            Cardinality::Scalar => 1,
            Cardinality::Small(n) | Cardinality::Medium(n) | Cardinality::Large(n) => *n,
        }
    }

    /// Check if this is suitable for SIMD execution.
    pub fn is_simd_friendly(&self) -> bool {
        matches!(self, Cardinality::Medium(_) | Cardinality::Large(_))
    }

    /// Check if this is suitable for GPU execution.
    pub fn is_gpu_friendly(&self) -> bool {
        matches!(self, Cardinality::Large(_))
    }
}

impl<P: VectorizedPrimitive> From<&P> for Cardinality {
    fn from(primitive: &P) -> Self {
        Cardinality::classify(primitive.cardinality())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MemberSignalId Tests
    // ========================================================================

    #[test]
    fn test_member_signal_id_from_path() {
        let id = MemberSignalId::from_path("human.person.age").unwrap();
        assert_eq!(id.entity_id.0, "human.person");
        assert_eq!(id.signal_name, "age");

        let id = MemberSignalId::from_path("stellar.star.mass").unwrap();
        assert_eq!(id.entity_id.0, "stellar.star");
        assert_eq!(id.signal_name, "mass");
    }

    #[test]
    fn test_member_signal_id_from_path_invalid() {
        assert!(MemberSignalId::from_path("single").is_none());
        assert!(MemberSignalId::from_path("").is_none());
    }

    #[test]
    fn test_member_signal_id_display() {
        let id = MemberSignalId::new("human.person".into(), "age");
        assert_eq!(format!("{}", id), "member.human.person.age");
    }

    // ========================================================================
    // GlobalSignal Tests
    // ========================================================================

    #[test]
    fn test_global_signal_cardinality() {
        let signal = GlobalSignal::new("terra.temperature".into(), ValueType::Scalar);
        assert_eq!(signal.cardinality(), 1);
        assert!(signal.is_scalar());
        assert!(!signal.is_vectorized());
    }

    #[test]
    fn test_global_signal_identity() {
        let signal = GlobalSignal::new("terra.temperature".into(), ValueType::Scalar);
        let identity = signal.identity_at(0);
        assert_eq!(identity.0, "terra.temperature");
    }

    #[test]
    #[should_panic(expected = "index must be 0")]
    fn test_global_signal_identity_out_of_bounds() {
        let signal = GlobalSignal::new("terra.temperature".into(), ValueType::Scalar);
        let _ = signal.identity_at(1);
    }

    #[test]
    fn test_global_signal_identities() {
        let signal = GlobalSignal::new("terra.temperature".into(), ValueType::Scalar);
        let ids: Vec<_> = signal.identities().collect();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].0, "terra.temperature");
    }

    // ========================================================================
    // MemberSignal Tests
    // ========================================================================

    #[test]
    fn test_member_signal_cardinality() {
        let signal = MemberSignal::new(
            MemberSignalId::new("human.person".into(), "age"),
            ValueType::Scalar,
            100,
        );
        assert_eq!(signal.cardinality(), 100);
        assert!(!signal.is_scalar());
        assert!(signal.is_vectorized());
    }

    #[test]
    fn test_member_signal_identity() {
        let signal = MemberSignal::new(
            MemberSignalId::new("human.person".into(), "age"),
            ValueType::Scalar,
            100,
        );

        let identity = signal.identity_at(42);
        assert_eq!(identity.entity_index.0, 42);
        assert_eq!(identity.signal_id.signal_name, "age");
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_member_signal_identity_out_of_bounds() {
        let signal = MemberSignal::new(
            MemberSignalId::new("human.person".into(), "age"),
            ValueType::Scalar,
            100,
        );
        let _ = signal.identity_at(100);
    }

    #[test]
    fn test_member_signal_identities() {
        let signal = MemberSignal::new(
            MemberSignalId::new("human.person".into(), "age"),
            ValueType::Scalar,
            3,
        );

        let ids: Vec<_> = signal.identities().collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].entity_index.0, 0);
        assert_eq!(ids[1].entity_index.0, 1);
        assert_eq!(ids[2].entity_index.0, 2);
    }

    // ========================================================================
    // FieldPrimitive Tests
    // ========================================================================

    #[test]
    fn test_field_primitive_cardinality() {
        let field = FieldPrimitive::new("terra.temperature".into(), ValueType::Scalar, 1000);
        assert_eq!(field.cardinality(), 1000);
        assert!(field.is_vectorized());
    }

    #[test]
    fn test_field_primitive_identity() {
        let field = FieldPrimitive::new("terra.temperature".into(), ValueType::Scalar, 100);
        let identity = field.identity_at(42);
        assert_eq!(identity.sample_index.0, 42);
        assert_eq!(identity.field_id.0, "terra.temperature");
    }

    // ========================================================================
    // FracturePrimitive Tests
    // ========================================================================

    #[test]
    fn test_fracture_primitive_global() {
        let fracture = FracturePrimitive::global("stress.threshold".into());
        assert_eq!(fracture.cardinality(), 1);
        assert!(fracture.is_global());
        assert!(!fracture.is_per_entity());
    }

    #[test]
    fn test_fracture_primitive_per_entity() {
        let fracture =
            FracturePrimitive::per_entity("bone.break".into(), "human.person".into(), 100);
        assert_eq!(fracture.cardinality(), 100);
        assert!(!fracture.is_global());
        assert!(fracture.is_per_entity());
    }

    #[test]
    fn test_fracture_primitive_identity_global() {
        let fracture = FracturePrimitive::global("stress.threshold".into());
        let identity = fracture.identity_at(0);
        assert!(identity.entity_index.is_none());
    }

    #[test]
    fn test_fracture_primitive_identity_per_entity() {
        let fracture =
            FracturePrimitive::per_entity("bone.break".into(), "human.person".into(), 100);
        let identity = fracture.identity_at(42);
        assert_eq!(identity.entity_index, Some(EntityIndex(42)));
    }

    // ========================================================================
    // IndexSpace Tests
    // ========================================================================

    #[test]
    fn test_index_space_basic() {
        let mut space = IndexSpace::new();

        let idx1 = space.insert("alice".to_string());
        let idx2 = space.insert("bob".to_string());
        let idx3 = space.insert("charlie".to_string());

        assert_eq!(idx1.0, 0);
        assert_eq!(idx2.0, 1);
        assert_eq!(idx3.0, 2);
        assert_eq!(space.len(), 3);
    }

    #[test]
    fn test_index_space_lookup() {
        let mut space = IndexSpace::new();
        space.insert("alice".to_string());
        space.insert("bob".to_string());

        assert_eq!(space.get_index("alice"), Some(EntityIndex(0)));
        assert_eq!(space.get_index("bob"), Some(EntityIndex(1)));
        assert_eq!(space.get_index("unknown"), None);
    }

    #[test]
    fn test_index_space_reverse_lookup() {
        let mut space = IndexSpace::new();
        space.insert("alice".to_string());
        space.insert("bob".to_string());

        assert_eq!(space.get_key(EntityIndex(0)), Some(&"alice".to_string()));
        assert_eq!(space.get_key(EntityIndex(1)), Some(&"bob".to_string()));
        assert_eq!(space.get_key(EntityIndex(2)), None);
    }

    #[test]
    fn test_index_space_duplicate_insert() {
        let mut space = IndexSpace::new();

        let idx1 = space.insert("alice".to_string());
        let idx2 = space.insert("alice".to_string()); // Duplicate

        assert_eq!(idx1.0, idx2.0); // Same index
        assert_eq!(space.len(), 1); // No new entry
    }

    #[test]
    fn test_index_space_iteration_order() {
        let mut space = IndexSpace::new();
        space.insert("charlie".to_string());
        space.insert("alice".to_string());
        space.insert("bob".to_string());

        let keys: Vec<_> = space.keys().collect();
        assert_eq!(keys, vec!["charlie", "alice", "bob"]); // Insertion order
    }

    // ========================================================================
    // Cardinality Tests
    // ========================================================================

    #[test]
    fn test_cardinality_classification() {
        assert_eq!(Cardinality::classify(0), Cardinality::Scalar);
        assert_eq!(Cardinality::classify(1), Cardinality::Scalar);
        assert_eq!(Cardinality::classify(8), Cardinality::Small(8));
        assert_eq!(Cardinality::classify(16), Cardinality::Small(16));
        assert_eq!(Cardinality::classify(17), Cardinality::Medium(17));
        assert_eq!(Cardinality::classify(1024), Cardinality::Medium(1024));
        assert_eq!(Cardinality::classify(1025), Cardinality::Large(1025));
    }

    #[test]
    fn test_cardinality_simd_friendly() {
        assert!(!Cardinality::Scalar.is_simd_friendly());
        assert!(!Cardinality::Small(8).is_simd_friendly());
        assert!(Cardinality::Medium(100).is_simd_friendly());
        assert!(Cardinality::Large(10000).is_simd_friendly());
    }

    #[test]
    fn test_cardinality_gpu_friendly() {
        assert!(!Cardinality::Scalar.is_gpu_friendly());
        assert!(!Cardinality::Small(8).is_gpu_friendly());
        assert!(!Cardinality::Medium(100).is_gpu_friendly());
        assert!(Cardinality::Large(10000).is_gpu_friendly());
    }

    #[test]
    fn test_cardinality_from_primitive() {
        let global = GlobalSignal::new("test".into(), ValueType::Scalar);
        let member = MemberSignal::new(
            MemberSignalId::new("entity".into(), "signal"),
            ValueType::Scalar,
            500,
        );

        let global_card: Cardinality = (&global).into();
        let member_card: Cardinality = (&member).into();

        assert_eq!(global_card, Cardinality::Scalar);
        assert_eq!(member_card, Cardinality::Medium(500));
    }

    // ========================================================================
    // SIMD Lane Boundary Tests
    // ========================================================================

    /// Test that identity generation works correctly at SIMD lane boundary sizes.
    /// SIMD lane widths are typically 4 (f32x4), 8 (f32x8), 16 (f32x16), 32.
    #[test]
    fn test_simd_lane_boundary_identity_access() {
        // Test at common SIMD lane boundaries and off-by-one values
        let boundary_sizes = [4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65];

        for size in boundary_sizes {
            let signal = MemberSignal::new(
                MemberSignalId::new("entity".into(), "signal"),
                ValueType::Scalar,
                size,
            );

            // Verify cardinality
            assert_eq!(
                signal.cardinality(),
                size,
                "Cardinality mismatch for size {}",
                size
            );

            // Verify all identities are accessible
            for i in 0..size {
                let identity = signal.identity_at(i);
                assert_eq!(
                    identity.entity_index.0, i,
                    "Identity index mismatch at {} for size {}",
                    i, size
                );
            }

            // Verify identities iterator returns exact count
            let collected: Vec<_> = signal.identities().collect();
            assert_eq!(
                collected.len(),
                size,
                "Identities iterator wrong count for size {}",
                size
            );
        }
    }

    /// Test boundary conditions for SIMD tail handling scenarios.
    /// When population isn't a multiple of SIMD width, tail elements must be handled.
    #[test]
    fn test_simd_tail_handling_boundaries() {
        // f32x4 lane width: 4 elements
        // Test sizes that produce different tail lengths (0, 1, 2, 3 tail elements)
        let sizes_for_f32x4 = [
            (4, 0),  // 4 elements = 1 full lane, 0 tail
            (5, 1),  // 5 elements = 1 full lane, 1 tail
            (6, 2),  // 6 elements = 1 full lane, 2 tail
            (7, 3),  // 7 elements = 1 full lane, 3 tail
            (8, 0),  // 8 elements = 2 full lanes, 0 tail
            (17, 1), // 17 elements = 4 full lanes, 1 tail
        ];

        for (size, expected_tail) in sizes_for_f32x4 {
            let signal = MemberSignal::new(
                MemberSignalId::new("entity".into(), "data"),
                ValueType::Scalar,
                size,
            );

            let actual_tail = size % 4;
            assert_eq!(
                actual_tail, expected_tail,
                "Tail calculation wrong for size {}",
                size
            );

            // Verify the last element (tail element if any) is accessible
            let last_identity = signal.identity_at(size - 1);
            assert_eq!(last_identity.entity_index.0, size - 1);

            // Verify iteration covers all elements including tail
            let count = signal.identities().count();
            assert_eq!(count, size);
        }
    }

    /// Test field primitive at SIMD boundary sizes for spatial sampling.
    #[test]
    fn test_field_primitive_simd_boundary_sizes() {
        let boundary_sizes = [4, 8, 16, 32, 64, 128, 255, 256, 257];

        for size in boundary_sizes {
            let field = FieldPrimitive::new("spatial.data".into(), ValueType::Scalar, size);

            assert_eq!(field.cardinality(), size);

            // Verify all sample identities are accessible
            for i in 0..size {
                let identity = field.identity_at(i);
                assert_eq!(identity.sample_index.0, i);
            }

            // Verify iterator count
            assert_eq!(field.identities().count(), size);
        }
    }

    /// Test cardinality classification at exact SIMD boundary values.
    #[test]
    fn test_cardinality_at_simd_boundaries() {
        // Test exact boundaries of Cardinality classification
        // Small: 2..=16, Medium: 17..=1024, Large: >1024

        // SIMD lane boundaries within Small range
        assert_eq!(Cardinality::classify(4), Cardinality::Small(4));
        assert_eq!(Cardinality::classify(8), Cardinality::Small(8));
        assert_eq!(Cardinality::classify(16), Cardinality::Small(16));

        // Boundary from Small to Medium (at 17)
        assert_eq!(Cardinality::classify(16), Cardinality::Small(16));
        assert_eq!(Cardinality::classify(17), Cardinality::Medium(17));

        // SIMD boundaries within Medium range
        assert_eq!(Cardinality::classify(32), Cardinality::Medium(32));
        assert_eq!(Cardinality::classify(64), Cardinality::Medium(64));
        assert_eq!(Cardinality::classify(128), Cardinality::Medium(128));
        assert_eq!(Cardinality::classify(256), Cardinality::Medium(256));
        assert_eq!(Cardinality::classify(512), Cardinality::Medium(512));
        assert_eq!(Cardinality::classify(1024), Cardinality::Medium(1024));

        // Boundary from Medium to Large (at 1025)
        assert_eq!(Cardinality::classify(1024), Cardinality::Medium(1024));
        assert_eq!(Cardinality::classify(1025), Cardinality::Large(1025));
    }

    /// Test per-entity fracture at SIMD boundary populations.
    #[test]
    fn test_fracture_simd_boundary_populations() {
        let boundary_sizes = [4, 8, 15, 16, 17, 32, 64, 100];

        for size in boundary_sizes {
            let fracture =
                FracturePrimitive::per_entity("stress.check".into(), "entity".into(), size);

            assert_eq!(fracture.cardinality(), size);
            assert!(fracture.is_per_entity());

            // Verify all entity indices are accessible
            for i in 0..size {
                let identity = fracture.identity_at(i);
                assert_eq!(identity.entity_index, Some(EntityIndex(i)));
            }

            // Verify iterator count
            assert_eq!(fracture.identities().count(), size);
        }
    }

    /// Test MemberSignal with actual MemberSignalBuffer storage at SIMD boundaries.
    #[test]
    fn test_member_signal_storage_simd_boundaries() {
        let boundary_sizes = [4, 7, 8, 15, 16, 17, 31, 32, 33];

        for size in boundary_sizes {
            let signal = MemberSignal::new(
                MemberSignalId::new("entity".into(), "energy"),
                ValueType::Scalar,
                size,
            );

            // Create storage
            let mut storage = MemberSignalBuffer::new();
            storage.register_signal("energy".to_string(), ValueType::Scalar);
            storage.init_instances(size);

            // Write values at all indices
            for i in 0..size {
                signal.set_current(&mut storage, EntityIndex(i), Value::Scalar(i as f64 * 10.0));
            }

            // Verify all values are readable including tail elements
            for i in 0..size {
                let value = signal.get_current(&storage, EntityIndex(i));
                assert_eq!(
                    value,
                    Some(Value::Scalar(i as f64 * 10.0)),
                    "Value mismatch at index {} for size {}",
                    i,
                    size
                );
            }

            // Verify scalar slice access returns correct length
            let slice = signal.scalar_slice(&storage).unwrap();
            assert_eq!(slice.len(), size, "Slice length mismatch for size {}", size);

            // Verify slice contents match including last (potentially tail) element
            assert_eq!(
                slice[size - 1],
                (size - 1) as f64 * 10.0,
                "Last element wrong for size {}",
                size
            );
        }
    }

    /// Test Vec3 member signal storage at SIMD boundaries.
    /// Vec3 has different SIMD implications (3 components per element).
    #[test]
    fn test_vec3_member_signal_simd_boundaries() {
        let boundary_sizes = [4, 8, 16, 17, 32, 33];

        for size in boundary_sizes {
            let signal = MemberSignal::new(
                MemberSignalId::new("entity".into(), "position"),
                ValueType::Vec3,
                size,
            );

            // Create storage
            let mut storage = MemberSignalBuffer::new();
            storage.register_signal("position".to_string(), ValueType::Vec3);
            storage.init_instances(size);

            // Write Vec3 values at all indices
            for i in 0..size {
                let v = i as f64;
                signal.set_current(
                    &mut storage,
                    EntityIndex(i),
                    Value::Vec3([v, v * 2.0, v * 3.0]),
                );
            }

            // Verify all values including tail elements
            for i in 0..size {
                let value = signal.get_current(&storage, EntityIndex(i));
                let v = i as f64;
                assert_eq!(
                    value,
                    Some(Value::Vec3([v, v * 2.0, v * 3.0])),
                    "Vec3 value mismatch at index {} for size {}",
                    i,
                    size
                );
            }

            // Verify vec3 slice access
            let slice = signal.vec3_slice(&storage).unwrap();
            assert_eq!(slice.len(), size);

            // Verify last element
            let v = (size - 1) as f64;
            assert_eq!(slice[size - 1], [v, v * 2.0, v * 3.0]);
        }
    }

    /// Regression test: ensure cardinality count accessor works for all sizes.
    #[test]
    fn test_cardinality_count_at_boundaries() {
        let test_cases = [
            (0, 1),       // Scalar
            (1, 1),       // Scalar
            (16, 16),     // Small
            (17, 17),     // Medium boundary
            (1024, 1024), // Medium max
            (1025, 1025), // Large boundary
        ];

        for (input, expected) in test_cases {
            let card = Cardinality::classify(input);
            assert_eq!(
                card.count(),
                expected,
                "count() mismatch for input {}",
                input
            );
        }
    }
}
