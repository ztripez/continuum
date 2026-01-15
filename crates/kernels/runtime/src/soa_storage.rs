//! SoA (Struct-of-Arrays) storage for vectorized simulation execution.
//!
//! This module provides cache-efficient, SIMD-friendly storage for member signals
//! and entity populations. Unlike the AoS (Array-of-Structs) pattern in `storage.rs`,
//! SoA storage organizes data by type into contiguous arrays for optimal vectorization.
//!
//! # Architecture
//!
//! ```text
//! AoS (current):              SoA (this module):
//! ┌─────────────┐             ┌─────────────────────────────────┐
//! │ Instance 0  │             │ scalars: [v0, v1, v2, v3, ...]  │
//! │  - mass: f64│             │ vec3s:   [p0, p1, p2, p3, ...]  │
//! │  - pos: Vec3│             │                                 │
//! ├─────────────┤             │ registry:                       │
//! │ Instance 1  │             │   "mass" → (Scalar, idx=0)      │
//! │  - mass: f64│             │   "pos"  → (Vec3, idx=0)        │
//! └─────────────┘             └─────────────────────────────────┘
//! ```
//!
//! # Benefits
//!
//! - **Cache efficiency**: Contiguous memory access patterns
//! - **SIMD-friendly**: 64-byte aligned buffers fit cache lines
//! - **No enum overhead**: Type-separated storage eliminates branching
//! - **Vectorization**: Compiler can auto-vectorize simple loops
//!
//! # Key Types
//!
//! - [`ValueType`] - Type discriminant for member signals
//! - [`AlignedBuffer`] - 64-byte aligned storage wrapper
//! - [`MemberSignalBuffer`] - Type-separated double-buffered storage
//! - [`PopulationStorage`] - Entity population with SoA member signals

use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr::NonNull;

use indexmap::IndexMap;

use crate::types::{EntityId, Value};
use continuum_foundation::{PrimitiveStorageClass, PrimitiveTypeId, primitive_type_by_name};

/// Alignment for SIMD-friendly allocation (64 bytes = cache line).
pub const SIMD_ALIGNMENT: usize = 64;

// ============================================================================
// Value Type Discriminant
// ============================================================================

/// Type discriminant for member signal values.
///
/// Unlike [`Value`], this struct stores only the type tag without data.
/// Used in the signal registry to route access to the correct buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueType {
    kind: ValueTypeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ValueTypeKind {
    Primitive(PrimitiveTypeId),
    Boolean,
    Integer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MemberBufferClass {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Boolean,
    Integer,
}

impl ValueType {
    pub fn scalar() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Scalar"))
    }

    pub fn vec2() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Vec2"))
    }

    pub fn vec3() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Vec3"))
    }

    pub fn vec4() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Vec4"))
    }

    pub fn quat() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Quat"))
    }

    pub fn boolean() -> Self {
        Self {
            kind: ValueTypeKind::Boolean,
        }
    }

    pub fn integer() -> Self {
        Self {
            kind: ValueTypeKind::Integer,
        }
    }

    pub fn from_primitive_id(primitive_id: PrimitiveTypeId) -> Self {
        Self {
            kind: ValueTypeKind::Primitive(primitive_id),
        }
    }

    pub fn primitive_id(&self) -> Option<PrimitiveTypeId> {
        match self.kind {
            ValueTypeKind::Primitive(id) => Some(id),
            _ => None,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Scalar")
    }

    pub fn is_quat(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Quat")
    }

    pub fn is_vec3(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Vec3")
    }

    pub fn is_vec2(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Vec2")
    }

    pub fn is_vec4(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Vec4")
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self.kind, ValueTypeKind::Boolean)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.kind, ValueTypeKind::Integer)
    }

    /// Size in bytes for one element of this type
    pub fn element_size(&self) -> usize {
        match self.buffer_class() {
            MemberBufferClass::Scalar => std::mem::size_of::<f64>(),
            MemberBufferClass::Vec2 => std::mem::size_of::<[f64; 2]>(),
            MemberBufferClass::Vec3 => std::mem::size_of::<[f64; 3]>(),
            MemberBufferClass::Vec4 => std::mem::size_of::<[f64; 4]>(),
            MemberBufferClass::Boolean => std::mem::size_of::<bool>(),
            MemberBufferClass::Integer => std::mem::size_of::<i64>(),
        }
    }

    /// Alignment requirement for this type
    pub fn alignment(&self) -> usize {
        match self.kind {
            ValueTypeKind::Boolean => std::mem::align_of::<bool>(),
            ValueTypeKind::Integer => std::mem::align_of::<i64>(),
            _ => std::mem::align_of::<f64>(),
        }
    }

    /// Infer type from a Value
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Scalar(_) => ValueType::scalar(),
            Value::Vec2(_) => ValueType::vec2(),
            Value::Vec3(_) => ValueType::vec3(),
            Value::Vec4(_) => ValueType::vec4(),
            Value::Quat(_) => ValueType::quat(),
            Value::Mat2(_) => ValueType::scalar(), // TODO: proper matrix value type
            Value::Mat3(_) => ValueType::scalar(), // TODO: proper matrix value type
            Value::Mat4(_) => ValueType::scalar(), // TODO: proper matrix value type
            Value::Boolean(_) => ValueType::boolean(),
            Value::Integer(_) => ValueType::integer(),
            Value::Map(_) => panic!("Map values are not supported in member signals"),
        }
    }

    /// Convert from a primitive storage class.
    pub fn from_storage_class(storage: PrimitiveStorageClass) -> Self {
        match storage {
            PrimitiveStorageClass::Scalar => ValueType::scalar(),
            PrimitiveStorageClass::Vec2 => ValueType::vec2(),
            PrimitiveStorageClass::Vec3 => ValueType::vec3(),
            PrimitiveStorageClass::Vec4 => ValueType::vec4(),
            PrimitiveStorageClass::Mat2 => ValueType::scalar(), // TODO: implement matrix types
            PrimitiveStorageClass::Mat3 => ValueType::scalar(), // TODO: implement matrix types
            PrimitiveStorageClass::Mat4 => ValueType::scalar(), // TODO: implement matrix types
            PrimitiveStorageClass::Tensor => ValueType::scalar(),
            PrimitiveStorageClass::Grid => ValueType::scalar(),
            PrimitiveStorageClass::Seq => ValueType::scalar(),
        }
    }

    fn buffer_class(&self) -> MemberBufferClass {
        match self.kind {
            ValueTypeKind::Boolean => MemberBufferClass::Boolean,
            ValueTypeKind::Integer => MemberBufferClass::Integer,
            ValueTypeKind::Primitive(id) => match primitive_type_by_name(id.name())
                .map(|def| def.storage)
                .unwrap_or(PrimitiveStorageClass::Scalar)
            {
                PrimitiveStorageClass::Scalar => MemberBufferClass::Scalar,
                PrimitiveStorageClass::Vec2 => MemberBufferClass::Vec2,
                PrimitiveStorageClass::Vec3 => MemberBufferClass::Vec3,
                PrimitiveStorageClass::Vec4 => MemberBufferClass::Vec4,
                PrimitiveStorageClass::Mat2 => MemberBufferClass::Scalar, // TODO: matrix buffer class
                PrimitiveStorageClass::Mat3 => MemberBufferClass::Scalar, // TODO: matrix buffer class
                PrimitiveStorageClass::Mat4 => MemberBufferClass::Scalar, // TODO: matrix buffer class
                PrimitiveStorageClass::Tensor => MemberBufferClass::Scalar,
                PrimitiveStorageClass::Grid => MemberBufferClass::Scalar,
                PrimitiveStorageClass::Seq => MemberBufferClass::Scalar,
            },
        }
    }
}

// ============================================================================
// Aligned Buffer
// ============================================================================

/// A raw buffer with guaranteed 64-byte alignment for SIMD operations.
///
/// This is a low-level building block that provides aligned memory allocation.
/// Higher-level typed buffers wrap this for type-safe access.
///
/// # Safety
///
/// This type uses raw pointer manipulation. All public methods maintain
/// memory safety invariants.
pub struct AlignedBuffer {
    /// Pointer to aligned memory (null if capacity is 0)
    ptr: Option<NonNull<u8>>,
    /// Allocated capacity in bytes
    capacity_bytes: usize,
    /// Used length in bytes
    len_bytes: usize,
    /// Element size in bytes
    element_size: usize,
}

impl AlignedBuffer {
    /// Create a new empty buffer for elements of the given size.
    pub fn new(element_size: usize) -> Self {
        Self {
            ptr: None,
            capacity_bytes: 0,
            len_bytes: 0,
            element_size,
        }
    }

    /// Create a buffer with pre-allocated capacity for `count` elements.
    pub fn with_capacity(element_size: usize, count: usize) -> Self {
        let mut buf = Self::new(element_size);
        if count > 0 {
            buf.reserve(count);
        }
        buf
    }

    /// Number of elements in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.len_bytes / self.element_size
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    /// Capacity in elements
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity_bytes / self.element_size
    }

    /// Reserve space for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len() + additional;
        if required <= self.capacity() {
            return;
        }

        // Grow by at least 2x or to required, whichever is larger
        let new_capacity = std::cmp::max(required, self.capacity() * 2).max(8);
        self.grow_to(new_capacity);
    }

    /// Grow buffer to hold at least `new_capacity` elements.
    fn grow_to(&mut self, new_capacity: usize) {
        let new_capacity_bytes = new_capacity * self.element_size;

        // Create aligned layout
        let layout = Layout::from_size_align(new_capacity_bytes, SIMD_ALIGNMENT)
            .expect("Invalid layout for aligned buffer");

        let new_ptr = if let Some(old_ptr) = self.ptr {
            // Reallocate
            let old_layout = Layout::from_size_align(self.capacity_bytes, SIMD_ALIGNMENT)
                .expect("Invalid old layout");

            // SAFETY: old_ptr is valid, old_layout matches previous allocation
            unsafe {
                let ptr = alloc::realloc(old_ptr.as_ptr(), old_layout, new_capacity_bytes);
                NonNull::new(ptr).expect("Allocation failed")
            }
        } else {
            // Fresh allocation
            // SAFETY: layout has non-zero size
            unsafe {
                let ptr = alloc::alloc(layout);
                NonNull::new(ptr).expect("Allocation failed")
            }
        };

        self.ptr = Some(new_ptr);
        self.capacity_bytes = new_capacity_bytes;
    }

    /// Get raw pointer to element at index.
    ///
    /// # Safety
    ///
    /// Caller must ensure index < len and use appropriate type.
    #[inline]
    pub unsafe fn get_ptr(&self, index: usize) -> *const u8 {
        debug_assert!(index < self.len());
        unsafe {
            self.ptr
                .map(|p| p.as_ptr().add(index * self.element_size) as *const u8)
                .unwrap_or(std::ptr::null())
        }
    }

    /// Get mutable raw pointer to element at index.
    ///
    /// # Safety
    ///
    /// Caller must ensure index < len and use appropriate type.
    #[inline]
    pub unsafe fn get_ptr_mut(&mut self, index: usize) -> *mut u8 {
        debug_assert!(index < self.len());
        unsafe {
            self.ptr
                .map(|p| p.as_ptr().add(index * self.element_size))
                .unwrap_or(std::ptr::null_mut())
        }
    }

    /// Push a value onto the buffer.
    ///
    /// # Safety
    ///
    /// Caller must ensure `value` points to data of `element_size` bytes.
    pub unsafe fn push_raw(&mut self, value: *const u8) {
        self.reserve(1);
        unsafe {
            let dst = self.ptr.unwrap().as_ptr().add(self.len_bytes);
            std::ptr::copy_nonoverlapping(value, dst, self.element_size);
        }
        self.len_bytes += self.element_size;
    }

    /// Set value at index.
    ///
    /// # Safety
    ///
    /// Caller must ensure index < len and value points to valid data.
    pub unsafe fn set_raw(&mut self, index: usize, value: *const u8) {
        debug_assert!(index < self.len());
        unsafe {
            let dst = self.get_ptr_mut(index);
            std::ptr::copy_nonoverlapping(value, dst, self.element_size);
        }
    }

    /// Clear the buffer without deallocating.
    pub fn clear(&mut self) {
        self.len_bytes = 0;
    }

    /// Resize to exactly `count` elements, filling new elements with zeros.
    pub fn resize(&mut self, count: usize) {
        let target_bytes = count * self.element_size;

        if target_bytes > self.capacity_bytes {
            self.grow_to(count);
        }

        // Zero-fill any new elements
        if target_bytes > self.len_bytes {
            if let Some(ptr) = self.ptr {
                unsafe {
                    let start = ptr.as_ptr().add(self.len_bytes);
                    std::ptr::write_bytes(start, 0, target_bytes - self.len_bytes);
                }
            }
        }

        self.len_bytes = target_bytes;
    }

    /// Copy all data from another buffer.
    ///
    /// # Panics
    ///
    /// Panics if element sizes don't match.
    pub fn copy_from(&mut self, other: &AlignedBuffer) {
        assert_eq!(
            self.element_size, other.element_size,
            "Element size mismatch in copy"
        );

        self.resize(other.len());

        if let (Some(dst), Some(src)) = (self.ptr, other.ptr) {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), other.len_bytes);
            }
        }
    }

    /// Get a slice view of the buffer as a specific type.
    ///
    /// # Safety
    ///
    /// Caller must ensure T matches the element type stored.
    #[inline]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        debug_assert_eq!(std::mem::size_of::<T>(), self.element_size);
        if let Some(ptr) = self.ptr {
            unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
        } else {
            &[]
        }
    }

    /// Get a mutable slice view of the buffer.
    ///
    /// # Safety
    ///
    /// Caller must ensure T matches the element type stored.
    #[inline]
    pub unsafe fn as_slice_mut<T>(&mut self) -> &mut [T] {
        debug_assert_eq!(std::mem::size_of::<T>(), self.element_size);
        if let Some(ptr) = self.ptr {
            unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
        } else {
            &mut []
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr {
            if self.capacity_bytes > 0 {
                let layout = Layout::from_size_align(self.capacity_bytes, SIMD_ALIGNMENT)
                    .expect("Invalid layout in drop");
                unsafe {
                    alloc::dealloc(ptr.as_ptr(), layout);
                }
            }
        }
    }
}

// SAFETY: AlignedBuffer is Send because:
// - It owns the memory pointed to by `ptr` (allocated via std::alloc)
// - No references to thread-local storage
// - Drop deallocates owned memory, safe from any thread
unsafe impl Send for AlignedBuffer {}

// SAFETY: AlignedBuffer is Sync because:
// - Shared references (&AlignedBuffer) provide read-only access
// - No interior mutability (no Cell, RefCell, etc.)
// - The `as_slice` method returns immutable slices safe for concurrent reads
// - Mutable access (`as_slice_mut`, `push_raw`, etc.) requires &mut self
unsafe impl Sync for AlignedBuffer {}

// ============================================================================
// Typed Buffer Wrapper
// ============================================================================

/// Type-safe wrapper around AlignedBuffer.
///
/// Provides a Vec-like interface with 64-byte alignment guarantee.
pub struct TypedBuffer<T> {
    inner: AlignedBuffer,
    _marker: PhantomData<T>,
}

impl<T: Copy + Default> TypedBuffer<T> {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self {
            inner: AlignedBuffer::new(std::mem::size_of::<T>()),
            _marker: PhantomData,
        }
    }

    /// Create a buffer with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: AlignedBuffer::with_capacity(std::mem::size_of::<T>(), capacity),
            _marker: PhantomData,
        }
    }

    /// Number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get element at index
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            unsafe { Some(&*(self.inner.get_ptr(index) as *const T)) }
        } else {
            None
        }
    }

    /// Get mutable element at index
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            unsafe { Some(&mut *(self.inner.get_ptr_mut(index) as *mut T)) }
        } else {
            None
        }
    }

    /// Push a value
    pub fn push(&mut self, value: T) {
        unsafe {
            self.inner.push_raw(&value as *const T as *const u8);
        }
    }

    /// Set value at index
    pub fn set(&mut self, index: usize, value: T) {
        assert!(index < self.len(), "Index out of bounds");
        unsafe {
            self.inner.set_raw(index, &value as *const T as *const u8);
        }
    }

    /// Get as slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.inner.as_slice() }
    }

    /// Get as mutable slice
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { self.inner.as_slice_mut() }
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Resize to count elements
    pub fn resize(&mut self, count: usize) {
        self.inner.resize(count);
    }

    /// Copy from another buffer
    pub fn copy_from(&mut self, other: &TypedBuffer<T>) {
        self.inner.copy_from(&other.inner);
    }
}

impl<T: Copy + Default> Default for TypedBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Member Signal Registry
// ============================================================================

/// Metadata for a registered member signal.
#[derive(Debug, Clone)]
pub struct MemberSignalMeta {
    /// Type of the signal value
    pub value_type: ValueType,
    /// Index within the type's buffer
    pub buffer_index: usize,
}

/// Registry mapping signal names to their storage location.
///
/// Each signal maps to a (type, index) pair that identifies which
/// type-separated buffer contains it and at what offset.
#[derive(Debug, Default)]
pub struct MemberSignalRegistry {
    /// Signal name → metadata
    signals: IndexMap<String, MemberSignalMeta>,
    /// Count of signals per buffer class (for assigning buffer indices)
    type_counts: HashMap<MemberBufferClass, usize>,
}

impl MemberSignalRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new member signal.
    ///
    /// Returns the assigned metadata, or panics if signal already exists.
    pub fn register(&mut self, name: String, value_type: ValueType) -> MemberSignalMeta {
        if self.signals.contains_key(&name) {
            panic!("Signal '{}' already registered", name);
        }

        let buffer_class = value_type.buffer_class();
        let buffer_index = *self.type_counts.get(&buffer_class).unwrap_or(&0);
        self.type_counts.insert(buffer_class, buffer_index + 1);

        let meta = MemberSignalMeta {
            value_type,
            buffer_index,
        };

        self.signals.insert(name, meta.clone());
        meta
    }

    /// Look up a signal by name.
    pub fn get(&self, name: &str) -> Option<&MemberSignalMeta> {
        self.signals.get(name)
    }

    /// Get count of signals for a type
    pub fn type_count(&self, value_type: ValueType) -> usize {
        *self
            .type_counts
            .get(&value_type.buffer_class())
            .unwrap_or(&0)
    }

    /// Iterate over all registered signals.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MemberSignalMeta)> {
        self.signals.iter()
    }

    /// Number of registered signals.
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }
}

// ============================================================================
// Double-Buffered SoA Storage
// ============================================================================

/// Double-buffered SoA storage for a single value type.
///
/// Stores current and previous tick values in separate aligned buffers.
/// Each buffer holds N instances × M signals of this type.
struct DoubleBuffer<T: Copy + Default> {
    current: TypedBuffer<T>,
    previous: TypedBuffer<T>,
    /// Number of signals of this type
    signal_count: usize,
    /// Number of instances
    instance_count: usize,
}

impl<T: Copy + Default> DoubleBuffer<T> {
    fn new() -> Self {
        Self {
            current: TypedBuffer::new(),
            previous: TypedBuffer::new(),
            signal_count: 0,
            instance_count: 0,
        }
    }

    /// Initialize for given signal count and instance count.
    fn init(&mut self, signal_count: usize, instance_count: usize) {
        self.signal_count = signal_count;
        self.instance_count = instance_count;
        let total = signal_count * instance_count;
        self.current.resize(total);
        self.previous.resize(total);
    }

    /// Get index for (signal_index, instance_index).
    /// Layout: signal-major (all instances of signal 0, then signal 1, etc.)
    #[inline]
    fn index(&self, signal_idx: usize, instance_idx: usize) -> usize {
        debug_assert!(signal_idx < self.signal_count);
        debug_assert!(instance_idx < self.instance_count);
        signal_idx * self.instance_count + instance_idx
    }

    /// Get current value for (signal, instance).
    #[inline]
    fn get_current(&self, signal_idx: usize, instance_idx: usize) -> Option<&T> {
        self.current.get(self.index(signal_idx, instance_idx))
    }

    /// Get previous value for (signal, instance).
    #[inline]
    fn get_previous(&self, signal_idx: usize, instance_idx: usize) -> Option<&T> {
        self.previous.get(self.index(signal_idx, instance_idx))
    }

    /// Set current value for (signal, instance).
    fn set_current(&mut self, signal_idx: usize, instance_idx: usize, value: T) {
        let idx = self.index(signal_idx, instance_idx);
        self.current.set(idx, value);
    }

    /// Advance tick: current becomes previous, current cleared.
    fn advance_tick(&mut self) {
        std::mem::swap(&mut self.current, &mut self.previous);
        // Copy previous to current for signals that won't be updated (gated)
        self.current.copy_from(&self.previous);
    }

    /// Get slice of all values for a signal (all instances).
    fn signal_slice(&self, signal_idx: usize) -> &[T] {
        let start = signal_idx * self.instance_count;
        let end = start + self.instance_count;
        &self.current.as_slice()[start..end]
    }

    /// Get mutable slice for a signal.
    fn signal_slice_mut(&mut self, signal_idx: usize) -> &mut [T] {
        let start = signal_idx * self.instance_count;
        let end = start + self.instance_count;
        &mut self.current.as_slice_mut()[start..end]
    }

    /// Get slice of previous tick's values for a signal (all instances).
    fn prev_signal_slice(&self, signal_idx: usize) -> &[T] {
        let start = signal_idx * self.instance_count;
        let end = start + self.instance_count;
        &self.previous.as_slice()[start..end]
    }
}

// ============================================================================
// Member Signal Buffer (Type-Erased)
// ============================================================================

/// SoA storage for member signals across an entity population.
///
/// Provides type-erased access to double-buffered, type-separated storage.
/// This is the main interface for resolvers accessing member signal data.
///
/// # Memory Layout
///
/// For N instances and M signals of each type:
///
/// ```text
/// scalars:  [sig0_inst0, sig0_inst1, ..., sig1_inst0, sig1_inst1, ...]
/// vec3s:    [sig0_inst0, sig0_inst1, ..., sig1_inst0, sig1_inst1, ...]
/// ```
///
/// This signal-major layout enables efficient vectorized iteration over
/// all instances of a single signal.
pub struct MemberSignalBuffer {
    /// Registry of signal names → (type, index)
    registry: MemberSignalRegistry,
    /// Maximum number of entity instances (used for storage allocation)
    instance_count: usize,
    /// Per-entity instance counts (entity_id → count)
    entity_instance_counts: std::collections::HashMap<String, usize>,
    /// Double-buffered scalar storage
    scalars: DoubleBuffer<f64>,
    /// Double-buffered Vec2 storage
    vec2s: DoubleBuffer<[f64; 2]>,
    /// Double-buffered Vec3 storage
    vec3s: DoubleBuffer<[f64; 3]>,
    /// Double-buffered Vec4 storage (Vec4 + Quat)
    vec4s: DoubleBuffer<[f64; 4]>,
    /// Double-buffered Boolean storage
    booleans: DoubleBuffer<bool>,
    /// Double-buffered Integer storage
    integers: DoubleBuffer<i64>,
}

impl MemberSignalBuffer {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self {
            registry: MemberSignalRegistry::new(),
            instance_count: 0,
            entity_instance_counts: std::collections::HashMap::new(),
            scalars: DoubleBuffer::new(),
            vec2s: DoubleBuffer::new(),
            vec3s: DoubleBuffer::new(),
            vec4s: DoubleBuffer::new(),
            booleans: DoubleBuffer::new(),
            integers: DoubleBuffer::new(),
        }
    }

    /// Register a member signal.
    ///
    /// Must be called before `init_instances` to register all signals.
    pub fn register_signal(&mut self, name: String, value_type: ValueType) {
        self.registry.register(name, value_type);
    }

    /// Initialize storage for the given instance count.
    ///
    /// Must be called after all signals are registered.
    pub fn init_instances(&mut self, instance_count: usize) {
        self.instance_count = instance_count;

        // Initialize type-separated buffers
        self.scalars.init(
            self.registry.type_count(ValueType::scalar()),
            instance_count,
        );
        self.vec2s
            .init(self.registry.type_count(ValueType::vec2()), instance_count);
        self.vec3s
            .init(self.registry.type_count(ValueType::vec3()), instance_count);
        self.vec4s
            .init(self.registry.type_count(ValueType::vec4()), instance_count);
        self.booleans.init(
            self.registry.type_count(ValueType::boolean()),
            instance_count,
        );
        self.integers.init(
            self.registry.type_count(ValueType::integer()),
            instance_count,
        );
    }

    /// Get the maximum number of instances (used for storage).
    pub fn instance_count(&self) -> usize {
        self.instance_count
    }

    /// Register the instance count for a specific entity.
    ///
    /// Call this before `init_instances` to track per-entity counts.
    pub fn register_entity_count(&mut self, entity_id: &str, count: usize) {
        self.entity_instance_counts
            .insert(entity_id.to_string(), count);
    }

    /// Get the instance count for a specific entity.
    ///
    /// Returns the registered count for that entity, or the global instance_count
    /// if no specific count was registered.
    pub fn instance_count_for_entity(&self, entity_id: &str) -> usize {
        self.entity_instance_counts
            .get(entity_id)
            .copied()
            .unwrap_or(self.instance_count)
    }

    /// Extract entity ID from a member signal name.
    ///
    /// Member signal names follow the pattern "entity.id.signal_name".
    /// For example: "terra.plate.age" → "terra.plate"
    pub fn entity_id_from_signal(&self, signal_name: &str) -> Option<String> {
        // Split by '.' and take all but the last component
        let parts: Vec<&str> = signal_name.split('.').collect();
        if parts.len() >= 2 {
            Some(parts[..parts.len() - 1].join("."))
        } else {
            None
        }
    }

    /// Get the instance count for a member signal by extracting its entity ID.
    ///
    /// Returns the entity-specific count, or the global instance_count if
    /// entity ID cannot be determined or is not registered.
    pub fn instance_count_for_signal(&self, signal_name: &str) -> usize {
        if let Some(entity_id) = self.entity_id_from_signal(signal_name) {
            self.instance_count_for_entity(&entity_id)
        } else {
            self.instance_count
        }
    }

    /// Get the registry for signal lookup.
    pub fn registry(&self) -> &MemberSignalRegistry {
        &self.registry
    }

    /// Get current value as a Value enum (for compatibility).
    pub fn get_current(&self, signal: &str, instance_idx: usize) -> Option<Value> {
        let meta = self.registry.get(signal)?;
        self.get_value(meta, instance_idx, false)
    }

    /// Get previous tick value.
    pub fn get_previous(&self, signal: &str, instance_idx: usize) -> Option<Value> {
        let meta = self.registry.get(signal)?;
        self.get_value(meta, instance_idx, true)
    }

    fn get_value(
        &self,
        meta: &MemberSignalMeta,
        instance_idx: usize,
        previous: bool,
    ) -> Option<Value> {
        match meta.value_type.buffer_class() {
            MemberBufferClass::Scalar => {
                let value = if previous {
                    self.scalars.get_previous(meta.buffer_index, instance_idx)
                } else {
                    self.scalars.get_current(meta.buffer_index, instance_idx)
                }?;
                Some(Value::Scalar(*value))
            }
            MemberBufferClass::Vec2 => {
                let value = if previous {
                    self.vec2s.get_previous(meta.buffer_index, instance_idx)
                } else {
                    self.vec2s.get_current(meta.buffer_index, instance_idx)
                }?;
                Some(Value::Vec2(*value))
            }
            MemberBufferClass::Vec3 => {
                let value = if previous {
                    self.vec3s.get_previous(meta.buffer_index, instance_idx)
                } else {
                    self.vec3s.get_current(meta.buffer_index, instance_idx)
                }?;
                Some(Value::Vec3(*value))
            }
            MemberBufferClass::Vec4 => self.get_vec4_value(meta, instance_idx, previous),
            MemberBufferClass::Boolean => {
                let value = if previous {
                    self.booleans.get_previous(meta.buffer_index, instance_idx)
                } else {
                    self.booleans.get_current(meta.buffer_index, instance_idx)
                }?;
                Some(Value::Boolean(*value))
            }
            MemberBufferClass::Integer => {
                let value = if previous {
                    self.integers.get_previous(meta.buffer_index, instance_idx)
                } else {
                    self.integers.get_current(meta.buffer_index, instance_idx)
                }?;
                Some(Value::Integer(*value))
            }
        }
    }

    fn get_vec4_value(
        &self,
        meta: &MemberSignalMeta,
        instance_idx: usize,
        previous: bool,
    ) -> Option<Value> {
        let value = if previous {
            self.vec4s
                .get_previous(meta.buffer_index, instance_idx)
                .copied()
        } else {
            self.vec4s
                .get_current(meta.buffer_index, instance_idx)
                .copied()
        }?;

        if meta.value_type.is_quat() {
            Some(Value::Quat(value))
        } else {
            Some(Value::Vec4(value))
        }
    }

    fn set_vec4_value(
        &mut self,
        meta: &MemberSignalMeta,
        instance_idx: usize,
        value: Value,
    ) -> Result<(), Value> {
        match (meta.value_type, value) {
            (value_type, Value::Vec4(v)) if value_type.is_vec4() => {
                self.vec4s.set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (value_type, Value::Quat(v)) if value_type.is_quat() => {
                self.vec4s.set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (_, value) => Err(value),
        }
    }

    /// Set current value from a Value enum.

    pub fn set_current(
        &mut self,
        signal: &str,
        instance_idx: usize,
        value: Value,
    ) -> Result<(), String> {
        let Some(meta) = self.registry.get(signal).cloned() else {
            return Err(format!("Unknown signal: {}", signal));
        };

        match (meta.value_type.buffer_class(), value) {
            (MemberBufferClass::Scalar, Value::Scalar(v)) => {
                self.scalars.set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Vec2, Value::Vec2(v)) => {
                self.vec2s.set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Vec3, Value::Vec3(v)) => {
                self.vec3s.set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Vec4, value) => {
                self.set_vec4_value(&meta, instance_idx, value)
                    .map_err(|value| {
                        format!(
                            "Type mismatch for signal {}: expected {:?}, got {:?}",
                            signal, meta.value_type, value
                        )
                    })?;
                Ok(())
            }
            (MemberBufferClass::Boolean, Value::Boolean(v)) => {
                self.booleans
                    .set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Integer, Value::Integer(v)) => {
                self.integers
                    .set_current(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (_, value) => Err(format!(
                "Type mismatch for signal {}: expected {:?}, got {:?}",
                signal, meta.value_type, value
            )),
        }
    }

    // ========================================================================
    // Direct typed access (zero-cost abstraction for hot paths)
    // ========================================================================

    /// Get scalar slice for a signal (all instances).
    ///
    /// Returns None if signal doesn't exist or isn't a scalar.
    pub fn scalar_slice(&self, signal: &str) -> Option<&[f64]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_scalar() {
            return None;
        }
        Some(self.scalars.signal_slice(meta.buffer_index))
    }

    /// Get mutable scalar slice for a signal.
    pub fn scalar_slice_mut(&mut self, signal: &str) -> Option<&mut [f64]> {
        let meta = self.registry.get(signal)?.clone();
        if !meta.value_type.is_scalar() {
            return None;
        }
        Some(self.scalars.signal_slice_mut(meta.buffer_index))
    }

    /// Get Vec3 slice for a signal.
    pub fn vec3_slice(&self, signal: &str) -> Option<&[[f64; 3]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_vec3() {
            return None;
        }
        Some(self.vec3s.signal_slice(meta.buffer_index))
    }

    /// Get mutable Vec3 slice for a signal.
    pub fn vec3_slice_mut(&mut self, signal: &str) -> Option<&mut [[f64; 3]]> {
        let meta = self.registry.get(signal)?.clone();
        if !meta.value_type.is_vec3() {
            return None;
        }
        Some(self.vec3s.signal_slice_mut(meta.buffer_index))
    }

    /// Get Quat slice for a signal.
    pub fn quat_slice(&self, signal: &str) -> Option<&[[f64; 4]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_quat() {
            return None;
        }
        Some(self.vec4s.signal_slice(meta.buffer_index))
    }

    /// Get mutable slice for a Quat signal.
    pub fn quat_slice_mut(&mut self, signal: &str) -> Option<&mut [[f64; 4]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_quat() {
            return None;
        }
        Some(self.vec4s.signal_slice_mut(meta.buffer_index))
    }

    // ========================================================================
    // Previous tick access (for lane kernel execution)
    // ========================================================================

    /// Get previous tick's scalar slice for a signal (all instances).
    ///
    /// This provides read-only access to the previous tick's values,
    /// enabling batch resolver execution.
    pub fn prev_scalar_slice(&self, signal: &str) -> Option<&[f64]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_scalar() {
            return None;
        }
        Some(self.scalars.prev_signal_slice(meta.buffer_index))
    }

    /// Get previous tick's Vec3 slice for a signal.
    pub fn prev_vec3_slice(&self, signal: &str) -> Option<&[[f64; 3]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_vec3() {
            return None;
        }
        Some(self.vec3s.prev_signal_slice(meta.buffer_index))
    }

    /// Get previous tick's Vec2 slice for a signal.
    pub fn prev_vec2_slice(&self, signal: &str) -> Option<&[[f64; 2]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_vec2() {
            return None;
        }
        Some(self.vec2s.prev_signal_slice(meta.buffer_index))
    }

    /// Get previous tick's Vec4 slice for a signal.
    pub fn prev_vec4_slice(&self, signal: &str) -> Option<&[[f64; 4]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_vec4() {
            return None;
        }
        Some(self.vec4s.prev_signal_slice(meta.buffer_index))
    }

    /// Get previous tick's Quat slice for a signal.
    pub fn prev_quat_slice(&self, signal: &str) -> Option<&[[f64; 4]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_quat() {
            return None;
        }
        Some(self.vec4s.prev_signal_slice(meta.buffer_index))
    }

    /// Advance tick: current becomes previous.
    pub fn advance_tick(&mut self) {
        self.scalars.advance_tick();
        self.vec2s.advance_tick();
        self.vec3s.advance_tick();
        self.vec4s.advance_tick();
        self.booleans.advance_tick();
        self.integers.advance_tick();
    }
}

impl Default for MemberSignalBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Population Storage
// ============================================================================

/// Storage for an entity population's member signals.
///
/// Combines stable instance IDs with SoA member signal storage.
/// The ID→index mapping allows string-based lookup while the underlying
/// storage uses contiguous indices for vectorization.
pub struct PopulationStorage {
    /// Entity type ID
    entity_id: EntityId,
    /// Stable instance ID → contiguous index
    id_to_index: IndexMap<String, usize>,
    /// Member signal storage
    signals: MemberSignalBuffer,
}

impl PopulationStorage {
    /// Create storage for an entity type.
    pub fn new(entity_id: EntityId) -> Self {
        Self {
            entity_id,
            id_to_index: IndexMap::new(),
            signals: MemberSignalBuffer::new(),
        }
    }

    /// Register a member signal type.
    pub fn register_signal(&mut self, name: String, value_type: ValueType) {
        self.signals.register_signal(name, value_type);
    }

    /// Register an instance and get its index.
    pub fn register_instance(&mut self, id: String) -> usize {
        let index = self.id_to_index.len();
        self.id_to_index.insert(id, index);
        index
    }

    /// Finalize registration and allocate storage.
    ///
    /// Must be called after all signals and instances are registered.
    pub fn finalize(&mut self) {
        self.signals.init_instances(self.id_to_index.len());
    }

    /// Get the entity ID.
    pub fn entity_id(&self) -> &EntityId {
        &self.entity_id
    }

    /// Get number of instances.
    pub fn instance_count(&self) -> usize {
        self.id_to_index.len()
    }

    /// Look up instance index by ID.
    pub fn instance_index(&self, id: &str) -> Option<usize> {
        self.id_to_index.get(id).copied()
    }

    /// Get instance ID by index.
    pub fn instance_id(&self, index: usize) -> Option<&String> {
        self.id_to_index.get_index(index).map(|(id, _)| id)
    }

    /// Iterate over instance IDs in deterministic order.
    pub fn instance_ids(&self) -> impl Iterator<Item = &String> {
        self.id_to_index.keys()
    }

    /// Get current signal value by instance ID.
    pub fn get_current(&self, instance_id: &str, signal: &str) -> Option<Value> {
        let idx = self.instance_index(instance_id)?;
        self.signals.get_current(signal, idx)
    }

    /// Get previous signal value by instance ID.
    pub fn get_previous(&self, instance_id: &str, signal: &str) -> Option<Value> {
        let idx = self.instance_index(instance_id)?;
        self.signals.get_previous(signal, idx)
    }

    /// Set current signal value by instance ID.
    pub fn set_current(&mut self, instance_id: &str, signal: &str, value: Value) {
        let idx = self.instance_index(instance_id).unwrap();
        let _ = self.signals.set_current(signal, idx, value);
    }

    /// Get direct access to member signal buffer.
    pub fn signals(&self) -> &MemberSignalBuffer {
        &self.signals
    }

    /// Get mutable access to member signal buffer.
    pub fn signals_mut(&mut self) -> &mut MemberSignalBuffer {
        &mut self.signals
    }

    /// Advance tick.
    pub fn advance_tick(&mut self) {
        self.signals.advance_tick();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_basic() {
        let mut buf: TypedBuffer<f64> = TypedBuffer::new();
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some(&1.0));
        assert_eq!(buf.get(1), Some(&2.0));
        assert_eq!(buf.get(2), Some(&3.0));
        assert_eq!(buf.get(3), None);
    }

    #[test]
    fn test_aligned_buffer_set() {
        let mut buf: TypedBuffer<f64> = TypedBuffer::new();
        buf.push(1.0);
        buf.push(2.0);

        buf.set(1, 42.0);
        assert_eq!(buf.get(1), Some(&42.0));
    }

    #[test]
    fn test_aligned_buffer_slice() {
        let mut buf: TypedBuffer<f64> = TypedBuffer::new();
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);

        let slice = buf.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_aligned_buffer_vec3() {
        let mut buf: TypedBuffer<[f64; 3]> = TypedBuffer::new();
        buf.push([1.0, 2.0, 3.0]);
        buf.push([4.0, 5.0, 6.0]);

        assert_eq!(buf.get(0), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(buf.get(1), Some(&[4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_member_signal_registry() {
        let mut registry = MemberSignalRegistry::new();

        let meta1 = registry.register("mass".to_string(), ValueType::scalar());
        assert_eq!(meta1.value_type, ValueType::scalar());
        assert_eq!(meta1.buffer_index, 0);

        let meta2 = registry.register("position".to_string(), ValueType::vec3());
        assert_eq!(meta2.value_type, ValueType::vec3());
        assert_eq!(meta2.buffer_index, 0); // First Vec3

        let meta3 = registry.register("velocity".to_string(), ValueType::vec3());
        assert_eq!(meta3.value_type, ValueType::vec3());
        assert_eq!(meta3.buffer_index, 1); // Second Vec3

        assert_eq!(registry.type_count(ValueType::scalar()), 1);
        assert_eq!(registry.type_count(ValueType::vec3()), 2);
    }

    #[test]
    fn test_member_signal_buffer_basic() {
        let mut buf = MemberSignalBuffer::new();

        buf.register_signal("mass".to_string(), ValueType::scalar());
        buf.register_signal("position".to_string(), ValueType::vec3());
        buf.init_instances(3);

        // Set values
        buf.set_current("mass", 0, Value::Scalar(100.0)).unwrap();
        buf.set_current("mass", 1, Value::Scalar(200.0)).unwrap();
        buf.set_current("mass", 2, Value::Scalar(300.0)).unwrap();

        buf.set_current("position", 0, Value::Vec3([1.0, 0.0, 0.0]))
            .unwrap();
        buf.set_current("position", 1, Value::Vec3([0.0, 1.0, 0.0]))
            .unwrap();
        buf.set_current("position", 2, Value::Vec3([0.0, 0.0, 1.0]))
            .unwrap();

        // Read back
        assert_eq!(buf.get_current("mass", 0), Some(Value::Scalar(100.0)));
        assert_eq!(buf.get_current("mass", 2), Some(Value::Scalar(300.0)));
        assert_eq!(
            buf.get_current("position", 1),
            Some(Value::Vec3([0.0, 1.0, 0.0]))
        );
    }

    #[test]
    fn test_member_signal_buffer_slices() {
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("mass".to_string(), ValueType::scalar());
        buf.init_instances(4);

        buf.set_current("mass", 0, Value::Scalar(1.0)).unwrap();
        buf.set_current("mass", 1, Value::Scalar(2.0)).unwrap();
        buf.set_current("mass", 2, Value::Scalar(3.0)).unwrap();
        buf.set_current("mass", 3, Value::Scalar(4.0)).unwrap();

        let slice = buf.scalar_slice("mass").unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_member_signal_buffer_tick_advance() {
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("mass".to_string(), ValueType::scalar());
        buf.init_instances(2);

        buf.set_current("mass", 0, Value::Scalar(100.0)).unwrap();
        buf.set_current("mass", 1, Value::Scalar(200.0)).unwrap();

        buf.advance_tick();

        // Previous now has old values
        assert_eq!(buf.get_previous("mass", 0), Some(Value::Scalar(100.0)));
        assert_eq!(buf.get_previous("mass", 1), Some(Value::Scalar(200.0)));

        // Set new current values
        buf.set_current("mass", 0, Value::Scalar(150.0)).unwrap();

        assert_eq!(buf.get_current("mass", 0), Some(Value::Scalar(150.0)));
        assert_eq!(buf.get_previous("mass", 0), Some(Value::Scalar(100.0)));
    }

    #[test]
    fn test_population_storage() {
        let mut pop = PopulationStorage::new("stellar.moon".into());

        pop.register_signal("mass".to_string(), ValueType::scalar());
        pop.register_signal("position".to_string(), ValueType::vec3());

        pop.register_instance("moon_1".to_string());
        pop.register_instance("moon_2".to_string());
        pop.finalize();

        assert_eq!(pop.instance_count(), 2);
        assert_eq!(pop.instance_index("moon_1"), Some(0));
        assert_eq!(pop.instance_index("moon_2"), Some(1));

        pop.set_current("moon_1", "mass", Value::Scalar(100.0));
        pop.set_current("moon_2", "mass", Value::Scalar(200.0));

        assert_eq!(
            pop.get_current("moon_1", "mass"),
            Some(Value::Scalar(100.0))
        );
        assert_eq!(
            pop.get_current("moon_2", "mass"),
            Some(Value::Scalar(200.0))
        );
    }

    #[test]
    fn test_population_storage_tick_advance() {
        let mut pop = PopulationStorage::new("stellar.moon".into());
        pop.register_signal("mass".to_string(), ValueType::scalar());
        pop.register_instance("moon_1".to_string());
        pop.finalize();

        pop.set_current("moon_1", "mass", Value::Scalar(100.0));
        pop.advance_tick();

        assert_eq!(
            pop.get_previous("moon_1", "mass"),
            Some(Value::Scalar(100.0))
        );

        pop.set_current("moon_1", "mass", Value::Scalar(150.0));
        assert_eq!(
            pop.get_current("moon_1", "mass"),
            Some(Value::Scalar(150.0))
        );
        assert_eq!(
            pop.get_previous("moon_1", "mass"),
            Some(Value::Scalar(100.0))
        );
    }

    // ========================================================================
    // Double-buffered tick semantics tests
    // ========================================================================

    #[test]
    fn test_double_buffer_tick_isolation() {
        // Verifies that writes to current buffer don't affect prev buffer
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("energy".to_string(), ValueType::scalar());
        buf.init_instances(3);

        // Write initial values to current tick
        buf.set_current("energy", 0, Value::Scalar(100.0)).unwrap();
        buf.set_current("energy", 1, Value::Scalar(200.0)).unwrap();
        buf.set_current("energy", 2, Value::Scalar(300.0)).unwrap();

        // Advance tick: current becomes previous
        buf.advance_tick();

        // Verify prev has the old values
        assert_eq!(buf.get_previous("energy", 0), Some(Value::Scalar(100.0)));
        assert_eq!(buf.get_previous("energy", 1), Some(Value::Scalar(200.0)));
        assert_eq!(buf.get_previous("energy", 2), Some(Value::Scalar(300.0)));

        // Write completely new values to current - SHOULD NOT affect prev
        buf.set_current("energy", 0, Value::Scalar(999.0)).unwrap();
        buf.set_current("energy", 1, Value::Scalar(888.0)).unwrap();
        buf.set_current("energy", 2, Value::Scalar(777.0)).unwrap();

        // Verify current has new values
        assert_eq!(buf.get_current("energy", 0), Some(Value::Scalar(999.0)));
        assert_eq!(buf.get_current("energy", 1), Some(Value::Scalar(888.0)));
        assert_eq!(buf.get_current("energy", 2), Some(Value::Scalar(777.0)));

        // CRITICAL: prev must still have original values (tick isolation)
        assert_eq!(buf.get_previous("energy", 0), Some(Value::Scalar(100.0)));
        assert_eq!(buf.get_previous("energy", 1), Some(Value::Scalar(200.0)));
        assert_eq!(buf.get_previous("energy", 2), Some(Value::Scalar(300.0)));
    }

    #[test]
    fn test_resolver_reads_prev_writes_current() {
        // Simulates a resolver operation:
        // - Read from prev_tick (snapshot of previous state)
        // - Compute new value based on prev
        // - Write to current_tick
        // - Verify reads from prev remain stable throughout
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("population".to_string(), ValueType::scalar());
        buf.init_instances(4);

        // Set initial population values
        buf.set_current("population", 0, Value::Scalar(1000.0))
            .unwrap();
        buf.set_current("population", 1, Value::Scalar(2000.0))
            .unwrap();
        buf.set_current("population", 2, Value::Scalar(3000.0))
            .unwrap();
        buf.set_current("population", 3, Value::Scalar(4000.0))
            .unwrap();

        // Advance tick to make these the "previous" values
        buf.advance_tick();

        // Simulate resolver: read prev, compute new, write current
        // Growth rate: 10% per tick
        let growth_rate = 0.1;

        for i in 0..4 {
            let prev = buf
                .get_previous("population", i)
                .unwrap()
                .as_scalar()
                .unwrap();
            let new_value = prev * (1.0 + growth_rate);
            buf.set_current("population", i, Value::Scalar(new_value))
                .unwrap();
        }

        // Verify final state
        assert_eq!(
            buf.get_current("population", 0),
            Some(Value::Scalar(1100.0))
        );
        assert_eq!(
            buf.get_current("population", 1),
            Some(Value::Scalar(2200.0))
        );
        assert_eq!(
            buf.get_previous("population", 0),
            Some(Value::Scalar(1000.0))
        );
        assert_eq!(
            buf.get_previous("population", 1),
            Some(Value::Scalar(2000.0))
        );
    }

    #[test]
    fn test_double_buffer_slice_isolation() {
        // Tests that slice access also maintains tick isolation
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("velocity".to_string(), ValueType::scalar());
        buf.init_instances(4);

        // Set initial velocities
        {
            let slice = buf.scalar_slice_mut("velocity").unwrap();
            slice[0] = 10.0;
            slice[1] = 20.0;
            slice[2] = 30.0;
            slice[3] = 40.0;
        }

        buf.advance_tick();

        // Verify prev slice has old values
        {
            let prev_slice = buf.prev_scalar_slice("velocity").unwrap();
            assert_eq!(prev_slice, &[10.0, 20.0, 30.0, 40.0]);
        }

        // Modify current slice
        {
            let current_slice = buf.scalar_slice_mut("velocity").unwrap();
            current_slice[0] = 100.0;
            current_slice[1] = 200.0;
            current_slice[2] = 300.0;
            current_slice[3] = 400.0;
        }

        // CRITICAL: prev slice must still have original values
        {
            let prev_slice = buf.prev_scalar_slice("velocity").unwrap();
            assert_eq!(prev_slice, &[10.0, 20.0, 30.0, 40.0]);
        }

        // Verify current slice has new values
        {
            let current_slice = buf.scalar_slice("velocity").unwrap();
            assert_eq!(current_slice, &[100.0, 200.0, 300.0, 400.0]);
        }
    }

    #[test]
    fn test_multiple_tick_advances() {
        // Tests that multiple tick advances maintain correct state progression
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("counter".to_string(), ValueType::scalar());
        buf.init_instances(1);

        // Tick 0: counter = 1
        buf.set_current("counter", 0, Value::Scalar(1.0)).unwrap();
        buf.advance_tick();
        assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(1.0)));

        buf.set_current("counter", 0, Value::Scalar(2.0)).unwrap();
        buf.advance_tick();
        assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(2.0)));

        buf.set_current("counter", 0, Value::Scalar(3.0)).unwrap();
        buf.advance_tick();
        assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(3.0)));

        buf.set_current("counter", 0, Value::Scalar(4.0)).unwrap();

        // Verify current state
        assert_eq!(buf.get_current("counter", 0), Some(Value::Scalar(4.0)));
        assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(3.0)));
    }

    #[test]
    fn test_vec3_double_buffer_isolation() {
        // Tests tick isolation for Vec3 type (not just scalars)
        let mut buf = MemberSignalBuffer::new();
        buf.register_signal("position".to_string(), ValueType::vec3());
        buf.init_instances(2);

        buf.set_current("position", 0, Value::Vec3([1.0, 2.0, 3.0]))
            .unwrap();
        buf.set_current("position", 1, Value::Vec3([4.0, 5.0, 6.0]))
            .unwrap();

        buf.advance_tick();

        assert_eq!(
            buf.get_previous("position", 0),
            Some(Value::Vec3([1.0, 2.0, 3.0]))
        );
        assert_eq!(
            buf.get_previous("position", 1),
            Some(Value::Vec3([4.0, 5.0, 6.0]))
        );

        // Set new values
        buf.set_current("position", 0, Value::Vec3([10.0, 20.0, 30.0]))
            .unwrap();
        buf.set_current("position", 1, Value::Vec3([40.0, 50.0, 60.0]))
            .unwrap();

        // CRITICAL: prev must be unchanged
        assert_eq!(
            buf.get_previous("position", 0),
            Some(Value::Vec3([1.0, 2.0, 3.0]))
        );
        assert_eq!(
            buf.get_previous("position", 1),
            Some(Value::Vec3([4.0, 5.0, 6.0]))
        );

        // Current has new values
        assert_eq!(
            buf.get_current("position", 0),
            Some(Value::Vec3([10.0, 20.0, 30.0]))
        );
    }
}
