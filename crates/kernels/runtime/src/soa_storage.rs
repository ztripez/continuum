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
use std::marker::PhantomData;
use std::ptr::NonNull;

use indexmap::IndexMap;

use crate::types::{EntityId, Value};
use continuum_foundation::{primitive_type_by_name, PrimitiveStorageClass, PrimitiveTypeId};

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
    /// Create a scalar value type.
    pub fn scalar() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Scalar"))
    }

    /// Create a 2D vector value type.
    pub fn vec2() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Vec2"))
    }

    /// Create a 3D vector value type.
    pub fn vec3() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Vec3"))
    }

    /// Create a 4D vector value type.
    pub fn vec4() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Vec4"))
    }

    /// Create a quaternion value type.
    pub fn quat() -> Self {
        Self::from_primitive_id(PrimitiveTypeId::new("Quat"))
    }

    /// Create a boolean value type.
    pub fn boolean() -> Self {
        Self {
            kind: ValueTypeKind::Boolean,
        }
    }

    /// Create an integer value type.
    pub fn integer() -> Self {
        Self {
            kind: ValueTypeKind::Integer,
        }
    }

    /// Create a value type from a primitive type identifier.
    pub fn from_primitive_id(primitive_id: PrimitiveTypeId) -> Self {
        Self {
            kind: ValueTypeKind::Primitive(primitive_id),
        }
    }

    /// Get the underlying primitive type identifier, if this is a primitive type.
    pub fn primitive_id(&self) -> Option<PrimitiveTypeId> {
        match self.kind {
            ValueTypeKind::Primitive(id) => Some(id),
            _ => None,
        }
    }

    /// Check if this is a scalar type.
    pub fn is_scalar(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Scalar")
    }

    /// Check if this is a quaternion type.
    pub fn is_quat(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Quat")
    }

    /// Check if this is a 3D vector type.
    pub fn is_vec3(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Vec3")
    }

    /// Check if this is a 2D vector type.
    pub fn is_vec2(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Vec2")
    }

    /// Check if this is a 4D vector type.
    pub fn is_vec4(&self) -> bool {
        self.primitive_id().is_some_and(|id| id.name() == "Vec4")
    }

    /// Check if this is a boolean type.
    pub fn is_boolean(&self) -> bool {
        matches!(self.kind, ValueTypeKind::Boolean)
    }

    /// Check if this is an integer type.
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
            Value::Mat2(_) => {
                unimplemented!("Matrix types not supported for member signals (Mat2)")
            }
            Value::Mat3(_) => {
                unimplemented!("Matrix types not supported for member signals (Mat3)")
            }
            Value::Mat4(_) => {
                unimplemented!("Matrix types not supported for member signals (Mat4)")
            }
            Value::Boolean(_) => ValueType::boolean(),
            Value::Integer(_) => ValueType::integer(),
            Value::String(_) => panic!("String values are not supported in member signals"),
            Value::Map(_) => panic!("Map values are not supported in member signals"),
            Value::Seq(_) => panic!("Seq values are not supported in member signals"),
            Value::Tensor(_) => panic!("Tensor values are not yet supported in member signals"),
            Value::EntitySelf => panic!("EntitySelf marker is not a storable value type"),
        }
    }

    /// Convert from a primitive storage class.
    pub fn from_storage_class(storage: PrimitiveStorageClass) -> Self {
        match storage {
            PrimitiveStorageClass::Scalar => ValueType::scalar(),
            PrimitiveStorageClass::Vec2 => ValueType::vec2(),
            PrimitiveStorageClass::Vec3 => ValueType::vec3(),
            PrimitiveStorageClass::Vec4 => ValueType::vec4(),
            PrimitiveStorageClass::Mat2 => {
                unimplemented!("Matrix types not supported for member signals (Mat2)")
            }
            PrimitiveStorageClass::Mat3 => {
                unimplemented!("Matrix types not supported for member signals (Mat3)")
            }
            PrimitiveStorageClass::Mat4 => {
                unimplemented!("Matrix types not supported for member signals (Mat4)")
            }
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
                .unwrap_or_else(|| {
                    panic!(
                        "unknown primitive type '{}': cannot determine buffer layout",
                        id.name()
                    )
                }) {
                PrimitiveStorageClass::Scalar => MemberBufferClass::Scalar,
                PrimitiveStorageClass::Vec2 => MemberBufferClass::Vec2,
                PrimitiveStorageClass::Vec3 => MemberBufferClass::Vec3,
                PrimitiveStorageClass::Vec4 => MemberBufferClass::Vec4,
                PrimitiveStorageClass::Mat2 => {
                    unimplemented!("Matrix types not supported for member signals (Mat2)")
                }
                PrimitiveStorageClass::Mat3 => {
                    unimplemented!("Matrix types not supported for member signals (Mat3)")
                }
                PrimitiveStorageClass::Mat4 => {
                    unimplemented!("Matrix types not supported for member signals (Mat4)")
                }
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
            let dst = self
                .ptr
                .expect("ptr must be Some after reserve")
                .as_ptr()
                .add(self.len_bytes);
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
        if target_bytes > self.len_bytes
            && let Some(ptr) = self.ptr {
                unsafe {
                    let start = ptr.as_ptr().add(self.len_bytes);
                    std::ptr::write_bytes(start, 0, target_bytes - self.len_bytes);
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
        if let Some(ptr) = self.ptr
            && self.capacity_bytes > 0 {
                let layout = Layout::from_size_align(self.capacity_bytes, SIMD_ALIGNMENT)
                    .expect("Invalid layout in drop");
                unsafe {
                    alloc::dealloc(ptr.as_ptr(), layout);
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
    type_counts: IndexMap<MemberBufferClass, usize>,
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

    /// Set previous value for (signal, instance).
    ///
    /// Used by `init_global` to set both current and previous buffers
    /// when initializing a signal for the first time.
    fn set_previous(&mut self, signal_idx: usize, instance_idx: usize, value: T) {
        let idx = self.index(signal_idx, instance_idx);
        self.previous.set(idx, value);
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
    entity_instance_counts: IndexMap<String, usize>,
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
            entity_instance_counts: IndexMap::new(),
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

    /// Get all entity instance counts (for checkpoint serialization).
    pub fn entity_instance_counts(&self) -> &IndexMap<String, usize> {
        &self.entity_instance_counts
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

    /// Get all signal names for a specific entity.
    ///
    /// Returns a vector of full signal paths (e.g., "hydrology.cell.temperature")
    /// that belong to the specified entity.
    ///
    /// # Example
    /// ```ignore
    /// let signals = buffer.signals_for_entity("hydrology.cell");
    /// // Returns: ["hydrology.cell.temperature", "hydrology.cell.pressure", ...]
    /// ```
    pub fn signals_for_entity(&self, entity_id: &str) -> Vec<String> {
        self.registry
            .iter()
            .filter_map(|(name, _)| {
                if self.entity_id_from_signal(name).as_deref() == Some(entity_id) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect()
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

    /// Get zero-copy slice of scalar values for all instances of a signal.
    ///
    /// Returns None if signal doesn't exist or is not a scalar type.
    /// This enables efficient aggregate operations without boxing.
    pub fn get_scalar_slice(&self, signal: &str) -> Option<&[f64]> {
        let meta = self.registry.get(signal)?;
        if meta.value_type.buffer_class() != MemberBufferClass::Scalar {
            return None;
        }
        Some(self.scalars.signal_slice(meta.buffer_index))
    }

    /// Get zero-copy slice of Vec2 values for all instances of a signal.
    pub fn get_vec2_slice(&self, signal: &str) -> Option<&[[f64; 2]]> {
        let meta = self.registry.get(signal)?;
        if meta.value_type.buffer_class() != MemberBufferClass::Vec2 {
            return None;
        }
        Some(self.vec2s.signal_slice(meta.buffer_index))
    }

    /// Get zero-copy slice of Vec3 values for all instances of a signal.
    pub fn get_vec3_slice(&self, signal: &str) -> Option<&[[f64; 3]]> {
        let meta = self.registry.get(signal)?;
        if meta.value_type.buffer_class() != MemberBufferClass::Vec3 {
            return None;
        }
        Some(self.vec3s.signal_slice(meta.buffer_index))
    }

    /// Get zero-copy slice of Vec4 values for all instances of a signal.
    ///
    /// Note: This also works for Quat types (which are stored as Vec4).
    pub fn get_vec4_slice(&self, signal: &str) -> Option<&[[f64; 4]]> {
        let meta = self.registry.get(signal)?;
        let class = meta.value_type.buffer_class();
        if class != MemberBufferClass::Vec4 {
            return None;
        }
        Some(self.vec4s.signal_slice(meta.buffer_index))
    }

    /// Get zero-copy slice of boolean values for all instances of a signal.
    pub fn get_boolean_slice(&self, signal: &str) -> Option<&[bool]> {
        let meta = self.registry.get(signal)?;
        if meta.value_type.buffer_class() != MemberBufferClass::Boolean {
            return None;
        }
        Some(self.booleans.signal_slice(meta.buffer_index))
    }

    /// Get zero-copy slice of integer values for all instances of a signal.
    pub fn get_integer_slice(&self, signal: &str) -> Option<&[i64]> {
        let meta = self.registry.get(signal)?;
        if meta.value_type.buffer_class() != MemberBufferClass::Integer {
            return None;
        }
        Some(self.integers.signal_slice(meta.buffer_index))
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
            Some(Value::Quat(continuum_foundation::Quat(value)))
        } else {
            Some(Value::Vec4(value))
        }
    }

    #[allow(clippy::result_large_err)]
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
                self.vec4s.set_current(meta.buffer_index, instance_idx, v.0);
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

    /// Get Vec2 slice for a signal.
    pub fn vec2_slice(&self, signal: &str) -> Option<&[[f64; 2]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_vec2() {
            return None;
        }
        Some(self.vec2s.signal_slice(meta.buffer_index))
    }

    /// Get mutable Vec2 slice for a signal.
    pub fn vec2_slice_mut(&mut self, signal: &str) -> Option<&mut [[f64; 2]]> {
        let meta = self.registry.get(signal)?.clone();
        if !meta.value_type.is_vec2() {
            return None;
        }
        Some(self.vec2s.signal_slice_mut(meta.buffer_index))
    }

    /// Get Vec4 slice for a signal.
    pub fn vec4_slice(&self, signal: &str) -> Option<&[[f64; 4]]> {
        let meta = self.registry.get(signal)?;
        if !meta.value_type.is_vec4() {
            return None;
        }
        Some(self.vec4s.signal_slice(meta.buffer_index))
    }

    /// Get mutable Vec4 slice for a signal.
    pub fn vec4_slice_mut(&mut self, signal: &str) -> Option<&mut [[f64; 4]]> {
        let meta = self.registry.get(signal)?.clone();
        if !meta.value_type.is_vec4() {
            return None;
        }
        Some(self.vec4s.signal_slice_mut(meta.buffer_index))
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

    // ========================================================================
    // Global signal API (instance_count=1, instance_idx=0)
    // ========================================================================
    //
    // Global signals are stored in the same SoA buffers as entity member signals.
    // They are registered with their entity set to the root entity ("__root")
    // which has instance_count=1. Access uses instance_idx=0 always.

    /// Name of the synthetic root entity that holds global signals.
    pub const ROOT_ENTITY: &'static str = "__root";

    /// Register a global signal (root entity, instance_count=1).
    ///
    /// Must be called before `init_instances`. The signal is stored
    /// alongside member signals in the same SoA buffers.
    pub fn register_global_signal(&mut self, name: &str, value_type: ValueType) {
        self.registry.register(name.to_string(), value_type);
    }

    /// Ensure the root entity is registered with instance_count=1.
    ///
    /// Idempotent — safe to call multiple times.
    pub fn register_root_entity(&mut self) {
        if !self.entity_instance_counts.contains_key(Self::ROOT_ENTITY) {
            self.entity_instance_counts
                .insert(Self::ROOT_ENTITY.to_string(), 1);
        }
    }

    /// Ensure a global signal is registered and its storage is allocated.
    ///
    /// Idempotent — safe to call multiple times for the same signal.
    /// This is the incremental counterpart to the batch
    /// `register_global_signal` + `init_instances` path used by `build_runtime`.
    ///
    /// When called on an already-initialized buffer, grows only the relevant
    /// typed buffer by `instance_count` elements (appending at the end, which
    /// is correct because new signals get the highest `buffer_index`).
    ///
    /// When called on a fresh buffer (instance_count=0), sets instance_count=1
    /// and allocates storage for all registered signals so far.
    pub fn ensure_global_signal(&mut self, name: &str, value_type: ValueType) {
        // 1. Root entity must exist
        self.register_root_entity();

        // 2. Already registered — nothing to do
        if self.registry.get(name).is_some() {
            return;
        }

        // 3. Register the signal (assigns next buffer_index for its type)
        self.registry.register(name.to_string(), value_type);

        // 4. Ensure instance_count >= 1 for global signals
        if self.instance_count == 0 {
            // Fresh buffer — init everything for all registered signals
            self.init_instances(1);
        } else {
            // Already initialized — grow only the relevant typed buffer
            // by instance_count elements (one per entity instance for the new signal)
            let ic = self.instance_count;
            match value_type.buffer_class() {
                MemberBufferClass::Scalar => {
                    let new_count = self.registry.type_count(ValueType::scalar());
                    self.scalars.init(new_count, ic);
                }
                MemberBufferClass::Vec2 => {
                    let new_count = self.registry.type_count(ValueType::vec2());
                    self.vec2s.init(new_count, ic);
                }
                MemberBufferClass::Vec3 => {
                    let new_count = self.registry.type_count(ValueType::vec3());
                    self.vec3s.init(new_count, ic);
                }
                MemberBufferClass::Vec4 => {
                    let new_count = self.registry.type_count(ValueType::vec4());
                    self.vec4s.init(new_count, ic);
                }
                MemberBufferClass::Boolean => {
                    let new_count = self.registry.type_count(ValueType::boolean());
                    self.booleans.init(new_count, ic);
                }
                MemberBufferClass::Integer => {
                    let new_count = self.registry.type_count(ValueType::integer());
                    self.integers.init(new_count, ic);
                }
            }
        }
    }

    /// Initialize a global signal: set both current and previous to the given value.
    ///
    /// Sets both current and previous buffers — ensures the signal has a value
    /// in both tick buffers from the start.
    pub fn init_global(&mut self, signal: &str, value: Value) -> Result<(), String> {
        // Set current
        self.set_current(signal, 0, value.clone())?;
        // Also copy into previous buffer for `prev` access on tick 0
        self.set_previous_value(signal, 0, value)?;
        Ok(())
    }

    /// Get the current value of a global signal.
    pub fn get_global(&self, signal: &str) -> Option<Value> {
        self.get_current(signal, 0)
    }

    /// Get the previous tick's value of a global signal.
    pub fn get_global_prev(&self, signal: &str) -> Option<Value> {
        self.get_previous(signal, 0)
    }

    /// Get the current value of a global signal, falling back to previous if not yet resolved.
    ///
    /// Used during Resolve phase for
    /// intra-tick dependencies.
    pub fn get_global_or_prev(&self, signal: &str) -> Option<Value> {
        self.get_current(signal, 0)
            .or_else(|| self.get_previous(signal, 0))
    }

    /// Set the current tick's value for a global signal.
    pub fn set_global(&mut self, signal: &str, value: Value) -> Result<(), String> {
        self.set_current(signal, 0, value)
    }

    /// Check if a global signal is registered.
    pub fn has_global(&self, signal: &str) -> bool {
        self.registry.get(signal).is_some()
    }

    /// Get all global signal names (signals belonging to the root entity).
    pub fn global_signal_names(&self) -> Vec<&String> {
        self.registry
            .iter()
            .filter_map(|(name, _)| {
                if self.entity_id_from_signal(name).as_deref() == Some(Self::ROOT_ENTITY) {
                    Some(name)
                } else if self.entity_id_from_signal(name).is_none() {
                    // Signals without a dot-separated entity prefix are also globals
                    // (e.g., "counter" as opposed to "terra.plate.age")
                    Some(name)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Set the previous tick's value for a signal at a specific instance.
    ///
    /// Used by `init_global()` to set both current and previous buffers
    /// when initializing a signal for the first time, and by checkpoint
    /// restore to reconstruct previous-tick state.
    pub fn set_previous_value(
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
                self.scalars
                    .set_previous(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Vec2, Value::Vec2(v)) => {
                self.vec2s
                    .set_previous(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Vec3, Value::Vec3(v)) => {
                self.vec3s
                    .set_previous(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Vec4, value) => match (meta.value_type, value) {
                (vt, Value::Vec4(v)) if vt.is_vec4() => {
                    self.vec4s
                        .set_previous(meta.buffer_index, instance_idx, v);
                    Ok(())
                }
                (vt, Value::Quat(v)) if vt.is_quat() => {
                    self.vec4s
                        .set_previous(meta.buffer_index, instance_idx, v.0);
                    Ok(())
                }
                (_, value) => Err(format!(
                    "Type mismatch for signal {} previous: expected {:?}, got {:?}",
                    signal, meta.value_type, value
                )),
            },
            (MemberBufferClass::Boolean, Value::Boolean(v)) => {
                self.booleans
                    .set_previous(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (MemberBufferClass::Integer, Value::Integer(v)) => {
                self.integers
                    .set_previous(meta.buffer_index, instance_idx, v);
                Ok(())
            }
            (_, value) => Err(format!(
                "Type mismatch for signal {} previous: expected {:?}, got {:?}",
                signal, meta.value_type, value
            )),
        }
    }

    // Note on gated signal preservation for globals:
    // Global signals that were not resolved this tick (gated strata) need their
    // values carried forward. The existing `advance_tick()` above handles this
    // correctly because `DoubleBuffer::advance_tick()` swaps and then copies
    // previous into current, preserving values for gated signals.

    /// Get all signal names (both global and member).
    pub fn all_signal_names(&self) -> impl Iterator<Item = &String> {
        self.registry.iter().map(|(name, _)| name)
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
    ///
    /// Returns an error if the instance ID doesn't exist, or if the signal name
    /// is unknown, or if the value type doesn't match the signal's registered type.
    pub fn set_current(
        &mut self,
        instance_id: &str,
        signal: &str,
        value: Value,
    ) -> Result<(), String> {
        let idx = self
            .instance_index(instance_id)
            .ok_or_else(|| format!("Unknown instance_id: {instance_id}"))?;
        self.signals.set_current(signal, idx, value)
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

    /// Helper: create a MemberSignalBuffer with global signals registered and initialized.
    fn make_global_buffer() -> MemberSignalBuffer {
        let mut buf = MemberSignalBuffer::new();
        buf.register_root_entity();
        buf.register_global_signal("counter", ValueType::scalar());
        buf.register_global_signal("velocity", ValueType::vec3());
        // Use instance_count=1 for root entity (globals)
        buf.init_instances(1);
        buf
    }

    #[test]
    fn test_global_signal_init_and_read() {
        let mut buf = make_global_buffer();

        buf.init_global("counter", Value::Scalar(42.0))
            .expect("init_global should succeed");

        // Both current and previous should have the value
        assert_eq!(buf.get_global("counter"), Some(Value::Scalar(42.0)));
        assert_eq!(buf.get_global_prev("counter"), Some(Value::Scalar(42.0)));
    }

    #[test]
    fn test_global_signal_set_and_advance() {
        let mut buf = make_global_buffer();

        buf.init_global("counter", Value::Scalar(1.0))
            .expect("init");

        // Set new value in current tick
        buf.set_global("counter", Value::Scalar(2.0))
            .expect("set_global");

        // Current = 2.0, previous = 1.0
        assert_eq!(buf.get_global("counter"), Some(Value::Scalar(2.0)));
        assert_eq!(buf.get_global_prev("counter"), Some(Value::Scalar(1.0)));

        // Advance tick
        buf.advance_tick();

        // After advance, previous = 2.0 (was current), current = 2.0 (copied from prev)
        assert_eq!(buf.get_global_prev("counter"), Some(Value::Scalar(2.0)));
        assert_eq!(buf.get_global("counter"), Some(Value::Scalar(2.0)));
    }

    #[test]
    fn test_global_signal_or_prev_fallback() {
        let mut buf = make_global_buffer();

        buf.init_global("counter", Value::Scalar(10.0))
            .expect("init");

        // get_global_or_prev returns current when available
        assert_eq!(
            buf.get_global_or_prev("counter"),
            Some(Value::Scalar(10.0))
        );
    }

    #[test]
    fn test_global_signal_has() {
        let buf = make_global_buffer();
        assert!(buf.has_global("counter"));
        assert!(!buf.has_global("nonexistent"));
    }

    #[test]
    fn test_global_signal_vec3() {
        let mut buf = make_global_buffer();

        buf.init_global("velocity", Value::Vec3([1.0, 2.0, 3.0]))
            .expect("init vec3");

        assert_eq!(
            buf.get_global("velocity"),
            Some(Value::Vec3([1.0, 2.0, 3.0]))
        );

        buf.set_global("velocity", Value::Vec3([4.0, 5.0, 6.0]))
            .expect("set vec3");

        assert_eq!(
            buf.get_global("velocity"),
            Some(Value::Vec3([4.0, 5.0, 6.0]))
        );
        assert_eq!(
            buf.get_global_prev("velocity"),
            Some(Value::Vec3([1.0, 2.0, 3.0]))
        );
    }

    #[test]
    fn test_global_and_member_coexistence() {
        let mut buf = MemberSignalBuffer::new();
        buf.register_root_entity();
        // Global scalar
        buf.register_global_signal("global.temp", ValueType::scalar());
        // Member scalar (entity signal)
        buf.register_signal("terra.plate.age".to_string(), ValueType::scalar());
        buf.register_entity_count("terra.plate", 4);
        // Init with max instance count
        buf.init_instances(4);

        // Init global
        buf.init_global("global.temp", Value::Scalar(300.0))
            .expect("init global");

        // Init member for each instance
        for i in 0..4 {
            buf.set_current("terra.plate.age", i, Value::Scalar(i as f64 * 100.0))
                .expect("set member");
        }

        // Global should be readable
        assert_eq!(buf.get_global("global.temp"), Some(Value::Scalar(300.0)));

        // Members should be readable
        assert_eq!(
            buf.get_current("terra.plate.age", 0),
            Some(Value::Scalar(0.0))
        );
        assert_eq!(
            buf.get_current("terra.plate.age", 3),
            Some(Value::Scalar(300.0))
        );
    }

    #[test]
    fn test_set_previous_value() {
        let mut buf = make_global_buffer();
        buf.init_global("counter", Value::Scalar(1.0))
            .expect("init");

        // Directly set previous
        buf.set_previous_value("counter", 0, Value::Scalar(99.0))
            .expect("set previous");

        assert_eq!(buf.get_global_prev("counter"), Some(Value::Scalar(99.0)));
        // Current unchanged
        assert_eq!(buf.get_global("counter"), Some(Value::Scalar(1.0)));
    }
}
