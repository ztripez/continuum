//! Signal and entity storage for simulation state.
//!
//! This module provides storage types for managing simulation state across
//! ticks. The storage system maintains both current and previous tick values,
//! enabling the `prev` keyword in DSL expressions.
//!
//! # Key Types
//!
//! - [`SignalStorage`] - Double-buffered signal values with tick advancement
//! - [`InputChannels`] - Accumulator for signal inputs during Collect phase
//! - [`FractureQueue`] - Queue for fracture outputs to be processed next tick
//! - [`FieldBuffer`] - Collector for measured field values
//! - [`EntityStorage`] - Storage for entity instances and their field values
//!
//! # Tick Lifecycle
//!
//! Each tick follows this storage pattern:
//!
//! 1. **Collect** - Inputs accumulated via [`InputChannels::accumulate`]
//! 2. **Resolve** - Signals read via [`SignalStorage::get`], written via `set_current`
//! 3. **Fracture** - Emissions queued via [`FractureQueue::queue`]
//! 4. **Measure** - Fields emitted via [`FieldBuffer::emit_scalar`]
//! 5. **Advance** - [`SignalStorage::advance_tick`] swaps current ↔ previous

use indexmap::IndexMap;

use crate::types::{EntityId, FieldId, InstanceId, SignalId, Value};

use serde::{Deserialize, Serialize};

/// Double-buffered storage for signal values across ticks.
///
/// SignalStorage maintains two value maps: `current` (being resolved this tick)
/// and `previous` (resolved last tick). This enables the `prev` keyword in DSL
/// expressions to access last tick's values while computing new ones.
///
/// # Tick Lifecycle
///
/// 1. **Resolve phase**: Read from `previous` via `get_prev()`, write to `current` via `set_current()`
/// 2. **Advance**: After tick completes, `current` becomes `previous` for next tick
///
/// # Gated Signals
///
/// When a stratum is gated (not executing), its signals are not resolved.
/// `advance_tick()` copies forward any unresolved signals from `previous`
/// to maintain state continuity.
///
/// # Example
///
/// ```
/// use continuum_runtime::storage::SignalStorage;
/// use continuum_runtime::{SignalId, Value};
///
/// let mut storage = SignalStorage::default();
/// let temp: SignalId = "terra.temp".into();
///
/// // Initialize signal
/// storage.init(temp.clone(), Value::Scalar(300.0));
///
/// // Resolve new value
/// storage.set_current(temp.clone(), Value::Scalar(301.0));
///
/// // Current tick sees new value
/// assert_eq!(storage.get(&temp), Some(&Value::Scalar(301.0)));
///
/// // Previous tick value still accessible
/// assert_eq!(storage.get_prev(&temp), Some(&Value::Scalar(300.0)));
///
/// // Advance to next tick
/// storage.advance_tick();
///
/// // Now 301.0 is the previous value
/// assert_eq!(storage.get_prev(&temp), Some(&Value::Scalar(301.0)));
/// ```
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SignalStorage {
    /// Values resolved in the current tick.
    current: IndexMap<SignalId, Value>,
    /// Values from the previous tick (for `prev` access).
    previous: IndexMap<SignalId, Value>,
}

impl SignalStorage {
    /// Initialize a signal with an initial value.
    pub fn init(&mut self, id: SignalId, value: Value) {
        self.previous.insert(id.clone(), value.clone());
        self.current.insert(id, value);
    }

    /// Get the previous tick's value for a signal
    pub fn get_prev(&self, id: &SignalId) -> Option<&Value> {
        self.previous.get(id)
    }

    /// Get a resolved signal value (current tick if resolved, else previous)
    /// Used during Resolve phase for intra-tick dependencies
    pub fn get(&self, id: &SignalId) -> Option<&Value> {
        self.current.get(id).or_else(|| self.previous.get(id))
    }

    /// Get the last fully resolved value (from previous tick)
    /// Used for external access after a tick completes
    pub fn get_resolved(&self, id: &SignalId) -> Option<&Value> {
        self.previous.get(id)
    }

    /// Set the resolved value for the current tick
    pub fn set_current(&mut self, id: SignalId, value: Value) {
        self.current.insert(id, value);
    }

    /// Advance to next tick. Gated signals keep their previous values.
    pub fn advance_tick(&mut self) {
        // Copy forward any signals not resolved this tick (gated strata)
        for (id, value) in &self.previous {
            if !self.current.contains_key(id) {
                self.current.insert(id.clone(), value.clone());
            }
        }
        std::mem::swap(&mut self.previous, &mut self.current);
        self.current.clear();
    }

    /// Get all signal IDs
    pub fn signal_ids(&self) -> impl Iterator<Item = &SignalId> {
        self.previous.keys()
    }
}

/// Accumulator for signal inputs during Collect phase
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InputChannels {
    /// Accumulated inputs per signal
    channels: IndexMap<SignalId, Vec<f64>>,
}

impl InputChannels {
    /// Accumulate an input value for a signal.
    pub fn accumulate(&mut self, id: &SignalId, value: f64) {
        self.channels.entry(id.clone()).or_default().push(value);
    }

    /// Sum all accumulated values for a signal and remove them from the channels.
    pub fn drain_sum(&mut self, id: &SignalId) -> f64 {
        self.channels
            .shift_remove(id)
            .map(|values| values.iter().sum())
            .unwrap_or(0.0)
    }
}

/// Queued fracture outputs for next tick
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FractureQueue {
    /// Outputs queued for next tick's Collect
    queue: Vec<(SignalId, f64)>,
}

impl FractureQueue {
    /// Add a signal emission to the next tick's queue.
    pub fn queue(&mut self, id: SignalId, value: f64) {
        self.queue.push((id, value));
    }

    /// Drain queued outputs into input channels
    pub fn drain_into(&mut self, channels: &mut InputChannels) {
        for (id, value) in self.queue.drain(..) {
            channels.accumulate(&id, value);
        }
    }
}

/// A single field sample (position + value)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSample {
    /// Position in field's coordinate space
    pub position: [f64; 3],
    /// Sample value
    pub value: Value,
}

/// Storage for field samples emitted during Measure phase
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FieldBuffer {
    /// Samples per field, collected during Measure phase
    samples: IndexMap<FieldId, Vec<FieldSample>>,
}

impl FieldBuffer {
    /// Emit a sample to a field
    pub fn emit(&mut self, field: FieldId, position: [f64; 3], value: Value) {
        self.samples
            .entry(field)
            .or_default()
            .push(FieldSample { position, value });
    }

    /// Emit a scalar sample (convenience for point values)
    pub fn emit_scalar(&mut self, field: FieldId, value: f64) {
        self.emit(field, [0.0, 0.0, 0.0], Value::Scalar(value));
    }

    /// Get samples for a field
    pub fn get_samples(&self, field: &FieldId) -> Option<&[FieldSample]> {
        self.samples.get(field).map(|v| v.as_slice())
    }

    /// Drain all samples (for observer consumption)
    pub fn drain(&mut self) -> IndexMap<FieldId, Vec<FieldSample>> {
        std::mem::take(&mut self.samples)
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Check if any samples exist
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get field IDs with samples
    pub fn field_ids(&self) -> impl Iterator<Item = &FieldId> {
        self.samples.keys()
    }

    /// Merge samples from another buffer into this one.
    ///
    /// Used for combining thread-local buffers after parallel emission.
    pub fn merge(&mut self, other: FieldBuffer) {
        for (field, mut samples) in other.samples {
            self.samples.entry(field).or_default().append(&mut samples);
        }
    }

    /// Get total sample count across all fields.
    pub fn sample_count(&self) -> usize {
        self.samples.values().map(|v| v.len()).sum()
    }
}

/// Thread-local field buffer for parallel emission.
///
/// Each thread gets its own `FieldBuffer`, allowing lock-free emission
/// during parallel Measure phase. After emission completes, call
/// [`merge_all`](Self::merge_all) to combine all thread-local buffers.
///
/// # Example
///
/// ```ignore
/// use rayon::prelude::*;
///
/// let parallel_buffer = ParallelFieldBuffer::new();
///
/// // Parallel emission
/// (0..1000).into_par_iter().for_each(|i| {
///     parallel_buffer.emit(field_id.clone(), [i as f64, 0.0, 0.0], Value::Scalar(i as f64));
/// });
///
/// // Merge all thread-local buffers
/// let result = parallel_buffer.merge_all();
/// ```
pub struct ParallelFieldBuffer {
    /// Per-thread buffers using thread_local crate.
    local_buffers: thread_local::ThreadLocal<std::cell::RefCell<FieldBuffer>>,
}

impl ParallelFieldBuffer {
    /// Create a new parallel field buffer.
    pub fn new() -> Self {
        Self {
            local_buffers: thread_local::ThreadLocal::new(),
        }
    }

    /// Emit a sample to a field (thread-safe, lock-free).
    pub fn emit(&self, field: FieldId, position: [f64; 3], value: Value) {
        self.local_buffers
            .get_or(|| std::cell::RefCell::new(FieldBuffer::default()))
            .borrow_mut()
            .emit(field, position, value);
    }

    /// Emit a scalar sample (convenience for point values).
    pub fn emit_scalar(&self, field: FieldId, value: f64) {
        self.emit(field, [0.0, 0.0, 0.0], Value::Scalar(value));
    }

    /// Merge all thread-local buffers into a single FieldBuffer.
    ///
    /// This consumes the parallel buffer and returns the merged result.
    pub fn merge_all(self) -> FieldBuffer {
        let mut result = FieldBuffer::default();
        for local in self.local_buffers.into_iter() {
            result.merge(local.into_inner());
        }
        result
    }

}

impl Default for ParallelFieldBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage for events emitted during Measure phase by chronicles.
///
/// Events are observer-only outputs that do not affect simulation causality.
/// They are collected here for logging, analytics, and external consumption.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EventBuffer {
    /// Events collected during Measure phase
    events: Vec<EmittedEventRecord>,
}

/// A recorded event with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedEventRecord {
    /// The event name (e.g., "climate.alert")
    pub name: String,
    /// Event fields as key-value pairs
    pub fields: Vec<(String, Value)>,
}

impl EventBuffer {
    /// Add an event to the buffer.
    pub fn emit(&mut self, name: String, fields: Vec<(String, Value)>) {
        self.events.push(EmittedEventRecord { name, fields });
    }

    /// Get all events.
    pub fn events(&self) -> &[EmittedEventRecord] {
        &self.events
    }

    /// Drain all events (for observer consumption).
    pub fn drain(&mut self) -> Vec<EmittedEventRecord> {
        std::mem::take(&mut self.events)
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Check if any events exist.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get event count.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Merge events from another buffer into this one.
    ///
    /// Used for combining thread-local buffers after parallel emission.
    pub fn merge(&mut self, mut other: EventBuffer) {
        self.events.append(&mut other.events);
    }
}

/// Data for a single entity instance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceData {
    /// Field values for this instance
    pub fields: IndexMap<String, Value>,
}

impl InstanceData {
    /// Create a new instance with the given fields
    pub fn new(fields: IndexMap<String, Value>) -> Self {
        Self { fields }
    }

    /// Get a field value
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.fields.get(field)
    }

    /// Set a field value
    pub fn set(&mut self, field: String, value: Value) {
        self.fields.insert(field, value);
    }
}

/// All instances of a single entity type, keyed by stable InstanceId
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityInstances {
    /// Map from instance ID to instance data (deterministic ordering via IndexMap)
    pub instances: IndexMap<InstanceId, InstanceData>,
}

impl EntityInstances {
    /// Create empty instances
    pub fn new() -> Self {
        Self {
            instances: IndexMap::new(),
        }
    }

    /// Add a new instance
    pub fn insert(&mut self, id: InstanceId, data: InstanceData) {
        self.instances.insert(id, data);
    }

    /// Get an instance by ID
    pub fn get(&self, id: &InstanceId) -> Option<&InstanceData> {
        self.instances.get(id)
    }

    /// Get mutable reference to an instance by ID
    pub fn get_mut(&mut self, id: &InstanceId) -> Option<&mut InstanceData> {
        self.instances.get_mut(id)
    }

    /// Get number of instances
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Iterate over all instance IDs in deterministic order
    pub fn instance_ids(&self) -> impl Iterator<Item = &InstanceId> {
        self.instances.keys()
    }

    /// Iterate over all instances in deterministic order
    pub fn iter(&self) -> impl Iterator<Item = (&InstanceId, &InstanceData)> {
        self.instances.iter()
    }

    /// Iterate over all instances mutably
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&InstanceId, &mut InstanceData)> {
        self.instances.iter_mut()
    }
}

/// Storage for entity instances across ticks
///
/// Uses stable InstanceIds (not numeric indexes) for deterministic iteration
/// across serialization/deserialization and parallel execution.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EntityStorage {
    /// Instances resolved in the current tick
    current: IndexMap<EntityId, EntityInstances>,
    /// Instances from the previous tick (for `prev` access)
    previous: IndexMap<EntityId, EntityInstances>,
}

impl EntityStorage {
    /// Initialize an entity type with its instances
    pub fn init_entity(&mut self, id: EntityId, instances: EntityInstances) {
        self.previous.insert(id.clone(), instances.clone());
        self.current.insert(id, instances);
    }

    /// Get the previous tick's data for an instance field
    pub fn get_prev_field(
        &self,
        entity: &EntityId,
        instance: &InstanceId,
        field: &str,
    ) -> Option<&Value> {
        self.previous
            .get(entity)
            .and_then(|e: &EntityInstances| e.get(instance))
            .and_then(|i: &InstanceData| i.get(field))
    }

    /// Get a field value for an instance (current if resolved, else previous)
    pub fn get_field(
        &self,
        entity: &EntityId,
        instance: &InstanceId,
        field: &str,
    ) -> Option<&Value> {
        self.current
            .get(entity)
            .and_then(|e: &EntityInstances| e.get(instance))
            .and_then(|i: &InstanceData| i.get(field))
            .or_else(|| self.get_prev_field(entity, instance, field))
    }

    /// Set a field value for the current tick
    pub fn set_field(
        &mut self,
        entity: &EntityId,
        instance: &InstanceId,
        field: String,
        value: Value,
    ) {
        if let Some(instances) = self.current.get_mut(entity) {
            if let Some(data) = instances.get_mut(instance) {
                data.set(field, value);
            }
        }
    }

    /// Get the number of instances for an entity type
    pub fn count(&self, entity: &EntityId) -> usize {
        self.previous
            .get(entity)
            .map(|e: &EntityInstances| e.count())
            .unwrap_or(0)
    }

    /// Get all instance IDs for an entity type (deterministic order)
    pub fn instance_ids(&self, entity: &EntityId) -> impl Iterator<Item = &InstanceId> {
        self.previous
            .get(entity)
            .into_iter()
            .flat_map(|e: &EntityInstances| e.instance_ids())
    }

    /// Get previous tick instances for an entity
    pub fn get_prev_instances(&self, entity: &EntityId) -> Option<&EntityInstances> {
        self.previous.get(entity)
    }

    /// Get current tick instances for an entity
    pub fn get_current_instances(&self, entity: &EntityId) -> Option<&EntityInstances> {
        self.current.get(entity)
    }

    /// Get mutable reference to current tick instances
    pub fn get_current_instances_mut(&mut self, entity: &EntityId) -> Option<&mut EntityInstances> {
        self.current.get_mut(entity)
    }

    /// Advance to next tick
    pub fn advance_tick(&mut self) {
        // Copy forward any entities not resolved this tick
        for (id, instances) in &self.previous {
            if !self.current.contains_key(id) {
                self.current.insert(id.clone(), instances.clone());
            }
        }
        std::mem::swap(&mut self.previous, &mut self.current);
        // Re-create current from previous (deep clone for field mutation)
        for (id, instances) in &self.previous {
            self.current.insert(id.clone(), instances.clone());
        }
    }

    /// Get all entity IDs
    pub fn entity_ids(&self) -> impl Iterator<Item = &EntityId> {
        self.previous.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_storage_tick_advance() {
        let mut storage = SignalStorage::default();
        let id: SignalId = "test.signal".into();

        storage.init(id.clone(), Value::Scalar(1.0));
        assert_eq!(storage.get_prev(&id), Some(&Value::Scalar(1.0)));

        storage.set_current(id.clone(), Value::Scalar(2.0));
        assert_eq!(storage.get(&id), Some(&Value::Scalar(2.0)));
        assert_eq!(storage.get_prev(&id), Some(&Value::Scalar(1.0)));

        storage.advance_tick();
        assert_eq!(storage.get_prev(&id), Some(&Value::Scalar(2.0)));
    }

    #[test]
    fn test_input_channels_accumulation() {
        let mut channels = InputChannels::default();
        let id: SignalId = "test.signal".into();

        channels.accumulate(&id, 1.0);
        channels.accumulate(&id, 2.0);
        channels.accumulate(&id, 3.0);

        assert_eq!(channels.drain_sum(&id), 6.0);
        assert_eq!(channels.drain_sum(&id), 0.0); // Drained
    }

    // ========================================================================
    // InstanceData Tests
    // ========================================================================

    #[test]
    fn test_instance_data_creation() {
        let mut fields = IndexMap::new();
        fields.insert("mass".to_string(), Value::Scalar(1000.0));
        fields.insert("position".to_string(), Value::Vec3([1.0, 2.0, 3.0]));

        let data = InstanceData::new(fields);
        assert_eq!(data.get("mass"), Some(&Value::Scalar(1000.0)));
        assert_eq!(data.get("position"), Some(&Value::Vec3([1.0, 2.0, 3.0])));
        assert_eq!(data.get("nonexistent"), None);
    }

    #[test]
    fn test_instance_data_set_field() {
        let mut data = InstanceData::default();
        data.set("mass".to_string(), Value::Scalar(500.0));

        assert_eq!(data.get("mass"), Some(&Value::Scalar(500.0)));

        // Update existing field
        data.set("mass".to_string(), Value::Scalar(600.0));
        assert_eq!(data.get("mass"), Some(&Value::Scalar(600.0)));
    }

    // ========================================================================
    // EntityInstances Tests
    // ========================================================================

    #[test]
    fn test_entity_instances_basic() {
        let mut instances = EntityInstances::new();
        let id1: InstanceId = "moon_1".into();
        let id2: InstanceId = "moon_2".into();

        let mut data1 = InstanceData::default();
        data1.set("mass".to_string(), Value::Scalar(100.0));

        let mut data2 = InstanceData::default();
        data2.set("mass".to_string(), Value::Scalar(200.0));

        instances.insert(id1.clone(), data1);
        instances.insert(id2.clone(), data2);

        assert_eq!(instances.count(), 2);
        assert_eq!(
            instances.get(&id1).unwrap().get("mass"),
            Some(&Value::Scalar(100.0))
        );
        assert_eq!(
            instances.get(&id2).unwrap().get("mass"),
            Some(&Value::Scalar(200.0))
        );
    }

    #[test]
    fn test_entity_instances_mutable_access() {
        let mut instances = EntityInstances::new();
        let id: InstanceId = "moon_1".into();

        let mut data = InstanceData::default();
        data.set("mass".to_string(), Value::Scalar(100.0));
        instances.insert(id.clone(), data);

        // Mutate via get_mut
        if let Some(inst) = instances.get_mut(&id) {
            inst.set("mass".to_string(), Value::Scalar(150.0));
        }

        assert_eq!(
            instances.get(&id).unwrap().get("mass"),
            Some(&Value::Scalar(150.0))
        );
    }

    #[test]
    fn test_entity_instances_iteration() {
        let mut instances = EntityInstances::new();
        let id1: InstanceId = "a".into();
        let id2: InstanceId = "b".into();

        instances.insert(id1.clone(), InstanceData::default());
        instances.insert(id2.clone(), InstanceData::default());

        let ids: Vec<_> = instances.instance_ids().collect();
        assert_eq!(ids.len(), 2);

        // iter() returns (id, data) pairs
        let pairs: Vec<_> = instances.iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    // ========================================================================
    // EntityStorage Tests
    // ========================================================================

    #[test]
    fn test_entity_storage_init() {
        let mut storage = EntityStorage::default();
        let entity_id: EntityId = "stellar.moon".into();
        let instance_id: InstanceId = "moon_1".into();

        let mut instances = EntityInstances::new();
        let mut data = InstanceData::default();
        data.set("mass".to_string(), Value::Scalar(1000.0));
        instances.insert(instance_id.clone(), data);

        storage.init_entity(entity_id.clone(), instances);

        // Verify count
        assert_eq!(storage.count(&entity_id), 1);

        // Verify field access
        assert_eq!(
            storage.get_field(&entity_id, &instance_id, "mass"),
            Some(&Value::Scalar(1000.0))
        );
    }

    #[test]
    fn test_entity_storage_set_field() {
        let mut storage = EntityStorage::default();
        let entity_id: EntityId = "stellar.moon".into();
        let instance_id: InstanceId = "moon_1".into();

        let mut instances = EntityInstances::new();
        let mut data = InstanceData::default();
        data.set("mass".to_string(), Value::Scalar(1000.0));
        instances.insert(instance_id.clone(), data);

        storage.init_entity(entity_id.clone(), instances);

        // Set new value
        storage.set_field(
            &entity_id,
            &instance_id,
            "mass".to_string(),
            Value::Scalar(1500.0),
        );

        // Current should have new value
        assert_eq!(
            storage.get_field(&entity_id, &instance_id, "mass"),
            Some(&Value::Scalar(1500.0))
        );

        // Previous should still have old value
        assert_eq!(
            storage.get_prev_field(&entity_id, &instance_id, "mass"),
            Some(&Value::Scalar(1000.0))
        );
    }

    #[test]
    fn test_entity_storage_advance_tick() {
        let mut storage = EntityStorage::default();
        let entity_id: EntityId = "stellar.moon".into();
        let instance_id: InstanceId = "moon_1".into();

        let mut instances = EntityInstances::new();
        let mut data = InstanceData::default();
        data.set("mass".to_string(), Value::Scalar(1000.0));
        instances.insert(instance_id.clone(), data);

        storage.init_entity(entity_id.clone(), instances);

        // Set new value
        storage.set_field(
            &entity_id,
            &instance_id,
            "mass".to_string(),
            Value::Scalar(1500.0),
        );

        // Advance tick
        storage.advance_tick();

        // Now previous has the 1500.0 value
        assert_eq!(
            storage.get_prev_field(&entity_id, &instance_id, "mass"),
            Some(&Value::Scalar(1500.0))
        );
    }

    #[test]
    fn test_entity_storage_multiple_entities() {
        let mut storage = EntityStorage::default();
        let moon_id: EntityId = "stellar.moon".into();
        let star_id: EntityId = "stellar.star".into();

        // Add moons
        let mut moon_instances = EntityInstances::new();
        let mut moon_data = InstanceData::default();
        moon_data.set("mass".to_string(), Value::Scalar(100.0));
        moon_instances.insert("moon_1".into(), moon_data);

        // Add stars
        let mut star_instances = EntityInstances::new();
        let mut star_data = InstanceData::default();
        star_data.set("luminosity".to_string(), Value::Scalar(5000.0));
        star_instances.insert("star_1".into(), star_data);

        storage.init_entity(moon_id.clone(), moon_instances);
        storage.init_entity(star_id.clone(), star_instances);

        assert_eq!(storage.count(&moon_id), 1);
        assert_eq!(storage.count(&star_id), 1);

        let entity_ids: Vec<_> = storage.entity_ids().collect();
        assert_eq!(entity_ids.len(), 2);
    }

    #[test]
    fn test_entity_storage_instance_ids() {
        let mut storage = EntityStorage::default();
        let entity_id: EntityId = "stellar.moon".into();

        let mut instances = EntityInstances::new();
        instances.insert("moon_1".into(), InstanceData::default());
        instances.insert("moon_2".into(), InstanceData::default());
        instances.insert("moon_3".into(), InstanceData::default());

        storage.init_entity(entity_id.clone(), instances);

        let ids: Vec<_> = storage.instance_ids(&entity_id).collect();
        assert_eq!(ids.len(), 3);
    }

    // ========================================================================
    // FieldBuffer Tests
    // ========================================================================

    #[test]
    fn test_field_buffer_emit_scalar() {
        let mut buffer = FieldBuffer::default();
        let field_id: FieldId = "terra.temp".into();

        buffer.emit_scalar(field_id.clone(), 300.0);
        buffer.emit_scalar(field_id.clone(), 301.0);

        let samples = buffer.get_samples(&field_id).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].value, Value::Scalar(300.0));
        assert_eq!(samples[1].value, Value::Scalar(301.0));
    }

    #[test]
    fn test_field_buffer_emit_with_position() {
        let mut buffer = FieldBuffer::default();
        let field_id: FieldId = "terra.temp".into();

        buffer.emit(
            field_id.clone(),
            [1.0, 2.0, 3.0],
            Value::Scalar(300.0),
        );

        let samples = buffer.get_samples(&field_id).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(samples[0].value, Value::Scalar(300.0));
    }

    #[test]
    fn test_field_buffer_drain() {
        let mut buffer = FieldBuffer::default();
        let field_id: FieldId = "terra.temp".into();

        buffer.emit_scalar(field_id.clone(), 300.0);
        assert!(!buffer.is_empty());

        let drained = buffer.drain();
        assert!(buffer.is_empty());
        assert!(drained.contains_key(&field_id));
    }

    #[test]
    fn test_field_buffer_clear() {
        let mut buffer = FieldBuffer::default();
        let field_id: FieldId = "terra.temp".into();

        buffer.emit_scalar(field_id.clone(), 300.0);
        buffer.clear();

        assert!(buffer.is_empty());
        assert!(buffer.get_samples(&field_id).is_none());
    }

    #[test]
    fn test_field_buffer_multiple_fields() {
        let mut buffer = FieldBuffer::default();
        let temp_id: FieldId = "terra.temp".into();
        let pressure_id: FieldId = "terra.pressure".into();

        buffer.emit_scalar(temp_id.clone(), 300.0);
        buffer.emit_scalar(pressure_id.clone(), 101.0);

        let field_ids: Vec<_> = buffer.field_ids().collect();
        assert_eq!(field_ids.len(), 2);
    }

    // ========================================================================
    // FractureQueue Tests
    // ========================================================================

    #[test]
    fn test_fracture_queue_basic() {
        let mut queue = FractureQueue::default();
        let signal_id: SignalId = "terra.stress".into();

        queue.queue(signal_id.clone(), 10.0);
        queue.queue(signal_id.clone(), 20.0);

        let mut channels = InputChannels::default();
        queue.drain_into(&mut channels);

        // Both values should be accumulated
        assert_eq!(channels.drain_sum(&signal_id), 30.0);
    }

    #[test]
    fn test_fracture_queue_drain_clears() {
        let mut queue = FractureQueue::default();
        let signal_id: SignalId = "terra.stress".into();

        queue.queue(signal_id.clone(), 10.0);

        let mut channels = InputChannels::default();
        queue.drain_into(&mut channels);

        // Draining again should produce nothing
        let mut channels2 = InputChannels::default();
        queue.drain_into(&mut channels2);
        assert_eq!(channels2.drain_sum(&signal_id), 0.0);
    }

    // ========================================================================
    // Gated Signal Tests
    // ========================================================================

    #[test]
    fn test_signal_storage_gated_signals_preserved() {
        let mut storage = SignalStorage::default();
        let fast_id: SignalId = "fast.signal".into();
        let slow_id: SignalId = "slow.signal".into();

        // Initialize both signals
        storage.init(fast_id.clone(), Value::Scalar(1.0));
        storage.init(slow_id.clone(), Value::Scalar(100.0));

        // Only resolve the fast signal (slow is gated)
        storage.set_current(fast_id.clone(), Value::Scalar(2.0));

        // Advance tick
        storage.advance_tick();

        // Fast signal has new value
        assert_eq!(storage.get_prev(&fast_id), Some(&Value::Scalar(2.0)));
        // Slow signal preserved from previous tick
        assert_eq!(storage.get_prev(&slow_id), Some(&Value::Scalar(100.0)));
    }

    // ========================================================================
    // ParallelFieldBuffer Tests
    // ========================================================================

    #[test]
    fn test_parallel_field_buffer_single_thread() {
        let buffer = ParallelFieldBuffer::new();
        let field_id: FieldId = "test.field".into();

        buffer.emit_scalar(field_id.clone(), 1.0);
        buffer.emit_scalar(field_id.clone(), 2.0);

        let merged = buffer.merge_all();
        let samples = merged.get_samples(&field_id).unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn test_parallel_field_buffer_with_positions() {
        let buffer = ParallelFieldBuffer::new();
        let field_id: FieldId = "spatial.field".into();

        buffer.emit(field_id.clone(), [1.0, 0.0, 0.0], Value::Scalar(10.0));
        buffer.emit(field_id.clone(), [2.0, 0.0, 0.0], Value::Scalar(20.0));

        let merged = buffer.merge_all();
        let samples = merged.get_samples(&field_id).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(samples[1].position, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_parallel_field_buffer_parallel_emission() {
        use rayon::prelude::*;

        let buffer = ParallelFieldBuffer::new();
        let field_id: FieldId = "parallel.field".into();

        // Emit from multiple threads
        (0..1000).into_par_iter().for_each(|i| {
            buffer.emit(
                field_id.clone(),
                [i as f64, 0.0, 0.0],
                Value::Scalar(i as f64),
            );
        });

        let merged = buffer.merge_all();
        let samples = merged.get_samples(&field_id).unwrap();

        // All 1000 samples should be present
        assert_eq!(samples.len(), 1000);

        // Verify values (order is not guaranteed)
        let sum: f64 = samples
            .iter()
            .filter_map(|s| match s.value {
                Value::Scalar(v) => Some(v),
                _ => None,
            })
            .sum();
        // Sum of 0..1000 = 499500
        assert_eq!(sum, 499500.0);
    }

    #[test]
    fn test_parallel_field_buffer_multiple_fields() {
        use rayon::prelude::*;

        let buffer = ParallelFieldBuffer::new();
        let field_a: FieldId = "field.a".into();
        let field_b: FieldId = "field.b".into();

        (0..100).into_par_iter().for_each(|i| {
            if i % 2 == 0 {
                buffer.emit_scalar(field_a.clone(), i as f64);
            } else {
                buffer.emit_scalar(field_b.clone(), i as f64);
            }
        });

        let merged = buffer.merge_all();

        // 50 samples in each field
        assert_eq!(merged.get_samples(&field_a).unwrap().len(), 50);
        assert_eq!(merged.get_samples(&field_b).unwrap().len(), 50);
    }

    #[test]
    fn test_field_buffer_merge() {
        let mut buffer1 = FieldBuffer::default();
        let mut buffer2 = FieldBuffer::default();
        let field_id: FieldId = "merged.field".into();

        buffer1.emit_scalar(field_id.clone(), 1.0);
        buffer1.emit_scalar(field_id.clone(), 2.0);

        buffer2.emit_scalar(field_id.clone(), 3.0);
        buffer2.emit_scalar(field_id.clone(), 4.0);

        buffer1.merge(buffer2);

        let samples = buffer1.get_samples(&field_id).unwrap();
        assert_eq!(samples.len(), 4);
    }

    #[test]
    fn test_field_buffer_sample_count() {
        let mut buffer = FieldBuffer::default();
        let field_a: FieldId = "field.a".into();
        let field_b: FieldId = "field.b".into();

        buffer.emit_scalar(field_a.clone(), 1.0);
        buffer.emit_scalar(field_a.clone(), 2.0);
        buffer.emit_scalar(field_b.clone(), 3.0);

        assert_eq!(buffer.sample_count(), 3);
    }
}
