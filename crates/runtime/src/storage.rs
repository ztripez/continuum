//! Signal storage
//!
//! Manages current and previous tick values, plus input channel accumulation.

use indexmap::IndexMap;

use crate::types::{FieldId, SignalId, Value};

/// Storage for signal values across ticks
#[derive(Debug, Default)]
pub struct SignalStorage {
    /// Values resolved in the current tick
    current: IndexMap<SignalId, Value>,
    /// Values from the previous tick (for `prev` access)
    previous: IndexMap<SignalId, Value>,
}

impl SignalStorage {
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
#[derive(Debug, Default)]
pub struct InputChannels {
    /// Accumulated inputs per signal
    channels: IndexMap<SignalId, Vec<f64>>,
}

impl InputChannels {
    pub fn accumulate(&mut self, id: &SignalId, value: f64) {
        self.channels.entry(id.clone()).or_default().push(value);
    }

    pub fn drain_sum(&mut self, id: &SignalId) -> f64 {
        self.channels
            .shift_remove(id)
            .map(|values| values.iter().sum())
            .unwrap_or(0.0)
    }
}

/// Queued fracture outputs for next tick
#[derive(Debug, Default)]
pub struct FractureQueue {
    /// Outputs queued for next tick's Collect
    queue: Vec<(SignalId, f64)>,
}

impl FractureQueue {
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
#[derive(Debug, Clone)]
pub struct FieldSample {
    /// Position in field's coordinate space
    pub position: [f64; 3],
    /// Sample value
    pub value: Value,
}

/// Storage for field samples emitted during Measure phase
#[derive(Debug, Default)]
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
}
