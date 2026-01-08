//! Signal storage
//!
//! Manages current and previous tick values, plus input channel accumulation.

use indexmap::IndexMap;

use crate::types::{SignalId, Value};

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
