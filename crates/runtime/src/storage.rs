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
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize a signal with a value
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

    /// Advance to next tick
    /// Signals that were resolved this tick have their new values in current.
    /// Signals that were gated keep their previous values.
    pub fn advance_tick(&mut self) {
        // For each signal: if it was resolved this tick (in current),
        // use that value; otherwise keep the previous value
        let to_preserve: Vec<_> = self
            .previous
            .iter()
            .filter(|(id, _)| !self.current.contains_key(*id))
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();

        for (id, value) in to_preserve {
            self.current.insert(id, value);
        }

        // Now swap - previous gets current (all resolved + preserved values)
        std::mem::swap(&mut self.previous, &mut self.current);
        // Clear current for next tick
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate a value into a signal's input channel
    pub fn accumulate(&mut self, id: &SignalId, value: f64) {
        self.channels.entry(id.clone()).or_default().push(value);
    }

    /// Drain and sum all accumulated inputs for a signal
    pub fn drain_sum(&mut self, id: &SignalId) -> f64 {
        self.channels
            .shift_remove(id)
            .map(|values| values.iter().sum())
            .unwrap_or(0.0)
    }

    /// Clear all channels (called after Resolve)
    pub fn clear(&mut self) {
        self.channels.clear();
    }
}

/// Queued fracture outputs for next tick
#[derive(Debug, Default)]
pub struct FractureQueue {
    /// Outputs queued for next tick's Collect
    queue: Vec<(SignalId, f64)>,
}

impl FractureQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a value to be applied next tick
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
        let mut storage = SignalStorage::new();
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
        let mut channels = InputChannels::new();
        let id: SignalId = "test.signal".into();

        channels.accumulate(&id, 1.0);
        channels.accumulate(&id, 2.0);
        channels.accumulate(&id, 3.0);

        assert_eq!(channels.drain_sum(&id), 6.0);
        assert_eq!(channels.drain_sum(&id), 0.0); // Drained
    }
}
