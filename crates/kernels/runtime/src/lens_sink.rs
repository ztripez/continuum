//! Lens Sink - Observer data output abstraction
//!
//! This module defines the sink interface for lens (observer) data output.
//!
//! # Philosophy
//!
//! Lens sinks handle **observer data only** (fields from Measure phase).
//! They are strictly non-causal and exist outside the observer boundary.
//!
//! **Separation of concerns**:
//! - **Checkpoints** → Causal state (signals, entities, members) → Resume
//! - **Lens Sinks** → Observer data (fields only) → Analysis/visualization
//!
//! # Sink Trait
//!
//! All sinks implement the `LensSink` trait:
//! - `emit_tick()` - Called after each Measure phase with field data
//! - `flush()` - Ensure buffered data is written
//! - `close()` - Clean shutdown (writes manifest, closes connections)
//!
//! # Implementations
//!
//! - `FileSink` - JSON files to disk (canonical format)
//! - `ParquetSink` - Column-oriented for analysis (future)
//! - `InfluxDBSink` - Time-series database (future)
//! - `NullSink` - Discard output (for performance testing)

pub mod file;

pub use file::{FileSink, FileSinkConfig};

use indexmap::IndexMap;

use crate::storage::FieldSample;
use crate::types::FieldId;

/// Result type for sink operations
pub type Result<T> = std::result::Result<T, LensSinkError>;

/// Errors that can occur during sink operations
#[derive(Debug, thiserror::Error)]
pub enum LensSinkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Sink configuration error: {0}")]
    Config(String),

    #[error("Sink already closed")]
    AlreadyClosed,
}

/// Lens data emitted per tick (fields only - no signals)
#[derive(Debug, Clone)]
pub struct LensData {
    /// Field samples emitted during Measure phase
    pub fields: IndexMap<FieldId, Vec<FieldSample>>,
}

/// Trait for lens data sinks
///
/// Sinks receive field data after each Measure phase and output it to
/// various backends (files, databases, streams, etc).
///
/// # Lifecycle
///
/// 1. Create sink with backend-specific configuration
/// 2. Call `emit_tick()` after each tick's Measure phase
/// 3. Call `flush()` periodically to ensure data is written
/// 4. Call `close()` when simulation ends to finalize output
pub trait LensSink: Send {
    /// Emit field data for a completed tick
    ///
    /// Called after Measure phase with all field samples for this tick.
    fn emit_tick(&mut self, tick: u64, time_seconds: f64, data: LensData) -> Result<()>;

    /// Flush any buffered data to backend
    ///
    /// Called periodically to ensure data durability.
    fn flush(&mut self) -> Result<()>;

    /// Close the sink and finalize output
    ///
    /// Called once at simulation end. Should write manifests, close
    /// connections, etc. After close(), the sink must not be used.
    fn close(&mut self) -> Result<()>;

    /// Returns the output path where this sink is writing data, if any.
    fn output_path(&self) -> Option<std::path::PathBuf> {
        None
    }
}

/// Null sink - discards all data (for performance testing)
pub struct NullSink;

impl LensSink for NullSink {
    fn emit_tick(&mut self, _tick: u64, _time_seconds: f64, _data: LensData) -> Result<()> {
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Configuration for lens sink behavior
#[derive(Debug, Clone)]
pub struct LensSinkConfig {
    /// Only emit data every N ticks
    pub stride: u64,

    /// Optional field filter (if empty, emit all fields)
    pub field_filter: Vec<FieldId>,
}

impl Default for LensSinkConfig {
    fn default() -> Self {
        Self {
            stride: 1,
            field_filter: Vec::new(),
        }
    }
}

/// Wrapper that applies stride and filtering to any sink
pub struct FilteredSink<S: LensSink> {
    inner: S,
    config: LensSinkConfig,
    last_emitted_tick: u64,
}

impl<S: LensSink> FilteredSink<S> {
    pub fn new(sink: S, config: LensSinkConfig) -> Self {
        Self {
            inner: sink,
            config,
            last_emitted_tick: 0,
        }
    }
}

impl<S: LensSink> LensSink for FilteredSink<S> {
    fn emit_tick(&mut self, tick: u64, time_seconds: f64, mut data: LensData) -> Result<()> {
        // Apply stride filter
        if tick < self.last_emitted_tick + self.config.stride {
            return Ok(());
        }
        self.last_emitted_tick = tick;

        // Apply field filter if specified
        if !self.config.field_filter.is_empty() {
            data.fields
                .retain(|field_id, _| self.config.field_filter.contains(field_id));
        }

        self.inner.emit_tick(tick, time_seconds, data)
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn output_path(&self) -> Option<std::path::PathBuf> {
        self.inner.output_path()
    }
}

/// Multi-sink - broadcast to multiple sinks simultaneously
pub struct MultiSink {
    sinks: Vec<Box<dyn LensSink>>,
}

impl MultiSink {
    pub fn new() -> Self {
        Self { sinks: Vec::new() }
    }

    pub fn add_sink(&mut self, sink: Box<dyn LensSink>) {
        self.sinks.push(sink);
    }
}

impl Default for MultiSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LensSink for MultiSink {
    fn emit_tick(&mut self, tick: u64, time_seconds: f64, data: LensData) -> Result<()> {
        if self.sinks.is_empty() {
            return Ok(());
        }

        let num_sinks = self.sinks.len();

        // Clone data for all sinks except the last
        for sink in &mut self.sinks[..num_sinks - 1] {
            sink.emit_tick(tick, time_seconds, data.clone())?;
        }

        // Move data to last sink
        self.sinks[num_sinks - 1].emit_tick(tick, time_seconds, data)?;

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        for sink in &mut self.sinks {
            sink.flush()?;
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        for sink in &mut self.sinks {
            sink.close()?;
        }
        Ok(())
    }

    fn output_path(&self) -> Option<std::path::PathBuf> {
        self.sinks.first().and_then(|s| s.output_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_sink() {
        let mut sink = NullSink;
        let data = LensData {
            fields: IndexMap::new(),
        };

        sink.emit_tick(0, 0.0, data.clone()).unwrap();
        sink.flush().unwrap();
        sink.close().unwrap();
    }

    #[test]
    fn test_filtered_sink_stride() {
        let null_sink = NullSink;
        let config = LensSinkConfig {
            stride: 10,
            field_filter: Vec::new(),
        };
        let mut filtered = FilteredSink::new(null_sink, config);

        let data = LensData {
            fields: IndexMap::new(),
        };

        // Should emit tick 0
        filtered.emit_tick(0, 0.0, data.clone()).unwrap();
        assert_eq!(filtered.last_emitted_tick, 0);

        // Should skip ticks 1-9
        for tick in 1..10 {
            filtered.emit_tick(tick, tick as f64, data.clone()).unwrap();
            assert_eq!(filtered.last_emitted_tick, 0);
        }

        // Should emit tick 10
        filtered.emit_tick(10, 10.0, data.clone()).unwrap();
        assert_eq!(filtered.last_emitted_tick, 10);
    }

    #[test]
    fn test_multi_sink() {
        let mut multi = MultiSink::new();
        multi.add_sink(Box::new(NullSink));
        multi.add_sink(Box::new(NullSink));

        let data = LensData {
            fields: IndexMap::new(),
        };

        multi.emit_tick(0, 0.0, data).unwrap();
        multi.flush().unwrap();
        multi.close().unwrap();
    }
}
