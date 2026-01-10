//! Continuum Lens (observer boundary) - core storage and ingest.
//!
//! Lens ingests field emissions and stores latest + bounded history per field.
//! It is observer-only and must not influence execution.

use std::collections::VecDeque;

use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::FieldId;
use indexmap::IndexMap;
use thiserror::Error;

/// Lens configuration.
#[derive(Debug, Clone, Copy)]
pub struct FieldLensConfig {
    /// Maximum number of frames to retain per field.
    pub max_frames_per_field: usize,
}

impl FieldLensConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), LensError> {
        if self.max_frames_per_field == 0 {
            return Err(LensError::InvalidConfig(
                "max_frames_per_field must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for FieldLensConfig {
    fn default() -> Self {
        Self {
            max_frames_per_field: 1000,
        }
    }
}

/// Single-field snapshot frame stored by Lens.
#[derive(Debug, Clone)]
pub struct FieldFrame {
    pub tick: u64,
    pub samples: Vec<FieldSample>,
}

/// Input payload for ingesting a single field snapshot.
#[derive(Debug, Clone)]
pub struct FieldSnapshot {
    pub field_id: FieldId,
    pub tick: u64,
    pub samples: Vec<FieldSample>,
}

#[derive(Debug, Default)]
struct FieldStorage {
    history: VecDeque<FieldFrame>,
}

impl FieldStorage {
    fn push(&mut self, frame: FieldFrame, max_frames: usize) {
        if self.history.len() == max_frames {
            self.history.pop_front();
        }
        self.history.push_back(frame);
    }

    fn latest(&self) -> Option<&FieldFrame> {
        self.history.back()
    }
}

/// Lens error types.
#[derive(Debug, Error)]
pub enum LensError {
    #[error("Invalid lens config: {0}")]
    InvalidConfig(String),
}

/// Canonical observer boundary for field history.
#[derive(Debug, Default)]
pub struct FieldLens {
    config: FieldLensConfig,
    fields: IndexMap<FieldId, FieldStorage>,
}

impl FieldLens {
    /// Create a new lens with validated config.
    pub fn new(config: FieldLensConfig) -> Result<Self, LensError> {
        config.validate()?;
        Ok(Self {
            config,
            fields: IndexMap::new(),
        })
    }

    /// Record a single field snapshot.
    pub fn record(&mut self, snapshot: FieldSnapshot) {
        let storage = self.fields.entry(snapshot.field_id).or_default();
        storage.push(
            FieldFrame {
                tick: snapshot.tick,
                samples: snapshot.samples,
            },
            self.config.max_frames_per_field,
        );
    }

    /// Record a batch of fields for a single tick, preserving field order.
    pub fn record_many(
        &mut self,
        tick: u64,
        fields: IndexMap<FieldId, Vec<FieldSample>>,
    ) {
        for (field_id, samples) in fields {
            self.record(FieldSnapshot {
                field_id,
                tick,
                samples,
            });
        }
    }

    /// Get the latest frame for a field.
    pub fn latest(&self, field_id: &FieldId) -> Option<&FieldFrame> {
        self.fields.get(field_id).and_then(FieldStorage::latest)
    }

    /// Get bounded history for a field.
    pub fn history(&self, field_id: &FieldId) -> Option<&VecDeque<FieldFrame>> {
        self.fields.get(field_id).map(|storage| &storage.history)
    }

    /// Iterate over field IDs in deterministic insertion order.
    pub fn field_ids(&self) -> impl Iterator<Item = &FieldId> {
        self.fields.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_runtime::types::Value;

    fn sample(v: f64) -> FieldSample {
        FieldSample {
            position: [0.0, 0.0, 0.0],
            value: Value::Scalar(v),
        }
    }

    #[test]
    fn record_eviction_is_bounded() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
        })
        .expect("config valid");

        let field_id: FieldId = "terra.temp".into();

        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 1,
            samples: vec![sample(1.0)],
        });
        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 2,
            samples: vec![sample(2.0)],
        });
        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 3,
            samples: vec![sample(3.0)],
        });

        let history = lens.history(&field_id).expect("history exists");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].tick, 2);
        assert_eq!(history[1].tick, 3);

        let latest = lens.latest(&field_id).expect("latest exists");
        assert_eq!(latest.tick, 3);
    }

    #[test]
    fn record_many_preserves_field_order() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
        })
        .expect("config valid");

        let mut fields = IndexMap::new();
        fields.insert("field.a".into(), vec![sample(1.0)]);
        fields.insert("field.b".into(), vec![sample(2.0)]);

        lens.record_many(1, fields);

        let ids: Vec<String> = lens.field_ids().map(|id| id.to_string()).collect();
        assert_eq!(ids, vec!["field.a", "field.b"]);
    }
}
