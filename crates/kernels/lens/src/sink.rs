use indexmap::IndexMap;

use continuum_runtime::lens_sink::{LensData, LensSink, LensSinkError, Result};
use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::Value;

use crate::{FieldLens, FieldLensConfig};

/// Lens sink that reconstructs fields before emitting.
///
/// Uses [`FieldLens`] to ensure observer outputs are reconstructed values,
/// not raw samples.
pub struct ReconstructedSink {
    lens: FieldLens,
    inner: Box<dyn LensSink>,
}

impl ReconstructedSink {
    /// Create a reconstructed sink using default lens configuration.
    pub fn new(inner: Box<dyn LensSink>) -> Result<Self> {
        Self::with_config(inner, FieldLensConfig::default())
    }

    /// Create a reconstructed sink with explicit lens config.
    pub fn with_config(inner: Box<dyn LensSink>, config: FieldLensConfig) -> Result<Self> {
        let lens = FieldLens::new(config)
            .map_err(|e| LensSinkError::Config(format!("Lens config error: {}", e)))?;
        Ok(Self { lens, inner })
    }
}

impl LensSink for ReconstructedSink {
    fn emit_tick(&mut self, tick: u64, time_seconds: f64, data: LensData) -> Result<()> {
        let raw_fields = data.fields;
        self.lens.record_many(tick, raw_fields.clone());

        let mut reconstructed = IndexMap::new();
        for (field_id, samples) in raw_fields {
            let reconstruction = self
                .lens
                .at(&field_id, tick)
                .map_err(|e| LensSinkError::Serialization(e.to_string()))?;
            let rebuilt: Vec<FieldSample> = samples
                .iter()
                .map(|sample| {
                    let value = match &sample.value {
                        Value::Scalar(_) => Value::Scalar(reconstruction.query(sample.position)),
                        Value::Vec3(_) => Value::Vec3(reconstruction.query_vector(sample.position)),
                        other => {
                            return Err(LensSinkError::Serialization(format!(
                                "unsupported field value for reconstruction: {:?}",
                                other
                            )));
                        }
                    };
                    Ok(FieldSample {
                        position: sample.position,
                        value,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            reconstructed.insert(field_id, rebuilt);
        }

        self.inner.emit_tick(
            tick,
            time_seconds,
            LensData {
                fields: reconstructed,
            },
        )
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
