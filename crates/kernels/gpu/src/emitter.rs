//! High-level field emitter API with GPU/CPU abstraction.
//!
//! The [`FieldEmitter`] provides a unified interface for field emission that
//! automatically selects between GPU and CPU backends based on availability
//! and configuration.

use indexmap::IndexMap;

use continuum_foundation::FieldId;
use continuum_runtime::storage::{FieldBuffer, FieldSample};
use continuum_runtime::types::Value;

use crate::error::GpuError;

/// Configuration for field emission.
#[derive(Debug, Clone)]
pub struct FieldEmitterConfig {
    /// Minimum sample count to use GPU (below this, CPU is faster).
    pub gpu_threshold: usize,
    /// Workgroup size for compute shaders.
    pub workgroup_size: u32,
    /// Whether to prefer GPU when available.
    pub prefer_gpu: bool,
}

impl Default for FieldEmitterConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 1000,
            workgroup_size: 64,
            prefer_gpu: true,
        }
    }
}

/// Backend selection for field emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitterBackend {
    /// CPU-based emission (always available).
    Cpu,
    /// GPU compute shader emission.
    #[cfg(feature = "gpu")]
    Gpu,
}

/// Field emission request describing what to compute.
#[derive(Debug, Clone)]
pub struct FieldEmission {
    /// Field ID to emit to.
    pub field_id: FieldId,
    /// Sample positions (empty for scalar fields).
    pub positions: Vec<[f64; 3]>,
    /// Signal values used in computation (f64 for precision).
    pub signal_inputs: Vec<f64>,
    /// Expression to evaluate (index into compiled expression cache).
    pub expression_id: usize,
}

/// Result of field emission.
#[derive(Debug)]
pub struct EmissionResult {
    /// Field samples produced.
    pub samples: Vec<FieldSample>,
    /// Backend used for emission.
    pub backend: EmitterBackend,
    /// Time taken in microseconds.
    pub time_us: u64,
}

/// Unified field emitter with automatic GPU/CPU selection.
///
/// The emitter maintains GPU resources when available and falls back to
/// CPU emission transparently.
pub struct FieldEmitter {
    config: FieldEmitterConfig,
    #[cfg(feature = "gpu")]
    gpu_pipeline: Option<crate::GpuFieldPipeline>,
}

impl FieldEmitter {
    /// Create a new field emitter with default configuration.
    pub fn new() -> Self {
        Self::with_config(FieldEmitterConfig::default())
    }

    /// Create a new field emitter with the given configuration.
    pub fn with_config(config: FieldEmitterConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "gpu")]
            gpu_pipeline: None,
        }
    }

    /// Initialize GPU resources if available.
    ///
    /// This is optional - GPU will be initialized lazily on first use.
    /// Call this explicitly to fail fast if GPU is required.
    #[cfg(feature = "gpu")]
    pub fn init_gpu(&mut self) -> Result<(), GpuError> {
        if self.gpu_pipeline.is_some() {
            return Ok(());
        }

        let context = crate::GpuContext::new()?;
        self.gpu_pipeline = Some(crate::GpuFieldPipeline::new(context, self.config.clone())?);
        Ok(())
    }

    /// Check if GPU is available.
    #[cfg(feature = "gpu")]
    pub fn gpu_available(&self) -> bool {
        self.gpu_pipeline.is_some()
    }

    /// Check if GPU is available (always false without gpu feature).
    #[cfg(not(feature = "gpu"))]
    pub fn gpu_available(&self) -> bool {
        false
    }

    /// Get the current backend that will be used.
    pub fn current_backend(&self, sample_count: usize) -> EmitterBackend {
        #[cfg(feature = "gpu")]
        {
            if self.config.prefer_gpu
                && self.gpu_available()
                && sample_count >= self.config.gpu_threshold
            {
                return EmitterBackend::Gpu;
            }
        }
        EmitterBackend::Cpu
    }

    /// Emit field samples for multiple fields.
    ///
    /// Automatically batches fields by backend for optimal performance.
    pub fn emit_batch(
        &mut self,
        emissions: Vec<FieldEmission>,
    ) -> Result<IndexMap<FieldId, EmissionResult>, GpuError> {
        let mut results = IndexMap::new();

        // For now, process each emission individually
        // TODO: Batch GPU emissions for shared signal uploads
        for emission in emissions {
            let result = self.emit_single(emission.clone())?;
            results.insert(emission.field_id, result);
        }

        Ok(results)
    }

    /// Emit field samples for a single field.
    fn emit_single(&mut self, emission: FieldEmission) -> Result<EmissionResult, GpuError> {
        let sample_count = if emission.positions.is_empty() {
            1
        } else {
            emission.positions.len()
        };

        let backend = self.current_backend(sample_count);

        match backend {
            EmitterBackend::Cpu => self.emit_cpu(emission),
            #[cfg(feature = "gpu")]
            EmitterBackend::Gpu => self.emit_gpu(emission),
        }
    }

    /// CPU-based field emission.
    fn emit_cpu(&self, emission: FieldEmission) -> Result<EmissionResult, GpuError> {
        let start = std::time::Instant::now();

        let samples = if emission.positions.is_empty() {
            // Scalar field - single sample at origin
            vec![FieldSample {
                position: [0.0, 0.0, 0.0],
                value: self.evaluate_expression_cpu(&emission.signal_inputs, emission.expression_id),
            }]
        } else {
            // Spatial field - sample at each position
            emission
                .positions
                .iter()
                .map(|&pos| FieldSample {
                    position: pos,
                    value: self.evaluate_expression_cpu(&emission.signal_inputs, emission.expression_id),
                })
                .collect()
        };

        let time_us = start.elapsed().as_micros() as u64;

        Ok(EmissionResult {
            samples,
            backend: EmitterBackend::Cpu,
            time_us,
        })
    }

    /// GPU-based field emission.
    #[cfg(feature = "gpu")]
    fn emit_gpu(&mut self, emission: FieldEmission) -> Result<EmissionResult, GpuError> {
        // Lazy GPU initialization
        if self.gpu_pipeline.is_none() {
            self.init_gpu()?;
        }

        let pipeline = self.gpu_pipeline.as_mut().ok_or(GpuError::NoAdapter)?;
        pipeline.emit_field(emission)
    }

    /// Evaluate an expression on CPU.
    ///
    /// TODO: This will integrate with the expression compilation system.
    /// For now, it's a placeholder that returns the sum of inputs.
    fn evaluate_expression_cpu(&self, inputs: &[f64], _expression_id: usize) -> Value {
        // Placeholder: sum all inputs
        let sum: f64 = inputs.iter().sum();
        Value::Scalar(sum)
    }

    /// Convert emission results to a FieldBuffer.
    pub fn to_field_buffer(results: IndexMap<FieldId, EmissionResult>) -> FieldBuffer {
        let mut buffer = FieldBuffer::default();

        for (field_id, result) in results {
            for sample in result.samples {
                buffer.emit(field_id.clone(), sample.position, sample.value);
            }
        }

        buffer
    }
}

impl Default for FieldEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = FieldEmitterConfig::default();
        assert_eq!(config.gpu_threshold, 1000);
        assert_eq!(config.workgroup_size, 64);
        assert!(config.prefer_gpu);
    }

    #[test]
    fn test_emitter_creation() {
        let emitter = FieldEmitter::new();
        assert!(!emitter.gpu_available());
    }

    #[test]
    fn test_cpu_backend_selection() {
        let emitter = FieldEmitter::new();
        // Without GPU, should always select CPU
        assert_eq!(emitter.current_backend(100), EmitterBackend::Cpu);
        assert_eq!(emitter.current_backend(10000), EmitterBackend::Cpu);
    }

    #[test]
    fn test_cpu_scalar_emission() {
        let emitter = FieldEmitter::new();
        let emission = FieldEmission {
            field_id: FieldId::from("test.field"),
            positions: vec![],
            signal_inputs: vec![1.0, 2.0, 3.0],
            expression_id: 0,
        };

        let result = emitter.emit_cpu(emission).unwrap();

        assert_eq!(result.backend, EmitterBackend::Cpu);
        assert_eq!(result.samples.len(), 1);
        assert_eq!(result.samples[0].value, Value::Scalar(6.0)); // 1+2+3
    }

    #[test]
    fn test_cpu_spatial_emission() {
        let emitter = FieldEmitter::new();
        let emission = FieldEmission {
            field_id: FieldId::from("test.field"),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            signal_inputs: vec![10.0],
            expression_id: 0,
        };

        let result = emitter.emit_cpu(emission).unwrap();

        assert_eq!(result.samples.len(), 3);
        for sample in &result.samples {
            assert_eq!(sample.value, Value::Scalar(10.0));
        }
    }

    #[test]
    fn test_to_field_buffer() {
        let mut results = IndexMap::new();
        results.insert(
            FieldId::from("field1"),
            EmissionResult {
                samples: vec![FieldSample {
                    position: [0.0, 0.0, 0.0],
                    value: Value::Scalar(42.0),
                }],
                backend: EmitterBackend::Cpu,
                time_us: 100,
            },
        );

        let buffer = FieldEmitter::to_field_buffer(results);
        let samples = buffer.get_samples(&FieldId::from("field1")).unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].value, Value::Scalar(42.0));
    }
}
