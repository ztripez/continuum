//! GPU compute pipeline for field emission.
//!
//! The [`GpuFieldPipeline`] manages compute shaders and buffers for
//! GPU-accelerated field sample generation.

use std::borrow::Cow;

use indexmap::IndexMap;

use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::Value;

use crate::context::GpuContext;
use crate::emitter::{EmissionResult, EmitterBackend, FieldEmission, FieldEmitterConfig};
use crate::error::GpuError;

/// GPU pipeline for field emission compute shaders.
pub struct GpuFieldPipeline {
    /// GPU context (device, queue).
    context: GpuContext,
    /// Configuration.
    config: FieldEmitterConfig,
    /// Cached compute pipelines by expression ID.
    pipelines: IndexMap<usize, ComputePipelineBundle>,
    /// Reusable staging buffer for readback.
    staging_buffer: Option<wgpu::Buffer>,
    /// Maximum staging buffer size seen.
    max_staging_size: u64,
}

/// Bundle of resources for a compute pipeline.
struct ComputePipelineBundle {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuFieldPipeline {
    /// Create a new GPU field pipeline.
    pub fn new(context: GpuContext, config: FieldEmitterConfig) -> Result<Self, GpuError> {
        Ok(Self {
            context,
            config,
            pipelines: IndexMap::new(),
            staging_buffer: None,
            max_staging_size: 0,
        })
    }

    /// Emit field samples using GPU compute.
    pub fn emit_field(&mut self, emission: FieldEmission) -> Result<EmissionResult, GpuError> {
        let start = std::time::Instant::now();

        // Determine sample count
        let sample_count = if emission.positions.is_empty() {
            1
        } else {
            emission.positions.len()
        };

        // Ensure pipeline exists (mutable borrow ends after this)
        self.ensure_pipeline_exists(emission.expression_id)?;

        // Convert signal inputs from f64 to f32
        let signal_inputs_f32: Vec<f32> =
            emission.signal_inputs.iter().map(|&v| v as f32).collect();

        // Convert positions to f32
        let positions_f32: Vec<[f32; 3]> = if emission.positions.is_empty() {
            vec![[0.0, 0.0, 0.0]]
        } else {
            emission
                .positions
                .iter()
                .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
                .collect()
        };

        // Create buffers
        let signal_buffer = self.create_signal_buffer(&signal_inputs_f32)?;
        let position_buffer = self.create_position_buffer(&positions_f32)?;
        let output_buffer = self.create_output_buffer(sample_count)?;

        // Create bind group and dispatch in a scope to release borrow before readback
        {
            let bundle = &self.pipelines[&emission.expression_id];
            let bind_group =
                self.create_bind_group(bundle, &signal_buffer, &position_buffer, &output_buffer);
            self.dispatch_compute(&bundle.pipeline, &bind_group, sample_count)?;
        }

        // Read results back (needs &mut self for staging buffer)
        let results_f32 = self.read_output_buffer(&output_buffer, sample_count)?;

        // Convert to samples
        let samples: Vec<FieldSample> = if emission.positions.is_empty() {
            vec![FieldSample {
                position: [0.0, 0.0, 0.0],
                value: Value::Scalar(results_f32[0] as f64),
            }]
        } else {
            emission
                .positions
                .iter()
                .zip(results_f32.iter())
                .map(|(&pos, &val)| FieldSample {
                    position: pos,
                    value: Value::Scalar(val as f64),
                })
                .collect()
        };

        let time_us = start.elapsed().as_micros() as u64;

        Ok(EmissionResult {
            samples,
            backend: EmitterBackend::Gpu,
            time_us,
        })
    }

    /// Ensure a compute pipeline exists for an expression.
    fn ensure_pipeline_exists(&mut self, expression_id: usize) -> Result<(), GpuError> {
        if !self.pipelines.contains_key(&expression_id) {
            let bundle = self.create_pipeline(expression_id)?;
            self.pipelines.insert(expression_id, bundle);
        }
        Ok(())
    }

    /// Create a compute pipeline for an expression.
    fn create_pipeline(&self, _expression_id: usize) -> Result<ComputePipelineBundle, GpuError> {
        // TODO: Generate shader from expression via Naga
        // For now, use a simple sum shader as placeholder
        let shader_source = self.generate_shader_source();

        let shader_module =
            self.context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Field Compute Shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_source)),
                });

        let bind_group_layout =
            self.context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Field Compute Bind Group Layout"),
                    entries: &[
                        // Signal inputs (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Positions (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            self.context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Field Compute Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    ..Default::default()
                });

        let pipeline =
            self.context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Field Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(ComputePipelineBundle {
            pipeline,
            bind_group_layout,
        })
    }

    /// Generate WGSL shader source.
    ///
    /// TODO: This will be generated from DSL expressions via Naga.
    fn generate_shader_source(&self) -> String {
        format!(
            r#"
// Field compute shader - placeholder implementation
// TODO: Generate from DSL expression via Naga

@group(0) @binding(0) var<storage, read> signals: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{
        return;
    }}

    // Placeholder: sum all signal inputs
    var sum: f32 = 0.0;
    let signal_count = arrayLength(&signals);
    for (var i: u32 = 0u; i < signal_count; i = i + 1u) {{
        sum = sum + signals[i];
    }}

    output[idx] = sum;
}}
"#,
            self.config.workgroup_size
        )
    }

    /// Create a buffer for signal inputs.
    fn create_signal_buffer(&self, signals: &[f32]) -> Result<wgpu::Buffer, GpuError> {
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signal Input Buffer"),
            size: (signals.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.context.upload_buffer(&buffer, signals);
        Ok(buffer)
    }

    /// Create a buffer for positions.
    fn create_position_buffer(&self, positions: &[[f32; 3]]) -> Result<wgpu::Buffer, GpuError> {
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Buffer"),
            size: (positions.len() * std::mem::size_of::<[f32; 3]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.context.upload_buffer(&buffer, positions);
        Ok(buffer)
    }

    /// Create a buffer for output values.
    fn create_output_buffer(&self, sample_count: usize) -> Result<wgpu::Buffer, GpuError> {
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (sample_count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(buffer)
    }

    /// Create a bind group for compute dispatch.
    fn create_bind_group(
        &self,
        bundle: &ComputePipelineBundle,
        signal_buffer: &wgpu::Buffer,
        position_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Field Compute Bind Group"),
                layout: &bundle.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: signal_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: position_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            })
    }

    /// Dispatch compute shader.
    fn dispatch_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        sample_count: usize,
    ) -> Result<(), GpuError> {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Field Compute Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);

            // Calculate workgroup count
            let workgroup_size = self.config.workgroup_size as usize;
            let workgroup_count = (sample_count + workgroup_size - 1) / workgroup_size;

            pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        self.context.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Read output buffer back to CPU.
    fn read_output_buffer(
        &mut self,
        output_buffer: &wgpu::Buffer,
        sample_count: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let size = (sample_count * std::mem::size_of::<f32>()) as u64;

        // Ensure staging buffer is large enough
        if self.staging_buffer.is_none() || size > self.max_staging_size {
            self.staging_buffer = Some(self.context.create_staging_buffer("Staging Buffer", size));
            self.max_staging_size = size;
        }

        let staging = self.staging_buffer.as_ref().unwrap();

        // Copy output to staging
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Encoder"),
                });

        encoder.copy_buffer_to_buffer(output_buffer, 0, staging, 0, size);

        self.context.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = staging.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self
            .context
            .device
            .poll(wgpu::PollType::wait_indefinitely());

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();

        drop(mapped);
        staging.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_foundation::FieldId;

    #[test]
    #[ignore = "requires GPU"]
    fn test_pipeline_creation() {
        let context = GpuContext::new().unwrap();
        let config = FieldEmitterConfig::default();
        let pipeline = GpuFieldPipeline::new(context, config).unwrap();

        // Pipeline created successfully
        assert!(pipeline.pipelines.is_empty());
    }

    #[test]
    #[ignore = "requires GPU"]
    fn test_field_emission() {
        let context = GpuContext::new().unwrap();
        let config = FieldEmitterConfig::default();
        let mut pipeline = GpuFieldPipeline::new(context, config).unwrap();

        let emission = FieldEmission {
            field_id: FieldId::from("test.field"),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            signal_inputs: vec![1.0, 2.0, 3.0],
            expression_id: 0,
        };

        let result = pipeline.emit_field(emission).unwrap();

        assert_eq!(result.backend, EmitterBackend::Gpu);
        assert_eq!(result.samples.len(), 2);
        // Both samples should have sum of inputs (1+2+3=6)
        for sample in &result.samples {
            assert_eq!(sample.value, Value::Scalar(6.0));
        }
    }
}
