//! Continuum Lens (observer boundary).
//!
//! Lens is the canonical observer boundary. End programs must query fields
//! through Lens APIs only. FieldSnapshot is internal transport only.
//!
//! Lens ingests field emissions, stores bounded history, and reconstructs
//! continuous field functions for observation. It is observer-only and must
//! never influence execution or causality.
//!
//! # Observer Boundary Contract
//!
//! - Lens ingests field emissions from the Measure phase
//! - Lens structures samples using virtual topology
//! - Lens reconstructs continuous field functions from discrete samples
//! - Lens provides deterministic query APIs with temporal interpolation
//!
//! **CRITICAL**: Removing Lens must never change simulation outcomes. Lens is strictly
//! non-causal and observer-only.
//!
//! # Internal Transport vs Public API
//!
//! [`FieldSnapshot`] is **internal transport only**. End programs must NEVER read
//! `FieldSnapshot` directly. All field access must go through [`FieldLens`] query methods:
//! - [`FieldLens::query`] - query scalar at fractional time
//! - [`FieldLens::query_vector`] - query vector at fractional time
//! - [`FieldLens::query_batch`] - batch query (GPU-accelerated when available)
//! - [`FieldLens::at`] - get reconstruction for a specific tick
//! - [`FieldLens::tile`] - get reconstruction for a spatial tile
//!
//! # Fields Are Functions
//!
//! A field is a function `f: Position -> Value`, not raw sample data. Reconstruction
//! is **mandatory** for all observer use. Samples are constraints, not final data.

mod config;
mod error;
mod lens;
mod playback;
mod reconstruction;
mod refinement;
mod sink;
mod storage;
mod topology;

// Re-export public API
pub use config::{FieldConfig, FieldLensConfig};
pub use error::LensError;
pub use lens::FieldLens;
pub use playback::PlaybackClock;
pub use reconstruction::{FieldReconstruction, NearestNeighborReconstruction};
pub use refinement::{
    RefinementHandle, RefinementRequest, RefinementRequestSpec, RefinementStatus, Region,
};
pub use sink::ReconstructedSink;
pub use storage::{FieldFrame, FieldSnapshot};
pub use topology::{CubedSphereTopology, TileId};

// GPU module (feature-gated)
#[cfg(feature = "gpu")]
pub mod gpu {
    //! GPU-accelerated field sampling backend.
    //!
    //! Provides batch nearest-neighbor queries for scalar fields using compute shaders.
    //! GPU compute is observer-only and uses f32 internally for performance.
    //!
    //! # Usage
    //!
    //! ```ignore
    //! let gpu_ctx = GpuContext::new().await?;
    //! let backend = GpuLensBackend::new(gpu_ctx);
    //! lens.set_gpu_backend(backend);
    //!
    //! // Now query_batch will use GPU automatically
    //! let results = lens.query_batch(&field_id, &positions, tick)?;
    //! ```

    use std::borrow::Cow;

    use bytemuck::{Pod, Zeroable};
    use continuum_gpu::GpuContext;
    use wgpu::util::DeviceExt;

    /// GPU backend handle for Lens reconstruction.
    ///
    /// Provides batch nearest-neighbor queries for scalar fields. GPU compute
    /// is observer-only and uses f32 internally.
    pub struct GpuLensBackend {
        context: GpuContext,
        pipeline: Option<NearestNeighborPipeline>,
    }

    impl GpuLensBackend {
        /// Create a GPU backend from an existing GPU context.
        pub fn new(context: GpuContext) -> Self {
            Self {
                context,
                pipeline: None,
            }
        }

        /// Access the underlying GPU context.
        pub fn context(&self) -> &GpuContext {
            &self.context
        }

        /// Query a batch of scalar positions using GPU nearest-neighbor.
        ///
        /// Returns f64 values but uses f32 computation on the GPU.
        ///
        /// # Arguments
        /// * `samples` - Field samples as (position, value) pairs
        /// * `positions` - Query positions to evaluate
        ///
        /// # Returns
        /// Vector of nearest-neighbor values for each query position.
        pub fn query_scalar_batch(
            &mut self,
            samples: &[([f64; 3], f64)],
            positions: &[[f64; 3]],
        ) -> Result<Vec<f64>, String> {
            if samples.is_empty() || positions.is_empty() {
                return Ok(Vec::new());
            }

            self.ensure_pipeline();
            let pipeline = self.pipeline.as_ref().expect("pipeline exists");

            let (sample_positions, sample_values, query_positions) =
                Self::prepare_buffers(samples, positions);
            let buffers = self.create_gpu_buffers(
                pipeline,
                &sample_positions,
                &sample_values,
                &query_positions,
            );
            self.dispatch_and_read(pipeline, &buffers, positions.len())
        }

        fn ensure_pipeline(&mut self) {
            if self.pipeline.is_none() {
                self.pipeline = Some(NearestNeighborPipeline::new(&self.context));
            }
        }

        fn prepare_buffers(
            samples: &[([f64; 3], f64)],
            positions: &[[f64; 3]],
        ) -> (Vec<[f32; 4]>, Vec<f32>, Vec<[f32; 4]>) {
            let sample_positions: Vec<[f32; 4]> = samples
                .iter()
                .map(|(pos, _)| [pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.0])
                .collect();
            let sample_values: Vec<f32> = samples.iter().map(|(_, v)| *v as f32).collect();
            let query_positions: Vec<[f32; 4]> = positions
                .iter()
                .map(|pos| [pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.0])
                .collect();
            (sample_positions, sample_values, query_positions)
        }

        fn create_gpu_buffers(
            &self,
            pipeline: &NearestNeighborPipeline,
            sample_positions: &[[f32; 4]],
            sample_values: &[f32],
            query_positions: &[[f32; 4]],
        ) -> GpuBuffers {
            let device = self.context.device();
            let sample_count = sample_positions.len() as u32;
            let query_count = query_positions.len() as u32;

            let sample_pos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_samples_pos"),
                contents: bytemuck::cast_slice(sample_positions),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let sample_val_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_samples_val"),
                contents: bytemuck::cast_slice(sample_values),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let query_pos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_query_pos"),
                contents: bytemuck::cast_slice(query_positions),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lens_output"),
                size: (query_count as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let params = Params {
                sample_count,
                query_count,
                _pad: [0; 2],
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lens_bind_group"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sample_pos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sample_val_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: query_pos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            GpuBuffers {
                output_buffer,
                bind_group,
                query_count,
            }
        }

        fn dispatch_and_read(
            &self,
            pipeline: &NearestNeighborPipeline,
            buffers: &GpuBuffers,
            query_count: usize,
        ) -> Result<Vec<f64>, String> {
            let device = self.context.device();

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lens_query_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("lens_query_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &buffers.bind_group, &[]);
                let workgroup_size = 64;
                let workgroups = (buffers.query_count + workgroup_size - 1) / workgroup_size;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lens_output_staging"),
                size: (query_count as u64) * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                &buffers.output_buffer,
                0,
                &staging,
                0,
                (query_count as u64) * 4,
            );
            self.context.queue().submit(Some(encoder.finish()));
            let _ = self
                .context
                .device()
                .poll(wgpu::PollType::wait_indefinitely());

            let buffer_slice = staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let _ = self
                .context
                .device()
                .poll(wgpu::PollType::wait_indefinitely());
            receiver
                .recv()
                .map_err(|e| format!("gpu map recv error: {e}"))?
                .map_err(|e| format!("gpu map error: {e:?}"))?;

            let data = buffer_slice.get_mapped_range();
            let out_f32: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();

            Ok(out_f32.into_iter().map(|v| v as f64).collect())
        }
    }

    struct GpuBuffers {
        output_buffer: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        query_count: u32,
    }

    #[derive(Clone, Copy, Pod, Zeroable)]
    #[repr(C)]
    struct Params {
        sample_count: u32,
        query_count: u32,
        _pad: [u32; 2],
    }

    struct NearestNeighborPipeline {
        pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
    }

    impl NearestNeighborPipeline {
        fn new(context: &GpuContext) -> Self {
            let device = context.device();
            let shader_source = Cow::Borrowed(SHADER_SOURCE);
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lens_nearest_neighbor_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source),
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("lens_bind_group_layout"),
                    entries: &[
                        storage_buffer_entry(0, true),
                        storage_buffer_entry(1, true),
                        storage_buffer_entry(2, true),
                        storage_buffer_entry(3, false),
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lens_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lens_nearest_neighbor_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            Self {
                pipeline,
                bind_group_layout,
            }
        }
    }

    fn storage_buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    const SHADER_SOURCE: &str = r#"
struct Params {
    sample_count: u32,
    query_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> samples_pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> samples_val: array<f32>;
@group(0) @binding(2) var<storage, read> query_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_val: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (idx >= params.query_count) {
        return;
    }

    let q = query_pos[idx].xyz;
    var best_dist: f32 = 1e30;
    var best_val: f32 = 0.0;

    var i: u32 = 0u;
    loop {
        if (i >= params.sample_count) { break; }
        let s = samples_pos[i].xyz;
        let d = s - q;
        let dist = dot(d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best_val = samples_val[i];
        }
        i = i + 1u;
    }

    out_val[idx] = best_val;
}
"#;
}

#[cfg(test)]
mod tests;
