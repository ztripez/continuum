//! Continuum Lens (observer boundary) - core storage and ingest.
//!
//! Lens ingests field emissions and stores latest + bounded history per field.
//! It is observer-only and must not influence execution.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::FieldId;
use indexmap::IndexMap;
use thiserror::Error;

#[cfg(feature = "gpu")]
pub mod gpu {
    use std::borrow::Cow;

    use bytemuck::{Pod, Zeroable};
    use continuum_gpu::GpuContext;
    use wgpu::util::DeviceExt;

    /// GPU backend handle for Lens reconstruction.
    pub struct GpuLensBackend {
        context: GpuContext,
        pipeline: Option<NearestNeighborPipeline>,
    }

    impl GpuLensBackend {
        pub fn new(context: GpuContext) -> Self {
            Self {
                context,
                pipeline: None,
            }
        }

        pub fn context(&self) -> &GpuContext {
            &self.context
        }

        pub fn query_scalar_batch(
            &mut self,
            samples: &[( [f64; 3], f64 )],
            positions: &[[f64; 3]],
        ) -> Result<Vec<f64>, String> {
            if samples.is_empty() || positions.is_empty() {
                return Ok(Vec::new());
            }

            if self.pipeline.is_none() {
                self.pipeline = Some(NearestNeighborPipeline::new(&self.context));
            }
            let pipeline = self.pipeline.as_ref().expect("pipeline exists");
            let device = self.context.device();

            let sample_count = samples.len() as u32;
            let query_count = positions.len() as u32;

            let sample_positions: Vec<[f32; 4]> = samples
                .iter()
                .map(|(pos, _)| [pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.0])
                .collect();
            let sample_values: Vec<f32> = samples.iter().map(|(_, v)| *v as f32).collect();
            let query_positions: Vec<[f32; 4]> = positions
                .iter()
                .map(|pos| [pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.0])
                .collect();

            let sample_pos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_samples_pos"),
                contents: bytemuck::cast_slice(&sample_positions),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let sample_val_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_samples_val"),
                contents: bytemuck::cast_slice(&sample_values),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let query_pos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lens_query_pos"),
                contents: bytemuck::cast_slice(&query_positions),
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

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lens_query_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("lens_query_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroup_size = 64;
                let workgroups = (query_count + workgroup_size - 1) / workgroup_size;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lens_output_staging"),
                size: (query_count as u64) * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, (query_count as u64) * 4);
            self.context.queue().submit(Some(encoder.finish()));
            let _ = self.context.device().poll(wgpu::PollType::wait_indefinitely());

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
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
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

/// Lens configuration.
#[derive(Debug, Clone, Copy)]
pub struct FieldLensConfig {
    /// Maximum number of frames to retain per field.
    pub max_frames_per_field: usize,
    /// Maximum cached reconstructions per field.
    pub max_cached_per_field: usize,
    /// Maximum refinement requests buffered.
    pub max_refinement_queue: usize,
}

impl FieldLensConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), LensError> {
        if self.max_frames_per_field == 0 {
            return Err(LensError::InvalidConfig(
                "max_frames_per_field must be > 0".to_string(),
            ));
        }
        if self.max_cached_per_field == 0 {
            return Err(LensError::InvalidConfig(
                "max_cached_per_field must be > 0".to_string(),
            ));
        }
        if self.max_refinement_queue == 0 {
            return Err(LensError::InvalidConfig(
                "max_refinement_queue must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for FieldLensConfig {
    fn default() -> Self {
        Self {
            max_frames_per_field: 1000,
            max_cached_per_field: 32,
            max_refinement_queue: 1024,
        }
    }
}

/// Single-field snapshot frame stored by Lens.
#[derive(Debug, Clone)]
pub(crate) struct FieldFrame {
    pub tick: u64,
    pub samples: Vec<FieldSample>,
}

/// Input payload for ingesting a single field snapshot.
#[derive(Debug, Clone)]
pub(crate) struct FieldSnapshot {
    pub field_id: FieldId,
    pub tick: u64,
    pub samples: Vec<FieldSample>,
}

struct FieldStorage {
    history: VecDeque<FieldFrame>,
    cache: VecDeque<(u64, Arc<dyn FieldReconstruction>)>,
}

impl FieldStorage {
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
            cache: VecDeque::new(),
        }
    }

    fn push(&mut self, frame: FieldFrame, max_frames: usize) {
        if self.history.len() == max_frames {
            self.history.pop_front();
        }
        self.history.push_back(frame);
        self.cache.clear();
    }

    fn latest(&self) -> Option<&FieldFrame> {
        self.history.back()
    }

    fn cache_get(&self, tick: u64) -> Option<Arc<dyn FieldReconstruction>> {
        self.cache
            .iter()
            .find(|(cached_tick, _)| *cached_tick == tick)
            .map(|(_, recon)| Arc::clone(recon))
    }

    fn cache_insert(
        &mut self,
        tick: u64,
        recon: Arc<dyn FieldReconstruction>,
        max_cached: usize,
    ) {
        if let Some(pos) = self.cache.iter().position(|(t, _)| *t == tick) {
            self.cache.remove(pos);
        }
        if self.cache.len() == max_cached {
            self.cache.pop_front();
        }
        self.cache.push_back((tick, recon));
    }
}

/// Lens error types.
#[derive(Debug, Error)]
pub enum LensError {
    #[error("Invalid lens config: {0}")]
    InvalidConfig(String),
    #[error("Field not found: {0}")]
    FieldNotFound(FieldId),
    #[error("No samples for field {field} at tick {tick}")]
    NoSamplesAtTick { field: FieldId, tick: u64 },
    #[error("Refinement queue full")]
    RefinementQueueFull,
    #[error("GPU backend not configured")]
    GpuUnavailable,
    #[error("GPU query failed: {0}")]
    GpuQuery(String),
    #[error("Non-scalar sample encountered for GPU query: {0}")]
    NonScalarSample(FieldId),
}

/// Tile identifier for virtual topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(u64);

impl TileId {
    pub fn from_parts(face: u8, lod: u8, morton: u64) -> Self {
        let id = ((face as u64) << 56) | ((lod as u64) << 48) | (morton & 0x0000_FFFF_FFFF_FFFF);
        Self(id)
    }

    pub fn lod(self) -> u8 {
        ((self.0 >> 48) & 0xFF) as u8
    }
}

/// Region selector for topology queries.
#[derive(Debug, Clone)]
pub enum Region {
    Tile(TileId),
    SphereCap { center: [f64; 3], radius_rad: f64 },
}

/// Virtual topology interface (observer-only).
pub trait VirtualTopology: Send + Sync {
    fn tile_at(&self, position: [f64; 3], lod: u8) -> TileId;
}

/// Minimal cubed-sphere topology.
#[derive(Debug, Default, Clone)]
pub struct CubedSphereTopology;

impl CubedSphereTopology {
    fn face_and_uv(position: [f64; 3]) -> (u8, f64, f64) {
        let (x, y, z) = (position[0], position[1], position[2]);
        let ax = x.abs();
        let ay = y.abs();
        let az = z.abs();
        if ax >= ay && ax >= az {
            if x >= 0.0 {
                (0, -z / ax, y / ax)
            } else {
                (1, z / ax, y / ax)
            }
        } else if ay >= ax && ay >= az {
            if y >= 0.0 {
                (2, x / ay, -z / ay)
            } else {
                (3, x / ay, z / ay)
            }
        } else if z >= 0.0 {
            (4, x / az, y / az)
        } else {
            (5, -x / az, y / az)
        }
    }

    fn uv_to_morton(u: f64, v: f64, lod: u8) -> u64 {
        let grid = 1u64 << lod;
        let u = ((u + 1.0) * 0.5 * grid as f64).clamp(0.0, (grid - 1) as f64) as u64;
        let v = ((v + 1.0) * 0.5 * grid as f64).clamp(0.0, (grid - 1) as f64) as u64;
        let mut morton = 0u64;
        for i in 0..lod {
            morton |= ((u >> i) & 1) << (2 * i);
            morton |= ((v >> i) & 1) << (2 * i + 1);
        }
        morton
    }
}

impl VirtualTopology for CubedSphereTopology {
    fn tile_at(&self, position: [f64; 3], lod: u8) -> TileId {
        let (face, u, v) = Self::face_and_uv(position);
        let morton = Self::uv_to_morton(u, v, lod);
        TileId::from_parts(face, lod, morton)
    }
}

/// Canonical observer boundary for field history.
pub struct FieldLens {
    config: FieldLensConfig,
    fields: IndexMap<FieldId, FieldStorage>,
    topology: Arc<dyn VirtualTopology>,
    field_configs: HashMap<FieldId, FieldConfig>,
    refinement_queue: VecDeque<RefinementRequest>,
    refinement_status: HashMap<RefinementHandle, RefinementStatus>,
    next_refinement_id: u64,
    #[cfg(feature = "gpu")]
    gpu_backend: Option<gpu::GpuLensBackend>,
}

/// Playback clock for observer queries (fractional tick time).
#[derive(Debug, Clone)]
pub struct PlaybackClock {
    current_time: f64,
    lag_ticks: f64,
    speed: f64,
}

impl PlaybackClock {
    pub fn new(lag_ticks: f64) -> Self {
        Self {
            current_time: 0.0,
            lag_ticks,
            speed: 1.0,
        }
    }

    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.max(0.0);
    }

    pub fn seek(&mut self, time: f64) {
        self.current_time = time.max(0.0);
    }

    pub fn advance(&mut self, sim_tick: u64) {
        let target_time = sim_tick as f64 - self.lag_ticks;
        self.current_time = target_time.max(0.0) * self.speed;
    }

    pub fn bracketing_ticks(&self) -> (u64, u64, f64) {
        let tick_prev = self.current_time.floor() as u64;
        let tick_next = self.current_time.ceil() as u64;
        let alpha = self.current_time.fract();
        (tick_prev, tick_next, alpha)
    }
}

/// Reconstructed field interface (observer-only).
pub trait FieldReconstruction: Send + Sync {
    /// Query scalar value at position.
    fn query(&self, position: [f64; 3]) -> f64;
    /// Query vector value at position (default: scalar -> zero vector).
    fn query_vector(&self, position: [f64; 3]) -> [f64; 3] {
        let v = self.query(position);
        [v, 0.0, 0.0]
    }
    // Raw sample access intentionally omitted to enforce observer boundary.
}

/// Nearest-neighbor reconstruction (MVP).
pub struct NearestNeighborReconstruction {
    samples: Vec<FieldSample>,
}

impl NearestNeighborReconstruction {
    pub fn new(samples: Vec<FieldSample>) -> Self {
        Self { samples }
    }
}

impl FieldReconstruction for NearestNeighborReconstruction {
    fn query(&self, position: [f64; 3]) -> f64 {
        let mut best_dist = f64::MAX;
        let mut best_value = 0.0;
        for sample in &self.samples {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < best_dist {
                best_dist = dist;
                best_value = sample.value.as_scalar().unwrap_or(0.0);
            }
        }
        best_value
    }

    fn query_vector(&self, position: [f64; 3]) -> [f64; 3] {
        let mut best_dist = f64::MAX;
        let mut best_value = [0.0, 0.0, 0.0];
        for sample in &self.samples {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < best_dist {
                best_dist = dist;
                best_value = sample
                    .value
                    .as_vec3()
                    .map(|v| [v[0], v[1], v[2]])
                    .unwrap_or([0.0, 0.0, 0.0]);
            }
        }
        best_value
    }

}

impl FieldLens {
    /// Create a new lens with validated config.
    pub fn new(config: FieldLensConfig) -> Result<Self, LensError> {
        config.validate()?;
        Ok(Self {
            config,
            fields: IndexMap::new(),
            topology: Arc::new(CubedSphereTopology::default()),
            field_configs: HashMap::new(),
            refinement_queue: VecDeque::new(),
            refinement_status: HashMap::new(),
            next_refinement_id: 1,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        })
    }

    /// Record a single field snapshot.
pub(crate) fn record(&mut self, snapshot: FieldSnapshot) {
        let storage = self
            .fields
            .entry(snapshot.field_id)
            .or_insert_with(FieldStorage::new);
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
    /// Get reconstruction for a specific tick (nearest-neighbor MVP).
    pub fn at(
        &mut self,
        field_id: &FieldId,
        tick: u64,
    ) -> Result<Arc<dyn FieldReconstruction>, LensError> {
        let storage = self
            .fields
            .get(field_id)
            .ok_or_else(|| LensError::FieldNotFound(field_id.clone()))?;

        if let Some(cached) = storage.cache_get(tick) {
            return Ok(cached);
        }

        let frame = storage
            .history
            .iter()
            .find(|frame| frame.tick == tick)
            .ok_or_else(|| LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick,
            })?;

        if frame.samples.is_empty() {
            return Err(LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick,
            });
        }

        let recon = Arc::new(NearestNeighborReconstruction::new(
            frame.samples.clone(),
        ));
        let recon: Arc<dyn FieldReconstruction> = recon;

        let max_cached = self
            .field_configs
            .get(field_id)
            .and_then(|cfg| cfg.max_cached_per_field)
            .unwrap_or(self.config.max_cached_per_field);

        let storage = self.fields.get_mut(field_id).expect("storage exists");
        storage.cache_insert(tick, Arc::clone(&recon), max_cached);

        Ok(recon)
    }

    /// Get reconstruction for latest tick (nearest-neighbor MVP).
    pub fn latest_reconstruction(
        &mut self,
        field_id: &FieldId,
    ) -> Result<Arc<dyn FieldReconstruction>, LensError> {
        let storage = self
            .fields
            .get(field_id)
            .ok_or_else(|| LensError::FieldNotFound(field_id.clone()))?;
        let frame = storage.latest().ok_or_else(|| LensError::NoSamplesAtTick {
            field: field_id.clone(),
            tick: 0,
        })?;
        if frame.samples.is_empty() {
            return Err(LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick: frame.tick,
            });
        }
        self.at(field_id, frame.tick)
    }

    /// Get reconstruction for a specific tile at tick.
    pub fn tile(
        &self,
        field_id: &FieldId,
        tile_id: TileId,
        tick: u64,
    ) -> Result<Arc<dyn FieldReconstruction>, LensError> {
        let storage = self
            .fields
            .get(field_id)
            .ok_or_else(|| LensError::FieldNotFound(field_id.clone()))?;

        let frame = storage
            .history
            .iter()
            .find(|frame| frame.tick == tick)
            .ok_or_else(|| LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick,
            })?;

        if frame.samples.is_empty() {
            return Err(LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick,
            });
        }

        let samples = frame
            .samples
            .iter()
            .filter(|sample| self.topology.tile_at(sample.position, tile_id.lod()) == tile_id)
            .cloned()
            .collect::<Vec<_>>();

        if samples.is_empty() {
            return Err(LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick,
            });
        }

        Ok(Arc::new(NearestNeighborReconstruction::new(samples)))
    }

    /// Query scalar at a specific tick (spatial only, no temporal interpolation).
    pub fn query_at_tick(
        &mut self,
        field_id: &FieldId,
        position: [f64; 3],
        tick: u64,
    ) -> Result<f64, LensError> {
        let reconstruction = self.at(field_id, tick)?;
        Ok(reconstruction.query(position))
    }

    /// Query scalar value at fractional time (temporal interpolation).
    pub fn query(
        &mut self,
        field_id: &FieldId,
        position: [f64; 3],
        time: f64,
    ) -> Result<f64, LensError> {
        let tick_prev = time.floor() as u64;
        let tick_next = time.ceil() as u64;
        let alpha = time.fract();

        if alpha == 0.0 {
            return self.query_at_tick(field_id, position, tick_prev);
        }

        let prev = self.query_at_tick(field_id, position, tick_prev)?;
        let next = self.query_at_tick(field_id, position, tick_next)?;
        Ok(prev * (1.0 - alpha) + next * alpha)
    }

    /// Query scalar value using playback clock.
    pub fn query_playback(
        &mut self,
        field_id: &FieldId,
        position: [f64; 3],
        playback: &PlaybackClock,
    ) -> Result<f64, LensError> {
        self.query(field_id, position, playback.current_time())
    }

    /// Query vector value at fractional time (temporal interpolation).
    pub fn query_vector(
        &mut self,
        field_id: &FieldId,
        position: [f64; 3],
        time: f64,
    ) -> Result<[f64; 3], LensError> {
        let tick_prev = time.floor() as u64;
        let tick_next = time.ceil() as u64;
        let alpha = time.fract();

        if alpha == 0.0 {
            let reconstruction = self.at(field_id, tick_prev)?;
            return Ok(reconstruction.query_vector(position));
        }

        let prev = self.at(field_id, tick_prev)?.query_vector(position);
        let next = self.at(field_id, tick_next)?.query_vector(position);

        let lerped = [
            prev[0] * (1.0 - alpha) + next[0] * alpha,
            prev[1] * (1.0 - alpha) + next[1] * alpha,
            prev[2] * (1.0 - alpha) + next[2] * alpha,
        ];

        let mag = (lerped[0] * lerped[0] + lerped[1] * lerped[1] + lerped[2] * lerped[2]).sqrt();
        if mag > 0.0 {
            Ok([lerped[0] / mag, lerped[1] / mag, lerped[2] / mag])
        } else {
            Ok([0.0, 0.0, 0.0])
        }
    }

    /// Query a batch of positions at a specific tick.
    ///
    /// If GPU is enabled and configured, uses GPU batch query automatically.
    pub fn query_batch(
        &mut self,
        field_id: &FieldId,
        positions: &[[f64; 3]],
        tick: u64,
    ) -> Result<Vec<f64>, LensError> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "gpu")]
        if self.gpu_backend.is_some() {
            return self.query_batch_gpu(field_id, positions, tick);
        }

        let reconstruction = self.at(field_id, tick)?;
        Ok(positions
            .iter()
            .map(|pos| reconstruction.query(*pos))
            .collect())
    }

    /// Get bounded history metadata for a field.
    pub fn history_ticks(&self, field_id: &FieldId) -> Option<Vec<u64>> {
        self.fields
            .get(field_id)
            .map(|storage| storage.history.iter().map(|frame| frame.tick).collect())
    }

    /// Iterate over field IDs in deterministic insertion order.
    pub fn field_ids(&self) -> impl Iterator<Item = &FieldId> {
        self.fields.keys()
    }

    /// Configure per-field overrides.
    pub fn configure_field(&mut self, field_id: FieldId, config: FieldConfig) {
        self.field_configs.insert(field_id, config);
    }

    #[cfg(feature = "gpu")]
    pub fn set_gpu_backend(&mut self, backend: gpu::GpuLensBackend) {
        self.gpu_backend = Some(backend);
    }

    #[cfg(feature = "gpu")]
    pub fn query_batch_gpu(
        &mut self,
        field_id: &FieldId,
        positions: &[[f64; 3]],
        tick: u64,
    ) -> Result<Vec<f64>, LensError> {
        let backend = self.gpu_backend.as_mut().ok_or(LensError::GpuUnavailable)?;

        let storage = self
            .fields
            .get(field_id)
            .ok_or_else(|| LensError::FieldNotFound(field_id.clone()))?;

        let frame = storage
            .history
            .iter()
            .find(|frame| frame.tick == tick)
            .ok_or_else(|| LensError::NoSamplesAtTick {
                field: field_id.clone(),
                tick,
            })?;

        let mut samples = Vec::with_capacity(frame.samples.len());
        for sample in &frame.samples {
            let value = sample
                .value
                .as_scalar()
                .ok_or_else(|| LensError::NonScalarSample(field_id.clone()))?;
            samples.push((sample.position, value));
        }

        backend
            .query_scalar_batch(&samples, positions)
            .map_err(LensError::GpuQuery)
    }

    /// Request refinement of a field region.
    pub fn request_refinement(
        &mut self,
        request: RefinementRequestSpec,
    ) -> Result<RefinementHandle, LensError> {
        if self.refinement_queue.len() >= self.config.max_refinement_queue {
            return Err(LensError::RefinementQueueFull);
        }
        let handle = RefinementHandle(self.next_refinement_id);
        self.next_refinement_id += 1;
        self.refinement_queue.push_back(RefinementRequest {
            handle,
            field_id: request.field_id,
            region: request.region,
            target_lod: request.target_lod,
            priority: request.priority,
        });
        self.refinement_status
            .insert(handle, RefinementStatus::Pending);
        Ok(handle)
    }

    /// Check refinement status.
    pub fn refinement_status(&self, handle: RefinementHandle) -> Option<RefinementStatus> {
        self.refinement_status.get(&handle).copied()
    }

    /// Cancel a refinement request.
    pub fn cancel_refinement(&mut self, handle: RefinementHandle) {
        self.refinement_queue.retain(|req| req.handle != handle);
        self.refinement_status.remove(&handle);
    }

    /// Drain up to `max` refinement requests (for measurement system).
    pub fn drain_refinements(&mut self, max: usize) -> Vec<RefinementRequest> {
        let mut drained = Vec::new();
        let count = max.min(self.refinement_queue.len());
        for _ in 0..count {
            if let Some(req) = self.refinement_queue.pop_front() {
                if let Some(status) = self.refinement_status.get_mut(&req.handle) {
                    *status = RefinementStatus::Sampling;
                }
                drained.push(req);
            }
        }
        drained
    }
}

/// Per-field overrides for Lens behavior.
#[derive(Debug, Clone, Default)]
pub struct FieldConfig {
    pub max_cached_per_field: Option<usize>,
}

/// Refinement request handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RefinementHandle(u64);

/// Refinement request status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementStatus {
    Pending,
    Sampling,
    Complete,
    Failed,
}

/// Refinement request (observer-only).
#[derive(Debug, Clone)]
pub struct RefinementRequest {
    pub(crate) handle: RefinementHandle,
    pub field_id: FieldId,
    pub region: Region,
    pub target_lod: u8,
    pub priority: u32,
}

/// Public refinement request spec (handle assigned by Lens).
#[derive(Debug, Clone)]
pub struct RefinementRequestSpec {
    pub field_id: FieldId,
    pub region: Region,
    pub target_lod: u8,
    pub priority: u32,
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
            max_cached_per_field: 4,
            max_refinement_queue: 16,
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

        let ticks = lens.history_ticks(&field_id).expect("history exists");
        assert_eq!(ticks, vec![2, 3]);
    }

    #[test]
    fn record_many_preserves_field_order() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
            max_cached_per_field: 4,
            max_refinement_queue: 16,
        })
        .expect("config valid");

        let mut fields = IndexMap::new();
        fields.insert("field.a".into(), vec![sample(1.0)]);
        fields.insert("field.b".into(), vec![sample(2.0)]);

        lens.record_many(1, fields);

        let ids: Vec<String> = lens.field_ids().map(|id| id.to_string()).collect();
        assert_eq!(ids, vec!["field.a", "field.b"]);
    }

    #[test]
    fn cache_clears_on_new_record() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 3,
            max_cached_per_field: 2,
            max_refinement_queue: 16,
        })
        .expect("config valid");

        let field_id: FieldId = "field.temp".into();
        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 1,
            samples: vec![sample(1.0)],
        });

        let _ = lens
            .query_at_tick(&field_id, [0.0, 0.0, 0.0], 1)
            .expect("query works");

        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 2,
            samples: vec![sample(2.0)],
        });

        let storage = lens.fields.get(&field_id).expect("storage exists");
        assert!(storage.cache.is_empty());
    }

    #[test]
    fn query_at_tick_returns_nearest_sample() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
            max_cached_per_field: 4,
            max_refinement_queue: 16,
        })
        .expect("config valid");

        let field_id: FieldId = "field.elevation".into();
        let samples = vec![
            FieldSample {
                position: [0.0, 0.0, 0.0],
                value: Value::Scalar(10.0),
            },
            FieldSample {
                position: [10.0, 0.0, 0.0],
                value: Value::Scalar(20.0),
            },
        ];

        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 5,
            samples,
        });

        let value = lens
            .query_at_tick(&field_id, [1.0, 0.0, 0.0], 5)
            .expect("query works");
        assert_eq!(value, 10.0);
    }

    #[test]
    fn query_at_tick_errors_on_missing_field() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
            max_cached_per_field: 4,
            max_refinement_queue: 16,
        })
        .expect("config valid");

        let err = lens
            .query_at_tick(&"missing.field".into(), [0.0, 0.0, 0.0], 0)
            .expect_err("should error");

        match err {
            LensError::FieldNotFound(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn query_at_tick_errors_on_missing_tick() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
            max_cached_per_field: 4,
            max_refinement_queue: 16,
        })
        .expect("config valid");

        let field_id: FieldId = "field.elevation".into();
        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 1,
            samples: vec![sample(1.0)],
        });

        let err = lens
            .query_at_tick(&field_id, [0.0, 0.0, 0.0], 2)
            .expect_err("should error");

        match err {
            LensError::NoSamplesAtTick { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn playback_clock_brackets_ticks() {
        let mut clock = PlaybackClock::new(1.0);
        clock.advance(10);
        let (prev, next, alpha) = clock.bracketing_ticks();
        assert_eq!(prev, 9);
        assert_eq!(next, 9);
        assert_eq!(alpha, 0.0);

        clock.seek(9.5);
        let (prev, next, alpha) = clock.bracketing_ticks();
        assert_eq!(prev, 9);
        assert_eq!(next, 10);
        assert_eq!(alpha, 0.5);
    }

    #[test]
    fn query_interpolates_between_ticks() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 3,
            max_cached_per_field: 4,
            max_refinement_queue: 16,
        })
        .expect("config valid");

        let field_id: FieldId = "field.temp".into();
        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 1,
            samples: vec![sample(0.0)],
        });
        lens.record(FieldSnapshot {
            field_id: field_id.clone(),
            tick: 2,
            samples: vec![sample(10.0)],
        });

        let value = lens
            .query(&field_id, [0.0, 0.0, 0.0], 1.5)
            .expect("query works");
        assert_eq!(value, 5.0);
    }

    #[test]
    fn topology_tile_at_is_deterministic() {
        let topo = CubedSphereTopology::default();
        let pos = [1.0, 0.2, -0.3];
        let a = topo.tile_at(pos, 3);
        let b = topo.tile_at(pos, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn refinement_queue_tracks_status() {
        let mut lens = FieldLens::new(FieldLensConfig {
            max_frames_per_field: 2,
            max_cached_per_field: 4,
            max_refinement_queue: 4,
        })
        .expect("config valid");

        let handle = lens
            .request_refinement(RefinementRequestSpec {
                field_id: "field.temp".into(),
                region: Region::Tile(TileId::from_parts(0, 1, 0)),
                target_lod: 2,
                priority: 1,
            })
            .expect("request ok");

        assert_eq!(lens.refinement_status(handle), Some(RefinementStatus::Pending));

        let drained = lens.drain_refinements(1);
        assert_eq!(drained.len(), 1);
        assert_eq!(lens.refinement_status(handle), Some(RefinementStatus::Sampling));

        lens.cancel_refinement(handle);
        assert_eq!(lens.refinement_status(handle), None);
    }
}
