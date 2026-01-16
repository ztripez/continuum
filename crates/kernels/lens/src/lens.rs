//! Core FieldLens implementation.

use std::collections::HashMap;
use std::sync::Arc;

use continuum_foundation::{FieldId, FieldSample};
use indexmap::IndexMap;

use crate::config::{FieldConfig, FieldLensConfig};
use crate::error::LensError;
use crate::playback::PlaybackClock;
use crate::reconstruction::{FieldReconstruction, NearestNeighborReconstruction};
use crate::refinement::{
    RefinementHandle, RefinementQueue, RefinementRequest, RefinementRequestSpec, RefinementStatus,
};
use crate::storage::{FieldFrame, FieldSnapshot, FieldStorage};
use crate::topology::{CubedSphereTopology, TileId};

/// Canonical observer boundary for field history and reconstruction.
///
/// Lens is observer-only. It must never influence execution or causality.
pub struct FieldLens {
    config: FieldLensConfig,
    fields: IndexMap<FieldId, FieldStorage>,
    topology: CubedSphereTopology,
    field_configs: HashMap<FieldId, FieldConfig>,
    refinement: RefinementQueue,
    #[cfg(feature = "gpu")]
    gpu_backend: Option<crate::gpu::GpuLensBackend>,
}

impl FieldLens {
    /// Create a new lens with validated config.
    pub fn new(config: FieldLensConfig) -> Result<Self, LensError> {
        config.validate()?;
        Ok(Self {
            refinement: RefinementQueue::new(config.max_refinement_queue),
            config,
            fields: IndexMap::new(),
            topology: CubedSphereTopology::default(),
            field_configs: HashMap::new(),
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        })
    }

    /// Record a single field snapshot (internal transport).
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
    ///
    /// Uses IndexMap order for deterministic ingestion.
    pub fn record_many(&mut self, tick: u64, fields: IndexMap<FieldId, Vec<FieldSample>>) {
        for (field_id, samples) in fields {
            self.record(FieldSnapshot {
                field_id,
                tick,
                samples,
            });
        }
    }

    // --- Storage lookup helpers (DRY) ---

    fn get_storage(&self, field_id: &FieldId) -> Result<&FieldStorage, LensError> {
        self.fields
            .get(field_id)
            .ok_or_else(|| LensError::FieldNotFound(field_id.clone()))
    }

    fn get_storage_mut(&mut self, field_id: &FieldId) -> Result<&mut FieldStorage, LensError> {
        self.fields
            .get_mut(field_id)
            .ok_or_else(|| LensError::FieldNotFound(field_id.clone()))
    }

    fn get_frame<'a>(
        &self,
        storage: &'a FieldStorage,
        field_id: &FieldId,
        tick: u64,
    ) -> Result<&'a FieldFrame, LensError> {
        storage
            .frame_at(tick)
            .ok_or_else(|| LensError::no_samples(field_id.clone(), tick))
    }

    fn require_non_empty(&self, frame: &FieldFrame, field_id: &FieldId) -> Result<(), LensError> {
        if frame.samples.is_empty() {
            Err(LensError::no_samples(field_id.clone(), frame.tick))
        } else {
            Ok(())
        }
    }

    // --- Reconstruction API ---

    /// Get reconstruction for a specific tick (nearest-neighbor MVP).
    ///
    /// Uses cache when available; returns errors if the field or tick is missing.
    pub fn at(
        &mut self,
        field_id: &FieldId,
        tick: u64,
    ) -> Result<Arc<dyn FieldReconstruction>, LensError> {
        let storage = self.get_storage(field_id)?;

        if let Some(cached) = storage.cache_get(tick) {
            return Ok(cached);
        }

        let frame = self.get_frame(storage, field_id, tick)?;
        self.require_non_empty(frame, field_id)?;

        let recon = Arc::new(NearestNeighborReconstruction::new(frame.samples.clone()));
        let recon: Arc<dyn FieldReconstruction> = recon;

        let max_cached = self
            .field_configs
            .get(field_id)
            .and_then(|cfg| cfg.max_cached_per_field)
            .unwrap_or(self.config.max_cached_per_field);

        let storage = self.get_storage_mut(field_id)?;
        storage.cache_insert(tick, Arc::clone(&recon), max_cached);

        Ok(recon)
    }

    /// Get reconstruction for latest tick (nearest-neighbor MVP).
    pub fn latest_reconstruction(
        &mut self,
        field_id: &FieldId,
    ) -> Result<Arc<dyn FieldReconstruction>, LensError> {
        let storage = self.get_storage(field_id)?;
        let frame = storage
            .latest()
            .ok_or_else(|| LensError::no_samples(field_id.clone(), 0))?;
        self.require_non_empty(frame, field_id)?;
        let tick = frame.tick;
        self.at(field_id, tick)
    }

    /// Get reconstruction for a specific tile at tick.
    ///
    /// Returns an error if the tile has no samples at this tick.
    pub fn tile(
        &self,
        field_id: &FieldId,
        tile_id: TileId,
        tick: u64,
    ) -> Result<Arc<dyn FieldReconstruction>, LensError> {
        let storage = self.get_storage(field_id)?;
        let frame = self.get_frame(storage, field_id, tick)?;
        self.require_non_empty(frame, field_id)?;

        let samples = frame
            .samples
            .iter()
            .filter(|sample| self.topology.tile_at(sample.position, tile_id.lod()) == tile_id)
            .cloned()
            .collect::<Vec<_>>();

        if samples.is_empty() {
            return Err(LensError::no_samples(field_id.clone(), tick));
        }

        Ok(Arc::new(NearestNeighborReconstruction::new(samples)))
    }

    // --- Query API ---

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
        let (tick_prev, tick_next, alpha) = temporal_bracket(time);

        if alpha == 0.0 {
            return self.query_at_tick(field_id, position, tick_prev);
        }

        let prev = self.query_at_tick(field_id, position, tick_prev)?;
        let next = self.query_at_tick(field_id, position, tick_next)?;
        Ok(lerp(prev, next, alpha))
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

    /// Query scalar value at the latest available tick.
    pub fn query_latest(
        &mut self,
        field_id: &FieldId,
        position: [f64; 3],
    ) -> Result<f64, LensError> {
        let reconstruction = self.latest_reconstruction(field_id)?;
        Ok(reconstruction.query(position))
    }

    /// Query vector value at fractional time (temporal interpolation).
    pub fn query_vector(
        &mut self,
        field_id: &FieldId,
        position: [f64; 3],
        time: f64,
    ) -> Result<[f64; 3], LensError> {
        let (tick_prev, tick_next, alpha) = temporal_bracket(time);

        if alpha == 0.0 {
            let reconstruction = self.at(field_id, tick_prev)?;
            return Ok(reconstruction.query_vector(position));
        }

        let prev = self.at(field_id, tick_prev)?.query_vector(position);
        let next = self.at(field_id, tick_next)?.query_vector(position);

        let lerped = [
            lerp(prev[0], next[0], alpha),
            lerp(prev[1], next[1], alpha),
            lerp(prev[2], next[2], alpha),
        ];

        // Normalize interpolated direction
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

    // --- Metadata API ---

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
    ///
    /// Call before issuing queries for the field.
    pub fn configure_field(&mut self, field_id: FieldId, config: FieldConfig) {
        self.field_configs.insert(field_id, config);
    }

    // --- GPU API ---

    #[cfg(feature = "gpu")]
    /// Set a GPU backend for batch queries.
    pub fn set_gpu_backend(&mut self, backend: crate::gpu::GpuLensBackend) {
        self.gpu_backend = Some(backend);
    }

    #[cfg(feature = "gpu")]
    /// Query a batch of positions on the GPU for a specific tick.
    ///
    /// Only supports scalar samples; returns NonScalarSample otherwise.
    pub fn query_batch_gpu(
        &mut self,
        field_id: &FieldId,
        positions: &[[f64; 3]],
        tick: u64,
    ) -> Result<Vec<f64>, LensError> {
        // First, gather the samples while borrowing self immutably
        let samples = {
            let storage = self.get_storage(field_id)?;
            let frame = self.get_frame(storage, field_id, tick)?;

            let mut samples = Vec::with_capacity(frame.samples.len());
            for sample in &frame.samples {
                let value = sample
                    .value
                    .as_scalar()
                    .ok_or_else(|| LensError::NonScalarSample(field_id.clone()))?;
                samples.push((sample.position, value));
            }
            samples
        };

        // Now borrow self mutably for the backend
        let backend = self.gpu_backend.as_mut().ok_or(LensError::GpuUnavailable)?;
        backend
            .query_scalar_batch(&samples, positions)
            .map_err(LensError::GpuQuery)
    }

    // --- Refinement API ---

    /// Request refinement of a field region.
    pub fn request_refinement(
        &mut self,
        request: RefinementRequestSpec,
    ) -> Result<RefinementHandle, LensError> {
        self.refinement.request(request)
    }

    /// Check refinement status.
    pub fn refinement_status(&self, handle: RefinementHandle) -> Option<RefinementStatus> {
        self.refinement.status(handle)
    }

    /// Cancel a refinement request.
    pub fn cancel_refinement(&mut self, handle: RefinementHandle) {
        self.refinement.cancel(handle);
    }

    /// Drain up to `max` refinement requests (for measurement system).
    pub fn drain_refinements(&mut self, max: usize) -> Vec<RefinementRequest> {
        self.refinement.drain(max)
    }
}

// --- Helper functions (DRY for temporal interpolation) ---

/// Compute temporal bracketing ticks and interpolation alpha.
fn temporal_bracket(time: f64) -> (u64, u64, f64) {
    let tick_prev = time.floor() as u64;
    let tick_next = time.ceil() as u64;
    let alpha = time.fract();
    (tick_prev, tick_next, alpha)
}

/// Linear interpolation.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}
