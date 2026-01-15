//! Refinement request types for LOD-based field sampling.
//!
//! Refinement allows observers to request higher-detail sampling in specific regions.

use std::collections::{HashMap, VecDeque};

use continuum_foundation::FieldId;

use crate::error::LensError;
use crate::topology::TileId;

/// Region selector for refinement requests.
///
/// Currently only tile-based regions are supported. Add SphereCap variant
/// when cone/frustum queries are implemented.
#[derive(Debug, Clone)]
pub enum Region {
    /// Single tile region.
    Tile(TileId),
}

/// Refinement request handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RefinementHandle(u64);

/// Refinement request status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementStatus {
    /// Enqueued, waiting for processing.
    Pending,
    /// Sampling/refinement in progress.
    Sampling,
    /// Refinement completed successfully.
    Complete,
    /// Refinement failed.
    Failed,
}

/// Refinement request (observer-only, internal handle).
#[derive(Debug, Clone)]
pub struct RefinementRequest {
    /// Internal request handle.
    pub(crate) handle: RefinementHandle,
    /// Target field to refine.
    pub field_id: FieldId,
    /// Region to sample.
    pub region: Region,
    /// Target level-of-detail.
    pub target_lod: u8,
    /// Priority hint (higher means sooner).
    pub priority: u32,
}

/// Public refinement request spec (handle assigned by Lens).
#[derive(Debug, Clone)]
pub struct RefinementRequestSpec {
    /// Target field to refine.
    pub field_id: FieldId,
    /// Region to sample.
    pub region: Region,
    /// Target level-of-detail.
    pub target_lod: u8,
    /// Priority hint (higher means sooner).
    pub priority: u32,
}

/// Refinement queue manager.
pub struct RefinementQueue {
    queue: VecDeque<RefinementRequest>,
    status: HashMap<RefinementHandle, RefinementStatus>,
    next_id: u64,
    max_queue: usize,
}

impl RefinementQueue {
    pub fn new(max_queue: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            status: HashMap::new(),
            next_id: 1,
            max_queue,
        }
    }

    /// Request refinement of a field region.
    pub fn request(&mut self, spec: RefinementRequestSpec) -> Result<RefinementHandle, LensError> {
        if self.queue.len() >= self.max_queue {
            return Err(LensError::RefinementQueueFull);
        }
        let handle = RefinementHandle(self.next_id);
        self.next_id += 1;
        self.queue.push_back(RefinementRequest {
            handle,
            field_id: spec.field_id,
            region: spec.region,
            target_lod: spec.target_lod,
            priority: spec.priority,
        });
        self.status.insert(handle, RefinementStatus::Pending);
        Ok(handle)
    }

    /// Check refinement status.
    pub fn status(&self, handle: RefinementHandle) -> Option<RefinementStatus> {
        self.status.get(&handle).copied()
    }

    /// Cancel a refinement request.
    pub fn cancel(&mut self, handle: RefinementHandle) {
        self.queue.retain(|req| req.handle != handle);
        self.status.remove(&handle);
    }

    /// Drain up to `max` refinement requests (for measurement system).
    ///
    /// Requests are drained in priority order (higher priority first).
    /// For requests with equal priority, the one submitted first (lower handle) is drained first.
    /// This ensures deterministic ordering.
    pub fn drain(&mut self, max: usize) -> Vec<RefinementRequest> {
        if self.queue.is_empty() || max == 0 {
            return Vec::new();
        }

        // Sort queue by priority (desc), then by handle (asc) for deterministic tie-break
        let mut sorted: Vec<_> = self.queue.drain(..).collect();
        sorted.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority) // Higher priority first
                .then_with(|| a.handle.0.cmp(&b.handle.0)) // Lower handle first (FIFO for ties)
        });

        let count = max.min(sorted.len());
        let mut drained = Vec::with_capacity(count);

        for req in sorted.drain(..count) {
            if let Some(status) = self.status.get_mut(&req.handle) {
                *status = RefinementStatus::Sampling;
            }
            drained.push(req);
        }

        // Put remaining items back in queue (already sorted by priority)
        self.queue = sorted.into();

        drained
    }
}
