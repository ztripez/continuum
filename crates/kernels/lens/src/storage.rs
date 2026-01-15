//! Internal storage types for field history and caching.

use std::collections::VecDeque;
use std::sync::Arc;

use continuum_foundation::{FieldId, FieldSample};

use crate::reconstruction::FieldReconstruction;

/// Single-field snapshot frame stored by Lens (internal transport).
#[derive(Debug, Clone)]
pub struct FieldFrame {
    pub tick: u64,
    pub samples: Vec<FieldSample>,
}

/// Input payload for ingesting a single field snapshot (internal transport).
#[derive(Debug, Clone)]
pub struct FieldSnapshot {
    pub field_id: FieldId,
    pub tick: u64,
    pub samples: Vec<FieldSample>,
}

/// Per-field storage with bounded history and reconstruction cache.
pub struct FieldStorage {
    pub(crate) history: VecDeque<FieldFrame>,
    pub(crate) cache: VecDeque<(u64, Arc<dyn FieldReconstruction>)>,
}

impl FieldStorage {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            cache: VecDeque::new(),
        }
    }

    pub fn push(&mut self, frame: FieldFrame, max_frames: usize) {
        if self.history.len() == max_frames {
            self.history.pop_front();
        }
        self.history.push_back(frame);
        self.cache.clear();
    }

    pub fn latest(&self) -> Option<&FieldFrame> {
        self.history.back()
    }

    /// Find a frame by tick.
    pub fn frame_at(&self, tick: u64) -> Option<&FieldFrame> {
        self.history.iter().find(|frame| frame.tick == tick)
    }

    pub fn cache_get(&self, tick: u64) -> Option<Arc<dyn FieldReconstruction>> {
        self.cache
            .iter()
            .find(|(cached_tick, _)| *cached_tick == tick)
            .map(|(_, recon)| Arc::clone(recon))
    }

    pub fn cache_insert(
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

impl Default for FieldStorage {
    fn default() -> Self {
        Self::new()
    }
}
