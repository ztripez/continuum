//! Lane kernel registry for dispatch during execution.
//!
//! The registry maps member signal IDs to their compiled lane kernels,
//! enabling the phase executor to dispatch to the correct kernel.

use crate::vectorized::MemberSignalId;

use super::lane_kernel::LaneKernel;

/// Registry of lane kernels for dispatch during execution.
///
/// The registry maps member signal IDs to their compiled lane kernels,
/// enabling the phase executor to dispatch to the correct kernel.
pub struct LaneKernelRegistry {
    kernels: indexmap::IndexMap<MemberSignalId, Box<dyn LaneKernel>>,
}

impl LaneKernelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            kernels: indexmap::IndexMap::new(),
        }
    }

    /// Register a lane kernel.
    pub fn register(&mut self, kernel: impl LaneKernel + 'static) {
        let id = kernel.member_signal_id().clone();
        self.kernels.insert(id, Box::new(kernel));
    }

    /// Get a kernel by member signal ID.
    pub fn get(&self, id: &MemberSignalId) -> Option<&dyn LaneKernel> {
        self.kernels.get(id).map(|k| k.as_ref())
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Iterate over all kernels.
    pub fn iter(&self) -> impl Iterator<Item = (&MemberSignalId, &dyn LaneKernel)> {
        self.kernels.iter().map(|(k, v)| (k, v.as_ref()))
    }
}

impl Default for LaneKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::executor::l1_kernels::ScalarL1Kernel;
    use crate::executor::lowering_strategy::LoweringStrategy;
    use crate::types::EntityId;

    fn make_member_signal_id(entity: &str, signal: &str) -> MemberSignalId {
        MemberSignalId::new(EntityId::from(entity), signal)
    }

    #[test]
    fn test_lane_kernel_registry() {
        let mut registry = LaneKernelRegistry::new();
        assert!(registry.is_empty());

        let id = make_member_signal_id("test.entity", "value");
        let kernel = ScalarL1Kernel::new(id.clone(), Arc::new(|ctx| ctx.prev + 1.0), 100);

        registry.register(kernel);
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().strategy(),
            LoweringStrategy::InstanceParallel
        );
    }
}
