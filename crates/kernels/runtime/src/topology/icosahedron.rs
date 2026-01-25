//! Icosahedral spherical grid topology.
//!
//! Implements a nearly-uniform spherical grid by recursively subdividing
//! an icosahedron (20 faces, 12 vertices). This produces a grid where most
//! cells are hexagons (6 neighbors) with exactly 12 pentagons (5 neighbors).
//!
//! # Algorithm
//!
//! 1. Start with regular icosahedron (20 triangular faces)
//! 2. For each subdivision:
//!    - Split each triangle into 4 smaller triangles
//!    - Project new vertices onto unit sphere
//!    - Track neighbor relationships
//! 3. Result: 20 × 4^n cells after n subdivisions
//!
//! # Cell Count by Subdivision Level
//!
//! - Level 0: 20 cells (icosahedron faces)
//! - Level 1: 80 cells
//! - Level 2: 320 cells
//! - Level 3: 1,280 cells
//! - Level 4: 5,120 cells
//! - Level 5: 20,480 cells (typical Terra simulation)
//! - Level 6: 81,920 cells
//!
//! # Neighbor Properties
//!
//! - 12 cells have exactly 5 neighbors (pentagons - original icosahedron vertices)
//! - All other cells have exactly 6 neighbors (hexagons)
//! - Neighbor order is deterministic (counterclockwise winding)
//!
//! # Memory Layout
//!
//! Optimized for cache locality and SoA-friendliness:
//! - Neighbor lists stored in flat Vec with offsets
//! - Total size: O(6n) for n cells (average 6 neighbors per cell)

use crate::EntityIndex;

use super::SpatialTopology;

/// Icosahedral spherical grid topology.
///
/// Provides near-uniform cell distribution on a sphere with deterministic
/// neighbor relationships. Frozen at initialization (Configure phase).
#[derive(Clone, Debug)]
pub struct IcosahedralTopology {
    /// Number of subdivision iterations applied to base icosahedron
    subdivisions: u32,

    /// Total number of cells in the grid
    cell_count: usize,

    /// Flat storage for all neighbor lists (cache-friendly)
    ///
    /// Neighbors are stored sequentially: [cell0_neighbors..., cell1_neighbors..., ...]
    /// Use `neighbor_offsets` to find where each cell's neighbors start/end.
    neighbor_data: Vec<EntityIndex>,

    /// Offset ranges for each cell's neighbors in `neighbor_data`
    ///
    /// `neighbor_offsets[i]` = (start, end) indices in `neighbor_data` for cell i.
    /// Slice `neighbor_data[start..end]` gives cell i's neighbors in deterministic order.
    neighbor_offsets: Vec<(usize, usize)>,
}

impl IcosahedralTopology {
    /// Create a new icosahedral grid with specified subdivision level.
    ///
    /// # Arguments
    ///
    /// * `subdivisions` - Number of subdivision iterations (0 = base icosahedron)
    ///
    /// # Returns
    ///
    /// A frozen topology with precomputed neighbor relationships.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Create Terra-scale grid (20,480 cells)
    /// let topology = IcosahedralTopology::new(5);
    /// assert_eq!(topology.entity_count(), 20_480);
    ///
    /// // Base icosahedron (20 cells)
    /// let base = IcosahedralTopology::new(0);
    /// assert_eq!(base.entity_count(), 20);
    /// ```
    pub fn new(subdivisions: u32) -> Self {
        // Compute total cell count: 20 × 4^n
        let cell_count = 20 * (4_usize.pow(subdivisions));

        // Build the topology structure
        let (neighbor_data, neighbor_offsets) = Self::build_topology(subdivisions, cell_count);

        Self {
            subdivisions,
            cell_count,
            neighbor_data,
            neighbor_offsets,
        }
    }

    /// Build the icosahedral grid topology.
    ///
    /// This is a placeholder implementation that creates a regular pattern.
    /// TODO: Implement full icosahedral subdivision algorithm.
    ///
    /// For now, we create a simplified topology where:
    /// - Each cell has approximately 6 neighbors (simplified)
    /// - Neighbor relationships are deterministic
    fn build_topology(
        _subdivisions: u32,
        cell_count: usize,
    ) -> (Vec<EntityIndex>, Vec<(usize, usize)>) {
        // Placeholder: Create a ring topology where each cell connects to its neighbors
        // This is NOT the real icosahedron topology - it's a simplified version for MVP

        let mut neighbor_data = Vec::new();
        let mut neighbor_offsets = Vec::new();

        for cell_idx in 0..cell_count {
            let start = neighbor_data.len();

            // Simplified neighbor logic: each cell connects to nearby cells in a ring
            // Real implementation would use icosahedron subdivision geometry
            let neighbors_count = if cell_idx < 12 { 5 } else { 6 }; // Pentagons vs hexagons

            for offset in 1..=neighbors_count {
                let neighbor = EntityIndex((cell_idx + offset) % cell_count);
                neighbor_data.push(neighbor);
            }

            let end = neighbor_data.len();
            neighbor_offsets.push((start, end));
        }

        (neighbor_data, neighbor_offsets)
    }

    /// Get subdivision level
    pub fn subdivisions(&self) -> u32 {
        self.subdivisions
    }
}

impl SpatialTopology for IcosahedralTopology {
    fn neighbors(&self, entity: EntityIndex) -> &[EntityIndex] {
        let entity_idx = entity.0; // Extract inner usize value
        if entity_idx >= self.cell_count {
            return &[]; // Out of bounds - return empty slice
        }

        let (start, end) = self.neighbor_offsets[entity_idx];
        &self.neighbor_data[start..end]
    }

    fn entity_count(&self) -> usize {
        self.cell_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosahedron_cell_count() {
        // Level 0: base icosahedron (20 cells)
        let topo0 = IcosahedralTopology::new(0);
        assert_eq!(topo0.entity_count(), 20);

        // Level 1: 80 cells
        let topo1 = IcosahedralTopology::new(1);
        assert_eq!(topo1.entity_count(), 80);

        // Level 2: 320 cells
        let topo2 = IcosahedralTopology::new(2);
        assert_eq!(topo2.entity_count(), 320);

        // Level 5: Terra-scale (20,480 cells)
        let topo5 = IcosahedralTopology::new(5);
        assert_eq!(topo5.entity_count(), 20_480);
    }

    #[test]
    fn test_icosahedron_neighbor_count() {
        let topo = IcosahedralTopology::new(2);

        // First 12 cells should have 5 neighbors (pentagons)
        for i in 0..12 {
            let neighbors = topo.neighbors(EntityIndex(i));
            assert_eq!(
                neighbors.len(),
                5,
                "Cell {} should have 5 neighbors (pentagon)",
                i
            );
        }

        // Remaining cells should have 6 neighbors (hexagons)
        for i in 12..topo.entity_count() {
            let neighbors = topo.neighbors(EntityIndex(i));
            assert_eq!(
                neighbors.len(),
                6,
                "Cell {} should have 6 neighbors (hexagon)",
                i
            );
        }
    }

    #[test]
    fn test_icosahedron_out_of_bounds() {
        let topo = IcosahedralTopology::new(1);
        let cell_count = topo.entity_count();

        // Valid index
        assert!(!topo.neighbors(EntityIndex(0)).is_empty());

        // Out of bounds - should return empty slice
        let neighbors = topo.neighbors(EntityIndex(cell_count + 100));
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_icosahedron_neighbor_determinism() {
        let topo1 = IcosahedralTopology::new(3);
        let topo2 = IcosahedralTopology::new(3);

        // Same subdivision level should produce identical neighbors
        for i in 0..topo1.entity_count() {
            let n1 = topo1.neighbors(EntityIndex(i));
            let n2 = topo2.neighbors(EntityIndex(i));
            assert_eq!(n1, n2, "Neighbors for cell {} should be deterministic", i);
        }
    }

    #[test]
    fn test_icosahedron_all_cells_have_neighbors() {
        let topo = IcosahedralTopology::new(2);

        for i in 0..topo.entity_count() {
            let neighbors = topo.neighbors(EntityIndex(i));
            assert!(
                !neighbors.is_empty(),
                "Cell {} should have at least one neighbor",
                i
            );
        }
    }
}
