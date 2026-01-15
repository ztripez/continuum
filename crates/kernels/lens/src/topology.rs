//! Virtual topology for spatial indexing of field samples.
//!
//! Provides cubed-sphere tile addressing for LOD-aware field queries.

/// Tile identifier for virtual topology.
///
/// Stable across runs for identical positions and LOD.
/// Encoding: [face:8][lod:8][morton:48]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(u64);

impl TileId {
    /// Build a tile id from face, LOD, and Morton code.
    ///
    /// Face numbering follows cubed-sphere order: +X, -X, +Y, -Y, +Z, -Z.
    pub fn from_parts(face: u8, lod: u8, morton: u64) -> Self {
        let id = ((face as u64) << 56) | ((lod as u64) << 48) | (morton & 0x0000_FFFF_FFFF_FFFF);
        Self(id)
    }

    /// Level of detail encoded in this tile id (0 = coarsest).
    pub fn lod(self) -> u8 {
        ((self.0 >> 48) & 0xFF) as u8
    }
}

/// Minimal cubed-sphere topology.
///
/// Uses cube faces and Morton-coded tiles for LOD partitioning.
/// This is the only topology implementation - the trait was removed per KISS principle.
#[derive(Debug, Default, Clone)]
pub struct CubedSphereTopology;

impl CubedSphereTopology {
    /// Map a position to a tile at the given LOD.
    pub fn tile_at(&self, position: [f64; 3], lod: u8) -> TileId {
        let (face, u, v) = Self::face_and_uv(position);
        let morton = Self::uv_to_morton(u, v, lod);
        TileId::from_parts(face, lod, morton)
    }

    fn face_and_uv(position: [f64; 3]) -> (u8, f64, f64) {
        let (x, y, z) = (position[0], position[1], position[2]);
        let ax = x.abs();
        let ay = y.abs();
        let az = z.abs();

        // Handle origin or extremely small vectors by defaulting to +X face center
        if ax < 1e-10 && ay < 1e-10 && az < 1e-10 {
            return (0, 0.0, 0.0);
        }

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
