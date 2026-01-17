//! Shape system for geometric structure of numeric types
//!
//! Shapes describe the geometric structure of values: scalars, vectors,
//! matrices, tensors, and specialized structured types.
//!
//! # Design Philosophy
//!
//! Named variants (`Scalar`, `Vector`, `Matrix`) cover the 95% case for
//! readable errors and self-documenting code. `Tensor` handles rank 3+
//! for ML, elasticity tensors, or other high-dimensional data.
//!
//! Structured types (`Complex`, `Quaternion`, `SymmetricMatrix`, etc.)
//! have specialized semantics beyond simple arrays.
//!
//! # Examples
//!
//! ```
//! # use continuum_cdsl::foundation::shape::*;
//! // Basic shapes
//! let scalar = Shape::Scalar;
//! assert_eq!(scalar.rank(), 0);
//! assert_eq!(scalar.dims(), vec![]);
//!
//! let vec3 = Shape::Vector { dim: 3 };
//! assert_eq!(vec3.rank(), 1);
//! assert_eq!(vec3.dims(), vec![3]);
//!
//! let mat3x3 = Shape::Matrix { rows: 3, cols: 3 };
//! assert_eq!(mat3x3.rank(), 2);
//! assert_eq!(mat3x3.dims(), vec![3, 3]);
//!
//! // Structured types
//! let quat = Shape::Quaternion;
//! assert_eq!(quat.component_count(), 4);
//!
//! let complex = Shape::Complex;
//! assert_eq!(complex.component_count(), 2);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Geometric structure of a value.
///
/// Shapes determine how many components a value has and how they're
/// organized. This enables compile-time validation of operations
/// (e.g., Vec3 + Vec2 → error).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Shape {
    /// Scalar value (rank 0, 1 component)
    Scalar,

    /// Vector (rank 1)
    Vector {
        /// Dimension (number of components)
        dim: u8,
    },

    /// Matrix (rank 2)
    Matrix {
        /// Number of rows
        rows: u8,
        /// Number of columns
        cols: u8,
    },

    /// General tensor (rank 3+)
    Tensor {
        /// Dimensions for each rank
        dims: Vec<u8>,
    },

    // === Structured types ===
    /// Complex number (real + imaginary, 2 components)
    Complex,

    /// Quaternion (rotation representation, 4 components: w, x, y, z)
    Quaternion,

    /// Symmetric matrix (n×n with n(n+1)/2 independent components)
    SymmetricMatrix {
        /// Matrix dimension (n for n×n matrix)
        dim: u8,
    },

    /// Skew-symmetric matrix (n×n with n(n-1)/2 independent components)
    SkewSymmetricMatrix {
        /// Matrix dimension (n for n×n matrix)
        dim: u8,
    },
}

impl Shape {
    /// Get the rank (number of dimensions).
    ///
    /// - Scalar: 0
    /// - Vector: 1
    /// - Matrix: 2
    /// - Tensor: dims.len()
    /// - Structured types: varies (see docs)
    pub fn rank(&self) -> usize {
        match self {
            Shape::Scalar => 0,
            Shape::Vector { .. } => 1,
            Shape::Matrix { .. } | Shape::Complex => 2,
            Shape::Tensor { dims } => dims.len(),
            Shape::Quaternion => 1,
            Shape::SymmetricMatrix { .. } | Shape::SkewSymmetricMatrix { .. } => 2,
        }
    }

    /// Get the dimensions as a vector.
    ///
    /// Provides uniform access for operations that don't care about
    /// named variants.
    pub fn dims(&self) -> Vec<u8> {
        match self {
            Shape::Scalar => vec![],
            Shape::Vector { dim } => vec![*dim],
            Shape::Matrix { rows, cols } => vec![*rows, *cols],
            Shape::Tensor { dims } => dims.clone(),
            Shape::Complex => vec![2],
            Shape::Quaternion => vec![4],
            Shape::SymmetricMatrix { dim } => vec![*dim, *dim],
            Shape::SkewSymmetricMatrix { dim } => vec![*dim, *dim],
        }
    }

    /// Get the total number of components.
    ///
    /// For structured types, this is the number of stored components,
    /// not necessarily rows × cols.
    pub fn component_count(&self) -> usize {
        match self {
            Shape::Scalar => 1,
            Shape::Vector { dim } => *dim as usize,
            Shape::Matrix { rows, cols } => (*rows as usize) * (*cols as usize),
            Shape::Tensor { dims } => dims.iter().map(|&d| d as usize).product(),
            Shape::Complex => 2,
            Shape::Quaternion => 4,
            Shape::SymmetricMatrix { dim } => {
                let n = *dim as usize;
                n * (n + 1) / 2
            }
            Shape::SkewSymmetricMatrix { dim } => {
                let n = *dim as usize;
                n * (n - 1) / 2
            }
        }
    }

    /// Check if this is a scalar.
    pub fn is_scalar(&self) -> bool {
        matches!(self, Shape::Scalar)
    }

    /// Check if this is a vector.
    pub fn is_vector(&self) -> bool {
        matches!(self, Shape::Vector { .. })
    }

    /// Check if this is a matrix.
    pub fn is_matrix(&self) -> bool {
        matches!(self, Shape::Matrix { .. })
    }

    /// Check if this is a general tensor.
    pub fn is_tensor(&self) -> bool {
        matches!(self, Shape::Tensor { .. })
    }

    /// Check if this is a structured type.
    pub fn is_structured(&self) -> bool {
        matches!(
            self,
            Shape::Complex
                | Shape::Quaternion
                | Shape::SymmetricMatrix { .. }
                | Shape::SkewSymmetricMatrix { .. }
        )
    }

    /// Check if two shapes are compatible for addition/subtraction.
    ///
    /// Shapes are compatible if they have the same structure
    /// (same variant and same dimensions).
    pub fn is_compatible(&self, other: &Shape) -> bool {
        self == other
    }

    /// Check if this shape can broadcast to another shape.
    ///
    /// Scalars can broadcast to any shape.
    /// Otherwise, shapes must match exactly.
    pub fn can_broadcast_to(&self, target: &Shape) -> bool {
        self.is_scalar() || self == target
    }

    // ============================================================================
    // Common constructors
    // ============================================================================

    /// Create a scalar shape.
    pub const fn scalar() -> Self {
        Shape::Scalar
    }

    /// Create a vector shape with given dimension.
    pub const fn vector(dim: u8) -> Self {
        Shape::Vector { dim }
    }

    /// Create a matrix shape with given rows and columns.
    pub const fn matrix(rows: u8, cols: u8) -> Self {
        Shape::Matrix { rows, cols }
    }

    /// Create a general tensor shape with given dimensions.
    pub fn tensor(dims: Vec<u8>) -> Self {
        Shape::Tensor { dims }
    }

    /// Create a complex number shape.
    pub const fn complex() -> Self {
        Shape::Complex
    }

    /// Create a quaternion shape.
    pub const fn quaternion() -> Self {
        Shape::Quaternion
    }

    /// Create a symmetric matrix shape.
    pub const fn symmetric_matrix(dim: u8) -> Self {
        Shape::SymmetricMatrix { dim }
    }

    /// Create a skew-symmetric matrix shape.
    pub const fn skew_symmetric_matrix(dim: u8) -> Self {
        Shape::SkewSymmetricMatrix { dim }
    }

    // Common named shapes
    /// Vec2 shape
    pub const fn vec2() -> Self {
        Self::vector(2)
    }

    /// Vec3 shape
    pub const fn vec3() -> Self {
        Self::vector(3)
    }

    /// Vec4 shape
    pub const fn vec4() -> Self {
        Self::vector(4)
    }

    /// Mat2 (2×2 matrix) shape
    pub const fn mat2() -> Self {
        Self::matrix(2, 2)
    }

    /// Mat3 (3×3 matrix) shape
    pub const fn mat3() -> Self {
        Self::matrix(3, 3)
    }

    /// Mat4 (4×4 matrix) shape
    pub const fn mat4() -> Self {
        Self::matrix(4, 4)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shape::Scalar => write!(f, "Scalar"),
            Shape::Vector { dim } => write!(f, "Vec{}", dim),
            Shape::Matrix { rows, cols } => write!(f, "Mat{}x{}", rows, cols),
            Shape::Tensor { dims } => {
                write!(
                    f,
                    "Tensor[{}]",
                    dims.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("×")
                )
            }
            Shape::Complex => write!(f, "Complex"),
            Shape::Quaternion => write!(f, "Quat"),
            Shape::SymmetricMatrix { dim } => write!(f, "SymMat{}", dim),
            Shape::SkewSymmetricMatrix { dim } => write!(f, "SkewMat{}", dim),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.dims(), vec![]);
        assert_eq!(s.component_count(), 1);
        assert!(s.is_scalar());
    }

    #[test]
    fn test_vector() {
        let v3 = Shape::vec3();
        assert_eq!(v3.rank(), 1);
        assert_eq!(v3.dims(), vec![3]);
        assert_eq!(v3.component_count(), 3);
        assert!(v3.is_vector());
    }

    #[test]
    fn test_matrix() {
        let m = Shape::mat3();
        assert_eq!(m.rank(), 2);
        assert_eq!(m.dims(), vec![3, 3]);
        assert_eq!(m.component_count(), 9);
        assert!(m.is_matrix());
    }

    #[test]
    fn test_tensor() {
        let t = Shape::tensor(vec![2, 3, 4]);
        assert_eq!(t.rank(), 3);
        assert_eq!(t.dims(), vec![2, 3, 4]);
        assert_eq!(t.component_count(), 24);
        assert!(t.is_tensor());
    }

    #[test]
    fn test_complex() {
        let c = Shape::complex();
        assert_eq!(c.component_count(), 2);
        assert!(c.is_structured());
    }

    #[test]
    fn test_quaternion() {
        let q = Shape::quaternion();
        assert_eq!(q.component_count(), 4);
        assert!(q.is_structured());
    }

    #[test]
    fn test_symmetric_matrix() {
        let sm3 = Shape::symmetric_matrix(3);
        // 3×3 symmetric matrix has 3*(3+1)/2 = 6 independent components
        assert_eq!(sm3.component_count(), 6);
        assert!(sm3.is_structured());
    }

    #[test]
    fn test_skew_symmetric_matrix() {
        let skew3 = Shape::skew_symmetric_matrix(3);
        // 3×3 skew-symmetric matrix has 3*(3-1)/2 = 3 independent components
        assert_eq!(skew3.component_count(), 3);
        assert!(skew3.is_structured());
    }

    #[test]
    fn test_compatibility() {
        let v3a = Shape::vec3();
        let v3b = Shape::vec3();
        let v2 = Shape::vec2();

        assert!(v3a.is_compatible(&v3b));
        assert!(!v3a.is_compatible(&v2));
    }

    #[test]
    fn test_broadcast() {
        let scalar = Shape::scalar();
        let v3 = Shape::vec3();
        let m = Shape::mat3();

        // Scalar broadcasts to anything
        assert!(scalar.can_broadcast_to(&v3));
        assert!(scalar.can_broadcast_to(&m));

        // Non-scalar only broadcasts to itself
        assert!(v3.can_broadcast_to(&v3));
        assert!(!v3.can_broadcast_to(&m));
    }

    #[test]
    fn test_display() {
        assert_eq!(Shape::scalar().to_string(), "Scalar");
        assert_eq!(Shape::vec3().to_string(), "Vec3");
        assert_eq!(Shape::mat3().to_string(), "Mat3x3");
        assert_eq!(Shape::complex().to_string(), "Complex");
        assert_eq!(Shape::quaternion().to_string(), "Quat");
    }
}
