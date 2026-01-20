//! Type Coercion Rules
//!
//! Defines what operations are valid between numeric types and what the result type should be.

use crate::primitives::{PrimitiveShape, PrimitiveTypeId};

/// Binary operations for type checking (subset of full BinaryOp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeCheckOp {
    /// Addition (+).
    Add,
    /// Subtraction (-).
    Sub,
    /// Multiplication (*).
    Mul,
    /// Division (/).
    Div,
}

/// Result of type checking an operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeCheckResult {
    /// Operation is valid, returns this type
    Valid(PrimitiveShape),
    /// Operation is invalid (dimension/semantic mismatch)
    Invalid(&'static str),
}

/// Check if a binary operation is valid between two types
///
/// # Coercion Rules
///
/// ## Addition/Subtraction (+, -)
/// - Scalar ± Scalar → Scalar
/// - Vec ± Vec (same dim) → Vec
/// - Mat ± Mat (same dim) → Mat (element-wise)
/// - Tensor ± Tensor (same shape) → Tensor (element-wise)
/// - Scalar ± Vec → Vec (broadcast)
/// - Vec ± Scalar → Vec (broadcast)
/// - Scalar ± Mat → Mat (broadcast)
/// - Mat ± Scalar → Mat (broadcast)
///
/// ## Multiplication (*)
/// - Scalar * Scalar → Scalar
/// - Vec * Vec (same dim) → Scalar (dot product - NOT IMPLEMENTED YET, returns Vec element-wise)
/// - Scalar * Vec → Vec (scale)
/// - Vec * Scalar → Vec (scale)
/// - Mat * Mat (same dim) → Mat (matrix multiply)
/// - Mat * Vec (compatible) → Vec (transform)
/// - Scalar * Mat → Mat (scale)
/// - Mat * Scalar → Mat (scale)
/// - Tensor * Tensor (compatible) → Tensor (matmul)
///
/// ## Division (/)
/// - Scalar / Scalar → Scalar
/// - Vec / Scalar → Vec (element-wise)
/// - Mat / Scalar → Mat (element-wise)
/// - Tensor / Scalar → Tensor (element-wise)
///
/// ## Invalid Operations
/// - Vec2 + Vec3 (dimension mismatch)
/// - Mat2 * Mat3 (dimension mismatch)
/// - Mat3 * Vec4 (dimension mismatch)
pub fn can_operate(
    op: TypeCheckOp,
    left: &PrimitiveShape,
    right: &PrimitiveShape,
) -> TypeCheckResult {
    use PrimitiveShape::*;

    match (op, left, right) {
        // Scalar operations
        (_, Scalar, Scalar) => TypeCheckResult::Valid(Scalar),

        // Vector operations - same dimension
        (TypeCheckOp::Add | TypeCheckOp::Sub, Vector { dim: d1 }, Vector { dim: d2 })
            if d1 == d2 =>
        {
            TypeCheckResult::Valid(Vector { dim: *d1 })
        }
        (TypeCheckOp::Mul, Vector { dim: d1 }, Vector { dim: d2 }) if d1 == d2 => {
            // TODO: Should be dot product returning Scalar, but currently element-wise
            TypeCheckResult::Valid(Vector { dim: *d1 })
        }

        // Vector-Scalar broadcast
        (TypeCheckOp::Add | TypeCheckOp::Sub | TypeCheckOp::Mul, Scalar, Vector { dim }) => {
            TypeCheckResult::Valid(Vector { dim: *dim })
        }
        (TypeCheckOp::Add | TypeCheckOp::Sub | TypeCheckOp::Mul, Vector { dim }, Scalar) => {
            TypeCheckResult::Valid(Vector { dim: *dim })
        }
        (TypeCheckOp::Div, Vector { dim }, Scalar) => TypeCheckResult::Valid(Vector { dim: *dim }),

        // Matrix operations - same dimension
        (
            TypeCheckOp::Add | TypeCheckOp::Sub,
            Matrix { rows: r1, cols: c1 },
            Matrix { rows: r2, cols: c2 },
        ) if r1 == r2 && c1 == c2 => TypeCheckResult::Valid(Matrix {
            rows: *r1,
            cols: *c1,
        }),

        // Matrix multiplication (rows1 × cols1) * (rows2 × cols2) → (rows1 × cols2) if cols1 == rows2
        (TypeCheckOp::Mul, Matrix { rows: r1, cols: c1 }, Matrix { rows: r2, cols: c2 })
            if c1 == r2 =>
        {
            TypeCheckResult::Valid(Matrix {
                rows: *r1,
                cols: *c2,
            })
        }

        // Matrix-Vector multiplication
        (TypeCheckOp::Mul, Matrix { rows, cols }, Vector { dim }) if cols == dim => {
            TypeCheckResult::Valid(Vector { dim: *rows })
        }

        // Matrix-Scalar operations
        (TypeCheckOp::Add | TypeCheckOp::Sub | TypeCheckOp::Mul, Scalar, Matrix { rows, cols }) => {
            TypeCheckResult::Valid(Matrix {
                rows: *rows,
                cols: *cols,
            })
        }
        (TypeCheckOp::Add | TypeCheckOp::Sub | TypeCheckOp::Mul, Matrix { rows, cols }, Scalar) => {
            TypeCheckResult::Valid(Matrix {
                rows: *rows,
                cols: *cols,
            })
        }
        (TypeCheckOp::Div, Matrix { rows, cols }, Scalar) => TypeCheckResult::Valid(Matrix {
            rows: *rows,
            cols: *cols,
        }),

        // Invalid: dimension mismatch
        (TypeCheckOp::Add | TypeCheckOp::Sub, Vector { .. }, Vector { .. }) => {
            TypeCheckResult::Invalid("Vector dimension mismatch for addition/subtraction")
        }
        (TypeCheckOp::Mul, Vector { .. }, Vector { .. }) => {
            TypeCheckResult::Invalid("Vector dimension mismatch for multiplication")
        }

        // Invalid: Matrix dimension mismatch
        (TypeCheckOp::Add | TypeCheckOp::Sub, Matrix { .. }, Matrix { .. }) => {
            TypeCheckResult::Invalid("Matrix dimension mismatch for element-wise operation")
        }
        (TypeCheckOp::Mul, Matrix { .. }, Matrix { .. }) => TypeCheckResult::Invalid(
            "Matrix multiplication dimension mismatch (left.cols must equal right.rows)",
        ),
        (TypeCheckOp::Mul, Matrix { .. }, Vector { .. }) => TypeCheckResult::Invalid(
            "Matrix-vector multiplication dimension mismatch (mat.cols must equal vec.dim)",
        ),

        // Catch-all: unsupported operation
        _ => TypeCheckResult::Invalid("Unsupported operation between these types"),
    }
}

/// Get the shape of a primitive type
pub fn type_shape(ty: &PrimitiveTypeId) -> Option<PrimitiveShape> {
    use crate::primitives::PRIMITIVE_TYPES;
    PRIMITIVE_TYPES
        .iter()
        .find(|t| t.id == *ty)
        .map(|t| t.shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_ops() {
        let result = can_operate(
            TypeCheckOp::Add,
            &PrimitiveShape::Scalar,
            &PrimitiveShape::Scalar,
        );
        assert_eq!(result, TypeCheckResult::Valid(PrimitiveShape::Scalar));
    }

    #[test]
    fn test_vec3_add_vec3() {
        let v3 = PrimitiveShape::Vector { dim: 3 };
        let result = can_operate(TypeCheckOp::Add, &v3, &v3);
        assert_eq!(
            result,
            TypeCheckResult::Valid(PrimitiveShape::Vector { dim: 3 })
        );
    }

    #[test]
    fn test_vec2_add_vec3_invalid() {
        let v2 = PrimitiveShape::Vector { dim: 2 };
        let v3 = PrimitiveShape::Vector { dim: 3 };
        let result = can_operate(TypeCheckOp::Add, &v2, &v3);
        assert!(matches!(result, TypeCheckResult::Invalid(_)));
    }

    #[test]
    fn test_scalar_mul_vec() {
        let v3 = PrimitiveShape::Vector { dim: 3 };
        let result = can_operate(TypeCheckOp::Mul, &PrimitiveShape::Scalar, &v3);
        assert_eq!(
            result,
            TypeCheckResult::Valid(PrimitiveShape::Vector { dim: 3 })
        );
    }

    #[test]
    fn test_mat_mul_mat() {
        let m23 = PrimitiveShape::Matrix { rows: 2, cols: 3 };
        let m34 = PrimitiveShape::Matrix { rows: 3, cols: 4 };
        let result = can_operate(TypeCheckOp::Mul, &m23, &m34);
        assert_eq!(
            result,
            TypeCheckResult::Valid(PrimitiveShape::Matrix { rows: 2, cols: 4 })
        );
    }

    #[test]
    fn test_mat_mul_vec() {
        let m33 = PrimitiveShape::Matrix { rows: 3, cols: 3 };
        let v3 = PrimitiveShape::Vector { dim: 3 };
        let result = can_operate(TypeCheckOp::Mul, &m33, &v3);
        assert_eq!(
            result,
            TypeCheckResult::Valid(PrimitiveShape::Vector { dim: 3 })
        );
    }

    #[test]
    fn test_mat3_mul_vec4_invalid() {
        let m33 = PrimitiveShape::Matrix { rows: 3, cols: 3 };
        let v4 = PrimitiveShape::Vector { dim: 4 };
        let result = can_operate(TypeCheckOp::Mul, &m33, &v4);
        assert!(matches!(result, TypeCheckResult::Invalid(_)));
    }

    #[test]
    fn test_vec_div_scalar() {
        let v3 = PrimitiveShape::Vector { dim: 3 };
        let result = can_operate(TypeCheckOp::Div, &v3, &PrimitiveShape::Scalar);
        assert_eq!(
            result,
            TypeCheckResult::Valid(PrimitiveShape::Vector { dim: 3 })
        );
    }
}
