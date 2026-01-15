//! Compile-time type checking for numeric operations.
//!
//! This module provides type inference and validation for binary operations
//! during IR lowering, catching type mismatches at compile time rather than runtime.

use continuum_dsl::ast::{self, BinaryOp, Expr, Literal, Span};
use continuum_foundation::{
    PrimitiveShape, SignalId,
    coercion::{self, TypeCheckResult},
};
use std::collections::HashMap;

use super::LowerError;

/// Inferred type shape for an expression during type checking.
/// This is a simplified representation used only for type checking binary operations.
#[derive(Debug, Clone, PartialEq)]
pub enum InferredType {
    /// A known primitive shape (Scalar, Vector, Matrix)
    Shape(PrimitiveShape),
    /// Type is unknown (e.g., signal reference without type info, or complex expressions)
    Unknown,
}

impl InferredType {
    /// Create a scalar type
    pub fn scalar() -> Self {
        InferredType::Shape(PrimitiveShape::Scalar)
    }

    /// Create a vector type with given dimension
    pub fn vector(dim: u8) -> Self {
        InferredType::Shape(PrimitiveShape::Vector { dim })
    }

    /// Create a matrix type with given dimensions
    pub fn matrix(rows: u8, cols: u8) -> Self {
        InferredType::Shape(PrimitiveShape::Matrix { rows, cols })
    }

    /// Get the underlying shape if known
    pub fn shape(&self) -> Option<&PrimitiveShape> {
        match self {
            InferredType::Shape(s) => Some(s),
            InferredType::Unknown => None,
        }
    }

    /// Get a human-readable name for this type
    pub fn type_name(&self) -> String {
        match self {
            InferredType::Shape(PrimitiveShape::Scalar) => "Scalar".to_string(),
            InferredType::Shape(PrimitiveShape::Vector { dim }) => format!("Vec{}", dim),
            InferredType::Shape(PrimitiveShape::Matrix { rows, cols }) => {
                if rows == cols {
                    format!("Mat{}", rows)
                } else {
                    format!("Mat{}x{}", rows, cols)
                }
            }
            InferredType::Shape(PrimitiveShape::Tensor) => "Tensor".to_string(),
            InferredType::Shape(PrimitiveShape::Grid) => "Grid".to_string(),
            InferredType::Shape(PrimitiveShape::Seq) => "Seq".to_string(),
            InferredType::Unknown => "unknown".to_string(),
        }
    }
}

/// Context for type checking, holding known signal types.
pub struct TypeCheckContext {
    /// Known signal types (signal ID -> inferred type)
    pub signal_types: HashMap<String, InferredType>,
    /// Known local variable types (from let bindings)
    pub local_types: HashMap<String, InferredType>,
}

impl TypeCheckContext {
    pub fn new() -> Self {
        Self {
            signal_types: HashMap::new(),
            local_types: HashMap::new(),
        }
    }

    /// Create a new context with a local variable binding
    pub fn with_local(&self, name: String, ty: InferredType) -> Self {
        let mut new_locals = self.local_types.clone();
        new_locals.insert(name, ty);
        Self {
            signal_types: self.signal_types.clone(),
            local_types: new_locals,
        }
    }
}

impl Default for TypeCheckContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Check binary operation type validity and return the result type.
///
/// Returns `Ok(result_type)` if the operation is valid, or `Err(LowerError::TypeError)`
/// if the types are incompatible.
pub fn check_binary_op(
    op: BinaryOp,
    left_type: &InferredType,
    right_type: &InferredType,
    span: &Span,
    file: Option<std::path::PathBuf>,
) -> Result<InferredType, LowerError> {
    // If either type is unknown, we can't check - allow it through
    let (left_shape, right_shape) = match (left_type.shape(), right_type.shape()) {
        (Some(l), Some(r)) => (l, r),
        _ => return Ok(InferredType::Unknown),
    };

    // Convert DSL BinaryOp to coercion BinaryOp for arithmetic operations
    let coercion_op = match op {
        BinaryOp::Add => Some(coercion::BinaryOp::Add),
        BinaryOp::Sub => Some(coercion::BinaryOp::Sub),
        BinaryOp::Mul => Some(coercion::BinaryOp::Mul),
        BinaryOp::Div => Some(coercion::BinaryOp::Div),
        // Comparison and logical operators are valid for any types
        BinaryOp::Eq
        | BinaryOp::Ne
        | BinaryOp::Lt
        | BinaryOp::Le
        | BinaryOp::Gt
        | BinaryOp::Ge
        | BinaryOp::And
        | BinaryOp::Or
        | BinaryOp::Pow => None,
    };

    // For comparison/logical ops, return Scalar (boolean result)
    let Some(coercion_op) = coercion_op else {
        return Ok(InferredType::scalar());
    };

    // Check if the operation is valid using the coercion rules
    match coercion::can_operate(coercion_op, left_shape, right_shape) {
        TypeCheckResult::Valid(result_shape) => Ok(InferredType::Shape(result_shape)),
        TypeCheckResult::Invalid(message) => Err(LowerError::TypeError {
            message: message.to_string(),
            left_type: left_type.type_name(),
            right_type: right_type.type_name(),
            op: format!("{:?}", op),
            file,
            span: span.clone(),
        }),
    }
}

/// Infer the type of an expression.
///
/// This performs a recursive traversal of the expression tree, inferring types
/// where possible. For binary operations, it also validates type compatibility.
pub fn infer_expr_type(
    expr: &Expr,
    ctx: &TypeCheckContext,
    span: &Span,
    file: Option<std::path::PathBuf>,
) -> Result<InferredType, LowerError> {
    match expr {
        // Literals are scalars
        Expr::Literal(Literal::Integer(_) | Literal::Float(_) | Literal::Bool(_)) => {
            Ok(InferredType::scalar())
        }
        Expr::Literal(Literal::String(_)) => Ok(InferredType::Unknown),
        Expr::LiteralWithUnit { .. } => Ok(InferredType::scalar()),

        // Time values are scalars
        Expr::DtRaw | Expr::SimTime => Ok(InferredType::scalar()),

        // Previous value - type depends on context, treat as unknown
        Expr::Prev | Expr::PrevField(_) => Ok(InferredType::Unknown),

        // Collected values - unknown type
        Expr::Collected => Ok(InferredType::Unknown),

        // Math constants are scalars
        Expr::MathConst(_) => Ok(InferredType::scalar()),

        // Signal/const/config references - look up in context or return unknown
        Expr::Path(path) => {
            let key = path.to_string();
            // Check locals first
            if let Some(ty) = ctx.local_types.get(&key) {
                return Ok(ty.clone());
            }
            // Then signals
            if let Some(ty) = ctx.signal_types.get(&key) {
                return Ok(ty.clone());
            }
            Ok(InferredType::Unknown)
        }
        Expr::SignalRef(path) => {
            let key = path.to_string();
            ctx.signal_types
                .get(&key)
                .cloned()
                .unwrap_or(InferredType::Unknown)
                .pipe(Ok)
        }
        Expr::ConstRef(_) | Expr::ConfigRef(_) => Ok(InferredType::scalar()),

        // Local variable lookup
        Expr::Let { name, value, body } => {
            let value_type = infer_expr_type(&value.node, ctx, &value.span, file.clone())?;
            let new_ctx = ctx.with_local(name.clone(), value_type);
            infer_expr_type(&body.node, &new_ctx, &body.span, file)
        }

        // Binary operations - check type compatibility
        Expr::Binary { op, left, right } => {
            let left_type = infer_expr_type(&left.node, ctx, &left.span, file.clone())?;
            let right_type = infer_expr_type(&right.node, ctx, &right.span, file.clone())?;
            check_binary_op(*op, &left_type, &right_type, span, file)
        }

        // Unary operations preserve type (for negation) or return scalar (for not)
        Expr::Unary { op, operand } => {
            let operand_type = infer_expr_type(&operand.node, ctx, &operand.span, file)?;
            match op {
                ast::UnaryOp::Neg => Ok(operand_type),
                ast::UnaryOp::Not => Ok(InferredType::scalar()),
            }
        }

        // Conditionals - type depends on branches, return unknown if they differ
        Expr::If {
            condition: _,
            then_branch,
            else_branch,
        } => {
            let then_type =
                infer_expr_type(&then_branch.node, ctx, &then_branch.span, file.clone())?;
            if let Some(else_br) = else_branch {
                let else_type = infer_expr_type(&else_br.node, ctx, &else_br.span, file)?;
                if then_type == else_type {
                    Ok(then_type)
                } else {
                    Ok(InferredType::Unknown)
                }
            } else {
                Ok(then_type)
            }
        }

        // Function calls - return unknown (could be enhanced with function signature lookup)
        Expr::Call { .. } | Expr::MethodCall { .. } => Ok(InferredType::Unknown),

        // Field access - return unknown (would need struct type info)
        Expr::FieldAccess { object, field } => {
            // Check for vector component access (x, y, z, w)
            if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                return Ok(InferredType::scalar());
            }
            // Check for matrix component access (m00, m01, etc.)
            if field.starts_with('m') && field.len() == 3 {
                return Ok(InferredType::scalar());
            }
            Ok(InferredType::Unknown)
        }

        // Vector literals - infer from element count
        Expr::Vector(elems) => {
            let dim = elems.len() as u8;
            if dim >= 2 && dim <= 4 {
                Ok(InferredType::vector(dim))
            } else {
                Ok(InferredType::Unknown)
            }
        }

        // Block - type is type of last expression
        Expr::Block(exprs) => {
            if let Some(last) = exprs.last() {
                infer_expr_type(&last.node, ctx, &last.span, file)
            } else {
                Ok(InferredType::Unknown)
            }
        }

        // Entity/aggregate expressions - return unknown
        Expr::SelfField(_)
        | Expr::EntityRef(_)
        | Expr::EntityAccess { .. }
        | Expr::Aggregate { .. }
        | Expr::Other(_)
        | Expr::Pairs(_)
        | Expr::Filter { .. }
        | Expr::First { .. }
        | Expr::Nearest { .. }
        | Expr::Within { .. } => Ok(InferredType::Unknown),

        // Impulse expressions
        Expr::Payload | Expr::PayloadField(_) => Ok(InferredType::Unknown),
        Expr::EmitSignal { .. } | Expr::EmitField { .. } => Ok(InferredType::Unknown),

        // Other expressions
        Expr::FieldRef(_)
        | Expr::Struct(_)
        | Expr::Map { .. }
        | Expr::Fold { .. }
        | Expr::For { .. } => Ok(InferredType::Unknown),
    }
}

/// Extension trait for pipe-style function application
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_dsl::ast::{BinaryOp, Spanned};

    fn make_span() -> Span {
        0..0
    }

    fn make_literal(val: f64) -> Box<Spanned<Expr>> {
        Box::new(Spanned {
            node: Expr::Literal(Literal::Float(val)),
            span: make_span(),
        })
    }

    #[test]
    fn test_scalar_add_scalar() {
        let result = check_binary_op(
            BinaryOp::Add,
            &InferredType::scalar(),
            &InferredType::scalar(),
            &make_span(),
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), InferredType::scalar());
    }

    #[test]
    fn test_vec3_add_vec3() {
        let result = check_binary_op(
            BinaryOp::Add,
            &InferredType::vector(3),
            &InferredType::vector(3),
            &make_span(),
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), InferredType::vector(3));
    }

    #[test]
    fn test_vec2_add_vec3_error() {
        let result = check_binary_op(
            BinaryOp::Add,
            &InferredType::vector(2),
            &InferredType::vector(3),
            &make_span(),
            None,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            LowerError::TypeError {
                left_type,
                right_type,
                ..
            } => {
                assert_eq!(left_type, "Vec2");
                assert_eq!(right_type, "Vec3");
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_mat3_mul_vec4_error() {
        let result = check_binary_op(
            BinaryOp::Mul,
            &InferredType::matrix(3, 3),
            &InferredType::vector(4),
            &make_span(),
            None,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            LowerError::TypeError {
                left_type,
                right_type,
                ..
            } => {
                assert_eq!(left_type, "Mat3");
                assert_eq!(right_type, "Vec4");
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_mat3_mul_vec3_valid() {
        let result = check_binary_op(
            BinaryOp::Mul,
            &InferredType::matrix(3, 3),
            &InferredType::vector(3),
            &make_span(),
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), InferredType::vector(3));
    }

    #[test]
    fn test_scalar_mul_vec3() {
        let result = check_binary_op(
            BinaryOp::Mul,
            &InferredType::scalar(),
            &InferredType::vector(3),
            &make_span(),
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), InferredType::vector(3));
    }

    #[test]
    fn test_unknown_types_pass_through() {
        // When types are unknown, we can't check - should pass through
        let result = check_binary_op(
            BinaryOp::Add,
            &InferredType::Unknown,
            &InferredType::vector(3),
            &make_span(),
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), InferredType::Unknown);
    }

    #[test]
    fn test_comparison_ops_return_scalar() {
        let result = check_binary_op(
            BinaryOp::Lt,
            &InferredType::vector(3),
            &InferredType::vector(3),
            &make_span(),
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), InferredType::scalar());
    }
}
