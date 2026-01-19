//! Expression typing pass - converts untyped Expr to TypedExpr
//!
//! This module implements type inference and checking for CDSL expressions.
//! It assigns types to expressions by looking up signal/field types, resolving
//! kernel signatures, and propagating types through operations.
//!
//! # What This Pass Does
//!
//! 1. **Type inference** - Assigns Type to each subexpression
//! 2. **Signal/Field lookup** - Resolves paths to their declared types
//! 3. **Kernel resolution** - Resolves kernel calls and derives return types
//! 4. **Context validation** - Ensures context-dependent expressions (prev, dt, etc.) are valid
//! 5. **Local binding** - Tracks let-bound variables and their types
//!
//! # What This Pass Does NOT Do
//!
//! - **No desugaring** - Binary/Unary/If should already be desugared to KernelCall
//! - **No validation** - Type compatibility checking happens in validation pass
//! - **No code generation** - Produces typed AST, not bytecode
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Res → Type Resolution → EXPR TYPING → Validation → Compilation
//!                                                  ^^^^^^^^^^^
//!                                                  YOU ARE HERE
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::expr_typing::{type_expression, TypingContext};
//! use continuum_cdsl::ast::untyped::Expr;
//!
//! let ctx = TypingContext::new(/* registries */);
//! let typed_expr = type_expression(&expr, &ctx)?;
//! ```

use crate::ast::{Expr, ExprKind, KernelRegistry, TypedExpr, UntypedKind};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{KernelType, Path, Shape, Type, Unit};
use crate::resolve::types::{TypeTable, resolve_unit_expr};
use std::collections::HashMap;

/// Context for expression typing
///
/// Provides access to type registries and tracks local bindings during typing.
///
/// # Parameters
///
/// - `type_table`: User-defined types for struct construction and field access
/// - `kernel_registry`: Kernel signatures for resolving call return types
/// - `signal_types`: Map from signal path to output type
/// - `field_types`: Map from field path to output type
/// - `local_bindings`: Currently in-scope let bindings
///
/// # Examples
///
/// ```rust,ignore
/// let ctx = TypingContext::new(
///     &type_table,
///     kernel_registry,
///     signal_types,
///     field_types,
/// );
/// ```
pub struct TypingContext<'a> {
    /// User-defined type definitions
    pub type_table: &'a TypeTable,

    /// Kernel signatures for call resolution
    pub kernel_registry: &'a KernelRegistry,

    /// Signal path → output type mapping
    pub signal_types: &'a HashMap<Path, Type>,

    /// Field path → output type mapping
    pub field_types: &'a HashMap<Path, Type>,

    /// Local let-bound variables (name → type)
    pub local_bindings: HashMap<String, Type>,
}

impl<'a> TypingContext<'a> {
    /// Create a new typing context
    ///
    /// # Parameters
    ///
    /// - `type_table`: User type definitions
    /// - `kernel_registry`: Kernel signatures
    /// - `signal_types`: Signal path → type mapping
    /// - `field_types`: Field path → type mapping
    ///
    /// # Returns
    ///
    /// A new typing context ready for use
    pub fn new(
        type_table: &'a TypeTable,
        kernel_registry: &'a KernelRegistry,
        signal_types: &'a HashMap<Path, Type>,
        field_types: &'a HashMap<Path, Type>,
    ) -> Self {
        Self {
            type_table,
            kernel_registry,
            signal_types,
            field_types,
            local_bindings: HashMap::new(),
        }
    }

    /// Fork context with additional local binding
    ///
    /// Creates a new context with the same registries but an extended local
    /// binding scope. Used for let expressions.
    ///
    /// # Parameters
    ///
    /// - `name`: Variable name to bind
    /// - `ty`: Type of the variable
    ///
    /// # Returns
    ///
    /// New context with extended local bindings
    fn with_binding(&self, name: String, ty: Type) -> Self {
        let mut ctx = Self {
            type_table: self.type_table,
            kernel_registry: self.kernel_registry,
            signal_types: self.signal_types,
            field_types: self.field_types,
            local_bindings: self.local_bindings.clone(),
        };
        ctx.local_bindings.insert(name, ty);
        ctx
    }
}

/// Type an untyped expression
///
/// Assigns types to all subexpressions by:
/// - Looking up signal/field types from registries
/// - Resolving kernel signatures for calls
/// - Propagating types through let bindings
/// - Inferring literal types from syntax
///
/// # Parameters
///
/// - `expr`: Untyped expression to type
/// - `ctx`: Typing context with registries and bindings
///
/// # Returns
///
/// `Ok(TypedExpr)` if typing succeeds, `Err` with list of errors otherwise.
///
/// # Errors
///
/// - `UnresolvedPath` - Signal/Field/Config/Const path not found
/// - `UnknownKernel` - Kernel call to unregistered kernel
/// - `TypeMismatch` - Incompatible types (checked in validation pass)
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::untyped::{Expr, ExprKind};
/// use continuum_cdsl::resolve::expr_typing::{type_expression, TypingContext};
///
/// let expr = Expr::new(ExprKind::Dt, span);
/// let typed_expr = type_expression(&expr, &ctx)?;
/// assert!(matches!(typed_expr.ty, Type::Kernel(_)));
/// ```
pub fn type_expression(expr: &Expr, ctx: &TypingContext) -> Result<TypedExpr, Vec<CompileError>> {
    let span = expr.span;
    let mut errors = Vec::new();

    let (kind, ty) = match &expr.kind {
        // === Literals ===
        UntypedKind::Literal { value, unit } => {
            let resolved_unit = match unit {
                Some(unit_expr) => resolve_unit_expr(Some(unit_expr), span).map_err(|e| vec![e])?,
                None => Unit::DIMENSIONLESS,
            };

            let kernel_type = KernelType {
                shape: Shape::Scalar,
                unit: resolved_unit,
                bounds: None,
            };

            (
                ExprKind::Literal {
                    value: *value,
                    unit: Some(resolved_unit),
                },
                Type::Kernel(kernel_type),
            )
        }

        UntypedKind::BoolLiteral(val) => (
            ExprKind::Literal {
                value: if *val { 1.0 } else { 0.0 },
                unit: None,
            },
            Type::Bool,
        ),

        // === References ===
        UntypedKind::Local(name) => {
            let ty = ctx.local_bindings.get(name).cloned().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!("local variable '{}' not in scope", name),
                )]
            })?;

            (ExprKind::Local(name.clone()), ty)
        }

        UntypedKind::Signal(path) => {
            let ty = ctx.signal_types.get(path).cloned().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!("signal '{}' not found", path),
                )]
            })?;

            (ExprKind::Signal(path.clone()), ty)
        }

        UntypedKind::Field(path) => {
            let ty = ctx.field_types.get(path).cloned().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!("field '{}' not found", path),
                )]
            })?;

            (ExprKind::Field(path.clone()), ty)
        }

        // === Context values ===
        UntypedKind::Dt => {
            let kernel_type = KernelType {
                shape: Shape::Scalar,
                unit: Unit::seconds(),
                bounds: None,
            };
            (ExprKind::Dt, Type::Kernel(kernel_type))
        }

        // === Not yet implemented ===
        UntypedKind::Vector(_)
        | UntypedKind::Config(_)
        | UntypedKind::Const(_)
        | UntypedKind::Prev
        | UntypedKind::Current
        | UntypedKind::Inputs
        | UntypedKind::Self_
        | UntypedKind::Other
        | UntypedKind::Payload
        | UntypedKind::Binary { .. }
        | UntypedKind::Unary { .. }
        | UntypedKind::If { .. }
        | UntypedKind::Let { .. }
        | UntypedKind::Aggregate { .. }
        | UntypedKind::Fold { .. }
        | UntypedKind::Call { .. }
        | UntypedKind::KernelCall { .. }
        | UntypedKind::Struct { .. }
        | UntypedKind::FieldAccess { .. }
        | UntypedKind::ParseError(_) => {
            errors.push(CompileError::new(
                ErrorKind::Internal,
                span,
                format!("expression typing not yet implemented for {:?}", expr.kind),
            ));
            return Err(errors);
        }
    };

    Ok(TypedExpr::new(kind, ty, span))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Span;

    fn make_context<'a>() -> TypingContext<'a> {
        let type_table = Box::leak(Box::new(TypeTable::new()));
        let kernel_registry = KernelRegistry::global();
        let signal_types = Box::leak(Box::new(HashMap::new()));
        let field_types = Box::leak(Box::new(HashMap::new()));

        TypingContext::new(type_table, kernel_registry, signal_types, field_types)
    }

    #[test]
    fn test_type_literal_dimensionless() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Literal {
                value: 42.0,
                unit: None,
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        assert!(matches!(typed.ty, Type::Kernel(_)));
    }

    #[test]
    fn test_type_bool_literal() {
        let ctx = make_context();
        let expr = Expr::new(UntypedKind::BoolLiteral(true), Span::new(0, 0, 10, 1));

        let typed = type_expression(&expr, &ctx).unwrap();
        assert!(matches!(typed.ty, Type::Bool));
    }

    #[test]
    fn test_type_dt() {
        let ctx = make_context();
        let expr = Expr::new(UntypedKind::Dt, Span::new(0, 0, 10, 1));

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::seconds());
                assert_eq!(kt.bounds, None);
            }
            _ => panic!("Expected Kernel type, got {:?}", typed.ty),
        }
    }

    #[test]
    fn test_type_local_not_in_scope() {
        let ctx = make_context();
        let expr = Expr::new(UntypedKind::Local("x".to_string()), Span::new(0, 0, 10, 1));

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    }
}
