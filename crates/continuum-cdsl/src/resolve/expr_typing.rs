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

/// Get kernel type from argument at index
///
/// Helper for SameAs derivation that extracts kernel type from an argument.
///
/// # Parameters
///
/// - `args`: Typed arguments
/// - `idx`: Parameter index to extract from
/// - `span`: Source span for error reporting
/// - `derivation_kind`: Description for error messages ("shape" or "unit")
///
/// # Returns
///
/// Reference to the kernel type at the specified index.
///
/// # Errors
///
/// - `Internal` - Index out of bounds or argument is not a kernel type
fn get_kernel_arg<'a>(
    args: &'a [TypedExpr],
    idx: usize,
    span: crate::foundation::Span,
    derivation_kind: &str,
) -> Result<&'a KernelType, Vec<CompileError>> {
    let arg = args.get(idx).ok_or_else(|| {
        vec![CompileError::new(
            ErrorKind::Internal,
            span,
            format!(
                "invalid parameter index {} in {} derivation",
                idx, derivation_kind
            ),
        )]
    })?;
    arg.ty.as_kernel().ok_or_else(|| {
        vec![CompileError::new(
            ErrorKind::Internal,
            span,
            format!("parameter {} is not a kernel type", idx),
        )]
    })
}

/// Derive return type from kernel signature and typed arguments
///
/// # Parameters
///
/// - `sig`: Kernel signature with return type derivation rules
/// - `args`: Typed arguments
/// - `span`: Source span for error reporting
///
/// # Returns
///
/// `Ok(Type)` if derivation succeeds, `Err` with errors otherwise.
///
/// # Bounds Derivation
///
/// - **Exact(shape)**: Returns unbounded (None)
/// - **Scalar**: Returns unbounded (None)
/// - **SameAs(idx)**: Copies bounds from argument at idx
/// - **FromBroadcast, VectorDim, MatrixDims**: Not yet implemented (errors)
/// - **Complex operations** (multiply, clamp, etc.): Require constraint
///   propagation (Phase 14/15)
///
/// # Errors
///
/// - `Internal` - Invalid parameter index in derivation
fn derive_return_type(
    sig: &crate::ast::KernelSignature,
    args: &[TypedExpr],
    span: crate::foundation::Span,
) -> Result<Type, Vec<CompileError>> {
    use crate::ast::{ShapeDerivation, UnitDerivation};

    // Derive shape and optionally bounds (when shape is SameAs)
    let (shape, bounds_from_shape) = match &sig.returns.shape {
        ShapeDerivation::Exact(s) => (s.clone(), None),
        ShapeDerivation::Scalar => (Shape::Scalar, None),
        ShapeDerivation::SameAs(idx) => {
            let kt = get_kernel_arg(args, *idx, span, "shape")?;
            (kt.shape.clone(), kt.bounds.clone())
        }
        _ => {
            return Err(vec![CompileError::new(
                ErrorKind::Internal,
                span,
                format!(
                    "shape derivation not yet implemented: {:?}",
                    sig.returns.shape
                ),
            )]);
        }
    };

    // Derive unit
    let unit = match &sig.returns.unit {
        UnitDerivation::Exact(u) => *u,
        UnitDerivation::Dimensionless => Unit::DIMENSIONLESS,
        UnitDerivation::SameAs(idx) => {
            let kt = get_kernel_arg(args, *idx, span, "unit")?;
            kt.unit
        }
        _ => {
            return Err(vec![CompileError::new(
                ErrorKind::Internal,
                span,
                format!(
                    "unit derivation not yet implemented: {:?}",
                    sig.returns.unit
                ),
            )]);
        }
    };

    // Bounds are derived from shape argument when using SameAs derivation.
    // For operations that transform values (multiply, clamp, etc.),
    // bounds derivation requires constraint propagation (Phase 14/15).
    Ok(Type::Kernel(KernelType {
        shape,
        unit,
        bounds: bounds_from_shape,
    }))
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

        // === Kernel calls ===
        UntypedKind::KernelCall { kernel, args } => {
            // Type each argument
            let typed_args: Vec<TypedExpr> = args
                .iter()
                .map(|arg| type_expression(arg, ctx))
                .collect::<Result<Vec<_>, _>>()?;

            // Lookup kernel signature
            let sig = ctx.kernel_registry.get(kernel).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    format!("unknown kernel: {:?}", kernel),
                )]
            })?;

            // Derive return type from signature + arg types
            // For MVP: just use the signature's return type derivation
            let return_type = derive_return_type(sig, &typed_args, span)?;

            (
                ExprKind::Call {
                    kernel: kernel.clone(),
                    args: typed_args,
                },
                return_type,
            )
        }

        // === Field access ===
        UntypedKind::FieldAccess { object, field } => {
            // Type the object first
            let typed_object = type_expression(object, ctx)?;

            // Extract field type based on object type
            let field_type = match &typed_object.ty {
                Type::User(type_id) => {
                    // Look up user type in type table
                    // Note: TypeTable doesn't have direct UserTypeId -> UserType lookup,
                    // so we iterate to find it. TODO: Add get_by_id method to TypeTable
                    let user_type = ctx
                        .type_table
                        .iter()
                        .find(|ut| ut.id() == type_id)
                        .ok_or_else(|| {
                            vec![CompileError::new(
                                ErrorKind::Internal,
                                span,
                                format!("user type {:?} not found in type table", type_id),
                            )]
                        })?;

                    // Look up field in user type
                    user_type.field(field).cloned().ok_or_else(|| {
                        vec![CompileError::new(
                            ErrorKind::UndefinedName,
                            span,
                            format!("field '{}' not found on type '{}'", field, user_type.name()),
                        )]
                    })?
                }
                Type::Kernel(kt) => {
                    // Vector component access (.x, .y, .z, .w)
                    match &kt.shape {
                        Shape::Vector { dim } => {
                            let component_index = match field.as_str() {
                                "x" => Some(0),
                                "y" => Some(1),
                                "z" => Some(2),
                                "w" => Some(3),
                                _ => None,
                            };

                            match component_index {
                                Some(idx) if idx < *dim as usize => {
                                    // Return scalar with same unit as vector
                                    Type::Kernel(KernelType {
                                        shape: Shape::Scalar,
                                        unit: kt.unit,
                                        bounds: None, // Component bounds not derived from vector bounds
                                    })
                                }
                                Some(_) => {
                                    return Err(vec![CompileError::new(
                                        ErrorKind::UndefinedName,
                                        span,
                                        format!(
                                            "component '{}' out of bounds for vector of dimension {}",
                                            field, dim
                                        ),
                                    )]);
                                }
                                None => {
                                    return Err(vec![CompileError::new(
                                        ErrorKind::UndefinedName,
                                        span,
                                        format!("invalid vector component '{}'", field),
                                    )]);
                                }
                            }
                        }
                        _ => {
                            return Err(vec![CompileError::new(
                                ErrorKind::TypeMismatch,
                                span,
                                format!("field access on non-struct, non-vector type"),
                            )]);
                        }
                    }
                }
                _ => {
                    return Err(vec![CompileError::new(
                        ErrorKind::TypeMismatch,
                        span,
                        format!("field access not supported on type {:?}", typed_object.ty),
                    )]);
                }
            };

            (
                ExprKind::FieldAccess {
                    object: Box::new(typed_object),
                    field: field.clone(),
                },
                field_type,
            )
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
        | UntypedKind::Struct { .. }
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

    #[test]
    fn test_derive_return_type_copies_bounds_from_same_as() {
        use crate::ast::{KernelReturn, KernelSignature, ShapeDerivation, UnitDerivation};
        use crate::foundation::Bounds;
        use continuum_kernel_types::KernelId;

        // Create a signature with SameAs shape derivation
        let sig = KernelSignature {
            id: KernelId::new("test", "identity"),
            params: vec![],
            returns: KernelReturn {
                shape: ShapeDerivation::SameAs(0),
                unit: UnitDerivation::SameAs(0),
            },
            purity: crate::ast::KernelPurity::Pure,
            requires_uses: None,
        };

        // Create a typed argument with bounds
        let arg_ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::meters(),
            bounds: Some(Bounds {
                min: Some(0.0),
                max: Some(100.0),
            }),
        });
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: 50.0,
                unit: Some(Unit::meters()),
            },
            arg_ty,
            Span::new(0, 0, 10, 1),
        );

        // Derive return type
        let result = derive_return_type(&sig, &[arg], Span::new(0, 0, 10, 1)).unwrap();

        // Verify bounds are copied
        match result {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::meters());
                assert_eq!(
                    kt.bounds,
                    Some(Bounds {
                        min: Some(0.0),
                        max: Some(100.0),
                    })
                );
            }
            _ => panic!("Expected Kernel type"),
        }
    }

    #[test]
    fn test_derive_return_type_no_bounds_for_exact_shape() {
        use crate::ast::{KernelReturn, KernelSignature, ShapeDerivation, UnitDerivation};
        use crate::foundation::Bounds;
        use continuum_kernel_types::KernelId;

        // Create a signature with Exact shape derivation
        let sig = KernelSignature {
            id: KernelId::new("test", "const"),
            params: vec![],
            returns: KernelReturn {
                shape: ShapeDerivation::Exact(Shape::Scalar),
                unit: UnitDerivation::Dimensionless,
            },
            purity: crate::ast::KernelPurity::Pure,
            requires_uses: None,
        };

        // Create a typed argument with bounds (should NOT be copied)
        let arg_ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::meters(),
            bounds: Some(Bounds {
                min: Some(0.0),
                max: Some(100.0),
            }),
        });
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: 50.0,
                unit: Some(Unit::meters()),
            },
            arg_ty,
            Span::new(0, 0, 10, 1),
        );

        // Derive return type
        let result = derive_return_type(&sig, &[arg], Span::new(0, 0, 10, 1)).unwrap();

        // Verify bounds are NOT copied (Exact shape doesn't inherit bounds)
        match result {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::DIMENSIONLESS);
                assert_eq!(kt.bounds, None);
            }
            _ => panic!("Expected Kernel type"),
        }
    }
}
