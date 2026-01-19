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
use continuum_kernel_types::KernelId;
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

    /// Config path → type mapping
    pub config_types: &'a HashMap<Path, Type>,

    /// Const path → type mapping
    pub const_types: &'a HashMap<Path, Type>,

    /// Local let-bound variables (name → type)
    pub local_bindings: HashMap<String, Type>,

    /// Current node output type (for `prev` and `current`)
    ///
    /// When typing execution blocks for a signal/field/operator, this is set
    /// to the node's output type. Expressions using `prev` or `current` resolve
    /// to this type.
    pub node_output: Option<Type>,

    /// Inputs type (for `inputs` expression)
    ///
    /// When typing execution blocks for a signal/member with inputs, this is set
    /// to the declared inputs type.
    pub inputs_type: Option<Type>,

    /// Payload type (for `payload` expression)
    ///
    /// When typing impulse handler blocks, this is set to the impulse's declared
    /// payload type.
    pub payload_type: Option<Type>,
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
    /// - `config_types`: Config path → type mapping
    /// - `const_types`: Const path → type mapping
    ///
    /// # Returns
    ///
    /// A new typing context ready for use with no execution context
    pub fn new(
        type_table: &'a TypeTable,
        kernel_registry: &'a KernelRegistry,
        signal_types: &'a HashMap<Path, Type>,
        field_types: &'a HashMap<Path, Type>,
        config_types: &'a HashMap<Path, Type>,
        const_types: &'a HashMap<Path, Type>,
    ) -> Self {
        Self {
            type_table,
            kernel_registry,
            signal_types,
            field_types,
            config_types,
            const_types,
            local_bindings: HashMap::new(),
            node_output: None,
            inputs_type: None,
            payload_type: None,
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
            config_types: self.config_types,
            const_types: self.const_types,
            local_bindings: self.local_bindings.clone(),
            node_output: self.node_output.clone(),
            inputs_type: self.inputs_type.clone(),
            payload_type: self.payload_type.clone(),
        };
        ctx.local_bindings.insert(name, ty);
        ctx
    }

    /// Set execution context for a node
    ///
    /// Creates a new context with the same registries but updated execution context.
    /// Used when typing execution blocks for a specific node.
    ///
    /// # Parameters
    ///
    /// - `node_output`: The node's output type (for `prev`/`current`)
    /// - `inputs_type`: The node's inputs type (for `inputs`)
    /// - `payload_type`: The impulse's payload type (for `payload`)
    ///
    /// # Returns
    ///
    /// New context with execution context set
    pub fn with_execution_context(
        &self,
        node_output: Option<Type>,
        inputs_type: Option<Type>,
        payload_type: Option<Type>,
    ) -> Self {
        Self {
            type_table: self.type_table,
            kernel_registry: self.kernel_registry,
            signal_types: self.signal_types,
            field_types: self.field_types,
            config_types: self.config_types,
            const_types: self.const_types,
            local_bindings: self.local_bindings.clone(),
            node_output,
            inputs_type,
            payload_type,
        }
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
        // === Field access ===
        //
        // Handles two distinct access patterns:
        //
        // 1. **User-defined struct field access**: `signal.position` → looks up field type
        //    - Requires TypeTable lookup by UserTypeId (currently O(n) via iter().find())
        //    - Returns the declared field type with its bounds
        //    - Errors if field doesn't exist on the user type
        //
        // 2. **Vector component access**: `velocity.x` → returns Scalar with same unit
        //    - Component names (x, y, z, w) map to indices (0, 1, 2, 3) within dimension
        //    - Returns Scalar shape with the vector's unit
        //    - Bounds are NOT propagated from vector to component (returns None)
        //    - Errors if component index >= vector dimension
        //
        // The bounds behavior differs between the two paths:
        // - Struct fields inherit their declared bounds from the type definition
        // - Vector components don't inherit vector bounds (semantically distinct values)
        UntypedKind::FieldAccess { object, field } => {
            // Helper to construct field access errors
            let err = |kind: ErrorKind, msg: String| -> Vec<CompileError> {
                vec![CompileError::new(kind, span, msg)]
            };

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
                            err(
                                ErrorKind::Internal,
                                format!("user type {:?} not found in type table", type_id),
                            )
                        })?;

                    // Look up field in user type
                    user_type.field(field).cloned().ok_or_else(|| {
                        err(
                            ErrorKind::UndefinedName,
                            format!("field '{}' not found on type '{}'", field, user_type.name()),
                        )
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
                                    return Err(err(
                                        ErrorKind::UndefinedName,
                                        format!(
                                            "component '{}' out of bounds for vector of dimension {}",
                                            field, dim
                                        ),
                                    ));
                                }
                                None => {
                                    return Err(err(
                                        ErrorKind::UndefinedName,
                                        format!("invalid vector component '{}'", field),
                                    ));
                                }
                            }
                        }
                        _ => {
                            return Err(err(
                                ErrorKind::TypeMismatch,
                                format!("field access on non-struct, non-vector type"),
                            ));
                        }
                    }
                }
                _ => {
                    return Err(err(
                        ErrorKind::TypeMismatch,
                        format!("field access not supported on type {:?}", typed_object.ty),
                    ));
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

        // === Config lookup ===
        UntypedKind::Config(path) => {
            let ty = ctx.config_types.get(path).cloned().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!("config path '{}' not found", path),
                )]
            })?;

            (ExprKind::Config(path.clone()), ty)
        }

        // === Const lookup ===
        UntypedKind::Const(path) => {
            let ty = ctx.const_types.get(path).cloned().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!("const path '{}' not found", path),
                )]
            })?;

            (ExprKind::Const(path.clone()), ty)
        }

        // === Prev (previous tick value) ===
        UntypedKind::Prev => {
            let ty = ctx.node_output.clone().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    "prev not available: no node output type in context".to_string(),
                )]
            })?;

            (ExprKind::Prev, ty)
        }

        // === Current (just-resolved value) ===
        UntypedKind::Current => {
            let ty = ctx.node_output.clone().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    "current not available: no node output type in context".to_string(),
                )]
            })?;

            (ExprKind::Current, ty)
        }

        // === Inputs (accumulated inputs) ===
        UntypedKind::Inputs => {
            let ty = ctx.inputs_type.clone().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    "inputs not available: no inputs type in context".to_string(),
                )]
            })?;

            (ExprKind::Inputs, ty)
        }

        // === Payload (impulse payload) ===
        UntypedKind::Payload => {
            let ty = ctx.payload_type.clone().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    "payload not available: no payload type in context".to_string(),
                )]
            })?;

            (ExprKind::Payload, ty)
        }

        // === Vector literal ===
        UntypedKind::Vector(elements) => {
            if elements.is_empty() {
                return Err(vec![CompileError::new(
                    ErrorKind::TypeMismatch,
                    span,
                    "vector literal cannot be empty".to_string(),
                )]);
            }

            if elements.len() > 4 {
                return Err(vec![CompileError::new(
                    ErrorKind::TypeMismatch,
                    span,
                    format!(
                        "vector literal has {} elements, maximum is 4",
                        elements.len()
                    ),
                )]);
            }

            // Type all elements
            let typed_elements: Result<Vec<_>, _> = elements
                .iter()
                .map(|elem| type_expression(elem, ctx))
                .collect();
            let typed_elements = typed_elements?;

            // All elements must be Kernel types
            let mut unit = None;
            for (i, elem) in typed_elements.iter().enumerate() {
                match &elem.ty {
                    Type::Kernel(kt) => {
                        if kt.shape != Shape::Scalar {
                            return Err(vec![CompileError::new(
                                ErrorKind::TypeMismatch,
                                elem.span,
                                format!(
                                    "vector element {} has shape {:?}, expected Scalar",
                                    i, kt.shape
                                ),
                            )]);
                        }

                        // All elements must have the same unit
                        if let Some(ref expected_unit) = unit {
                            if kt.unit != *expected_unit {
                                return Err(vec![CompileError::new(
                                    ErrorKind::TypeMismatch,
                                    elem.span,
                                    format!(
                                        "vector element {} has unit {}, expected {}",
                                        i, kt.unit, expected_unit
                                    ),
                                )]);
                            }
                        } else {
                            unit = Some(kt.unit);
                        }
                    }
                    _ => {
                        return Err(vec![CompileError::new(
                            ErrorKind::TypeMismatch,
                            elem.span,
                            format!("vector element {} has non-kernel type {:?}", i, elem.ty),
                        )]);
                    }
                }
            }

            let dim = elements.len() as u8;
            let unit = unit.unwrap(); // Safe: verified non-empty above

            (
                ExprKind::Vector(typed_elements),
                Type::Kernel(KernelType {
                    shape: Shape::Vector { dim },
                    unit,
                    bounds: None, // Vector bounds not derived from element bounds
                }),
            )
        }

        // === Let binding ===
        UntypedKind::Let { name, value, body } => {
            // Type the value expression
            let typed_value = type_expression(value, ctx)?;

            // Create extended context with the binding
            let extended_ctx = ctx.with_binding(name.clone(), typed_value.ty.clone());

            // Type the body in the extended context
            let typed_body = type_expression(body, &extended_ctx)?;

            // Result type is the body's type
            let ty = typed_body.ty.clone();

            (
                ExprKind::Let {
                    name: name.clone(),
                    value: Box::new(typed_value),
                    body: Box::new(typed_body),
                },
                ty,
            )
        }

        // === Struct literal ===
        UntypedKind::Struct {
            ty: ty_path,
            fields,
        } => {
            // Look up the user type
            let type_id = ctx
                .type_table
                .get_id(ty_path)
                .ok_or_else(|| {
                    vec![CompileError::new(
                        ErrorKind::UndefinedName,
                        span,
                        format!("unknown type '{}'", ty_path),
                    )]
                })?
                .clone();

            let user_type = ctx.type_table.get(ty_path).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    format!("type '{}' not found in type table", ty_path),
                )]
            })?;

            // Type all field expressions
            let mut typed_fields = Vec::new();
            for (field_name, field_expr) in fields {
                let typed_expr = type_expression(field_expr, ctx)?;

                // Verify field exists in type
                let expected_type = user_type.field(field_name).ok_or_else(|| {
                    vec![CompileError::new(
                        ErrorKind::UndefinedName,
                        field_expr.span,
                        format!("field '{}' not found on type '{}'", field_name, ty_path),
                    )]
                })?;

                // Verify type matches
                if &typed_expr.ty != expected_type {
                    return Err(vec![CompileError::new(
                        ErrorKind::TypeMismatch,
                        field_expr.span,
                        format!(
                            "field '{}' has type {:?}, expected {:?}",
                            field_name, typed_expr.ty, expected_type
                        ),
                    )]);
                }

                typed_fields.push((field_name.clone(), typed_expr));
            }

            // Verify all declared fields are provided
            for (declared_field, _) in user_type.fields() {
                if !fields.iter().any(|(name, _)| name == declared_field) {
                    return Err(vec![CompileError::new(
                        ErrorKind::TypeMismatch,
                        span,
                        format!("missing field '{}' in struct literal", declared_field),
                    )]);
                }
            }

            (
                ExprKind::Struct {
                    ty: type_id.clone(),
                    fields: typed_fields,
                },
                Type::User(type_id),
            )
        }

        // === Binary operator (desugar to kernel call) ===
        UntypedKind::Binary { op, left, right } => {
            let kernel_id = op.kernel();

            // Type operands
            let typed_left = type_expression(left, ctx)?;
            let typed_right = type_expression(right, ctx)?;

            // Look up kernel signature
            let sig = ctx.kernel_registry.get(&kernel_id).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!(
                        "kernel '{:?}' not found for unary operator {:?}",
                        kernel_id, op
                    ),
                )]
            })?;

            // Derive return type
            let return_type =
                derive_return_type(sig, &[typed_left.clone(), typed_right.clone()], span)?;

            (
                ExprKind::Call {
                    kernel: kernel_id,
                    args: vec![typed_left, typed_right],
                },
                return_type,
            )
        }

        // === Unary operator (desugar to kernel call) ===
        UntypedKind::Unary { op, operand } => {
            let kernel_id = op.kernel();

            // Type operand
            let typed_operand = type_expression(operand, ctx)?;

            // Look up kernel signature
            let sig = ctx.kernel_registry.get(&kernel_id).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::UndefinedName,
                    span,
                    format!(
                        "kernel '{:?}' not found for unary operator {:?}",
                        kernel_id, op
                    ),
                )]
            })?;

            // Derive return type
            let return_type = derive_return_type(sig, &[typed_operand.clone()], span)?;

            (
                ExprKind::Call {
                    kernel: kernel_id,
                    args: vec![typed_operand],
                },
                return_type,
            )
        }

        // === If-then-else (desugar to logic.select) ===
        UntypedKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let kernel_id = KernelId::new("logic", "select");

            // Type all branches
            let typed_condition = type_expression(condition, ctx)?;
            let typed_then = type_expression(then_branch, ctx)?;
            let typed_else = type_expression(else_branch, ctx)?;

            // Look up kernel signature
            let sig = ctx.kernel_registry.get(&kernel_id).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    "kernel 'logic.select' not found (required for if-then-else)".to_string(),
                )]
            })?;

            // Derive return type
            let return_type = derive_return_type(
                sig,
                &[
                    typed_condition.clone(),
                    typed_then.clone(),
                    typed_else.clone(),
                ],
                span,
            )?;

            (
                ExprKind::Call {
                    kernel: kernel_id,
                    args: vec![typed_condition, typed_then, typed_else],
                },
                return_type,
            )
        }

        // === Not yet implemented ===
        UntypedKind::Self_
        | UntypedKind::Other
        | UntypedKind::Aggregate { .. }
        | UntypedKind::Fold { .. }
        | UntypedKind::Call { .. }
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
    use crate::foundation::{Path, Span, UserType};
    use continuum_foundation::TypeId;

    fn make_context<'a>() -> TypingContext<'a> {
        let type_table = Box::leak(Box::new(TypeTable::new()));
        let kernel_registry = KernelRegistry::global();
        let signal_types = Box::leak(Box::new(HashMap::new()));
        let field_types = Box::leak(Box::new(HashMap::new()));
        let config_types = Box::leak(Box::new(HashMap::new()));
        let const_types = Box::leak(Box::new(HashMap::new()));

        TypingContext::new(
            type_table,
            kernel_registry,
            signal_types,
            field_types,
            config_types,
            const_types,
        )
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

    // === FieldAccess Tests ===

    #[test]
    fn test_type_field_access_vector_component_x() {
        let mut ctx = make_context();

        // Create a Vec3 with velocity unit (m/s)
        let velocity_unit = Unit::meters().divide(&Unit::seconds()).unwrap();
        ctx.local_bindings.insert(
            "velocity".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 3 },
                unit: velocity_unit,
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("velocity".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "x".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();

        // Should return Scalar with same unit
        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, velocity_unit);
                assert_eq!(kt.bounds, None);
            }
            _ => panic!("Expected Kernel type, got {:?}", typed.ty),
        }
    }

    #[test]
    fn test_type_field_access_vector_component_y_on_vec2() {
        let mut ctx = make_context();

        ctx.local_bindings.insert(
            "pos".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 2 },
                unit: Unit::meters(),
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("pos".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "y".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();

        // Should return Scalar with meters unit
        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::meters());
                assert_eq!(kt.bounds, None);
            }
            _ => panic!("Expected Kernel type, got {:?}", typed.ty),
        }
    }

    #[test]
    fn test_type_field_access_w_on_vec4() {
        let mut ctx = make_context();

        ctx.local_bindings.insert(
            "quaternion".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 4 },
                unit: Unit::DIMENSIONLESS,
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("quaternion".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "w".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();

        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::DIMENSIONLESS);
            }
            _ => panic!("Expected Kernel type"),
        }
    }

    #[test]
    fn test_type_field_access_w_on_vec3_fails() {
        let mut ctx = make_context();

        ctx.local_bindings.insert(
            "vec3".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 3 },
                unit: Unit::meters(),
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("vec3".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "w".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
        assert!(errors[0].message.contains("out of bounds"));
        assert!(errors[0].message.contains("dimension 3"));
    }

    #[test]
    fn test_type_field_access_z_on_vec2_fails() {
        let mut ctx = make_context();

        ctx.local_bindings.insert(
            "vec2".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 2 },
                unit: Unit::meters(),
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("vec2".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "z".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    }

    #[test]
    fn test_type_field_access_invalid_component_name() {
        let mut ctx = make_context();

        ctx.local_bindings.insert(
            "vec3".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 3 },
                unit: Unit::meters(),
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("vec3".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "foo".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
        assert!(errors[0].message.contains("invalid vector component"));
    }

    #[test]
    fn test_type_field_access_on_scalar_fails() {
        let mut ctx = make_context();

        ctx.local_bindings.insert(
            "scalar".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::meters(),
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("scalar".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "x".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
        assert!(errors[0].message.contains("non-struct, non-vector"));
    }

    #[test]
    fn test_type_field_access_on_bool_fails() {
        let mut ctx = make_context();

        ctx.local_bindings.insert("flag".to_string(), Type::Bool);

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("flag".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "x".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
        assert!(errors[0].message.contains("not supported"));
    }

    #[test]
    fn test_type_field_access_user_struct() {
        let type_table = Box::leak(Box::new({
            let mut table = TypeTable::new();

            // Create a user type "Position" with fields x, y
            let type_id = TypeId::from("Position");
            let user_type = UserType::new(
                type_id.clone(),
                Path::from("Position"),
                vec![
                    (
                        "x".to_string(),
                        Type::Kernel(KernelType {
                            shape: Shape::Scalar,
                            unit: Unit::meters(),
                            bounds: None,
                        }),
                    ),
                    (
                        "y".to_string(),
                        Type::Kernel(KernelType {
                            shape: Shape::Scalar,
                            unit: Unit::meters(),
                            bounds: None,
                        }),
                    ),
                ],
            );
            table.register(user_type);
            table
        }));

        let kernel_registry = KernelRegistry::global();
        let signal_types = Box::leak(Box::new(HashMap::new()));
        let field_types = Box::leak(Box::new(HashMap::new()));
        let config_types = Box::leak(Box::new(HashMap::new()));
        let const_types = Box::leak(Box::new(HashMap::new()));
        let mut ctx = TypingContext::new(
            type_table,
            kernel_registry,
            signal_types,
            field_types,
            config_types,
            const_types,
        );

        // Create a local variable of type Position
        let type_id = TypeId::from("Position");
        ctx.local_bindings
            .insert("pos".to_string(), Type::User(type_id));

        // Access pos.x
        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("pos".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "x".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();

        // Should return the field type
        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::meters());
            }
            _ => panic!("Expected Kernel type, got {:?}", typed.ty),
        }
    }

    #[test]
    fn test_type_field_access_unknown_struct_field() {
        let type_table = Box::leak(Box::new({
            let mut table = TypeTable::new();

            let type_id = TypeId::from("Position");
            let user_type = UserType::new(
                type_id.clone(),
                Path::from("Position"),
                vec![(
                    "x".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::meters(),
                        bounds: None,
                    }),
                )],
            );
            table.register(user_type);
            table
        }));

        let kernel_registry = KernelRegistry::global();
        let signal_types = Box::leak(Box::new(HashMap::new()));
        let field_types = Box::leak(Box::new(HashMap::new()));
        let config_types = Box::leak(Box::new(HashMap::new()));
        let const_types = Box::leak(Box::new(HashMap::new()));
        let mut ctx = TypingContext::new(
            type_table,
            kernel_registry,
            signal_types,
            field_types,
            config_types,
            const_types,
        );

        let type_id = TypeId::from("Position");
        ctx.local_bindings
            .insert("pos".to_string(), Type::User(type_id));

        // Try to access non-existent field pos.z
        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("pos".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "z".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
        assert!(errors[0].message.contains("field 'z' not found"));
    }

    #[test]
    fn test_type_field_access_vector_component_preserves_unit() {
        let mut ctx = make_context();

        // Create a vector with a custom unit (force = kg·m/s²)
        let newton = Unit::kilograms()
            .multiply(&Unit::meters())
            .unwrap()
            .divide(&Unit::seconds())
            .unwrap()
            .divide(&Unit::seconds())
            .unwrap();

        ctx.local_bindings.insert(
            "force".to_string(),
            Type::Kernel(KernelType {
                shape: Shape::Vector { dim: 3 },
                unit: newton,
                bounds: None,
            }),
        );

        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(
                    UntypedKind::Local("force".to_string()),
                    Span::new(0, 0, 5, 1),
                )),
                field: "z".to_string(),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();

        // Component should have same unit as vector
        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, newton);
                assert_eq!(kt.bounds, None);
            }
            _ => panic!("Expected Kernel type"),
        }
    }
}
