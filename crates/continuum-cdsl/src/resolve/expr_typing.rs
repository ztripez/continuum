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
//! - **No code generation** - Produces typed AST, not bytecode
//!
//! # What This Pass ALSO Does (Implementation Details)
//!
//! - **Operator desugaring** - Binary/Unary/If operators are desugared to KernelCall expressions
//!   by the dedicated desugar pass before typing runs.
//! - **Type compatibility validation** - Some type checks happen during typing where needed for inference
//!   (Full semantic validation happens in separate validation pass)
//! - **Phase boundary enforcement** - Field expressions are validated to only appear in Measure phase
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Res → Type Resolution → EXPR TYPING → Validation → Compilation
//!                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!                        YOU ARE HERE
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
use crate::resolve::types::TypeTable;
use crate::resolve::units::resolve_unit_expr;
use continuum_foundation::Phase;
use std::collections::HashMap;

/// Context for expression typing
///
/// Provides access to type registries and tracks local bindings and execution
/// context during typing.
///
/// The context carries:
/// - **Global Registries**: Type definitions and kernel signatures.
/// - **Authoritative State**: Declared types for signals, fields, configs, and constants.
/// - **Local Scope**: Types for variables bound in `let` expressions.
/// - **Execution Context**: Types for context-dependent values like `self`, `inputs`, or `node_output`.
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::resolve::expr_typing::TypingContext;
///
/// let ctx = TypingContext::new(
///     &type_table,
///     &kernel_registry,
///     &signal_types,
///     &field_types,
///     &config_types,
///     &const_types,
/// );
/// ```
#[derive(Clone)]
pub struct TypingContext<'a> {
    /// User-defined type definitions for struct construction and field access.
    pub type_table: &'a TypeTable,

    /// Kernel signatures used to resolve function calls and derive return types.
    pub kernel_registry: &'a KernelRegistry,

    /// Mapping from signal path to its authoritative resolved type.
    pub signal_types: &'a HashMap<Path, Type>,

    /// Mapping from field path to its authoritative observation type.
    pub field_types: &'a HashMap<Path, Type>,

    /// Mapping from world configuration path to its value type.
    pub config_types: &'a HashMap<Path, Type>,

    /// Mapping from CDSL constant path to its declared type.
    pub const_types: &'a HashMap<Path, Type>,

    /// Currently in-scope local variable bindings (name → type).
    pub local_bindings: HashMap<String, Type>,

    /// The type of the `self` entity instance in the current execution block.
    pub self_type: Option<Type>,

    /// The type of the `other` entity instance (used in n-body interaction blocks).
    pub other_type: Option<Type>,

    /// The output type of the node being typed (for `prev`, `current`, and aggregates).
    ///
    /// When typing a signal or field's execution block, this is set to the node's
    /// own output type.
    pub node_output: Option<Type>,

    /// The type of the inputs struct (for the `inputs` expression).
    pub inputs_type: Option<Type>,

    /// The type of the impulse payload (for `payload` expressions in impulse handlers).
    pub payload_type: Option<Type>,

    /// The execution phase for phase-based boundary enforcement.
    ///
    /// Used to ensure that `Field` expressions are only accessed in the `Measure` phase,
    /// preserving the absolute observer boundary.
    pub phase: Option<Phase>,
}

impl<'a> TypingContext<'a> {
    /// Create a new typing context with the provided registries.
    ///
    /// # Parameters
    /// - `type_table`: User type definitions.
    /// - `kernel_registry`: Kernel signatures.
    /// - `signal_types`: Signal path → type mapping.
    /// - `field_types`: Field path → type mapping.
    /// - `config_types`: Config path → type mapping.
    /// - `const_types`: Const path → type mapping.
    ///
    /// # Returns
    /// A new typing context ready for use with no execution context.
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
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: None,
        }
    }

    /// Fork context with an additional local variable binding.
    ///
    /// Creates a new context with the same registries but an extended local
    /// binding scope. Used for `let` expressions.
    ///
    /// # Parameters
    /// - `name`: Variable name to bind.
    /// - `ty`: Type of the variable.
    ///
    /// # Returns
    /// New context with extended local bindings.
    pub fn with_binding(&self, name: String, ty: Type) -> Self {
        let mut ctx = Self {
            type_table: self.type_table,
            kernel_registry: self.kernel_registry,
            signal_types: self.signal_types,
            field_types: self.field_types,
            config_types: self.config_types,
            const_types: self.const_types,
            local_bindings: self.local_bindings.clone(),
            self_type: self.self_type.clone(),
            other_type: self.other_type.clone(),
            node_output: self.node_output.clone(),
            inputs_type: self.inputs_type.clone(),
            payload_type: self.payload_type.clone(),
            phase: self.phase,
        };
        ctx.local_bindings.insert(name, ty);
        ctx
    }

    /// Set execution context for a specific node.
    ///
    /// Creates a new context with the same registries but updated execution context.
    /// Used when typing execution blocks for a specific node.
    ///
    /// # Parameters
    /// - `self_type`: The type of the 'self' entity.
    /// - `other_type`: The type of the 'other' entity.
    /// - `node_output`: The node's output type (for `prev`/`current`).
    /// - `inputs_type`: The node's inputs type (for `inputs`).
    /// - `payload_type`: The impulse's payload type (for `payload`).
    ///
    /// # Returns
    /// New context with execution context set.
    pub fn with_execution_context(
        &self,
        self_type: Option<Type>,
        other_type: Option<Type>,
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
            self_type,
            other_type,
            node_output,
            inputs_type,
            payload_type,
            phase: self.phase,
        }
    }

    /// Set phase context for boundary enforcement.
    ///
    /// Creates a new context with the specified execution phase.
    /// Used to enforce phase boundaries (e.g., fields only in Measure).
    ///
    /// # Parameters
    /// - `phase`: The active execution phase.
    ///
    /// # Returns
    /// New context with phase set.
    pub fn with_phase(&self, phase: Phase) -> Self {
        Self {
            type_table: self.type_table,
            kernel_registry: self.kernel_registry,
            signal_types: self.signal_types,
            field_types: self.field_types,
            config_types: self.config_types,
            const_types: self.const_types,
            local_bindings: self.local_bindings.clone(),
            self_type: self.self_type.clone(),
            other_type: self.other_type.clone(),
            node_output: self.node_output.clone(),
            inputs_type: self.inputs_type.clone(),
            payload_type: self.payload_type.clone(),
            phase: Some(phase),
        }
    }
}

/// Extracts the [`KernelType`] from a typed argument at the specified index.
///
/// Used by return type derivation rules (like `SameAs`) to resolve dependencies
/// between input and output types.
///
/// # Parameters
/// - `args`: The list of already typed arguments.
/// - `idx`: The zero-based parameter index to extract.
/// - `span`: Source location for error reporting.
/// - `derivation_kind`: Description used in error messages (e.g., "shape", "unit").
///
/// # Returns
/// A reference to the kernel type of the argument.
///
/// # Errors
/// Returns an internal compiler error if the index is out of bounds or the
/// argument is not a kernel-compatible type.
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

/// Derives the result type of a kernel call based on its signature and arguments.
///
/// This implements the type propagation logic for Continuum engine kernels,
/// handling shape inheritance, unit multiplication/division, and numeric promotion.
///
/// # Parameters
/// - `sig`: The registered signature of the kernel being called.
/// - `args`: The list of typed arguments provided to the call.
/// - `span`: Source location of the call expression.
///
/// # Returns
/// The successfully derived [`Type`] for the call result.
///
/// # Errors
/// Returns a list of compilation errors if type derivation fails or if the
/// signature requires an unimplemented derivation rule.
fn derive_return_type(
    sig: &crate::ast::KernelSignature,
    args: &[TypedExpr],
    span: crate::foundation::Span,
) -> Result<Type, Vec<CompileError>> {
    use crate::ast::{ShapeDerivation, UnitDerivation};

    // Check value type to distinguish between boolean and numeric returns
    // Boolean kernels return Type::Bool directly without shape/unit derivation
    use continuum_kernel_types::ValueType;
    if sig.returns.value_type == ValueType::Bool {
        return Ok(Type::Bool);
    }

    if !sig.returns.value_type.is_numeric() {
        return Err(vec![CompileError::new(
            ErrorKind::Internal,
            span,
            format!(
                "non-boolean kernel return should be numeric, got {:?}",
                sig.returns.value_type
            ),
        )]);
    }

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
                ErrorKind::UnsupportedDSLFeature,
                span,
                format!(
                    "shape derivation variant not yet implemented: {:?}",
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
        UnitDerivation::Multiply(indices) => {
            // Multiply units of all specified parameters
            // Start with dimensionless (multiplicative identity)
            let mut result = Unit::DIMENSIONLESS;
            for &idx in indices {
                let kt = get_kernel_arg(args, idx, span, "unit multiply")?;
                result = result.multiply(&kt.unit).ok_or_else(|| {
                    vec![CompileError::new(
                        ErrorKind::Internal,
                        span,
                        format!(
                            "cannot multiply non-multiplicative units (parameter {})",
                            idx
                        ),
                    )]
                })?;
            }
            result
        }
        UnitDerivation::Divide(a, b) => {
            // Divide unit of parameter a by unit of parameter b
            let kt_a = get_kernel_arg(args, *a, span, "unit divide numerator")?;
            let kt_b = get_kernel_arg(args, *b, span, "unit divide denominator")?;
            kt_a.unit.divide(&kt_b.unit).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    format!(
                        "cannot divide non-multiplicative units (parameters {} / {})",
                        a, b
                    ),
                )]
            })?
        }
        UnitDerivation::Sqrt(_idx) | UnitDerivation::Inverse(_idx) => {
            return Err(vec![CompileError::new(
                ErrorKind::UnsupportedDSLFeature,
                span,
                format!(
                    "unit derivation variant not yet implemented: {:?}",
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

/// Infers the type of a numeric literal and its associated unit.
///
/// # Parameters
/// - `span`: Source location of the literal.
/// - `value`: The literal numeric value.
/// - `unit`: Optional unit expression associated with the literal (e.g., `42[m]`).
fn type_literal(
    span: crate::foundation::Span,
    value: f64,
    unit: Option<&crate::ast::UnitExpr>,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let resolved_unit = match unit {
        Some(unit_expr) => resolve_unit_expr(Some(unit_expr), span).map_err(|e| vec![e])?,
        None => Unit::DIMENSIONLESS,
    };

    let kernel_type = KernelType {
        shape: Shape::Scalar,
        unit: resolved_unit,
        bounds: None,
    };

    Ok((
        ExprKind::Literal {
            value,
            unit: Some(resolved_unit),
        },
        Type::Kernel(kernel_type),
    ))
}

/// Types a field or vector component access (e.g., `obj.field` or `vec.x`).
///
/// Handles both named field lookup for user-defined structs and component
/// extraction for primitive vector types.
///
/// # Parameters
/// - `ctx`: The typing context providing type and registry lookups.
/// - `object`: The expression being accessed.
/// - `field`: The name of the field or component.
/// - `span`: Source location of the access expression.
///
/// # Returns
/// A tuple containing the typed expression kind and its derived type.
///
/// # Errors
/// Returns a list of compilation errors if the object type does not support
/// field access or if the specified field is undefined.
fn type_field_access(
    ctx: &TypingContext,
    object: &Expr,
    field: &str,
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    // Type the object first
    let typed_object = type_expression(object, ctx)?;

    // Extract field type based on object type
    let field_type = match &typed_object.ty {
        Type::User(type_id) => {
            let user_type = ctx.type_table.get_by_id(type_id).ok_or_else(|| {
                err_internal(
                    span,
                    format!("user type {:?} not found in type table", type_id),
                )
            })?;

            // Look up field in user type
            user_type.field(field).cloned().ok_or_else(|| {
                err_undefined(
                    span,
                    field,
                    &format!("field on type '{}'", user_type.name()),
                )
            })?
        }
        Type::Kernel(kt) => {
            // Vector component access (.x, .y, .z, .w)
            match &kt.shape {
                Shape::Vector { dim } => {
                    let component_index = match field {
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
                            return Err(err_undefined(
                                span,
                                field,
                                &format!("component out of bounds for vector of dimension {}", dim),
                            ));
                        }
                        None => {
                            return Err(err_undefined(span, field, "invalid vector component"));
                        }
                    }
                }
                _ => {
                    return Err(err_internal(
                        span,
                        "field access on non-struct, non-vector type",
                    ));
                }
            }
        }
        _ => {
            return Err(err_internal(
                span,
                format!("field access not supported on type {:?}", typed_object.ty),
            ));
        }
    };

    Ok((
        ExprKind::FieldAccess {
            object: Box::new(typed_object),
            field: field.to_string(),
        },
        field_type,
    ))
}

/// Types a vector literal expression (e.g., `[1.0, 2.0, 3.0]`).
///
/// Ensures all elements are compatible scalars and have matching units.
///
/// # Parameters
/// - `ctx`: The typing context providing type lookups.
/// - `elements`: The list of component expressions.
/// - `span`: Source location of the vector literal.
///
/// # Returns
/// A tuple containing the typed expression kind and the derived vector type.
///
/// # Errors
/// Returns a list of compilation errors if elements are not scalars, have
/// mismatched units, or exceed the maximum dimension (4).
fn type_vector(
    ctx: &TypingContext,
    elements: &[Expr],
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
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
    let unit = unit.ok_or_else(|| err_internal(span, "vector literal unit resolution failed"))?;

    Ok((
        ExprKind::Vector(typed_elements),
        Type::Kernel(KernelType {
            shape: Shape::Vector { dim },
            unit,
            bounds: None,
        }),
    ))
}

/// Types a `let` binding expression, extending the context for its body.
///
/// # Parameters
/// - `ctx`: The typing context to extend.
/// - `name`: The variable name being bound.
/// - `value`: The expression whose value is being bound.
/// - `body`: The expression in which the binding is in scope.
/// - `_span`: Source location (currently unused).
///
/// # Returns
/// A tuple containing the typed expression kind and the body's derived type.
///
/// # Errors
/// Returns errors encountered while typing either the value or the body.
fn type_let(
    ctx: &TypingContext,
    name: &str,
    value: &Expr,
    body: &Expr,
    _span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_value = type_expression(value, ctx)?;
    let extended_ctx = ctx.with_binding(name.to_string(), typed_value.ty.clone());
    let typed_body = type_expression(body, &extended_ctx)?;
    let ty = typed_body.ty.clone();

    Ok((
        ExprKind::Let {
            name: name.to_string(),
            value: Box::new(typed_value),
            body: Box::new(typed_body),
        },
        ty,
    ))
}

/// Types a struct literal expression (e.g., `Point { x: 1.0, y: 2.0 }`).
///
/// Validates that the type exists, all fields are provided, and their types match.
///
/// # Parameters
/// - `ctx`: The typing context providing type table lookups.
/// - `ty_path`: The canonical path of the user-defined struct type.
/// - `fields`: The list of field names and their initialization expressions.
/// - `span`: Source location of the struct literal.
///
/// # Returns
/// A tuple containing the typed expression kind and the user-defined type.
///
/// # Errors
/// Returns a list of compilation errors if the type is undefined, fields are
/// missing, or field types do not match the declaration.
fn type_struct(
    ctx: &TypingContext,
    ty_path: &Path,
    fields: &[(String, Expr)],
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let type_id = ctx
        .type_table
        .get_id(ty_path)
        .ok_or_else(|| err_undefined(span, &ty_path.to_string(), "type"))?
        .clone();

    let user_type = ctx
        .type_table
        .get(ty_path)
        .ok_or_else(|| err_internal(span, format!("type '{}' not found in table", ty_path)))?;

    let mut typed_fields = Vec::new();
    let mut seen_fields = std::collections::HashSet::new();

    for (field_name, field_expr) in fields {
        if !seen_fields.insert(field_name.clone()) {
            return Err(vec![CompileError::new(
                ErrorKind::TypeMismatch,
                field_expr.span,
                format!("field '{}' specified multiple times", field_name),
            )]);
        }

        let typed_expr = type_expression(field_expr, ctx)?;
        let expected_type = user_type.field(field_name).ok_or_else(|| {
            err_undefined(
                field_expr.span,
                field_name,
                &format!("field on type '{}'", ty_path),
            )
        })?;

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

    for (declared_field, _) in user_type.fields() {
        if !fields.iter().any(|(name, _)| name == declared_field) {
            return Err(vec![CompileError::new(
                ErrorKind::TypeMismatch,
                span,
                format!("missing field '{}'", declared_field),
            )]);
        }
    }

    Ok((
        ExprKind::Struct {
            ty: type_id.clone(),
            fields: typed_fields,
        },
        Type::User(type_id),
    ))
}

/// Types an aggregate operation (sum, map, count, etc.) over an entity set.
///
/// Binds the current entity instance to `binding` within the `body` expression.
///
/// # Parameters
/// - `ctx`: The typing context providing type and registry lookups.
/// - `op`: The aggregate operation to perform.
/// - `entity`: The entity set being iterated over.
/// - `binding`: The name of the variable representing the current instance.
/// - `body`: The expression evaluated for each instance.
/// - `_span`: Source location (currently unused).
///
/// # Returns
/// A tuple containing the typed expression kind and the derived aggregate type.
///
/// # Errors
/// Returns errors encountered while typing the aggregate body.
fn type_aggregate(
    ctx: &TypingContext,
    op: &crate::ast::AggregateOp,
    entity: &crate::ast::Entity,
    binding: &str,
    body: &Expr,
    _span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let element_ty = Type::User(continuum_foundation::TypeId::from(entity.id.0.to_string()));
    let extended_ctx = ctx.with_binding(binding.to_string(), element_ty);
    let typed_body = type_expression(body, &extended_ctx)?;

    let aggregate_ty = match op {
        crate::ast::AggregateOp::Map => Type::Seq(Box::new(typed_body.ty.clone())),
        crate::ast::AggregateOp::Sum => typed_body.ty.clone(),
        crate::ast::AggregateOp::Max | crate::ast::AggregateOp::Min => typed_body.ty.clone(),
        crate::ast::AggregateOp::Count => Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        }),
        crate::ast::AggregateOp::Any | crate::ast::AggregateOp::All => Type::Bool,
    };

    Ok((
        ExprKind::Aggregate {
            op: *op,
            entity: entity.id.clone(),
            binding: binding.to_string(),
            body: Box::new(typed_body),
        },
        aggregate_ty,
    ))
}

/// Types a stateful fold operation over an entity set.
///
/// Binds `acc` to the current accumulator value and `elem` to the current
/// entity instance within the `body` expression.
///
/// # Parameters
/// - `ctx`: The typing context providing type lookups.
/// - `entity`: The entity set being iterated over.
/// - `init`: The initial value of the accumulator.
/// - `acc`: The name of the accumulator variable.
/// - `elem`: The name of the current instance variable.
/// - `body`: The expression yielding the next accumulator value.
/// - `span`: Source location of the fold expression.
///
/// # Returns
/// A tuple containing the typed expression kind and the accumulator's type.
///
/// # Errors
/// Returns a list of compilation errors if the body type does not match the
/// accumulator type or if either expression fails to type.
fn type_fold(
    ctx: &TypingContext,
    entity: &crate::ast::Entity,
    init: &Expr,
    acc: &str,
    elem: &str,
    body: &Expr,
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_init = type_expression(init, ctx)?;
    let elem_ty = Type::User(continuum_foundation::TypeId::from(entity.id.0.to_string()));

    let mut extended_ctx = ctx.with_binding(acc.to_string(), typed_init.ty.clone());
    extended_ctx
        .local_bindings
        .insert(elem.to_string(), elem_ty);

    let typed_body = type_expression(body, &extended_ctx)?;

    if typed_body.ty != typed_init.ty {
        return Err(vec![CompileError::new(
            ErrorKind::TypeMismatch,
            span,
            format!(
                "fold body type {:?} does not match accumulator type {:?}",
                typed_body.ty, typed_init.ty
            ),
        )]);
    }

    Ok((
        ExprKind::Fold {
            entity: entity.id.clone(),
            init: Box::new(typed_init),
            acc: acc.to_string(),
            elem: elem.to_string(),
            body: Box::new(typed_body.clone()),
        },
        typed_body.ty,
    ))
}

/// Types a function or kernel call by its path.
///
/// # Parameters
/// - `ctx`: The typing context providing kernel registry lookups.
/// - `func`: The path to the function or kernel.
/// - `args`: The list of argument expressions.
/// - `span`: Source location of the call.
///
/// # Returns
/// A tuple containing the typed expression kind and the derived return type.
///
/// # Errors
/// Returns a list of compilation errors if the kernel is undefined, path is
/// invalid, or return type derivation fails.
fn type_call(
    ctx: &TypingContext,
    func: &Path,
    args: &[Expr],
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_args: Vec<TypedExpr> = args
        .iter()
        .map(|arg| type_expression(arg, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    let segments = func.segments();
    if segments.is_empty() {
        return Err(err_internal(span, "kernel path is empty"));
    }

    if segments.len() > 2 {
        return Err(err_internal(
            span,
            format!("kernel path '{}' must be namespace.name or bare name", func),
        ));
    }

    let (namespace, name) = if segments.len() == 1 {
        ("", segments[0].as_str())
    } else {
        (segments[0].as_str(), segments[1].as_str())
    };

    let sig = ctx
        .kernel_registry
        .get_by_name(namespace, name)
        .ok_or_else(|| err_undefined(span, &func.to_string(), "kernel"))?;

    let kernel_id = sig.id.clone();
    let return_type = derive_return_type(sig, &typed_args, span)?;

    Ok((
        ExprKind::Call {
            kernel: kernel_id,
            args: typed_args,
        },
        return_type,
    ))
}

/// Types a kernel call when the [`KernelId`] is already known.
///
/// # Parameters
/// - `ctx`: The typing context providing kernel registry lookups.
/// - `kernel`: The unique identifier of the kernel.
/// - `args`: The list of argument expressions.
/// - `span`: Source location of the call.
///
/// # Returns
/// A tuple containing the typed expression kind and the derived return type.
///
/// # Errors
/// Returns a list of compilation errors if the kernel is unknown or if
/// return type derivation fails.
fn type_as_kernel_call(
    ctx: &TypingContext,
    kernel: &continuum_kernel_types::KernelId,
    args: &[Expr],
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_args: Vec<TypedExpr> = args
        .iter()
        .map(|arg| type_expression(arg, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    let sig = ctx
        .kernel_registry
        .get(kernel)
        .ok_or_else(|| err_internal(span, format!("unknown kernel: {:?}", kernel)))?;

    let return_type = derive_return_type(sig, &typed_args, span)?;

    Ok((
        ExprKind::Call {
            kernel: kernel.clone(),
            args: typed_args,
        },
        return_type,
    ))
}

/// Types a function or kernel call by its path.
///
/// # Parameters
/// - `ctx`: The typing context providing kernel registry lookups.
/// - `func`: The path to the function or kernel.
/// - `args`: The list of argument expressions.
/// - `span`: Source location of the call.
///
/// # Returns
/// A tuple containing the typed expression kind and the derived return type.
///
/// # Errors
/// Returns a list of compilation errors if the kernel is undefined, path is
/// invalid, or return type derivation fails.
fn type_call(
    ctx: &TypingContext,
    func: &Path,
    args: &[Expr],
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_args: Vec<TypedExpr> = args
        .iter()
        .map(|arg| type_expression(arg, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    let segments = func.segments();
    if segments.is_empty() {
        return Err(err_internal(span, "kernel path is empty"));
    }

    if segments.len() > 2 {
        return Err(err_internal(
            span,
            format!("kernel path '{}' must be namespace.name or bare name", func),
        ));
    }

    let (namespace, name) = if segments.len() == 1 {
        ("", segments[0].as_str())
    } else {
        (segments[0].as_str(), segments[1].as_str())
    };

    let sig = ctx
        .kernel_registry
        .get_by_name(namespace, name)
        .ok_or_else(|| err_undefined(span, &func.to_string(), "kernel"))?;

    let kernel_id = sig.id.clone();
    let return_type = derive_return_type(sig, &typed_args, span)?;

    Ok((
        ExprKind::Call {
            kernel: kernel_id,
            args: typed_args,
        },
        return_type,
    ))
}

/// Types a kernel call when the [`KernelId`] is already known.
///
/// # Parameters
/// - `ctx`: The typing context providing kernel registry lookups.
/// - `kernel`: The unique identifier of the kernel.
/// - `args`: The list of argument expressions.
/// - `span`: Source location of the call.
///
/// # Returns
/// A tuple containing the typed expression kind and the derived return type.
///
/// # Errors
/// Returns a list of compilation errors if the kernel is unknown or if
/// return type derivation fails.
fn type_as_kernel_call(
    ctx: &TypingContext,
    kernel: &continuum_kernel_types::KernelId,
    args: &[Expr],
    span: crate::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_args: Vec<TypedExpr> = args
        .iter()
        .map(|arg| type_expression(arg, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    let sig = ctx
        .kernel_registry
        .get(kernel)
        .ok_or_else(|| err_internal(span, format!("unknown kernel: {:?}", kernel)))?;

    let return_type = derive_return_type(sig, &typed_args, span)?;

    Ok((
        ExprKind::Call {
            kernel: kernel.clone(),
            args: typed_args,
        },
        return_type,
    ))
}

fn test_type_current_with_node_output() {
    let ctx = make_context();
    let output_type = Type::Kernel(KernelType {
        shape: Shape::Vector { dim: 3 },
        unit: Unit::seconds(),
        bounds: None,
    });
    let ctx = ctx.with_execution_context(None, None, Some(output_type.clone()), None, None);

    let expr = Expr::new(UntypedKind::Current, Span::new(0, 0, 7, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, output_type);
}

#[test]
fn test_type_current_without_node_output() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Current, Span::new(0, 0, 7, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("current"));
}

#[test]
fn test_type_inputs_with_inputs_type() {
    let ctx = make_context();
    let inputs_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::kilograms(),
        bounds: None,
    });
    let ctx = ctx.with_execution_context(None, None, None, Some(inputs_type.clone()), None);

    let expr = Expr::new(UntypedKind::Inputs, Span::new(0, 0, 6, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, inputs_type);
}

#[test]
fn test_type_inputs_without_inputs_type() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Inputs, Span::new(0, 0, 6, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("inputs"));
}

#[test]
fn test_type_payload_with_payload_type() {
    let ctx = make_context();
    let payload_type = Type::User(TypeId::from("ImpulseData"));
    let ctx = ctx.with_execution_context(None, None, None, None, Some(payload_type.clone()));

    let expr = Expr::new(UntypedKind::Payload, Span::new(0, 0, 7, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, payload_type);
}

#[test]
fn test_type_payload_without_payload_type() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Payload, Span::new(0, 0, 7, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("payload"));
}

// ============================================================================
// Tests for Config/Const lookups
// ============================================================================

#[test]
fn test_type_config_found() {
    let mut ctx = make_context();
    let config_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::dimensionless(),
        bounds: None,
    });

    let path = Box::leak(Box::new(Path::from("gravity")));
    let config_types = Box::leak(Box::new({
        let mut map = HashMap::new();
        map.insert(path.clone(), config_type.clone());
        map
    }));
    ctx.config_types = config_types;

    let expr = Expr::new(UntypedKind::Config(path.clone()), Span::new(0, 0, 7, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, config_type);
}

#[test]
fn test_type_config_not_found() {
    let ctx = make_context();
    let path = Path::from("unknown_config");
    let expr = Expr::new(UntypedKind::Config(path), Span::new(0, 0, 14, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("config"));
}

#[test]
fn test_type_const_found() {
    let mut ctx = make_context();
    let const_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::meters(),
        bounds: None,
    });

    let path = Box::leak(Box::new(Path::from("PI")));
    let const_types = Box::leak(Box::new({
        let mut map = HashMap::new();
        map.insert(path.clone(), const_type.clone());
        map
    }));
    ctx.const_types = const_types;

    let expr = Expr::new(UntypedKind::Const(path.clone()), Span::new(0, 0, 2, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, const_type);
}

#[test]
fn test_type_const_not_found() {
    let ctx = make_context();
    let path = Path::from("UNKNOWN");
    let expr = Expr::new(UntypedKind::Const(path), Span::new(0, 0, 7, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("const"));
}

// ============================================================================
// Tests for Vector literal construction
// ============================================================================

#[test]
fn test_type_vector_2d() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Vector(vec![
            Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
        ]),
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Vector { dim: 2 });
            assert_eq!(kt.unit, Unit::dimensionless());
        }
        _ => panic!("Expected Kernel type"),
    }
}

#[test]
fn test_type_vector_3d() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Vector(vec![
            Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 3.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
        ]),
        Span::new(0, 0, 15, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Vector { dim: 3 });
            assert_eq!(kt.unit, Unit::dimensionless());
        }
        _ => panic!("Expected Kernel type"),
    }
}

#[test]
fn test_type_vector_4d() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Vector(vec![
            Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 3.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 4.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
        ]),
        Span::new(0, 0, 20, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Vector { dim: 4 });
            assert_eq!(kt.unit, Unit::dimensionless());
        }
        _ => panic!("Expected Kernel type"),
    }
}

#[test]
fn test_type_vector_empty_fails() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Vector(vec![]), Span::new(0, 0, 2, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("empty"));
}

#[test]
fn test_type_vector_too_large_fails() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Vector(vec![
            Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 3.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 4.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 5.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
        ]),
        Span::new(0, 0, 25, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("maximum is 4"));
}

#[test]
fn test_type_vector_mixed_units_fails() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Vector(vec![
            Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(UntypedKind::Dt, Span::new(0, 0, 2, 1)),
        ]),
        Span::new(0, 0, 10, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("unit"));
}

#[test]
fn test_type_vector_non_scalar_element_fails() {
    let ctx = make_context();
    // Try to create a vector containing a vector
    let expr = Expr::new(
        UntypedKind::Vector(vec![Expr::new(
            UntypedKind::Vector(vec![
                Expr::new(
                    UntypedKind::Literal {
                        value: 1.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                ),
                Expr::new(
                    UntypedKind::Literal {
                        value: 2.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                ),
            ]),
            Span::new(0, 0, 10, 1),
        )]),
        Span::new(0, 0, 12, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("scalar") || errors[0].message.contains("Scalar"));
}

#[test]
fn test_type_vector_bool_element_fails() {
    let ctx = make_context();
    // Try to create a vector containing a boolean
    let expr = Expr::new(
        UntypedKind::Vector(vec![Expr::new(
            UntypedKind::BoolLiteral(true),
            Span::new(0, 0, 4, 1),
        )]),
        Span::new(0, 0, 6, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("non-kernel") || errors[0].message.contains("Bool"));
}

#[test]
fn test_type_let_nested_bindings() {
    let ctx = make_context();
    // let x = 5.0; let y = x; y
    let expr = Expr::new(
        UntypedKind::Let {
            name: "x".to_string(),
            value: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 5.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
            body: Box::new(Expr::new(
                UntypedKind::Let {
                    name: "y".to_string(),
                    value: Box::new(Expr::new(
                        UntypedKind::Local("x".to_string()),
                        Span::new(0, 0, 1, 1),
                    )),
                    body: Box::new(Expr::new(
                        UntypedKind::Local("y".to_string()),
                        Span::new(0, 0, 1, 1),
                    )),
                },
                Span::new(0, 0, 10, 1),
            )),
        },
        Span::new(0, 0, 20, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Scalar);
            assert_eq!(kt.unit, Unit::dimensionless());
        }
        _ => panic!("Expected Kernel type"),
    }
}

#[test]
fn test_type_let_shadowing() {
    let ctx = make_context();
    // let x = 5.0; let x = dt; x (inner binding shadows outer)
    let expr = Expr::new(
        UntypedKind::Let {
            name: "x".to_string(),
            value: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 5.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
            body: Box::new(Expr::new(
                UntypedKind::Let {
                    name: "x".to_string(),
                    value: Box::new(Expr::new(UntypedKind::Dt, Span::new(0, 0, 2, 1))),
                    body: Box::new(Expr::new(
                        UntypedKind::Local("x".to_string()),
                        Span::new(0, 0, 1, 1),
                    )),
                },
                Span::new(0, 0, 10, 1),
            )),
        },
        Span::new(0, 0, 20, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    // Should resolve to inner x (dt = seconds), not outer x (dimensionless)
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Scalar);
            assert_eq!(kt.unit, Unit::seconds());
        }
        _ => panic!("Expected Kernel type"),
    }
}

#[test]
fn test_type_let_binding_available_in_body() {
    let ctx = make_context();
    // Verify that let binding is correctly available within the let body
    let expr = Expr::new(
        UntypedKind::Let {
            name: "temp".to_string(),
            value: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 10.0,
                    unit: None,
                },
                Span::new(0, 0, 4, 1),
            )),
            body: Box::new(Expr::new(
                UntypedKind::Local("temp".to_string()),
                Span::new(0, 0, 4, 1),
            )),
        },
        Span::new(0, 0, 15, 1),
    );

    // This should succeed - temp is in scope within body
    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Kernel(_)));
}

#[test]
fn test_type_let_value_error_propagates() {
    let ctx = make_context();
    // Let binding value expression fails - error should propagate
    let expr = Expr::new(
        UntypedKind::Let {
            name: "x".to_string(),
            value: Box::new(Expr::new(
                UntypedKind::Local("undefined_var".to_string()),
                Span::new(0, 0, 13, 1),
            )),
            body: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
        },
        Span::new(0, 0, 20, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("undefined_var"));
}

#[test]
fn test_type_let_body_error_propagates() {
    let ctx = make_context();
    // Let binding body expression fails - error should propagate
    let expr = Expr::new(
        UntypedKind::Let {
            name: "x".to_string(),
            value: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 42.0,
                    unit: None,
                },
                Span::new(0, 0, 4, 1),
            )),
            body: Box::new(Expr::new(
                UntypedKind::Local("undefined_var".to_string()),
                Span::new(0, 0, 13, 1),
            )),
        },
        Span::new(0, 0, 20, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("undefined_var"));
}

// ============================================================================
// Tests for Struct literal construction
// ============================================================================

#[test]
fn test_type_struct_valid_construction() {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        let type_id = TypeId::from("Position");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Position"),
            vec![
                (
                    "x".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::dimensionless(),
                        bounds: None,
                    }),
                ),
                (
                    "y".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::dimensionless(),
                        bounds: None,
                    }),
                ),
            ],
        );
        table.register(user_type);
        table
    }));

    let ctx = TypingContext::new(
        type_table,
        KernelRegistry::global(),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
    );

    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("Position"),
            fields: vec![
                (
                    "x".to_string(),
                    Expr::new(
                        UntypedKind::Literal {
                            value: 10.0,
                            unit: None,
                        },
                        Span::new(0, 0, 4, 1),
                    ),
                ),
                (
                    "y".to_string(),
                    Expr::new(
                        UntypedKind::Literal {
                            value: 20.0,
                            unit: None,
                        },
                        Span::new(0, 0, 4, 1),
                    ),
                ),
            ],
        },
        Span::new(0, 0, 30, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, Type::User(TypeId::from("Position")));
}

#[test]
fn test_type_struct_unknown_type() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("UnknownType"),
            fields: vec![],
        },
        Span::new(0, 0, 15, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("UnknownType"));
}

#[test]
fn test_type_struct_unknown_field() {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        let type_id = TypeId::from("Point");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Point"),
            vec![(
                "x".to_string(),
                Type::Kernel(KernelType {
                    shape: Shape::Scalar,
                    unit: Unit::dimensionless(),
                    bounds: None,
                }),
            )],
        );
        table.register(user_type);
        table
    }));

    let ctx = TypingContext::new(
        type_table,
        KernelRegistry::global(),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
    );

    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("Point"),
            fields: vec![(
                "unknown_field".to_string(),
                Expr::new(
                    UntypedKind::Literal {
                        value: 5.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                ),
            )],
        },
        Span::new(0, 0, 20, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("unknown_field"));
}

#[test]
fn test_type_struct_type_mismatch() {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        let type_id = TypeId::from("Data");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Data"),
            vec![(
                "value".to_string(),
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

    let ctx = TypingContext::new(
        type_table,
        KernelRegistry::global(),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
    );

    // Provide dt (seconds) instead of meters
    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("Data"),
            fields: vec![(
                "value".to_string(),
                Expr::new(UntypedKind::Dt, Span::new(0, 0, 2, 1)),
            )],
        },
        Span::new(0, 0, 20, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
}

#[test]
fn test_type_struct_missing_field() {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        let type_id = TypeId::from("Vec2");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Vec2"),
            vec![
                (
                    "x".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::dimensionless(),
                        bounds: None,
                    }),
                ),
                (
                    "y".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::dimensionless(),
                        bounds: None,
                    }),
                ),
            ],
        );
        table.register(user_type);
        table
    }));

    let ctx = TypingContext::new(
        type_table,
        KernelRegistry::global(),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
    );

    // Only provide x, missing y
    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("Vec2"),
            fields: vec![(
                "x".to_string(),
                Expr::new(
                    UntypedKind::Literal {
                        value: 10.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                ),
            )],
        },
        Span::new(0, 0, 20, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("missing"));
}

#[test]
fn test_type_struct_duplicate_field() {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        let type_id = TypeId::from("Point");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Point"),
            vec![(
                "x".to_string(),
                Type::Kernel(KernelType {
                    shape: Shape::Scalar,
                    unit: Unit::dimensionless(),
                    bounds: None,
                }),
            )],
        );
        table.register(user_type);
        table
    }));

    let ctx = TypingContext::new(
        type_table,
        KernelRegistry::global(),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
    );

    // Provide x twice
    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("Point"),
            fields: vec![
                (
                    "x".to_string(),
                    Expr::new(
                        UntypedKind::Literal {
                            value: 10.0,
                            unit: None,
                        },
                        Span::new(0, 0, 4, 1),
                    ),
                ),
                (
                    "x".to_string(),
                    Expr::new(
                        UntypedKind::Literal {
                            value: 20.0,
                            unit: None,
                        },
                        Span::new(0, 0, 4, 1),
                    ),
                ),
            ],
        },
        Span::new(0, 0, 30, 1),
    );

    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
    assert!(errors[0].message.contains("multiple"));
}

#[test]
fn test_type_self_found() {
    let mut ctx = make_context();
    let entity_type = Type::User(TypeId::from("Plate"));
    ctx.self_type = Some(entity_type.clone());

    let expr = Expr::new(UntypedKind::Self_, Span::new(0, 0, 4, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, entity_type);
}

#[test]
fn test_type_other_found() {
    let mut ctx = make_context();
    let entity_type = Type::User(TypeId::from("Plate"));
    ctx.other_type = Some(entity_type.clone());

    let expr = Expr::new(UntypedKind::Other, Span::new(0, 0, 5, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, entity_type);
}

#[test]
fn test_type_self_not_found() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Self_, Span::new(0, 0, 4, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("self"));
}

#[test]
fn test_type_aggregate_map() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Aggregate {
            op: crate::ast::AggregateOp::Map,
            entity: entity.clone(),
            binding: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Seq(inner) => {
            assert!(matches!(**inner, Type::Kernel(_)));
        }
        _ => panic!("Expected Seq type, got {:?}", typed.ty),
    }
}

#[test]
fn test_type_aggregate_sum() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Aggregate {
            op: crate::ast::AggregateOp::Sum,
            entity: entity.clone(),
            binding: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Kernel(_)));
}

#[test]
fn test_type_aggregate_count() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Aggregate {
            op: crate::ast::AggregateOp::Count,
            entity: entity.clone(),
            binding: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::BoolLiteral(true),
                Span::new(0, 0, 4, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Scalar);
            assert_eq!(kt.unit, Unit::DIMENSIONLESS);
        }
        _ => panic!("Expected dimensionless Scalar for Count"),
    }
}

#[test]
fn test_type_aggregate_any() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Aggregate {
            op: crate::ast::AggregateOp::Any,
            entity: entity.clone(),
            binding: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::BoolLiteral(true),
                Span::new(0, 0, 4, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Bool));
}

#[test]
fn test_type_fold_valid() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Fold {
            entity: entity.clone(),
            init: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
            acc: "acc".to_string(),
            elem: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Local("acc".to_string()),
                Span::new(0, 0, 3, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Kernel(_)));
}

#[test]
fn test_type_fold_mismatch_fails() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Fold {
            entity: entity.clone(),
            init: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
            acc: "acc".to_string(),
            elem: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::BoolLiteral(true),
                Span::new(0, 0, 4, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let result = type_expression(&expr, &ctx);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
}

#[test]
fn test_type_nested_aggregates() {
    let ctx = make_context();
    let entity1 = continuum_foundation::EntityId::new("Plate");
    let entity2 = continuum_foundation::EntityId::new("Point");

    // aggregate(Plate, p) { aggregate(Point, pt) { 1.0 } }
    let expr = Expr::new(
        UntypedKind::Aggregate {
            op: crate::ast::AggregateOp::Map,
            entity: entity1,
            binding: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Aggregate {
                    op: crate::ast::AggregateOp::Map,
                    entity: entity2,
                    binding: "pt".to_string(),
                    body: Box::new(Expr::new(
                        UntypedKind::Literal {
                            value: 1.0,
                            unit: None,
                        },
                        Span::new(0, 0, 3, 1),
                    )),
                },
                Span::new(0, 0, 10, 1),
            )),
        },
        Span::new(0, 0, 20, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    // Result should be Seq(Seq(Scalar))
    match &typed.ty {
        Type::Seq(inner1) => match &**inner1 {
            Type::Seq(inner2) => {
                assert!(matches!(**inner2, Type::Kernel(_)));
            }
            _ => panic!("Expected nested Seq type, got {:?}", inner1),
        },
        _ => panic!("Expected Seq type, got {:?}", typed.ty),
    }
}

#[test]
fn test_type_self_other_field_access() {
    let type_name = "Plate";
    let ctx = make_context_with_types(&[(
        type_name,
        &[(
            "mass",
            Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::kilograms(),
                bounds: None,
            }),
        )],
    )]);

    let mut ctx = ctx;
    let plate_ty = Type::User(TypeId::from(type_name));
    ctx.self_type = Some(plate_ty.clone());
    ctx.other_type = Some(plate_ty);

    // self.mass
    let expr_self = Expr::new(
        UntypedKind::FieldAccess {
            object: Box::new(Expr::new(UntypedKind::Self_, Span::new(0, 0, 4, 1))),
            field: "mass".to_string(),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed_self = type_expression(&expr_self, &ctx).unwrap();
    assert!(matches!(typed_self.ty, Type::Kernel(ref kt) if kt.unit == Unit::kilograms()));

    // other.mass
    let expr_other = Expr::new(
        UntypedKind::FieldAccess {
            object: Box::new(Expr::new(UntypedKind::Other, Span::new(0, 0, 5, 1))),
            field: "mass".to_string(),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed_other = type_expression(&expr_other, &ctx).unwrap();
    assert!(matches!(typed_other.ty, Type::Kernel(ref kt) if kt.unit == Unit::kilograms()));
}

#[test]
fn test_type_fold_with_nested_aggregate() {
    let ctx = make_context();
    let entity1 = continuum_foundation::EntityId::new("Plate");
    let entity2 = continuum_foundation::EntityId::new("Point");

    // fold(Plate, 0.0, |acc, p| acc + sum(Point, |pt| 1.0))
    let expr = Expr::new(
        UntypedKind::Fold {
            entity: entity1,
            init: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
            acc: "acc".to_string(),
            elem: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Aggregate {
                    op: crate::ast::AggregateOp::Sum,
                    entity: entity2,
                    binding: "pt".to_string(),
                    body: Box::new(Expr::new(
                        UntypedKind::Literal {
                            value: 1.0,
                            unit: None,
                        },
                        Span::new(0, 0, 3, 1),
                    )),
                },
                Span::new(0, 0, 10, 1),
            )),
        },
        Span::new(0, 0, 20, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Kernel(_)));
}
