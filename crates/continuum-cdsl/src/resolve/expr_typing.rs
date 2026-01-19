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
//!   (Note: Architecturally, this should be a separate desugar pass, but is currently done inline)
//! - **Type compatibility validation** - Some type checks happen during typing where needed for inference
//!   (Full semantic validation happens in separate validation pass)
//! - **Phase boundary enforcement** - Field expressions are validated to only appear in Measure phase
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
use continuum_foundation::Phase;
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

    /// Current execution phase (for phase boundary enforcement)
    ///
    /// When typing execution blocks, this is set to the phase in which the block
    /// executes. Used to enforce that fields can only be read in Measure phase.
    /// None indicates context-free typing (e.g., tests without phase constraints).
    pub phase: Option<Phase>,
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
            phase: None,
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
            phase: self.phase,
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
            phase: self.phase,
        }
    }

    /// Set phase context for boundary enforcement
    ///
    /// Creates a new context with the specified execution phase.
    /// Used to enforce phase boundaries (e.g., fields only in Measure).
    ///
    /// # Parameters
    ///
    /// - `phase`: The execution phase
    ///
    /// # Returns
    ///
    /// New context with phase set
    pub fn with_phase(&self, phase: Phase) -> Self {
        Self {
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
            phase: Some(phase),
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

    // Check value type to distinguish between boolean and numeric returns
    // Boolean kernels return Type::Bool directly without shape/unit derivation
    use continuum_kernel_types::ValueType;
    if sig.returns.value_type == ValueType::Bool {
        return Ok(Type::Bool);
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
        UnitDerivation::Sqrt(idx) | UnitDerivation::Inverse(idx) => {
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
            // Phase boundary enforcement: Fields can only be read in Measure phase
            if let Some(phase) = ctx.phase {
                if phase != Phase::Measure {
                    return Err(vec![CompileError::new(
                        ErrorKind::PhaseBoundaryViolation,
                        span,
                        format!(
                            "field '{}' cannot be read in {:?} phase (fields are only accessible in Measure phase)",
                            path, phase
                        ),
                    )]);
                }
            }

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
        //
        // Resolves a reference to a world-level configuration value.
        // Config values are read-only constants defined in world.yaml
        // and accessible from any execution block.
        //
        // Type derivation: Looks up path in ctx.config_types
        // Error: UndefinedName if path not found in config registry
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
        //
        // Resolves a reference to a CDSL-defined constant.
        // Constants are compile-time values declared in the DSL.
        //
        // Type derivation: Looks up path in ctx.const_types
        // Error: UndefinedName if path not found in const registry
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
        //
        // References the value from the previous execution tick.
        // Only valid in resolve phase operators with node_output context.
        //
        // Type derivation: Returns ctx.node_output type
        // Error: Internal if node_output not set in context
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
        //
        // References the just-computed value in the current tick.
        // Only valid in resolve phase operators with node_output context.
        //
        // Type derivation: Returns ctx.node_output type
        // Error: Internal if node_output not set in context
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
        //
        // References accumulated signal inputs collected during this tick.
        // Only valid in resolve phase operators with inputs_type context.
        //
        // Type derivation: Returns ctx.inputs_type
        // Error: Internal if inputs_type not set in context
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
        //
        // References the payload data carried by an impulse.
        // Only valid in impulse handler operators with payload_type context.
        //
        // Type derivation: Returns ctx.payload_type
        // Error: Internal if payload_type not set in context
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
        //
        // Constructs a vector from 2-4 scalar elements.
        // All elements must have the same unit.
        //
        // Type derivation: Vector<dim> with unit from first element
        // Errors:
        // - TypeMismatch if empty or >4 elements
        // - TypeMismatch if elements not all scalar
        // - TypeMismatch if elements have different units
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
            let unit = unit.expect("invariant: non-empty vector must have unit set");

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
        //
        // Creates a local binding for use in the body expression.
        // The binding is scoped to the body only.
        //
        // Type derivation: Body's type becomes result type
        // The value is typed first, then body is typed with binding in scope
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
        //
        // Constructs an instance of a user-defined type.
        // All fields must be provided exactly once with correct types.
        //
        // Type derivation: Returns Type::User(type_id)
        // Errors:
        // - UndefinedName if type not found
        // - UndefinedName if field not in type definition
        // - TypeMismatch if field type doesn't match definition
        // - TypeMismatch if missing fields
        // - TypeMismatch if duplicate fields
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
            let mut seen_fields = std::collections::HashSet::new();

            for (field_name, field_expr) in fields {
                // Check for duplicate fields
                if !seen_fields.insert(field_name.clone()) {
                    return Err(vec![CompileError::new(
                        ErrorKind::TypeMismatch,
                        field_expr.span,
                        format!(
                            "field '{}' specified multiple times in struct literal",
                            field_name
                        ),
                    )]);
                }

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
        //
        // Desugars binary operators (+, -, *, /, <, >, &&, ||, etc.)
        // to their corresponding kernel calls (maths.*, compare.*, logic.*).
        //
        // Type derivation: Determined by kernel signature
        // Error: UndefinedName if kernel not found for operator
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
                        "kernel '{:?}' not found for binary operator {:?}",
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
        //
        // Desugars unary operators (-, !) to their corresponding
        // kernel calls (maths.neg, logic.not).
        //
        // Type derivation: Determined by kernel signature
        // Error: UndefinedName if kernel not found for operator
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
        //
        // Desugars if-then-else to logic.select(condition, then, else) kernel call.
        // Condition must be Bool, branches must have compatible types.
        //
        // Type derivation: Determined by kernel signature (typically matches branch types)
        // Error: Internal if logic.select kernel not found
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
    use crate::ast::{BinaryOp, UnaryOp};
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

    /// Create a typing context with pre-registered user types
    ///
    /// Reduces test boilerplate by encapsulating TypeTable construction and leaking.
    ///
    /// # Parameters
    ///
    /// - `types`: Array of (type_name, fields) tuples where fields is array of (field_name, field_type) tuples
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ctx = make_context_with_types(&[(
    ///     "Position",
    ///     &[
    ///         ("x", Type::Kernel(KernelType { shape: Shape::Scalar, unit: Unit::meters(), bounds: None })),
    ///         ("y", Type::Kernel(KernelType { shape: Shape::Scalar, unit: Unit::meters(), bounds: None })),
    ///     ],
    /// )]);
    /// ```
    fn make_context_with_types<'a>(types: &[(&str, &[(&str, Type)])]) -> TypingContext<'a> {
        let type_table = Box::leak(Box::new({
            let mut table = TypeTable::new();
            for (type_name, fields) in types {
                let type_id = TypeId::from(*type_name);
                let user_type = UserType::new(
                    type_id.clone(),
                    Path::from(*type_name),
                    fields
                        .iter()
                        .map(|(name, ty)| (name.to_string(), ty.clone()))
                        .collect(),
                );
                table.register(user_type);
            }
            table
        }));

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

    /// Create a dimensionless scalar literal expression for testing
    ///
    /// Reduces boilerplate in vector literal tests where multiple literals are constructed.
    ///
    /// # Parameters
    ///
    /// - `value`: Numeric value for the literal
    ///
    /// # Returns
    ///
    /// Untyped expression with dummy span
    fn test_literal(value: f64) -> Expr {
        Expr::new(
            UntypedKind::Literal { value, unit: None },
            Span::new(0, 0, 3, 1),
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
                value_type: continuum_kernel_types::ValueType::F64,
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
                value_type: continuum_kernel_types::ValueType::F64,
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

    // ============================================================================
    // Tests for execution context expressions (Prev/Current/Inputs/Payload)
    // ============================================================================

    #[test]
    fn test_type_prev_with_node_output() {
        let ctx = make_context();
        let output_type = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::meters(),
            bounds: None,
        });
        let ctx = ctx.with_execution_context(Some(output_type.clone()), None, None);

        let expr = Expr::new(UntypedKind::Prev, Span::new(0, 0, 4, 1));
        let typed = type_expression(&expr, &ctx).unwrap();
        assert_eq!(typed.ty, output_type);
    }

    #[test]
    fn test_type_prev_without_node_output() {
        let ctx = make_context();
        let expr = Expr::new(UntypedKind::Prev, Span::new(0, 0, 4, 1));
        let errors = type_expression(&expr, &ctx).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::Internal));
        assert!(errors[0].message.contains("prev"));
    }

    #[test]
    fn test_type_current_with_node_output() {
        let ctx = make_context();
        let output_type = Type::Kernel(KernelType {
            shape: Shape::Vector { dim: 3 },
            unit: Unit::seconds(),
            bounds: None,
        });
        let ctx = ctx.with_execution_context(Some(output_type.clone()), None, None);

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
        let ctx = ctx.with_execution_context(None, Some(inputs_type.clone()), None);

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
        let ctx = ctx.with_execution_context(None, None, Some(payload_type.clone()));

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

    // ============================================================================
    // Tests for Let bindings
    // ============================================================================

    #[test]
    fn test_type_let_simple_binding() {
        let ctx = make_context();
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
                    UntypedKind::Local("x".to_string()),
                    Span::new(0, 0, 1, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        // Body returns x, which is dimensionless scalar
        match &typed.ty {
            Type::Kernel(kt) => {
                assert_eq!(kt.shape, Shape::Scalar);
                assert_eq!(kt.unit, Unit::dimensionless());
            }
            _ => panic!("Expected Kernel type"),
        }
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

    // ============================================================================
    // Tests for operator desugaring (Binary/Unary/If)
    // ============================================================================

    #[test]
    fn test_type_binary_add_desugars_to_kernel() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 5.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 10.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        // Should desugar to maths.add kernel call
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "maths");
                assert_eq!(kernel.name, "add");
            }
            _ => panic!("Expected Call, got {:?}", typed.expr),
        }
    }

    #[test]
    fn test_type_binary_subtract_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 20.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 5.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "maths");
                assert_eq!(kernel.name, "sub");
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_type_binary_multiply_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 3.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 7.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "maths");
                assert_eq!(kernel.name, "mul");
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_type_binary_divide_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::Div,
                left: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 20.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 4.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "maths");
                assert_eq!(kernel.name, "div");
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_type_binary_comparison_less_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::Lt,
                left: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 5.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 10.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "compare");
                assert_eq!(kernel.name, "lt");
            }
            _ => panic!("Expected Call"),
        }
        // Comparison returns Bool
        assert_eq!(typed.ty, Type::Bool);
    }

    #[test]
    fn test_type_binary_comparison_equal_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::Eq,
                left: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 5.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 5.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "compare");
                assert_eq!(kernel.name, "eq");
            }
            _ => panic!("Expected Call"),
        }
        assert_eq!(typed.ty, Type::Bool);
    }

    #[test]
    fn test_type_binary_logical_and_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Binary {
                op: BinaryOp::And,
                left: Box::new(Expr::new(
                    UntypedKind::BoolLiteral(true),
                    Span::new(0, 0, 4, 1),
                )),
                right: Box::new(Expr::new(
                    UntypedKind::BoolLiteral(false),
                    Span::new(0, 0, 5, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "logic");
                assert_eq!(kernel.name, "and");
            }
            _ => panic!("Expected Call"),
        }
        assert_eq!(typed.ty, Type::Bool);
    }

    #[test]
    fn test_type_unary_negate_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 42.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
            },
            Span::new(0, 0, 6, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "maths");
                assert_eq!(kernel.name, "neg");
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_type_unary_not_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Expr::new(
                    UntypedKind::BoolLiteral(true),
                    Span::new(0, 0, 4, 1),
                )),
            },
            Span::new(0, 0, 6, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "logic");
                assert_eq!(kernel.name, "not");
            }
            _ => panic!("Expected Call"),
        }
        assert_eq!(typed.ty, Type::Bool);
    }

    #[test]
    fn test_type_if_then_else_desugars() {
        let ctx = make_context();
        let expr = Expr::new(
            UntypedKind::If {
                condition: Box::new(Expr::new(
                    UntypedKind::BoolLiteral(true),
                    Span::new(0, 0, 4, 1),
                )),
                then_branch: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 10.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
                else_branch: Box::new(Expr::new(
                    UntypedKind::Literal {
                        value: 20.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
            },
            Span::new(0, 0, 20, 1),
        );

        let typed = type_expression(&expr, &ctx).unwrap();
        // Should desugar to logic.select(condition, then, else)
        match &typed.expr {
            ExprKind::Call { kernel, .. } => {
                assert_eq!(kernel.namespace, "logic");
                assert_eq!(kernel.name, "select");
            }
            _ => panic!("Expected Call"),
        }
        // Result type should be dimensionless scalar (from branches)
        assert!(matches!(typed.ty, Type::Kernel(_)));
    }
}
