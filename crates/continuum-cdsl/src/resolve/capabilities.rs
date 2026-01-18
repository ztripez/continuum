//! Capability validation for CDSL expressions.
//!
//! This module validates that expressions only access capabilities that are available
//! in their execution context. Capabilities represent what runtime context is accessible
//! during execution (e.g., previous tick values, accumulated inputs, time step).
//!
//! # Capabilities
//!
//! - **Scoping** - Access to config/const values (available in all phases)
//! - **Signals** - Read signal values (available in most phases)
//! - **Prev** - Previous tick value (available in Resolve, Assert)
//! - **Current** - Just-resolved value (available in Assert, Measure, Fracture)
//! - **Inputs** - Accumulated inputs (available in Resolve)
//! - **Dt** - Time step (available in all tick phases)
//! - **Payload** - Impulse payload (available only in impulse handlers)
//! - **Emit** - Emit to signal (available in Collect phase and impulse handlers)
//! - **Index** - Entity self-reference (available in per-entity contexts)
//!
//! # What This Pass Does
//!
//! 1. **Context detection** - Determines available capabilities from operator metadata
//! 2. **Expression scanning** - Recursively finds all capability-requiring expressions
//! 3. **Access validation** - Checks that required capabilities are available in context
//!
//! # What This Pass Does NOT Do
//!
//! - **No type checking** - Types must already be validated
//! - **No effect validation** - Effects are validated separately
//! - **No structure validation** - This is a separate validation pass
//!
//! # Pipeline Position
//!
//! This pass runs after type validation and effect validation, during Phase 12.
//! It validates that capability access is legal before proceeding to structure validation.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_foundation::{Phase, CapabilitySet, Capability};
//!
//! // Configure phase has no Prev capability
//! let ctx = CapabilityContext::new(
//!     CapabilitySet::empty()
//!         .with(Capability::Scoping)
//!         .with(Capability::Dt)
//! );
//!
//! // Expression using Prev will fail validation
//! let errors = validate_capability_access(&typed_expr, &ctx);
//! assert!(errors.iter().any(|e| matches!(e.kind, ErrorKind::MissingCapability)));
//! ```

use crate::ast::{ExprKind, TypedExpr};
use crate::error::{CompileError, ErrorKind};
use continuum_foundation::{Capability, CapabilitySet};
use continuum_kernel_types::KernelId;

/// Execution context for capability validation.
///
/// Tracks which capabilities are available in the current execution context,
/// determined by the operator's role and execution phase.
///
/// # Capabilities
///
/// - **Scoping** - Access to config/const values
/// - **Signals** - Read signal values
/// - **Prev** - Previous tick value (only in Resolve, Assert phases)
/// - **Current** - Just-resolved value (only in Fracture, Measure, Assert phases)
/// - **Inputs** - Accumulated inputs (only in Resolve phase)
/// - **Dt** - Time step (available in all tick phases)
/// - **Payload** - Impulse payload (only in impulse handler `apply` block)
/// - **Emit** - Emit to signal (only in Collect phase and impulse handlers)
/// - **Index** - Entity self-reference (only in per-entity member contexts)
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_foundation::{Capability, CapabilitySet};
///
/// // Resolve phase for a signal operator
/// let ctx = CapabilityContext::new(
///     CapabilitySet::empty()
///         .with(Capability::Scoping)
///         .with(Capability::Signals)
///         .with(Capability::Prev)
///         .with(Capability::Inputs)
///         .with(Capability::Dt)
/// );
///
/// assert!(ctx.has_capability(Capability::Prev));
/// assert!(!ctx.has_capability(Capability::Current));
/// ```
#[derive(Debug, Clone)]
pub struct CapabilityContext {
    /// Set of capabilities available in this context
    pub capabilities: CapabilitySet,
}

impl CapabilityContext {
    /// Create a new capability context with the given available capabilities.
    pub fn new(capabilities: CapabilitySet) -> Self {
        Self { capabilities }
    }

    /// Check if a specific capability is available in this context.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let ctx = CapabilityContext::new(
    ///     CapabilitySet::empty().with(Capability::Prev)
    /// );
    ///
    /// assert!(ctx.has_capability(Capability::Prev));
    /// assert!(!ctx.has_capability(Capability::Current));
    /// ```
    pub fn has_capability(&self, capability: Capability) -> bool {
        self.capabilities.contains(capability)
    }
}

/// Validates that an expression only accesses capabilities available in the given context.
///
/// This function recursively scans the expression tree and checks that all capability-requiring
/// operations (Prev, Current, Inputs, Dt, Payload, emit calls) are permitted in the context.
///
/// # Capability Requirements
///
/// The following expression kinds require specific capabilities:
///
/// - **`ExprKind::Prev`** - Requires `Capability::Prev`
/// - **`ExprKind::Current`** - Requires `Capability::Current`
/// - **`ExprKind::Inputs`** - Requires `Capability::Inputs`
/// - **`ExprKind::Dt`** - Requires `Capability::Dt`
/// - **`ExprKind::Payload`** - Requires `Capability::Payload`
/// - **`ExprKind::Call { kernel: emit, ... }`** - Requires `Capability::Emit`
///
/// # Error Kinds
///
/// Returns `Vec<CompileError>` with errors of kind:
///
/// - **`ErrorKind::MissingCapability`** - Required capability not available in context
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_foundation::{Capability, CapabilitySet};
///
/// // Context without Prev capability
/// let ctx = CapabilityContext::new(
///     CapabilitySet::empty().with(Capability::Dt)
/// );
///
/// // Expression using Prev
/// let expr = /* TypedExpr with ExprKind::Prev */;
///
/// let errors = validate_capability_access(&expr, &ctx);
/// assert_eq!(errors.len(), 1);
/// assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
/// ```
pub fn validate_capability_access(expr: &TypedExpr, ctx: &CapabilityContext) -> Vec<CompileError> {
    let mut errors = Vec::new();
    scan_for_capability_violations(expr, ctx, &mut errors);
    errors
}

/// Recursively scans an expression tree for capability access violations.
///
/// This internal function traverses all expression nodes and validates that
/// capability-requiring operations are permitted in the current context.
///
/// # Parameters
///
/// - `expr`: The expression to scan
/// - `ctx`: The capability context (available capabilities)
/// - `errors`: Accumulator for validation errors
fn scan_for_capability_violations(
    expr: &TypedExpr,
    ctx: &CapabilityContext,
    errors: &mut Vec<CompileError>,
) {
    match &expr.expr {
        // === Capability-requiring expressions ===
        ExprKind::Prev => {
            if !ctx.has_capability(Capability::Prev) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!("prev cannot be accessed in this context (requires Capability::Prev)"),
                ));
            }
        }

        ExprKind::Current => {
            if !ctx.has_capability(Capability::Current) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!(
                        "current cannot be accessed in this context (requires Capability::Current)"
                    ),
                ));
            }
        }

        ExprKind::Inputs => {
            if !ctx.has_capability(Capability::Inputs) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!(
                        "inputs cannot be accessed in this context (requires Capability::Inputs)"
                    ),
                ));
            }
        }

        ExprKind::Dt => {
            if !ctx.has_capability(Capability::Dt) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!("dt cannot be accessed in this context (requires Capability::Dt)"),
                ));
            }
        }

        ExprKind::Payload => {
            if !ctx.has_capability(Capability::Payload) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!(
                        "payload cannot be accessed in this context (requires Capability::Payload)"
                    ),
                ));
            }
        }

        ExprKind::Call { kernel, args } => {
            // Check if this is an emit call (requires Emit capability)
            if kernel.namespace.is_empty() && kernel.name == "emit" {
                if !ctx.has_capability(Capability::Emit) {
                    errors.push(CompileError::new(
                        ErrorKind::MissingCapability,
                        expr.span,
                        format!(
                            "emit() cannot be called in this context (requires Capability::Emit)"
                        ),
                    ));
                }
            }

            // Recursively validate arguments
            for arg in args {
                scan_for_capability_violations(arg, ctx, errors);
            }
        }

        // === Recursive traversal for other expression kinds ===
        ExprKind::Vector(elements) => {
            for elem in elements {
                scan_for_capability_violations(elem, ctx, errors);
            }
        }

        ExprKind::Let { value, body, .. } => {
            scan_for_capability_violations(value, ctx, errors);
            scan_for_capability_violations(body, ctx, errors);
        }

        ExprKind::Struct { fields, .. } => {
            for (_, field_expr) in fields {
                scan_for_capability_violations(field_expr, ctx, errors);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            scan_for_capability_violations(object, ctx, errors);
        }

        ExprKind::Aggregate { body, .. } => {
            scan_for_capability_violations(body, ctx, errors);
        }

        ExprKind::Fold { init, body, .. } => {
            scan_for_capability_violations(init, ctx, errors);
            scan_for_capability_violations(body, ctx, errors);
        }

        // === Non-capability-requiring expressions (leaf nodes or pure) ===
        ExprKind::Literal { .. }
        | ExprKind::Local(_)
        | ExprKind::Signal(_)
        | ExprKind::Field(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Self_
        | ExprKind::Other => {
            // These don't require special capabilities
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{Shape, Span, Type, Unit};

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    #[test]
    fn test_capability_context_has_capability() {
        let ctx = CapabilityContext::new(
            CapabilitySet::empty()
                .with(Capability::Prev)
                .with(Capability::Dt),
        );

        assert!(ctx.has_capability(Capability::Prev));
        assert!(ctx.has_capability(Capability::Dt));
        assert!(!ctx.has_capability(Capability::Current));
        assert!(!ctx.has_capability(Capability::Inputs));
    }

    #[test]
    fn test_prev_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Prev));

        let expr = TypedExpr::new(
            ExprKind::Prev,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_prev_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Dt));

        let expr = TypedExpr::new(
            ExprKind::Prev,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("prev"));
        assert!(errors[0].message.contains("Capability::Prev"));
    }

    #[test]
    fn test_current_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Current));

        let expr = TypedExpr::new(
            ExprKind::Current,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_current_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Prev));

        let expr = TypedExpr::new(
            ExprKind::Current,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("current"));
        assert!(errors[0].message.contains("Capability::Current"));
    }

    #[test]
    fn test_inputs_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Inputs));

        let expr = TypedExpr::new(
            ExprKind::Inputs,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_inputs_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty());

        let expr = TypedExpr::new(
            ExprKind::Inputs,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("inputs"));
        assert!(errors[0].message.contains("Capability::Inputs"));
    }

    #[test]
    fn test_dt_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Dt));

        let expr = TypedExpr::new(
            ExprKind::Dt,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_dt_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Prev));

        let expr = TypedExpr::new(
            ExprKind::Dt,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("dt"));
        assert!(errors[0].message.contains("Capability::Dt"));
    }

    #[test]
    fn test_payload_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Payload));

        let expr = TypedExpr::new(
            ExprKind::Payload,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_payload_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty());

        let expr = TypedExpr::new(
            ExprKind::Payload,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("payload"));
        assert!(errors[0].message.contains("Capability::Payload"));
    }

    #[test]
    fn test_emit_call_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Emit));

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Local("signal_id".to_string()),
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Literal {
                            value: 1.0,
                            unit: None,
                        },
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&call, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_emit_call_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Dt));

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Local("signal_id".to_string()),
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Literal {
                            value: 1.0,
                            unit: None,
                        },
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&call, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("emit()"));
        assert!(errors[0].message.contains("Capability::Emit"));
    }

    #[test]
    fn test_nested_capability_violation() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Dt));

        // Kernel call: maths.add(dt, prev) where prev is not allowed
        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Dt,
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Prev,
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("prev"));
    }

    #[test]
    fn test_multiple_violations() {
        let ctx = CapabilityContext::new(CapabilitySet::empty());

        // Kernel call: maths.add(prev, current) - neither capability available
        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Prev,
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Current,
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().any(|e| e.message.contains("prev")));
        assert!(errors.iter().any(|e| e.message.contains("current")));
    }

    #[test]
    fn test_non_emit_kernel_call_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty());

        // Regular kernel call (not emit) should be allowed regardless of Emit capability
        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("math", "sqrt"),
                args: vec![TypedExpr::new(
                    ExprKind::Literal {
                        value: 4.0,
                        unit: None,
                    },
                    Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                    test_span(),
                )],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&call, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_let_binding_with_capability_violation() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Dt));

        // Let binding where value uses prev (not allowed) but dt is allowed
        let expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(TypedExpr::new(
                    ExprKind::Prev,
                    Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                    test_span(),
                )),
                body: Box::new(TypedExpr::new(
                    ExprKind::Dt,
                    Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                    test_span(),
                )),
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("prev"));
    }

    #[test]
    fn test_field_access_on_capability_expr() {
        use crate::foundation::UserTypeId;

        let ctx = CapabilityContext::new(CapabilitySet::empty());

        // FieldAccess on payload (payload not available)
        let expr = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(TypedExpr::new(
                    ExprKind::Payload,
                    Type::User(UserTypeId::new("CollisionData")),
                    test_span(),
                )),
                field: "impulse".to_string(),
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("payload"));
    }

    #[test]
    fn test_struct_field_with_capability_violation() {
        use crate::foundation::UserTypeId;

        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Dt));

        // Struct construction with field value using inputs (not allowed)
        let expr = TypedExpr::new(
            ExprKind::Struct {
                ty: UserTypeId::new("State"),
                fields: vec![
                    (
                        "time".to_string(),
                        TypedExpr::new(
                            ExprKind::Dt,
                            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                            test_span(),
                        ),
                    ),
                    (
                        "accumulated".to_string(),
                        TypedExpr::new(
                            ExprKind::Inputs,
                            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                            test_span(),
                        ),
                    ),
                ],
            },
            Type::User(UserTypeId::new("State")),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("inputs"));
    }

    #[test]
    fn test_all_capabilities_available() {
        let ctx = CapabilityContext::new(
            CapabilitySet::empty()
                .with(Capability::Prev)
                .with(Capability::Current)
                .with(Capability::Inputs)
                .with(Capability::Dt)
                .with(Capability::Payload)
                .with(Capability::Emit),
        );

        // All capability-requiring expressions should pass
        let test_cases = vec![
            ExprKind::Prev,
            ExprKind::Current,
            ExprKind::Inputs,
            ExprKind::Dt,
            ExprKind::Payload,
        ];

        for kind in test_cases {
            let expr = TypedExpr::new(
                kind,
                Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                test_span(),
            );
            let errors = validate_capability_access(&expr, &ctx);
            assert_eq!(errors.len(), 0, "Expected no errors for {:#?}", expr.expr);
        }

        // Test emit call
        let emit = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Local("target".to_string()),
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Literal {
                            value: 1.0,
                            unit: None,
                        },
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&emit, &ctx);
        assert_eq!(errors.len(), 0);
    }
}
