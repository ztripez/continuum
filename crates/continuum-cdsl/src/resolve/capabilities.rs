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
//! - **Fields** - Read field values (available in Measure, Assert phases for Field role only)
//! - **Prev** - Previous tick value (available in Resolve, Assert)
//! - **Current** - Just-resolved value (available in Assert, Measure, Fracture)
//! - **Inputs** - Accumulated inputs (available in Resolve)
//! - **Dt** - Time step (available in all tick phases)
//! - **Payload** - Impulse payload (available only in impulse handlers)
//! - **Emit** - Emit to signal (available in Collect phase and impulse handlers)
//! - **Index** - Entity self-reference (available in per-entity contexts)
//!
//! # Phase→Capability Mapping
//!
//! The authoritative mapping of which capabilities are available in each phase
//! is defined in `crate::ast::RoleSpec::capabilities_for_phase()`. This ensures
//! a single source of truth - no ad-hoc context construction with potentially
//! invalid capability sets.
//!
//! # What This Pass Does
//!
//! 1. **Context detection** - Determines available capabilities from RoleSpec + Phase
//! 2. **Expression scanning** - Recursively finds all capability-requiring expressions
//! 3. **Access validation** - Checks that required capabilities are available in context
//!
//! # What This Pass Does NOT Do
//!
//! - **No type checking** - Types must already be validated before this pass
//! - **No effect validation** - Effect purity (emit/spawn/destroy) is validated by a separate pass
//! - **No structure validation** - Structural checks (cycles, collisions) are separate
//!
//! # Validation Pipeline
//!
//! This pass runs after type checking and effect purity validation.
//! It ensures expressions only access runtime context that will be available during execution.
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
    ///
    /// A capability context represents what runtime state is accessible during
    /// expression execution (e.g., previous tick values, time step, accumulated inputs).
    ///
    /// # Parameters
    ///
    /// - `capabilities`: The set of capabilities available in this execution context
    ///
    /// # Returns
    ///
    /// A new `CapabilityContext` with the specified capabilities
    pub fn new(capabilities: CapabilitySet) -> Self {
        Self { capabilities }
    }

    /// Check if a specific capability is available in this context.
    ///
    /// # Parameters
    ///
    /// - `capability`: The capability to check (e.g., `Capability::Prev`, `Capability::Dt`)
    ///
    /// # Returns
    ///
    /// `true` if the capability is available in this context, `false` otherwise
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
/// operations (Prev, Current, Inputs, Dt, Payload, Self_, Other, and emit calls) are permitted
/// in the context.
///
/// **Capabilities** represent what runtime context is accessible during execution:
/// - Previous tick values, current resolved values, accumulated inputs
/// - Time step, impulse payload data
/// - Entity self-reference and other-entity access
/// - Signal emission permission
///
/// # Parameters
///
/// - `expr`: The typed expression to validate
/// - `ctx`: The capability context (available capabilities for this execution phase)
///
/// # Returns
///
/// A vector of `CompileError` with `ErrorKind::MissingCapability` for each violation.
/// Returns empty vector if all capability access is valid.
///
/// # Capability Requirements
///
/// The following expression kinds require specific capabilities:
///
/// - **`ExprKind::Prev`** - Requires `Capability::Prev` (previous tick value)
/// - **`ExprKind::Current`** - Requires `Capability::Current` (just-resolved value)
/// - **`ExprKind::Inputs`** - Requires `Capability::Inputs` (accumulated signal inputs)
/// - **`ExprKind::Dt`** - Requires `Capability::Dt` (time step)
/// - **`ExprKind::Payload`** - Requires `Capability::Payload` (impulse trigger data)
/// - **`ExprKind::Self_`** - Requires `Capability::Index` (entity self-reference)
/// - **`ExprKind::Other`** - Requires `Capability::Index` (other entity access)
/// - **`ExprKind::Signal`** - Requires `Capability::Signals` (signal read access)
/// - **`ExprKind::Field`** - Requires `Capability::Fields` (field read access, observer-only)
/// - **`ExprKind::Call { kernel: emit, ... }`** - Requires `Capability::Emit` (signal emission)
///
/// Note: **emit** is identified by empty namespace and name "emit" (bare kernel call).
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
/// assert!(errors[0].message.contains("Capability::Prev"));
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

        ExprKind::Self_ => {
            if !ctx.has_capability(Capability::Index) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!("self cannot be accessed in this context (requires Capability::Index)"),
                ));
            }
        }

        ExprKind::Other => {
            if !ctx.has_capability(Capability::Index) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!(
                        "other cannot be accessed in this context (requires Capability::Index)"
                    ),
                ));
            }
        }

        ExprKind::Signal(_) => {
            if !ctx.has_capability(Capability::Signals) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!(
                        "signal access requires Capability::Signals (only available in signal-reading phases)"
                    ),
                ));
            }
        }

        ExprKind::Field(_) => {
            if !ctx.has_capability(Capability::Fields) {
                errors.push(CompileError::new(
                    ErrorKind::MissingCapability,
                    expr.span,
                    format!(
                        "field access requires Capability::Fields (only available in observer contexts)"
                    ),
                ));
            }
        }

        // === Non-capability-requiring expressions (leaf nodes or pure) ===
        ExprKind::Literal { .. }
        | ExprKind::Local(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_) => {
            // These don't require special capabilities
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::KernelId;
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

    #[test]
    fn test_self_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Index));

        let expr = TypedExpr::new(
            ExprKind::Self_,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_self_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty());

        let expr = TypedExpr::new(
            ExprKind::Self_,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("self"));
        assert!(errors[0].message.contains("Capability::Index"));
    }

    #[test]
    fn test_other_access_allowed() {
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Index));

        let expr = TypedExpr::new(
            ExprKind::Other,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_other_access_denied() {
        let ctx = CapabilityContext::new(CapabilitySet::empty());

        let expr = TypedExpr::new(
            ExprKind::Other,
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("other"));
        assert!(errors[0].message.contains("Capability::Index"));
    }

    #[test]
    fn test_signal_access_allowed() {
        use crate::foundation::Path;

        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Signals));

        let expr = TypedExpr::new(
            ExprKind::Signal(Path::from_path_str("body.velocity")),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_signal_access_denied() {
        use crate::foundation::Path;

        let ctx = CapabilityContext::new(CapabilitySet::empty());

        let expr = TypedExpr::new(
            ExprKind::Signal(Path::from_path_str("body.velocity")),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("signal"));
        assert!(errors[0].message.contains("Capability::Signals"));
    }

    #[test]
    fn test_field_access_allowed() {
        use crate::foundation::Path;

        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Fields));

        let expr = TypedExpr::new(
            ExprKind::Field(Path::from_path_str("temperature")),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_field_access_denied() {
        use crate::foundation::Path;

        let ctx = CapabilityContext::new(CapabilitySet::empty());

        let expr = TypedExpr::new(
            ExprKind::Field(Path::from_path_str("temperature")),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("field"));
        assert!(errors[0].message.contains("Capability::Fields"));
    }

    #[test]
    fn test_signal_in_causal_phase_denied() {
        use crate::foundation::Path;

        // Configure phase has no Signals capability
        let ctx = CapabilityContext::new(CapabilitySet::empty().with(Capability::Scoping));

        let expr = TypedExpr::new(
            ExprKind::Signal(Path::from_path_str("body.mass")),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
    }

    #[test]
    fn test_field_in_causal_phase_denied() {
        use crate::foundation::Path;

        // Resolve phase has no Fields capability (fields are observer-only)
        let ctx = CapabilityContext::new(
            CapabilitySet::empty()
                .with(Capability::Signals)
                .with(Capability::Prev)
                .with(Capability::Dt),
        );

        let expr = TypedExpr::new(
            ExprKind::Field(Path::from_path_str("elevation")),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_capability_access(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingCapability);
        assert!(errors[0].message.contains("observer"));
    }
}
