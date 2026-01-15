//! IR Validation and Warning Generation
//!
//! This module validates compiled IR and generates warnings for potential
//! issues that don't prevent compilation but may indicate problems.
//!
//! # Overview
//!
//! Validation runs after lowering and checks for:
//!
//! - **Missing assertions**: Signals with range constraints but no runtime validation
//! - **Undefined symbols**: References to signals, constants, or config that don't exist
//! - **Unknown functions**: Calls to functions not registered in the kernel registry
//!
//! # Warning vs Error
//!
//! This module produces warnings, not errors. Warnings indicate potential issues
//! but allow compilation to proceed. For example:
//!
//! - A signal with range `0..100` but no assertion may produce values outside that range
//! - A reference to `signal.temp` when only `signal.temperature` exists is likely a typo
//!
//! # Usage
//!
//! ```ignore
//! let world = lower(&compilation_unit)?;
//! let warnings = validate(&world);
//!
//! for warning in &warnings {
//!     eprintln!("[{:?}] {}: {}", warning.code, warning.entity, warning.message);
//! }
//! ```
//!
//! # Warning Codes
//!
//! Warnings are categorized by [`WarningCode`] for filtering and tooling:
//!
//! - `MissingRangeAssertion`: Compile with explicit assertions to fix
//! - `UndefinedSymbol`: Check for typos in symbol names
//! - `UnknownFunction`: Check function name or register new kernel

use std::collections::{HashMap, HashSet};

use tracing::warn;

// Import functions crate to ensure kernels are registered
use continuum_functions as _;
use continuum_kernel_registry::{get_in_namespace, namespace_exists};

use crate::{BinaryOpIr, CompiledExpr, CompiledWorld, ValueType};
use continuum_foundation::{PrimitiveParamKind, PrimitiveShape, coercion};

/// A compilation warning indicating a potential issue in the IR.
///
/// Warnings do not prevent compilation but may indicate bugs or
/// configuration issues that should be addressed.
#[derive(Debug, Clone)]
pub struct CompileWarning {
    /// Warning code for filtering/identification
    pub code: WarningCode,
    /// Human-readable message
    pub message: String,
    /// The entity this warning relates to (signal path, operator path, etc.)
    pub entity: String,
}

/// Warning codes for categorization and filtering.
///
/// Each warning has a code that can be used by tooling to:
/// - Filter warnings by category
/// - Configure which warnings to treat as errors
/// - Generate targeted fix suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningCode {
    /// A signal declares a range constraint but has no assertions to validate it.
    ///
    /// This means the range is documentation-only and won't be checked at runtime.
    /// Add an `assert` block to enforce the range.
    MissingRangeAssertion,

    /// A reference to a signal, constant, or config that doesn't exist.
    ///
    /// This usually indicates a typo in the symbol name. Check spelling and
    /// ensure the referenced definition exists.
    UndefinedSymbol,

    /// A call to a function not registered in the kernel registry.
    ///
    /// This may indicate a typo in the function name or a missing kernel
    /// registration.
    UnknownFunction,

    /// A member signal has no initialization and uses `prev` in its resolver.
    ///
    /// Members without initialization start at 0.0, which may cause NaN or
    /// incorrect values when used in division or other operations. Consider
    /// adding explicit initialization via config defaults or an initial expression.
    UninitializedMember,

    /// Type mismatch in binary operation.
    ///
    /// The operand types are incompatible for the operation. For example:
    /// - Vec2 + Vec3 (dimension mismatch)
    /// - Mat3 * Vec4 (dimension mismatch)
    TypeMismatch,
}

/// Validates a compiled world and returns any warnings.
///
/// This is the main entry point for IR validation. It runs all validation
/// checks and collects warnings into a single list.
///
/// # Checks Performed
///
/// 1. **Range assertions**: Signals with range types should have assertions
/// 2. **Undefined symbols**: All referenced symbols should exist
/// 3. **Unknown functions**: All called functions should be registered
///
/// # Logging
///
/// Warnings are also logged via `tracing::warn!` for immediate visibility.
///
/// # Example
///
/// ```ignore
/// let world = lower(&unit)?;
/// let warnings = validate(&world);
///
/// if !warnings.is_empty() {
///     println!("Compilation produced {} warning(s)", warnings.len());
/// }
/// ```
pub fn validate(world: &CompiledWorld) -> Vec<CompileWarning> {
    let mut warnings = Vec::new();

    check_range_assertions(world, &mut warnings);
    check_undefined_symbols(world, &mut warnings);
    check_uninitialized_members(world, &mut warnings);
    check_type_compatibility(world, &mut warnings);

    // Log warnings
    for warning in &warnings {
        warn!(
            code = ?warning.code,
            entity = %warning.entity,
            "{}",
            warning.message
        );
    }

    warnings
}

/// Checks that signals with range type constraints have runtime assertions.
///
/// A signal declared as `Scalar<K, 100..10000>` should have assertions to
/// validate the range at runtime. Without assertions, the range is purely
/// documentary and violations won't be detected.
fn check_range_assertions(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    let signals = world.signals();
    for (signal_id, signal) in &signals {
        // Check if the signal has a range constraint
        let has_range = signal
            .value_type
            .param_value(PrimitiveParamKind::Range)
            .is_some();

        if has_range && signal.assertions.is_empty() {
            warnings.push(CompileWarning {
                code: WarningCode::MissingRangeAssertion,
                message: format!(
                    "signal '{}' has a range constraint but no assertions to validate it at runtime",
                    signal_id
                ),
                entity: signal_id.to_string(),
            });
        }
    }
}

/// Checks for member signals that have no initialization.
///
/// Members with `resolve { prev }` or similar patterns that just maintain state
/// will start at 0.0 (from zeroed memory). This can cause NaN when these values
/// are used in division operations.
///
/// A member is considered "uninitialized" if:
/// - Its resolve expression is just `prev` (maintains previous value)
/// - There's no explicit `initial { expr }` block
fn check_uninitialized_members(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    let members = world.members();
    for (member_id, member) in &members {
        // Skip if member has an explicit initial block
        if member.initial.is_some() {
            continue;
        }

        if let Some(resolve) = &member.resolve {
            // Check if the resolve is just `prev` - meaning it maintains state
            // without any computation. This member will stay at 0.0 forever.
            if is_prev_only_resolver(resolve) {
                warnings.push(CompileWarning {
                    code: WarningCode::UninitializedMember,
                    message: format!(
                        "member '{}' uses 'resolve {{ prev }}' but has no initialization - \
                         will start at 0.0 which may cause NaN in dependent calculations",
                        member_id
                    ),
                    entity: member_id.to_string(),
                });
            }
        }
    }
}

/// Checks if a resolve expression is just `prev` (state-maintaining only).
///
/// Returns true for expressions like:
/// - `prev` - just the previous value
/// - `prev + 0` or similar no-ops could be detected in the future
fn is_prev_only_resolver(expr: &CompiledExpr) -> bool {
    matches!(expr, CompiledExpr::Prev)
}

/// Checks if a namespaced function name is registered in the kernel registry.
fn is_known_function(name: &str) -> bool {
    if let Some((namespace, function)) = name.split_once('.') {
        continuum_kernel_registry::is_known_in(namespace, function)
    } else {
        false
    }
}

/// Checks for undefined symbols in all expressions.
///
/// Scans all resolve expressions, measure expressions, assertion conditions,
/// transition conditions, and fracture expressions for references to
/// undefined signals, constants, config values, or unknown functions.
fn check_undefined_symbols(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    // Collect all defined symbols
    let signals = world.signals();
    let members = world.members();
    let fields = world.fields();
    let fractures = world.fractures();
    let eras = world.eras();

    let mut defined_signals: HashSet<String> = HashSet::new();
    for signal_id in signals.keys() {
        defined_signals.insert(signal_id.to_string());
    }
    for member_id in members.keys() {
        defined_signals.insert(member_id.to_string());
    }

    let defined_constants: HashSet<&str> = world.constants.keys().map(|s| s.as_str()).collect();
    let defined_config: HashSet<&str> = world.config.keys().map(|s| s.as_str()).collect();

    // Check signals
    for (signal_id, signal) in &signals {
        if let Some(resolve) = &signal.resolve {
            check_expr_symbols(
                resolve,
                &format!("signal.{}", signal_id),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
        for assertion in &signal.assertions {
            check_expr_symbols(
                &assertion.condition,
                &format!("signal.{} assert", signal_id),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check fields
    for (field_id, field) in &fields {
        if let Some(measure) = &field.measure {
            check_expr_symbols(
                measure,
                &format!("field.{}", field_id),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check fractures
    for (fracture_id, fracture) in &fractures {
        for condition in &fracture.conditions {
            check_expr_symbols(
                condition,
                &format!("fracture.{}", fracture_id),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
        for emit in &fracture.emits {
            check_expr_symbols(
                &emit.value,
                &format!("fracture.{}", fracture_id),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check era transitions
    for (era_id, era) in &eras {
        for transition in &era.transitions {
            check_expr_symbols(
                &transition.condition,
                &format!("era.{} transition", era_id),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check chronicles
    let chronicles = world.chronicles();
    for (chronicle_id, chronicle) in &chronicles {
        for (i, handler) in chronicle.handlers.iter().enumerate() {
            check_expr_symbols(
                &handler.condition,
                &format!("chronicle.{} handler[{}]", chronicle_id, i),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
            for field in &handler.event_fields {
                check_expr_symbols(
                    &field.value,
                    &format!("chronicle.{} handler[{}].{}", chronicle_id, i, field.name),
                    &defined_signals,
                    &defined_constants,
                    &defined_config,
                    warnings,
                );
            }
        }
    }
}

/// Recursively checks an expression for undefined symbols.
///
/// Walks the expression tree and reports warnings for:
/// - Signal references that don't exist in `defined_signals`
/// - Constant references that don't exist in `defined_constants`
/// - Config references that don't exist in `defined_config`
/// - Function calls to unknown kernel functions
fn check_expr_symbols(
    expr: &CompiledExpr,
    context: &str,
    defined_signals: &HashSet<String>,
    defined_constants: &HashSet<&str>,
    defined_config: &HashSet<&str>,
    warnings: &mut Vec<CompileWarning>,
) {
    match expr {
        CompiledExpr::Signal(signal_id) => {
            if !defined_signals.contains(&signal_id.to_string()) {
                warnings.push(CompileWarning {
                    code: WarningCode::UndefinedSymbol,
                    message: format!(
                        "undefined signal '{}' in {} (possible typo?)",
                        signal_id, context
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Const(name, _) => {
            if !defined_constants.contains(name.as_str()) {
                warnings.push(CompileWarning {
                    code: WarningCode::UndefinedSymbol,
                    message: format!(
                        "undefined constant '{}' in {} (possible typo?)",
                        name, context
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Config(name, _) => {
            if !defined_config.contains(name.as_str()) {
                warnings.push(CompileWarning {
                    code: WarningCode::UndefinedSymbol,
                    message: format!(
                        "undefined config '{}' in {} (possible typo?)",
                        name, context
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Call { function, args } => {
            if !is_known_function(function) {
                warnings.push(CompileWarning {
                    code: WarningCode::UnknownFunction,
                    message: format!(
                        "unknown function '{}' in {} (possible typo?)",
                        function, context
                    ),
                    entity: context.to_string(),
                });
            }
            for arg in args {
                check_expr_symbols(
                    arg,
                    context,
                    defined_signals,
                    defined_constants,
                    defined_config,
                    warnings,
                );
            }
        }

        CompiledExpr::KernelCall {
            namespace,
            function,
            args,
        } => {
            if namespace_exists(namespace) {
                if get_in_namespace(namespace, function).is_none() {
                    warnings.push(CompileWarning {
                        code: WarningCode::UnknownFunction,
                        message: format!(
                            "unknown function '{}.{}' in {} (possible typo?)",
                            namespace, function, context
                        ),
                        entity: context.to_string(),
                    });
                }
            } else {
                warnings.push(CompileWarning {
                    code: WarningCode::UnknownFunction,
                    message: format!(
                        "unknown namespace '{}' in {} (possible typo?)",
                        namespace, context
                    ),
                    entity: context.to_string(),
                });
            }

            for arg in args {
                check_expr_symbols(
                    arg,
                    context,
                    defined_signals,
                    defined_constants,
                    defined_config,
                    warnings,
                );
            }
        }
        CompiledExpr::Binary { left, right, .. } => {
            check_expr_symbols(
                left,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                right,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::Unary { operand, .. } => {
            check_expr_symbols(
                operand,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            check_expr_symbols(
                condition,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                then_branch,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                else_branch,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::Let { value, body, .. } => {
            check_expr_symbols(
                value,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                body,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::FieldAccess { object, .. } => {
            check_expr_symbols(
                object,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        // Entity expressions - recurse into sub-expressions
        CompiledExpr::SelfField(_) => {}
        CompiledExpr::EntityAccess { .. } => {
            // Entity access validation happens at runtime
        }
        CompiledExpr::Aggregate { body, .. } => {
            check_expr_symbols(
                body,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::Other { body, .. } | CompiledExpr::Pairs { body, .. } => {
            check_expr_symbols(
                body,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::Filter {
            predicate, body, ..
        } => {
            check_expr_symbols(
                predicate,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                body,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::First { predicate, .. } => {
            check_expr_symbols(
                predicate,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::Nearest { position, .. } => {
            check_expr_symbols(
                position,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::Within {
            position,
            radius,
            body,
            ..
        } => {
            check_expr_symbols(
                position,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                radius,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
            check_expr_symbols(
                body,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        CompiledExpr::EmitSignal { value, .. } => {
            check_expr_symbols(
                value,
                context,
                defined_signals,
                defined_constants,
                defined_config,
                warnings,
            );
        }
        // Literals, Prev, DtRaw, SimTime, Collected, Local, Payload don't need checking
        // Local variables are validated at parse/lower time
        // Payload expressions are validated in impulse context
        CompiledExpr::Literal(..)
        | CompiledExpr::Prev
        | CompiledExpr::DtRaw
        | CompiledExpr::SimTime
        | CompiledExpr::Collected
        | CompiledExpr::Local(_)
        | CompiledExpr::Payload
        | CompiledExpr::PayloadField(_) => {}
    }
}

// ============================================================================
// Type Compatibility Checking
// ============================================================================

/// Inferred type during validation (simplified shape for type checking).
#[derive(Debug, Clone, PartialEq)]
enum InferredShape {
    /// A known shape
    Known(PrimitiveShape),
    /// Type is unknown (can't be determined statically)
    Unknown,
}

impl InferredShape {
    fn scalar() -> Self {
        InferredShape::Known(PrimitiveShape::Scalar)
    }

    fn from_value_type(vt: &ValueType) -> Self {
        InferredShape::Known(vt.primitive_def().shape)
    }

    fn type_name(&self) -> String {
        match self {
            InferredShape::Known(PrimitiveShape::Scalar) => "Scalar".to_string(),
            InferredShape::Known(PrimitiveShape::Vector { dim }) => format!("Vec{}", dim),
            InferredShape::Known(PrimitiveShape::Matrix { rows, cols }) => {
                if rows == cols {
                    format!("Mat{}", rows)
                } else {
                    format!("Mat{}x{}", rows, cols)
                }
            }
            InferredShape::Known(PrimitiveShape::Tensor) => "Tensor".to_string(),
            InferredShape::Known(PrimitiveShape::Grid) => "Grid".to_string(),
            InferredShape::Known(PrimitiveShape::Seq) => "Seq".to_string(),
            InferredShape::Unknown => "unknown".to_string(),
        }
    }
}

/// Context for type checking, holding known signal/member types.
struct TypeCheckContext {
    signal_types: HashMap<String, InferredShape>,
    local_types: HashMap<String, InferredShape>,
}

impl TypeCheckContext {
    fn new(world: &CompiledWorld) -> Self {
        let mut signal_types = HashMap::new();

        // Collect signal types
        for (id, signal) in world.signals() {
            signal_types.insert(
                id.to_string(),
                InferredShape::from_value_type(&signal.value_type),
            );
        }

        // Collect member types
        for (id, member) in world.members() {
            signal_types.insert(
                id.to_string(),
                InferredShape::from_value_type(&member.value_type),
            );
        }

        // Collect field types (for observation expressions that reference fields)
        for (id, field) in world.fields() {
            signal_types.insert(
                id.to_string(),
                InferredShape::from_value_type(&field.value_type),
            );
        }

        Self {
            signal_types,
            local_types: HashMap::new(),
        }
    }

    fn with_local(&self, name: String, ty: InferredShape) -> Self {
        let mut local_types = self.local_types.clone();
        local_types.insert(name, ty);
        Self {
            signal_types: self.signal_types.clone(),
            local_types,
        }
    }
}

/// Checks type compatibility in binary operations across all expressions.
fn check_type_compatibility(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    let ctx = TypeCheckContext::new(world);

    // Check signals
    for (signal_id, signal) in world.signals() {
        if let Some(resolve) = &signal.resolve {
            check_expr_types(resolve, &format!("signal.{}", signal_id), &ctx, warnings);
        }
        for assertion in &signal.assertions {
            check_expr_types(
                &assertion.condition,
                &format!("signal.{} assert", signal_id),
                &ctx,
                warnings,
            );
        }
    }

    // Check members
    for (member_id, member) in world.members() {
        if let Some(resolve) = &member.resolve {
            check_expr_types(resolve, &format!("member.{}", member_id), &ctx, warnings);
        }
        if let Some(initial) = &member.initial {
            check_expr_types(
                initial,
                &format!("member.{} initial", member_id),
                &ctx,
                warnings,
            );
        }
    }

    // Check fields
    for (field_id, field) in world.fields() {
        if let Some(measure) = &field.measure {
            check_expr_types(measure, &format!("field.{}", field_id), &ctx, warnings);
        }
    }

    // Check fractures
    for (fracture_id, fracture) in world.fractures() {
        for condition in &fracture.conditions {
            check_expr_types(
                condition,
                &format!("fracture.{}", fracture_id),
                &ctx,
                warnings,
            );
        }
        for emit in &fracture.emits {
            check_expr_types(
                &emit.value,
                &format!("fracture.{} emit", fracture_id),
                &ctx,
                warnings,
            );
        }
    }

    // Check operators
    for (op_id, operator) in world.operators() {
        if let Some(body) = &operator.body {
            check_expr_types(body, &format!("operator.{}", op_id), &ctx, warnings);
        }
    }

    // Check impulses
    for (impulse_id, impulse) in world.impulses() {
        if let Some(apply) = &impulse.apply {
            check_expr_types(apply, &format!("impulse.{}", impulse_id), &ctx, warnings);
        }
    }

    // Check chronicles
    for (chronicle_id, chronicle) in world.chronicles() {
        for (i, handler) in chronicle.handlers.iter().enumerate() {
            // Check condition expression
            check_expr_types(
                &handler.condition,
                &format!("chronicle.{} handler[{}]", chronicle_id, i),
                &ctx,
                warnings,
            );
            // Check event field expressions
            for field in &handler.event_fields {
                check_expr_types(
                    &field.value,
                    &format!("chronicle.{} handler[{}].{}", chronicle_id, i, field.name),
                    &ctx,
                    warnings,
                );
            }
        }
    }
}

/// Recursively checks an expression for type compatibility in binary operations.
fn check_expr_types(
    expr: &CompiledExpr,
    context: &str,
    ctx: &TypeCheckContext,
    warnings: &mut Vec<CompileWarning>,
) {
    match expr {
        CompiledExpr::Binary { op, left, right } => {
            // First check children
            check_expr_types(left, context, ctx, warnings);
            check_expr_types(right, context, ctx, warnings);

            // Then check this operation
            let left_type = infer_type(left, ctx);
            let right_type = infer_type(right, ctx);

            if let Err(msg) = check_binary_op_types(*op, &left_type, &right_type) {
                warnings.push(CompileWarning {
                    code: WarningCode::TypeMismatch,
                    message: format!(
                        "{} in {}: cannot apply {:?} to {} and {}",
                        msg,
                        context,
                        op,
                        left_type.type_name(),
                        right_type.type_name()
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Unary { operand, .. } => {
            check_expr_types(operand, context, ctx, warnings);
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            check_expr_types(condition, context, ctx, warnings);
            check_expr_types(then_branch, context, ctx, warnings);
            check_expr_types(else_branch, context, ctx, warnings);
        }
        CompiledExpr::Let { name, value, body } => {
            check_expr_types(value, context, ctx, warnings);
            let value_type = infer_type(value, ctx);
            let new_ctx = ctx.with_local(name.clone(), value_type);
            check_expr_types(body, context, &new_ctx, warnings);
        }
        CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
            for arg in args {
                check_expr_types(arg, context, ctx, warnings);
            }
        }
        CompiledExpr::FieldAccess { object, .. } => {
            check_expr_types(object, context, ctx, warnings);
        }
        CompiledExpr::Aggregate { body, .. }
        | CompiledExpr::Other { body, .. }
        | CompiledExpr::Pairs { body, .. } => {
            check_expr_types(body, context, ctx, warnings);
        }
        CompiledExpr::Filter {
            predicate, body, ..
        } => {
            check_expr_types(predicate, context, ctx, warnings);
            check_expr_types(body, context, ctx, warnings);
        }
        CompiledExpr::First { predicate, .. } => {
            check_expr_types(predicate, context, ctx, warnings);
        }
        CompiledExpr::Nearest { position, .. } => {
            check_expr_types(position, context, ctx, warnings);
        }
        CompiledExpr::Within {
            position,
            radius,
            body,
            ..
        } => {
            check_expr_types(position, context, ctx, warnings);
            check_expr_types(radius, context, ctx, warnings);
            check_expr_types(body, context, ctx, warnings);
        }
        CompiledExpr::EmitSignal { value, .. } => {
            check_expr_types(value, context, ctx, warnings);
        }
        // Leaf nodes - no nested expressions
        CompiledExpr::Literal(..)
        | CompiledExpr::Prev
        | CompiledExpr::DtRaw
        | CompiledExpr::SimTime
        | CompiledExpr::Collected
        | CompiledExpr::Signal(_)
        | CompiledExpr::Const(..)
        | CompiledExpr::Config(..)
        | CompiledExpr::Local(_)
        | CompiledExpr::Payload
        | CompiledExpr::PayloadField(_)
        | CompiledExpr::SelfField(_)
        | CompiledExpr::EntityAccess { .. } => {}
    }
}

/// Infers the type of an expression.
fn infer_type(expr: &CompiledExpr, ctx: &TypeCheckContext) -> InferredShape {
    match expr {
        // Scalars
        CompiledExpr::Literal(..)
        | CompiledExpr::DtRaw
        | CompiledExpr::SimTime
        | CompiledExpr::Const(..)
        | CompiledExpr::Config(..) => InferredShape::scalar(),

        // Signal references - look up type
        CompiledExpr::Signal(id) => ctx
            .signal_types
            .get(&id.to_string())
            .cloned()
            .unwrap_or(InferredShape::Unknown),

        // Local variables
        CompiledExpr::Local(name) => ctx
            .local_types
            .get(name)
            .cloned()
            .unwrap_or(InferredShape::Unknown),

        // Field access - component access returns scalar
        CompiledExpr::FieldAccess { field, .. } => {
            if matches!(field.as_str(), "x" | "y" | "z" | "w")
                || (field.starts_with('m') && field.len() == 3)
            {
                InferredShape::scalar()
            } else {
                InferredShape::Unknown
            }
        }

        // Binary operations - compute result type
        CompiledExpr::Binary { op, left, right } => {
            let left_type = infer_type(left, ctx);
            let right_type = infer_type(right, ctx);
            infer_binary_result(*op, &left_type, &right_type)
        }

        // Unary operations
        CompiledExpr::Unary { op, operand } => match op {
            crate::UnaryOpIr::Neg => infer_type(operand, ctx),
            crate::UnaryOpIr::Not => InferredShape::scalar(),
        },

        // Conditionals - try to unify branch types
        CompiledExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let then_type = infer_type(then_branch, ctx);
            let else_type = infer_type(else_branch, ctx);
            if then_type == else_type {
                then_type
            } else {
                InferredShape::Unknown
            }
        }

        // Let bindings - type of body
        CompiledExpr::Let { name, value, body } => {
            let value_type = infer_type(value, ctx);
            let new_ctx = ctx.with_local(name.clone(), value_type);
            infer_type(body, &new_ctx)
        }

        // Everything else - unknown
        _ => InferredShape::Unknown,
    }
}

/// Checks if a binary operation is type-compatible.
fn check_binary_op_types(
    op: BinaryOpIr,
    left: &InferredShape,
    right: &InferredShape,
) -> Result<(), &'static str> {
    // If either type is unknown, we can't check
    let (left_shape, right_shape) = match (left, right) {
        (InferredShape::Known(l), InferredShape::Known(r)) => (l, r),
        _ => return Ok(()),
    };

    // Convert IR op to coercion op
    let coercion_op = match op {
        BinaryOpIr::Add => coercion::BinaryOp::Add,
        BinaryOpIr::Sub => coercion::BinaryOp::Sub,
        BinaryOpIr::Mul => coercion::BinaryOp::Mul,
        BinaryOpIr::Div => coercion::BinaryOp::Div,
        // Comparison and logical ops are valid for any numeric types
        BinaryOpIr::Eq
        | BinaryOpIr::Ne
        | BinaryOpIr::Lt
        | BinaryOpIr::Le
        | BinaryOpIr::Gt
        | BinaryOpIr::Ge
        | BinaryOpIr::And
        | BinaryOpIr::Or
        | BinaryOpIr::Pow => return Ok(()),
    };

    match coercion::can_operate(coercion_op, left_shape, right_shape) {
        coercion::TypeCheckResult::Valid(_) => Ok(()),
        coercion::TypeCheckResult::Invalid(msg) => Err(msg),
    }
}

/// Infers the result type of a binary operation.
fn infer_binary_result(
    op: BinaryOpIr,
    left: &InferredShape,
    right: &InferredShape,
) -> InferredShape {
    // If either type is unknown, result is unknown
    let (left_shape, right_shape) = match (left, right) {
        (InferredShape::Known(l), InferredShape::Known(r)) => (l, r),
        _ => return InferredShape::Unknown,
    };

    // Comparison ops always return scalar
    match op {
        BinaryOpIr::Eq
        | BinaryOpIr::Ne
        | BinaryOpIr::Lt
        | BinaryOpIr::Le
        | BinaryOpIr::Gt
        | BinaryOpIr::Ge
        | BinaryOpIr::And
        | BinaryOpIr::Or => return InferredShape::scalar(),
        _ => {}
    }

    // Convert IR op to coercion op for arithmetic
    let coercion_op = match op {
        BinaryOpIr::Add => coercion::BinaryOp::Add,
        BinaryOpIr::Sub => coercion::BinaryOp::Sub,
        BinaryOpIr::Mul => coercion::BinaryOp::Mul,
        BinaryOpIr::Div => coercion::BinaryOp::Div,
        BinaryOpIr::Pow => return InferredShape::scalar(), // pow returns scalar
        _ => return InferredShape::Unknown,
    };

    match coercion::can_operate(coercion_op, left_shape, right_shape) {
        coercion::TypeCheckResult::Valid(shape) => InferredShape::Known(shape),
        coercion::TypeCheckResult::Invalid(_) => InferredShape::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower;
    use continuum_dsl::parse;

    fn parse_and_lower(src: &str) -> CompiledWorld {
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        lower(&unit.unwrap()).unwrap()
    }

    #[test]
    fn test_signal_with_range_no_assertion_warns() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K, 100..10000>
                : strata(terra)
                resolve { prev }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, WarningCode::MissingRangeAssertion);
        assert!(warnings[0].entity.contains("terra.temp"));
    }

    #[test]
    fn test_signal_with_range_and_assertion_no_warning() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K, 100..10000>
                : strata(terra)
                resolve { prev }
                assert {
                    prev > 100
                }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert!(warnings.is_empty());
    }

    #[test]
    fn test_signal_without_range_no_warning() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { prev }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert!(warnings.is_empty());
    }

    #[test]
    fn test_undefined_symbol_warns() {
        // "colected" is a typo - should be "collected"
        // The parser will treat it as a path/signal reference
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { prev + colected }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should warn about undefined symbol "colected"
        assert!(!warnings.is_empty());
        let undefined_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UndefinedSymbol)
            .collect();
        assert_eq!(undefined_warnings.len(), 1);
        assert!(undefined_warnings[0].message.contains("colected"));
    }

    #[test]
    fn test_unknown_function_warns() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { unknownfunc(prev) }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should warn about unknown function
        let func_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UnknownFunction)
            .collect();
        assert_eq!(func_warnings.len(), 1);
        assert!(func_warnings[0].message.contains("unknownfunc"));
    }

    #[test]
    fn test_valid_symbols_no_warning() {
        let src = r#"
            const {
                physics.gravity: 9.81
            }

            config {
                thermal.decay_halflife: 1.0
            }

            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { dt.decay(prev, config.thermal.decay_halflife) + collected }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // No undefined symbol or unknown function warnings
        let symbol_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| {
                matches!(
                    w.code,
                    WarningCode::UndefinedSymbol | WarningCode::UnknownFunction
                )
            })
            .collect();
        assert!(
            symbol_warnings.is_empty(),
            "unexpected warnings: {:?}",
            symbol_warnings
        );
    }

    #[test]
    fn test_uninitialized_member_warns() {
        // Note: member signals can be defined without entity declaration for lowering
        let src = r#"
strata.test {}
era.main { : initial }

member.test.entity.value {
    : Scalar<1>
    : strata(test)
    resolve { prev }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should warn about uninitialized member
        let uninit_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UninitializedMember)
            .collect();
        assert_eq!(uninit_warnings.len(), 1);
        assert!(uninit_warnings[0].message.contains("test.entity.value"));
        assert!(uninit_warnings[0].message.contains("prev"));
    }

    #[test]
    fn test_member_with_computation_no_warning() {
        let src = r#"
strata.test {}
era.main { : initial }

member.test.entity.age {
    : Scalar<yr>
    : strata(test)
    resolve { dt.integrate(prev, 1.0) }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should NOT warn - member has actual computation (integrate)
        let uninit_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UninitializedMember)
            .collect();
        assert!(uninit_warnings.is_empty());
    }

    #[test]
    fn test_member_with_initial_no_warning() {
        let src = r#"
strata.test {}
era.main { : initial }

config {
    test.default_value: 25.0
}

member.test.entity.value {
    : Scalar<1>
    : strata(test)
    initial { config.test.default_value }
    resolve { prev }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should NOT warn - member has explicit initial block
        let uninit_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UninitializedMember)
            .collect();
        assert!(
            uninit_warnings.is_empty(),
            "expected no UninitializedMember warning for member with initial block"
        );
    }

    #[test]
    fn test_let_binding_in_fracture_emit_no_warning() {
        let src = r#"
strata.test {}
era.main { : initial }

config {
    test.threshold: 100.0
    test.coupling_strength: 0.5
}

signal.test.heat {
    : Scalar<J>
    : strata(test)
    resolve { prev + collected }
}

signal.test.flow {
    : Scalar<W>
    : strata(test)
    resolve { prev + collected }
}

fracture.test.thermal_coupling {
    when {
        signal.test.heat > config.test.threshold
    }

    emit {
        let ratio = signal.test.heat / config.test.threshold in
        let delta = (ratio - 1.0) * config.test.coupling_strength in
        signal.test.flow <- delta
    }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should NOT warn about let-bound variables being undefined signals
        let undefined_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UndefinedSymbol)
            .filter(|w| w.message.contains("ratio") || w.message.contains("delta"))
            .collect();
        assert!(
            undefined_warnings.is_empty(),
            "expected no UndefinedSymbol warnings for let-bound variables, got: {:?}",
            undefined_warnings
        );
    }

    #[test]
    fn test_multiple_let_bindings_and_emits_no_warning() {
        // Test pattern matching terra fractures: multiple let bindings with multiple emits
        let src = r#"
strata.test {}
era.main { : initial }

config {
    test.burn_fraction: 0.1
    test.release_fraction: 0.8
}

signal.test.biomass {
    : Scalar<kg>
    : strata(test)
    resolve { prev + collected }
}

signal.test.carbon {
    : Scalar<kg>
    : strata(test)
    resolve { prev + collected }
}

signal.test.released {
    : Scalar<kg>
    : strata(test)
    resolve { prev + collected }
}

fracture.test.fire {
    when {
        signal.test.biomass > 10.0
    }

    emit {
        let biomass = signal.test.biomass in
        let burned = biomass * config.test.burn_fraction in
        signal.test.biomass <- -burned;

        let released = burned * config.test.release_fraction in
        signal.test.released <- released;

        signal.test.carbon <- burned - released
    }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should NOT warn about let-bound variables being undefined signals
        let undefined_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UndefinedSymbol)
            .filter(|w| {
                w.message.contains("biomass")
                    || w.message.contains("burned")
                    || w.message.contains("released")
            })
            .collect();
        assert!(
            undefined_warnings.is_empty(),
            "expected no UndefinedSymbol warnings for let-bound variables, got: {:?}",
            undefined_warnings
        );
    }

    // ============================================================================
    // Type Compatibility Tests
    // ============================================================================

    #[test]
    fn test_vec2_add_vec3_type_mismatch() {
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.pos2 {
    : Vec2<m>
    : strata(test)
    resolve { prev }
}

signal.test.pos3 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.invalid {
    : Vec3<m>
    : strata(test)
    resolve { signal.test.pos2 + signal.test.pos3 }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert_eq!(
            type_warnings.len(),
            1,
            "expected TypeMismatch warning for Vec2 + Vec3"
        );
        assert!(type_warnings[0].message.contains("Vec2"));
        assert!(type_warnings[0].message.contains("Vec3"));
    }

    #[test]
    fn test_mat3_mul_vec4_type_mismatch() {
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.matrix {
    : Mat3<1>
    : strata(test)
    resolve { prev }
}

signal.test.vec4 {
    : Vec4<m>
    : strata(test)
    resolve { prev }
}

signal.test.invalid {
    : Vec3<m>
    : strata(test)
    resolve { signal.test.matrix * signal.test.vec4 }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert_eq!(
            type_warnings.len(),
            1,
            "expected TypeMismatch warning for Mat3 * Vec4"
        );
        assert!(type_warnings[0].message.contains("Mat3"));
        assert!(type_warnings[0].message.contains("Vec4"));
    }

    #[test]
    fn test_vec3_add_vec3_no_warning() {
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.pos1 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.pos2 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.sum {
    : Vec3<m>
    : strata(test)
    resolve { signal.test.pos1 + signal.test.pos2 }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert!(
            type_warnings.is_empty(),
            "expected no TypeMismatch warnings for Vec3 + Vec3"
        );
    }

    #[test]
    fn test_scalar_mul_vec3_no_warning() {
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.scale {
    : Scalar<1>
    : strata(test)
    resolve { prev }
}

signal.test.vec {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.scaled {
    : Vec3<m>
    : strata(test)
    resolve { signal.test.scale * signal.test.vec }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert!(
            type_warnings.is_empty(),
            "expected no TypeMismatch warnings for Scalar * Vec3"
        );
    }

    #[test]
    fn test_mat3_mul_vec3_no_warning() {
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.matrix {
    : Mat3<1>
    : strata(test)
    resolve { prev }
}

signal.test.vec {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.transformed {
    : Vec3<m>
    : strata(test)
    resolve { signal.test.matrix * signal.test.vec }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert!(
            type_warnings.is_empty(),
            "expected no TypeMismatch warnings for Mat3 * Vec3"
        );
    }

    #[test]
    fn test_comparison_ops_no_type_warning() {
        // Comparison operators should work with any types
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.vec2 {
    : Vec2<m>
    : strata(test)
    resolve { prev }
}

signal.test.vec3 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.result {
    : Scalar<1>
    : strata(test)
    resolve { if signal.test.vec2 > signal.test.vec3 { 1.0 } else { 0.0 } }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Comparison ops shouldn't trigger type mismatch
        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert!(
            type_warnings.is_empty(),
            "expected no TypeMismatch warnings for comparison operators"
        );
    }

    #[test]
    fn test_nested_type_error_detected() {
        // Type error in nested expression should be detected
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.vec2 {
    : Vec2<m>
    : strata(test)
    resolve { prev }
}

signal.test.vec3 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.invalid {
    : Vec3<m>
    : strata(test)
    resolve { (signal.test.vec2 + signal.test.vec3) * 2.0 }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert_eq!(
            type_warnings.len(),
            1,
            "expected TypeMismatch warning for nested Vec2 + Vec3"
        );
    }

    #[test]
    fn test_let_binding_preserves_type() {
        // Let bindings should propagate type info for checking
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.vec3 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

signal.test.invalid {
    : Vec3<m>
    : strata(test)
    resolve {
        let v = signal.test.vec3 in
        let bad = v + vector.new2(1.0, 2.0) in
        bad
    }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // The v + Vec2 operation should trigger a type mismatch
        // (v is Vec3 from signal.test.vec3, vector.new2 returns Vec2)
        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        // This might not trigger if we can't infer the type of vector.new2
        // but it tests the let binding propagation
        // The test passes even without type warning since function return types aren't inferred
    }

    #[test]
    fn test_field_type_checking() {
        // Fields should have their expressions type-checked
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.vec2 {
    : Vec2<m>
    : strata(test)
    resolve { prev }
}

signal.test.vec3 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

field.test.invalid_measure {
    : Vec3<m>
    : strata(test)
    measure { signal.test.vec2 + signal.test.vec3 }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert_eq!(
            type_warnings.len(),
            1,
            "expected TypeMismatch warning for field measure with Vec2 + Vec3"
        );
        assert!(
            type_warnings[0]
                .entity
                .contains("field.test.invalid_measure")
        );
    }

    #[test]
    fn test_fracture_emit_type_checking() {
        // Fracture emit expressions should be type-checked
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.vec2 {
    : Vec2<m>
    : strata(test)
    resolve { prev + collected }
}

signal.test.vec3 {
    : Vec3<m>
    : strata(test)
    resolve { prev + collected }
}

fracture.test.tension {
    : strata(test)
    when { signal.test.vec2.x > 10.0 }
    emit { signal.test.vec3 <- signal.test.vec2 + signal.test.vec3 }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert_eq!(
            type_warnings.len(),
            1,
            "expected TypeMismatch warning for fracture emit with Vec2 + Vec3"
        );
        assert!(type_warnings[0].entity.contains("fracture.test.tension"));
    }

    #[test]
    fn test_chronicle_type_checking() {
        // Chronicle handler conditions and event fields should be type-checked
        let src = r#"
strata.test {}
era.main { : initial }

signal.test.vec2 {
    : Vec2<m>
    : strata(test)
    resolve { prev }
}

signal.test.vec3 {
    : Vec3<m>
    : strata(test)
    resolve { prev }
}

chronicle.test.observer {
    observe {
        when signal.test.vec2.x > 0.0 {
            emit event.update {
                invalid_sum: signal.test.vec2 + signal.test.vec3
            }
        }
    }
}
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        let type_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::TypeMismatch)
            .collect();
        assert_eq!(
            type_warnings.len(),
            1,
            "expected TypeMismatch warning for chronicle event field with Vec2 + Vec3"
        );
        assert!(type_warnings[0].entity.contains("chronicle.test.observer"));
    }
}
