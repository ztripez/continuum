//! Execution block compilation pass.
//!
//! Converts raw execution blocks from the parser into compiled `Execution` structures.
//! Validates that phases are appropriate for the node's role and extracts
//! signal/field dependencies for DAG construction.
//!
//! # What This Pass Does
//!
//! 1. **Phase Validation** - Verifies phase names and role compatibility.
//! 2. **Purity Enforcement** - Ensures pure phases don't have statement blocks.
//! 3. **Dependency Extraction** - Walks expression trees to find read dependencies.
//! 4. **Execution Creation** - Populates the `executions` list on the node.
//! 5. **Lifecycle Management** - Clears `execution_blocks` after successful compilation.
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Block Compilation → Validation
//!                                                         ^^^^^^^^^^^^
//!                                                         YOU ARE HERE
//! ```

use crate::ast::{
    BlockBody, Execution, ExecutionBody, Expr, Index, Node, RoleId, Stmt, TypedStmt,
};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::Phase;
use crate::resolve::dependencies::{extract_dependencies, extract_stmt_dependencies};
use crate::resolve::effects::{validate_effect_purity, EffectContext};
use crate::resolve::expr_typing::{type_expression, TypingContext};
use crate::resolve::utils::sort_unique;
use std::collections::HashSet;

/// Compiles a sequence of untyped statements into a type-validated IR representation.
///
/// This pass transforms procedural simulation logic ([`Stmt<Expr>`]) into a typed
/// execution IR ([`TypedStmt`]), performing the following operations:
///
/// 1. **Expression Typing**: Recursively invokes [`type_expression`] on all expressions
///    within each statement (e.g., let-values, assignment sources).
/// 2. **Local Scoping**: Manages a block-local symbol table for `let` bindings. Variable
///    types are registered as they are encountered, allowing subsequent statements
///    in the same block to reference them.
/// 3. **Effect Validation**: Verifies that assignment targets (signals and fields)
///    are valid within the current [`Phase`] context.
///
/// # Emissions and Causality
///
/// The resulting [`TypedStmt`] list is used by Phase 13 (DAG Construction) to extract:
/// - **Read Dependencies**: All signals, fields, and entities referenced in expressions.
///   These form the incoming edges in the execution DAG.
/// - **Emissions (Side Effects)**: All signals or fields targeted by assignments.
///   These form the outgoing edges and define the causality boundary.
///
/// # Errors
///
/// Returns an aggregated list of [`CompileError`]s if any expression fails typing,
/// a variable is undefined, or an assignment target is invalid.
///
/// # Parameters
/// - `stmts`: Untyped statements to type-check.
/// - `ctx`: Typing context used for expression typing and phase checks.
///
/// # Returns
/// Typed statements suitable for dependency extraction and execution.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::ast::{Expr, Stmt};
/// use continuum_cdsl::ast::KernelRegistry;
/// use continuum_cdsl::foundation::Span;
/// use continuum_cdsl::resolve::blocks::compile_statements;
/// use continuum_cdsl::resolve::expr_typing::TypingContext;
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_foundation::Phase;
/// use std::collections::HashMap;
///
/// let span = Span::new(0, 0, 0, 1);
/// let stmts = vec![Stmt::Expr(Expr::literal(1.0, None, span))];
///
/// let registry = KernelRegistry::global();
/// let type_table = TypeTable::new();
/// let signal_types = HashMap::new();
/// let field_types = HashMap::new();
/// let config_types = HashMap::new();
/// let const_types = HashMap::new();
/// let ctx = TypingContext::new(
///     &type_table,
///     &registry,
///     &signal_types,
///     &field_types,
///     &config_types,
///     &const_types,
/// )
/// .with_phase(Phase::Collect);
///
/// let typed = compile_statements(&stmts, &ctx).unwrap();
/// assert_eq!(typed.len(), 1);
/// ```
pub fn compile_statements(
    stmts: &[Stmt<Expr>],
    ctx: &TypingContext,
) -> Result<Vec<TypedStmt>, Vec<CompileError>> {
    let mut typed_stmts = Vec::new();
    let mut errors = Vec::new();
    let mut current_ctx = ctx.clone();

    for stmt in stmts {
        match stmt {
            Stmt::Let { name, value, span } => match type_expression(value, &current_ctx) {
                Ok(typed_value) => {
                    current_ctx
                        .local_bindings
                        .insert(name.clone(), typed_value.ty.clone());
                    typed_stmts.push(TypedStmt::Let {
                        name: name.clone(),
                        value: typed_value,
                        span: *span,
                    });
                }
                Err(mut e) => errors.append(&mut e),
            },
            Stmt::SignalAssign {
                target,
                value,
                span,
            } => {
                // Phase boundary enforcement: Signals in Collect or Fracture phase
                if let Some(phase) = current_ctx.phase
                    && phase != Phase::Collect
                    && phase != Phase::Fracture
                {
                    errors.push(CompileError::new(
                        ErrorKind::PhaseBoundaryViolation,
                        *span,
                        format!(
                            "signal '{}' cannot be assigned in {:?} phase (signals are only assignable in Collect or Fracture phases)",
                            target, phase
                        ),
                    ));
                }

                match type_expression(value, &current_ctx) {
                    Ok(typed_value) => {
                        // Validate target signal exists and matches type
                        if let Some(expected_ty) = current_ctx.signal_types.get(target) {
                            if &typed_value.ty != expected_ty {
                                errors.push(CompileError::new(
                                    ErrorKind::TypeMismatch,
                                    *span,
                                    format!(
                                        "target signal '{}' has type {:?}, but assigned value has type {:?}",
                                        target, expected_ty, typed_value.ty
                                    ),
                                ));
                            }
                        } else {
                            errors.push(CompileError::new(
                                ErrorKind::UndefinedName,
                                *span,
                                format!("target signal '{}' not found", target),
                            ));
                        }

                        typed_stmts.push(TypedStmt::SignalAssign {
                            target: target.clone(),
                            value: typed_value,
                            span: *span,
                        });
                    }
                    Err(mut e) => errors.append(&mut e),
                }
            }
            Stmt::FieldAssign {
                target,
                position,
                value,
                span,
            } => {
                // Phase boundary enforcement: Fields only in Measure phase
                if let Some(phase) = current_ctx.phase
                    && phase != Phase::Measure
                {
                    errors.push(CompileError::new(
                        ErrorKind::PhaseBoundaryViolation,
                        *span,
                        format!(
                            "field '{}' cannot be assigned in {:?} phase (fields are only assignable in Measure phase)",
                            target, phase
                        ),
                    ));
                }

                let typed_pos = type_expression(position, &current_ctx);
                let typed_val = type_expression(value, &current_ctx);

                match (typed_pos, typed_val) {
                    (Ok(p), Ok(v)) => {
                        // Validate target field exists and matches type
                        if let Some(expected_ty) = current_ctx.field_types.get(target) {
                            if &v.ty != expected_ty {
                                errors.push(CompileError::new(
                                    ErrorKind::TypeMismatch,
                                    *span,
                                    format!(
                                        "target field '{}' has type {:?}, but assigned value has type {:?}",
                                        target, expected_ty, v.ty
                                    ),
                                ));
                            }
                        } else {
                            errors.push(CompileError::new(
                                ErrorKind::UndefinedName,
                                *span,
                                format!("target field '{}' not found", target),
                            ));
                        }

                        typed_stmts.push(TypedStmt::FieldAssign {
                            target: target.clone(),
                            position: p,
                            value: v,
                            span: *span,
                        });
                    }
                    (Err(mut e1), Err(mut e2)) => {
                        errors.append(&mut e1);
                        errors.append(&mut e2);
                    }
                    (Err(mut e), _) | (_, Err(mut e)) => errors.append(&mut e),
                }
            }
            Stmt::Expr(expr) => match type_expression(expr, &current_ctx) {
                Ok(typed_expr) => typed_stmts.push(TypedStmt::Expr(typed_expr)),
                Err(mut e) => errors.append(&mut e),
            },
        }
    }

    // Validate effect purity: ensure pure phases don't call effect kernels
    if let Some(phase) = ctx.phase {
        let effect_ctx = EffectContext::new(phase);
        for stmt in &typed_stmts {
            match stmt {
                TypedStmt::Expr(expr)
                | TypedStmt::Let { value: expr, .. }
                | TypedStmt::SignalAssign { value: expr, .. }
                | TypedStmt::FieldAssign { value: expr, .. } => {
                    errors.extend(validate_effect_purity(
                        expr,
                        &effect_ctx,
                        ctx.kernel_registry,
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(typed_stmts)
    } else {
        Err(errors)
    }
}

/// Parses a string phase name into a Phase enum value.
///
/// # Parameters
/// - `name`: Phase name string.
/// - `span`: Source span used for diagnostics.
///
/// # Returns
/// Parsed [`Phase`] value.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidCapability`] if the name is unrecognized or
/// is a legacy name like "apply" or "emit".
///
/// # Examples
/// ```rust
/// use continuum_cdsl::resolve::blocks::parse_phase_name;
/// use continuum_cdsl::foundation::Span;
/// use continuum_foundation::Phase;
///
/// let span = Span::new(0, 0, 0, 1);
/// let phase = parse_phase_name("resolve", span).unwrap();
/// assert_eq!(phase, Phase::Resolve);
/// ```
pub fn parse_phase_name(name: &str, span: crate::foundation::Span) -> Result<Phase, CompileError> {
    match name.to_lowercase().as_str() {
        "resolve" => Ok(Phase::Resolve),
        "collect" => Ok(Phase::Collect),
        "fracture" => Ok(Phase::Fracture),
        "measure" => Ok(Phase::Measure),
        "assert" => Ok(Phase::Assert),
        "configure" => Ok(Phase::Configure),

        // Handle legacy names with helpful error messages
        "apply" | "emit" => Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "legacy execution phase '{}' is no longer supported. Use 'collect' for signal inputs or 'measure' for observations.",
                name
            ),
        )),

        _ => Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "unknown execution phase '{}'. Valid phases are: resolve, collect, fracture, measure, assert, configure",
                name
            ),
        )),
    }
}

/// Validate that a phase is allowed for a node's role.
///
/// Uses the role's spec to check if the phase is in the allowed set.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidCapability`] if the phase is not allowed for this role.
///
/// # Examples
///
/// ```rust,ignore
/// // Signal role allows Resolve phase
/// validate_phase_for_role(Phase::Resolve, RoleId::Signal, node.span)?;
///
/// // Signal role does NOT allow Collect phase
/// validate_phase_for_role(Phase::Collect, RoleId::Signal, node.span)?; // Error
/// ```
fn validate_phase_for_role(
    phase: Phase,
    role_id: RoleId,
    span: crate::foundation::Span,
) -> Result<(), CompileError> {
    let spec = role_id.spec();
    if spec.allowed_phases.contains(phase) {
        Ok(())
    } else {
        Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "{:?} role cannot have {:?} phase execution block",
                role_id, phase
            ),
        ))
    }
}

/// Compile execution blocks for a node and aggregate dependencies.
///
/// Converts raw `execution_blocks` from the parser into typed [`Execution`] structs.
/// This pass performs:
/// - Phase name validation and role compatibility checks.
/// - Purity enforcement (pure phases cannot contain statements).
/// - Recursive dependency extraction from expression trees.
/// - Aggregation of all read dependencies (from both executions and assertions)
///   into [`Node::reads`].
///
/// Populates `node.executions` and `node.reads`, and clears `node.execution_blocks`.
///
/// # Errors
///
/// Returns errors for:
/// - Unknown phase names
/// - Phases not allowed for role
/// - Statement blocks in pure phases
/// - Type resolution failures (should not happen if type resolution passed)
///
/// # Examples
///
/// # Parameters
/// - `node`: Node whose execution blocks will be compiled.
/// - `ctx`: Typing context used for expression and statement typing.
///
/// # Returns
/// `Ok(())` after populating `node.executions` and related metadata.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::ast::{BlockBody, Expr, KernelRegistry, Node, RoleData};
/// use continuum_cdsl::foundation::{KernelType, Path, Shape, Span, Type, Unit};
/// use continuum_cdsl::resolve::blocks::compile_execution_blocks;
/// use continuum_cdsl::resolve::expr_typing::TypingContext;
/// use continuum_cdsl::resolve::types::TypeTable;
/// use std::collections::HashMap;
///
/// let span = Span::new(0, 0, 0, 1);
/// let mut node = Node::new(Path::from_path_str("demo.counter"), span, RoleData::Signal, ());
/// node.output = Some(Type::Kernel(KernelType {
///     shape: Shape::Scalar,
///     unit: Unit::DIMENSIONLESS,
///     bounds: None,
/// }));
/// node.execution_blocks.push((
///     "resolve".to_string(),
///     BlockBody::Expression(Expr::literal(1.0, None, span)),
/// ));
///
/// let registry = KernelRegistry::global();
/// let type_table = TypeTable::new();
/// let signal_types = HashMap::new();
/// let field_types = HashMap::new();
/// let config_types = HashMap::new();
/// let const_types = HashMap::new();
/// let ctx = TypingContext::new(
///     &type_table,
///     &registry,
///     &signal_types,
///     &field_types,
///     &config_types,
///     &const_types,
/// );
///
/// compile_execution_blocks(&mut node, &ctx).unwrap();
/// assert_eq!(node.executions.len(), 1);
/// ```
pub fn compile_execution_blocks<I: Index>(
    node: &mut Node<I>,
    ctx: &TypingContext,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();
    let mut executions = Vec::new();

    let role_id = node.role.id();
    let base_ctx = ctx.with_execution_context(None, None, node.output.clone(), None, None);

    // Process each execution block
    for (phase_name, block_body) in &node.execution_blocks {
        // 1. Parse phase name
        let phase = match parse_phase_name(phase_name, node.span) {
            Ok(p) => p,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };

        // 2. Validate phase for role
        if let Err(e) = validate_phase_for_role(phase, role_id, node.span) {
            errors.push(e);
            continue;
        }

        // 3. Validate block body type matches phase purity
        //
        // Resolve and Assert phases are strictly pure and must use single expressions.
        // Measure phase is theoretically pure (no side effects on authoritative state)
        // but can contain FieldAssign statements to emit observations.
        let is_pure_phase = matches!(phase, Phase::Resolve | Phase::Assert);
        let has_statements = matches!(block_body, BlockBody::Statements(_));

        if is_pure_phase && has_statements {
            errors.push(CompileError::new(
                ErrorKind::EffectInPureContext,
                node.span,
                format!(
                    "{:?} phase is pure and cannot contain statement blocks",
                    phase
                ),
            ));
            continue;
        }

        // 4. Extract body
        let body = match block_body {
            BlockBody::TypedExpression(typed_expr) => ExecutionBody::Expr(typed_expr.clone()),
            BlockBody::TypedStatements(typed_stmts) => {
                ExecutionBody::Statements(typed_stmts.clone())
            }
            BlockBody::Expression(expr) => {
                // Type untyped expression
                let block_ctx = base_ctx.with_phase(phase);
                match type_expression(expr, &block_ctx) {
                    Ok(typed_expr) => ExecutionBody::Expr(typed_expr),
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    }
                }
            }
            BlockBody::Statements(stmts) => {
                // Compile untyped statements
                let block_ctx = base_ctx.with_phase(phase);
                match compile_statements(stmts, &block_ctx) {
                    Ok(typed_stmts) => ExecutionBody::Statements(typed_stmts),
                    Err(mut e) => {
                        errors.append(&mut e);
                        continue;
                    }
                }
            }
        };

        // 5. Extract dependencies and emissions
        let (reads, temporal_reads, mut emits) = match &body {
            ExecutionBody::Expr(expr) => {
                let (r, t) = extract_dependencies(expr, &node.path);
                (r, t, Vec::new())
            }
            ExecutionBody::Statements(stmts) => {
                let mut reads = HashSet::new();
                let mut temporal_reads = HashSet::new();
                let mut emits = HashSet::new();
                for stmt in stmts {
                    let (s_reads, s_temporal, s_emits) =
                        extract_stmt_dependencies(stmt, &node.path);
                    for r in s_reads {
                        reads.insert(r);
                    }
                    for t in s_temporal {
                        temporal_reads.insert(t);
                    }
                    for e in s_emits {
                        emits.insert(e);
                    }
                }
                (
                    sort_unique(reads),
                    sort_unique(temporal_reads),
                    sort_unique(emits),
                )
            }
        };

        // 5.1 Enforce emission rules (continuum-6lc9)
        // Signal resolve blocks should have empty emits (implicit self-emission).
        // Resolve phase is pure - no explicit emissions allowed.
        if phase == Phase::Resolve && !emits.is_empty() {
            errors.push(CompileError::new(
                ErrorKind::PhaseBoundaryViolation,
                node.span,
                format!(
                    "Resolve phase execution at '{}' has explicit emissions: {:?}. \
                     Resolve phase must be pure; signals implicitly resolve to themselves.",
                    node.path, emits
                ),
            ));
            continue;
        }

        // 5.2 Populate Resolve emissions (continuum-4rnt)
        // Signals produce themselves in Resolve phase. Making this explicit in IR.
        if phase == Phase::Resolve && role_id == RoleId::Signal {
            emits = vec![node.path.clone()];
        }

        // 6. Create Execution
        let execution = Execution::new(
            phase_name.clone(),
            phase,
            body,
            reads,
            temporal_reads,
            emits,
            node.span,
        );
        executions.push(execution);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // 7. Update node
    node.executions = executions;
    node.execution_blocks.clear();

    // 8. Populate node-level reads for cycle detection (Phase 12 structure validation)
    // Union of all per-execution reads AND assertion reads
    let mut all_reads = HashSet::new();
    let mut all_temporal = HashSet::new();

    // Collect from executions
    for execution in &node.executions {
        for read in &execution.reads {
            all_reads.insert(read.clone());
        }
        for read in &execution.temporal_reads {
            all_temporal.insert(read.clone());
        }
    }

    // Collect from assertions
    for assertion in &node.assertions {
        let (r, t) = extract_dependencies(&assertion.condition, &node.path);
        for read in r {
            all_reads.insert(read);
        }
        for read in t {
            all_temporal.insert(read);
        }
    }

    node.reads = sort_unique(all_reads);
    node.temporal_reads = sort_unique(all_temporal);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, ExprKind, KernelRegistry, TypedExpr, TypedStmt, UntypedKind};
    use crate::foundation::{KernelType, Path, Shape, Span, Type, Unit};
    use std::collections::HashMap;

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn scalar_type() -> Type {
        Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::dimensionless(),
            bounds: None,
        })
    }

    fn make_test_context<'a>(
        type_table: &'a crate::resolve::types::TypeTable,
        registry: &'a KernelRegistry,
        signal_types: &'a HashMap<Path, Type>,
        field_types: &'a HashMap<Path, Type>,
        config_types: &'a HashMap<Path, Type>,
        const_types: &'a HashMap<Path, Type>,
    ) -> TypingContext<'a> {
        TypingContext::new(
            type_table,
            registry,
            signal_types,
            field_types,
            config_types,
            const_types,
        )
    }

    #[test]
    fn test_compile_statements_basic() {
        let registry = KernelRegistry::global();
        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("signal.target"), scalar_type());

        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();

        let ctx = TypingContext {
            type_table: &type_table,
            kernel_registry: &registry,
            signal_types: &signal_types,
            field_types: &field_types,
            config_types: &config_types,
            const_types: &const_types,
            local_bindings: HashMap::new(),
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: Some(Phase::Collect),
        };

        let span = test_span();
        let stmts = vec![
            Stmt::Let {
                name: "x".to_string(),
                value: Expr::literal(1.0, None, span),
                span,
            },
            Stmt::SignalAssign {
                target: Path::from_path_str("signal.target"),
                value: Expr::local("x".to_string(), span),
                span,
            },
        ];

        let result = compile_statements(&stmts, &ctx).unwrap();
        assert_eq!(result.len(), 2);

        if let TypedStmt::Let { name, value, .. } = &result[0] {
            assert_eq!(name, "x");
            assert!(matches!(value.expr, ExprKind::Literal { .. }));
        } else {
            panic!("Expected Let statement");
        }

        if let TypedStmt::SignalAssign { target, value, .. } = &result[1] {
            assert_eq!(target.to_string(), "signal.target");
            assert!(matches!(value.expr, ExprKind::Local(ref n) if n == "x"));
        } else {
            panic!("Expected SignalAssign statement");
        }
    }

    #[test]
    fn test_compile_statements_nested_let() {
        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();

        let ctx = TypingContext {
            type_table: &type_table,
            kernel_registry: &registry,
            signal_types: &signal_types,
            field_types: &field_types,
            config_types: &config_types,
            const_types: &const_types,
            local_bindings: HashMap::new(),
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: Some(Phase::Collect),
        };

        let span = test_span();
        let stmts = vec![
            Stmt::Let {
                name: "x".to_string(),
                value: Expr::literal(1.0, None, span),
                span,
            },
            Stmt::Let {
                name: "y".to_string(),
                value: Expr::local("x".to_string(), span),
                span,
            },
        ];

        let result = compile_statements(&stmts, &ctx).unwrap();
        assert_eq!(result.len(), 2);

        if let TypedStmt::Let { name, value, .. } = &result[1] {
            assert_eq!(name, "y");
            assert!(matches!(value.expr, ExprKind::Local(ref n) if n == "x"));
        } else {
            panic!("Expected second Let statement to reference first");
        }
    }

    #[test]
    fn test_compile_statements_type_mismatch() {
        let registry = KernelRegistry::global();
        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("signal.target"), scalar_type());

        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();

        let ctx = TypingContext {
            type_table: &type_table,
            kernel_registry: &registry,
            signal_types: &signal_types,
            field_types: &field_types,
            config_types: &config_types,
            const_types: &const_types,
            local_bindings: HashMap::new(),
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: Some(Phase::Collect),
        };

        let span = test_span();
        let stmts = vec![Stmt::SignalAssign {
            target: Path::from_path_str("signal.target"),
            value: Expr::new(UntypedKind::BoolLiteral(true), span), // Boolean instead of Scalar
            span,
        }];

        let result = compile_statements(&stmts, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(
            errors[0]
                .message
                .contains("target signal 'signal.target' has type")
        );
    }

    #[test]
    fn test_compile_statements_undefined_signal() {
        let registry = KernelRegistry::global();
        let signal_types = HashMap::new(); // Empty
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();

        let ctx = TypingContext {
            type_table: &type_table,
            kernel_registry: &registry,
            signal_types: &signal_types,
            field_types: &field_types,
            config_types: &config_types,
            const_types: &const_types,
            local_bindings: HashMap::new(),
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: Some(Phase::Collect),
        };

        let span = test_span();
        let stmts = vec![Stmt::SignalAssign {
            target: Path::from_path_str("signal.missing"),
            value: Expr::literal(1.0, None, span),
            span,
        }];

        let result = compile_statements(&stmts, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UndefinedName);
        assert!(
            errors[0]
                .message
                .contains("signal 'signal.missing' not found")
        );
    }

    #[test]
    fn test_compile_statements_phase_boundary_violation() {
        let registry = KernelRegistry::global();
        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("signal.target"), scalar_type());

        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();

        let ctx = TypingContext {
            type_table: &type_table,
            kernel_registry: &registry,
            signal_types: &signal_types,
            field_types: &field_types,
            config_types: &config_types,
            const_types: &const_types,
            local_bindings: HashMap::new(),
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: Some(Phase::Resolve),
        };

        let span = test_span();
        let stmts = vec![Stmt::SignalAssign {
            target: Path::from_path_str("signal.target"),
            value: Expr::literal(1.0, None, span),
            span,
        }];

        let result = compile_statements(&stmts, &ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::PhaseBoundaryViolation);
        assert!(
            errors[0]
                .message
                .contains("cannot be assigned in Resolve phase")
        );
    }

    #[test]
    fn test_extract_stmt_dependencies_let() {
        let span = test_span();
        let ty = scalar_type();
        let path = Path::from_path_str("signal.test");

        let stmt = TypedStmt::Let {
            name: "x".to_string(),
            value: TypedExpr::new(ExprKind::Signal(path.clone()), ty, span),
            span,
        };

        let (reads, _, emits) = extract_stmt_dependencies(&stmt, &Path::from_path_str("test"));
        assert_eq!(emits.len(), 0);
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0], path);
    }

    #[test]
    fn test_extract_stmt_dependencies_field_assign() {
        let span = test_span();
        let ty = scalar_type();
        let path_field = Path::from_path_str("field.temperature");
        let path_pos = Path::from_path_str("signal.pos");
        let path_val = Path::from_path_str("signal.val");

        let stmt = TypedStmt::FieldAssign {
            target: path_field.clone(),
            position: TypedExpr::new(ExprKind::Signal(path_pos.clone()), ty.clone(), span),
            value: TypedExpr::new(ExprKind::Signal(path_val.clone()), ty, span),
            span,
        };

        let (reads, _, emits) = extract_stmt_dependencies(&stmt, &Path::from_path_str("test"));
        assert_eq!(emits.len(), 1);
        assert_eq!(emits[0], path_field);
        assert_eq!(reads.len(), 2);
        assert!(reads.contains(&path_pos));
        assert!(reads.contains(&path_val));
    }

    #[test]
    fn test_extract_stmt_dependencies_expr() {
        let span = test_span();
        let ty = scalar_type();
        let path = Path::from_path_str("signal.test");

        let stmt = TypedStmt::Expr(TypedExpr::new(ExprKind::Signal(path.clone()), ty, span));

        let (reads, _, emits) = extract_stmt_dependencies(&stmt, &Path::from_path_str("test"));
        assert_eq!(emits.len(), 0);
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0], path);
    }

    #[test]
    fn test_extract_stmt_dependencies() {
        let span = test_span();
        let ty = scalar_type();
        let path_in = Path::from_path_str("signal.in");
        let path_out = Path::from_path_str("signal.out");

        let stmt = TypedStmt::SignalAssign {
            target: path_out.clone(),
            value: TypedExpr::new(ExprKind::Signal(path_in.clone()), ty, span),
            span,
        };

        let (reads, _, emits) = extract_stmt_dependencies(&stmt, &Path::from_path_str("test"));
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0], path_in);
        assert_eq!(emits.len(), 1);
        assert_eq!(emits[0], path_out);
    }

    #[test]
    fn test_parse_phase_name_valid() {
        let span = test_span();
        assert_eq!(parse_phase_name("resolve", span).unwrap(), Phase::Resolve);
        assert_eq!(parse_phase_name("collect", span).unwrap(), Phase::Collect);
        assert_eq!(parse_phase_name("fracture", span).unwrap(), Phase::Fracture);
        assert_eq!(parse_phase_name("measure", span).unwrap(), Phase::Measure);
        assert_eq!(parse_phase_name("assert", span).unwrap(), Phase::Assert);
        assert_eq!(
            parse_phase_name("configure", span).unwrap(),
            Phase::Configure
        );
    }

    #[test]
    fn test_parse_phase_name_legacy_rejected() {
        let span = test_span();
        // "apply" and "emit" are legacy names that should be rejected
        assert!(parse_phase_name("apply", span).is_err());
        assert!(parse_phase_name("emit", span).is_err());
    }

    #[test]
    fn test_parse_phase_name_invalid() {
        let span = test_span();
        let result = parse_phase_name("invalid", span);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidCapability);
        assert!(err.message.contains("unknown execution phase"));
    }

    #[test]
    fn test_validate_phase_for_role_signal_resolve() {
        let span = test_span();
        assert!(validate_phase_for_role(Phase::Resolve, RoleId::Signal, span).is_ok());
    }

    #[test]
    fn test_validate_phase_for_role_signal_collect_invalid() {
        let span = test_span();
        let result = validate_phase_for_role(Phase::Collect, RoleId::Signal, span);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidCapability);
        assert!(err.message.contains("cannot have"));
    }

    #[test]
    fn test_validate_phase_for_role_operator_fracture() {
        let span = test_span();
        // Operator allows Fracture phase
        assert!(validate_phase_for_role(Phase::Fracture, RoleId::Operator, span).is_ok());
    }

    #[test]
    fn test_extract_dependencies_empty() {
        let span = test_span();
        let ty = scalar_type();
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            ty,
            span,
        );
        let (deps, _) = extract_dependencies(&expr, &Path::from_path_str("test"));
        assert_eq!(deps.len(), 0);
    }

    #[test]
    fn test_extract_dependencies_signal() {
        let span = test_span();
        let ty = scalar_type();
        let path = Path::from_path_str("signal.temperature");
        let expr = TypedExpr::new(ExprKind::Signal(path.clone()), ty, span);
        let (deps, _) = extract_dependencies(&expr, &Path::from_path_str("test"));

        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&path));
    }

    #[test]
    fn test_extract_dependencies_nested() {
        use crate::ast::KernelId;

        let span = test_span();
        let ty = scalar_type();
        let path1 = Path::from_path_str("signal.a");
        let path2 = Path::from_path_str("field.b");

        let left = TypedExpr::new(ExprKind::Signal(path1.clone()), ty.clone(), span);
        let right = TypedExpr::new(ExprKind::Field(path2.clone()), ty.clone(), span);

        // Binary ops desugar to Call(maths.add, [left, right])
        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![left, right],
            },
            ty,
            span,
        );

        let (deps, _) = extract_dependencies(&expr, &Path::from_path_str("test"));

        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&path1));
        assert!(deps.contains(&path2));
    }

    #[test]
    fn test_compile_execution_blocks_with_typed_expression() {
        use crate::ast::RoleData;
        use crate::foundation::{KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");

        // Create a typed expression (simple literal)
        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });
        let typed_expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: Some(Unit::DIMENSIONLESS),
            },
            ty,
            span,
        );

        // Create node with resolve block containing typed expression
        let mut node = Node::new(path.clone(), span, RoleData::Signal, ());
        node.execution_blocks = vec![(
            "resolve".to_string(),
            BlockBody::TypedExpression(typed_expr),
        )];

        // Compile execution blocks
        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = compile_execution_blocks(&mut node, &ctx);
        assert!(result.is_ok(), "Compilation should succeed: {:?}", result);

        // Verify executions populated
        assert_eq!(node.executions.len(), 1, "Should have 1 execution");
        assert_eq!(node.executions[0].name, "resolve");
        assert_eq!(node.executions[0].phase, Phase::Resolve);
        match &node.executions[0].body {
            ExecutionBody::Expr(e) => {
                assert!(matches!(e.expr, ExprKind::Literal { .. }));
            }
            _ => panic!("Expected Expr body"),
        }

        // Verify execution_blocks cleared
        assert!(
            node.execution_blocks.is_empty(),
            "execution_blocks should be cleared"
        );

        // Verify node-level reads populated (from typed_expr, which is empty literal here)
        assert!(
            node.reads.is_empty(),
            "node.reads should be empty for literal-only execution"
        );
    }

    #[test]
    fn test_compile_execution_blocks_populates_node_reads() {
        use crate::ast::RoleData;
        use crate::foundation::{KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");
        let signal_path = Path::from_path_str("other.signal");

        // Create a typed expression that reads a signal
        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });
        let typed_expr = TypedExpr::new(ExprKind::Signal(signal_path.clone()), ty, span);

        // Create node with resolve block
        let mut node = Node::new(path.clone(), span, RoleData::Signal, ());
        node.execution_blocks = vec![(
            "resolve".to_string(),
            BlockBody::TypedExpression(typed_expr),
        )];

        // Compile execution blocks
        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = compile_execution_blocks(&mut node, &ctx);
        assert!(result.is_ok());

        // Verify node.reads contains the signal
        assert_eq!(node.reads.len(), 1);
        assert_eq!(node.reads[0], signal_path);
    }

    #[test]
    fn test_compile_execution_blocks_union_multiple_blocks() {
        use crate::ast::RoleData;
        use crate::foundation::{KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");
        let path_a = Path::from_path_str("signal.a");
        let path_b = Path::from_path_str("signal.b");

        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });

        let mut node = Node::new(path, span, RoleData::Operator, ());
        // Block 1 reads 'signal.b'
        node.execution_blocks.push((
            "collect".to_string(),
            BlockBody::TypedExpression(TypedExpr::new(
                ExprKind::Signal(path_b.clone()),
                ty.clone(),
                span,
            )),
        ));
        // Block 2 reads 'signal.a'
        node.execution_blocks.push((
            "resolve".to_string(),
            BlockBody::TypedExpression(TypedExpr::new(ExprKind::Signal(path_a.clone()), ty, span)),
        ));

        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        compile_execution_blocks(&mut node, &ctx).unwrap();

        // Verify union is sorted: [signal.a, signal.b]
        assert_eq!(node.reads.len(), 2);
        assert_eq!(node.reads[0], path_a);
        assert_eq!(node.reads[1], path_b);
    }

    #[test]
    fn test_compile_execution_blocks_includes_assertions() {
        use crate::ast::{Assertion, RoleData};
        use crate::foundation::{AssertionSeverity, KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");
        let assert_path = Path::from_path_str("signal.limit");

        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });

        let mut node = Node::new(path, span, RoleData::Signal, ());

        // Add an assertion that reads 'signal.limit'
        node.assertions.push(Assertion::new(
            TypedExpr::new(ExprKind::Signal(assert_path.clone()), ty, span),
            None,
            AssertionSeverity::Error,
            span,
        ));

        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        compile_execution_blocks(&mut node, &ctx).unwrap();

        // Verify assertion dependency is in node.reads
        assert!(node.reads.contains(&assert_path));
    }

    #[test]
    fn test_compile_execution_blocks_untyped_expression_success() {
        use crate::ast::{Expr, RoleData, UntypedKind};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");

        // Create untyped expression
        let untyped_expr = Expr::new(
            UntypedKind::Literal {
                value: 42.0,
                unit: None,
            },
            span,
        );

        // Create node with resolve block containing UNTYPED expression
        let mut node = Node::new(path.clone(), span, RoleData::Signal, ());
        node.execution_blocks = vec![("resolve".to_string(), BlockBody::Expression(untyped_expr))];

        // Compile execution blocks - should now succeed (types on the fly)
        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = compile_execution_blocks(&mut node, &ctx);
        assert!(result.is_ok(), "Should now succeed with untyped expression");
        assert_eq!(node.executions.len(), 1);
    }

    #[test]
    fn test_extract_dependencies_aggregate() {
        use crate::foundation::{AggregateOp, EntityId};

        let span = test_span();
        let ty = scalar_type();
        let entity = EntityId::new("plate");

        let body = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty.clone(),
            span,
        );

        let expr = TypedExpr::new(
            ExprKind::Aggregate {
                op: AggregateOp::Sum,
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(entity.clone()),
                    ty.clone(),
                    span,
                )),
                binding: "p".to_string(),
                body: Box::new(body),
            },
            ty,
            span,
        );

        let (deps, _) = extract_dependencies(&expr, &Path::from_path_str("test"));
        assert_eq!(deps.len(), 1);
        // Entity set dependency is captured
        assert_eq!(deps[0], Path::from_path_str("plate"));
    }

    #[test]
    fn test_extract_dependencies_fold() {
        use crate::foundation::EntityId;

        let span = test_span();
        let ty = scalar_type();
        let entity = EntityId::new("plate");

        let init = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            ty.clone(),
            span,
        );
        let body = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty.clone(),
            span,
        );

        let expr = TypedExpr::new(
            ExprKind::Fold {
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(entity.clone()),
                    ty.clone(),
                    span,
                )),
                init: Box::new(init),
                acc: "acc".to_string(),
                elem: "p".to_string(),
                body: Box::new(body),
            },
            ty,
            span,
        );

        let (deps, _) = extract_dependencies(&expr, &Path::from_path_str("test"));
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], Path::from_path_str("plate"));
    }

    #[test]
    fn test_compile_execution_blocks_union_duplicates() {
        use crate::ast::{Assertion, RoleData};
        use crate::foundation::{AssertionSeverity, KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");
        let path_a = Path::from_path_str("signal.a");

        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });

        let mut node = Node::new(path, span, RoleData::Signal, ());

        // Block reads 'signal.a'
        node.execution_blocks.push((
            "resolve".to_string(),
            BlockBody::TypedExpression(TypedExpr::new(
                ExprKind::Signal(path_a.clone()),
                ty.clone(),
                span,
            )),
        ));

        // Assertion also reads 'signal.a'
        node.assertions.push(Assertion::new(
            TypedExpr::new(ExprKind::Signal(path_a.clone()), ty, span),
            None,
            AssertionSeverity::Error,
            span,
        ));

        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        compile_execution_blocks(&mut node, &ctx).unwrap();

        // Verify union has only one entry
        assert_eq!(node.reads.len(), 1);
        assert_eq!(node.reads[0], path_a);
    }

    #[test]
    fn test_compile_execution_blocks_multiple_assertions() {
        use crate::ast::{Assertion, RoleData};
        use crate::foundation::{AssertionSeverity, KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");
        let path_1 = Path::from_path_str("signal.1");
        let path_2 = Path::from_path_str("signal.2");

        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });

        let mut node = Node::new(path, span, RoleData::Signal, ());

        node.assertions.push(Assertion::new(
            TypedExpr::new(ExprKind::Signal(path_1.clone()), ty.clone(), span),
            None,
            AssertionSeverity::Error,
            span,
        ));

        node.assertions.push(Assertion::new(
            TypedExpr::new(ExprKind::Signal(path_2.clone()), ty, span),
            None,
            AssertionSeverity::Error,
            span,
        ));

        let registry = KernelRegistry::global();
        let signal_types = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let type_table = crate::resolve::types::TypeTable::new();
        let ctx = make_test_context(
            &type_table,
            &registry,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        compile_execution_blocks(&mut node, &ctx).unwrap();

        // Verify both assertion dependencies are in node.reads
        assert_eq!(node.reads.len(), 2);
        assert!(node.reads.contains(&path_1));
        assert!(node.reads.contains(&path_2));
    }

    #[test]
    fn test_extract_dependencies_field_access_member() {
        use crate::foundation::TypeId;

        let span = test_span();
        let scalar_ty = scalar_type();
        let plate_ty_id = TypeId::from("plate");
        let plate_ty = Type::User(plate_ty_id.clone());

        // Local variable 'p' of type 'plate'
        let object = TypedExpr::new(ExprKind::Local("p".to_string()), plate_ty, span);

        // p.mass
        let expr = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "mass".to_string(),
            },
            scalar_ty,
            span,
        );

        let (deps, _) = extract_dependencies(&expr, &Path::from_path_str("test"));
        assert_eq!(deps.len(), 1);
        // Member dependency 'plate.mass' should be captured
        assert_eq!(deps[0], Path::from_path_str("plate.mass"));
    }

    #[test]
    fn test_extract_dependencies_config_const() {
        let span = test_span();
        let ty = scalar_type();
        let config_path = Path::from_path_str("config.max_temp");
        let const_path = Path::from_path_str("const.PI");

        let config_expr = TypedExpr::new(ExprKind::Config(config_path.clone()), ty.clone(), span);
        let const_expr = TypedExpr::new(ExprKind::Const(const_path.clone()), ty, span);

        let (deps_config, _) = extract_dependencies(&config_expr, &Path::from_path_str("test"));
        assert_eq!(deps_config.len(), 1);
        assert_eq!(deps_config[0], config_path);

        let (deps_const, _) = extract_dependencies(&const_expr, &Path::from_path_str("test"));
        assert_eq!(deps_const.len(), 1);
        assert_eq!(deps_const[0], const_path);
    }

    #[test]
    fn test_prev_field_extraction_fix() {
        use crate::foundation::TypeId;

        let span = test_span();
        let scalar_ty = scalar_type();
        let plate_ty_id = TypeId::from("plate");
        let plate_ty = Type::User(plate_ty_id.clone());

        // prev object of type 'plate'
        let object = TypedExpr::new(ExprKind::Prev, plate_ty, span);

        // prev.mass
        let expr = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "mass".to_string(),
            },
            scalar_ty,
            span,
        );

        let node_path = Path::from_path_str("plate.mass");
        let (reads, temporal_reads) = extract_dependencies(&expr, &node_path);

        // Should NOT have causal reads (prev is temporal)
        assert!(
            reads.is_empty(),
            "Temporal field access should not create causal dependency, got: {:?}",
            reads
        );

        // Should have temporal read of self
        assert_eq!(temporal_reads.len(), 1);
        assert_eq!(temporal_reads[0], node_path);
    }
}
