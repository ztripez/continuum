//! Tests for execution block compilation pass.

use crate::error::ErrorKind;
use crate::resolve::blocks::{
    compile_execution_blocks, compile_statements, parse_phase_name, validate_phase_for_role,
};
use crate::resolve::dependencies::{extract_dependencies, extract_stmt_dependencies};
use crate::resolve::expr_typing::TypingContext;
use continuum_cdsl_ast::foundation::{KernelType, Path, Phase, Shape, Span, Type, Unit};
use continuum_cdsl_ast::{
    BlockBody, ExecutionBody, Expr, ExprKind, KernelRegistry, Node, RoleId, Stmt, TypedExpr,
    TypedStmt, UntypedKind,
};
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
    function_table: &'a HashMap<Path, continuum_cdsl_ast::FunctionDecl>,
    signal_types: &'a HashMap<Path, Type>,
    field_types: &'a HashMap<Path, Type>,
    config_types: &'a HashMap<Path, Type>,
    const_types: &'a HashMap<Path, Type>,
) -> TypingContext<'a> {
    TypingContext::new(
        type_table,
        registry,
        function_table,
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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();

    let ctx = TypingContext {
        type_table: &type_table,
        kernel_registry: &registry,
        function_table: &function_table,
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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();

    let ctx = TypingContext {
        type_table: &type_table,
        kernel_registry: &registry,
        function_table: &function_table,
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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();

    let ctx = TypingContext {
        type_table: &type_table,
        kernel_registry: &registry,
        function_table: &function_table,
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
    assert!(errors[0]
        .message
        .contains("target signal 'signal.target' has type"));
}

#[test]
fn test_compile_statements_undefined_signal() {
    let registry = KernelRegistry::global();
    let signal_types = HashMap::new(); // Empty
    let field_types = HashMap::new();
    let config_types = HashMap::new();
    let const_types = HashMap::new();
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();

    let ctx = TypingContext {
        type_table: &type_table,
        kernel_registry: &registry,
        function_table: &function_table,
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
    assert!(errors[0]
        .message
        .contains("signal 'signal.missing' not found"));
}

#[test]
fn test_compile_statements_phase_boundary_violation() {
    let registry = KernelRegistry::global();
    let mut signal_types = HashMap::new();
    signal_types.insert(Path::from_path_str("signal.target"), scalar_type());

    let field_types = HashMap::new();
    let config_types = HashMap::new();
    let const_types = HashMap::new();
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();

    let ctx = TypingContext {
        type_table: &type_table,
        kernel_registry: &registry,
        function_table: &function_table,
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
        phase: Some(Phase::Resolve), // Signal assignment is invalid in Resolve phase
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
    assert!(errors[0]
        .message
        .contains("cannot be assigned in Resolve phase"));
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
    use continuum_cdsl_ast::KernelId;

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
    use continuum_cdsl_ast::foundation::{KernelType, Shape, Unit};
    use continuum_cdsl_ast::RoleData;

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::foundation::{KernelType, Shape, Unit};
    use continuum_cdsl_ast::RoleData;

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::foundation::{KernelType, Shape, Unit};
    use continuum_cdsl_ast::RoleData;

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::foundation::{AssertionSeverity, KernelType, Shape, Unit};
    use continuum_cdsl_ast::{Assertion, RoleData};

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::{Expr, RoleData, UntypedKind};

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::foundation::{AggregateOp, EntityId};

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
    use continuum_cdsl_ast::foundation::EntityId;

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
    use continuum_cdsl_ast::foundation::{AssertionSeverity, KernelType, Shape, Unit};
    use continuum_cdsl_ast::{Assertion, RoleData};

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::foundation::{AssertionSeverity, KernelType, Shape, Unit};
    use continuum_cdsl_ast::{Assertion, RoleData};

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
    let function_table = HashMap::new();
    let type_table = crate::resolve::types::TypeTable::new();
    let ctx = make_test_context(
        &type_table,
        &registry,
        &function_table,
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
    use continuum_cdsl_ast::foundation::TypeId;

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
    // Config and Const are intentionally excluded from the signal dependency graph.
    // They are resolved statically during Configure phase and cannot participate
    // in signal resolution cycles (see dependencies.rs lines 52-57).
    let span = test_span();
    let ty = scalar_type();
    let config_path = Path::from_path_str("config.max_temp");
    let const_path = Path::from_path_str("const.PI");

    let config_expr = TypedExpr::new(ExprKind::Config(config_path.clone()), ty.clone(), span);
    let const_expr = TypedExpr::new(ExprKind::Const(const_path.clone()), ty, span);

    let (deps_config, _) = extract_dependencies(&config_expr, &Path::from_path_str("test"));
    assert_eq!(deps_config.len(), 0); // Config not tracked as dependency

    let (deps_const, _) = extract_dependencies(&const_expr, &Path::from_path_str("test"));
    assert_eq!(deps_const.len(), 0); // Const not tracked as dependency
}

#[test]
fn test_prev_field_extraction_fix() {
    use continuum_cdsl_ast::foundation::TypeId;

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
