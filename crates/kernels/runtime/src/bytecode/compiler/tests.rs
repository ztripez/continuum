use super::*;
use continuum_cdsl::ast::expr::TypedExpr;
use continuum_cdsl::ast::{Execution, ExecutionBody, ExprKind};
use continuum_foundation::{Phase, Span, Type};

fn make_literal(value: f64) -> TypedExpr {
    TypedExpr {
        expr: ExprKind::Literal { value, unit: None },
        ty: Type::Bool,
        span: Span::new(0, 0, 0, 0),
    }
}

#[test]
fn test_compile_literal_expr() {
    let mut compiler = Compiler::new();
    let execution = Execution {
        name: "resolve".to_string(),
        phase: Phase::Resolve,
        body: ExecutionBody::Expr(make_literal(1.0)),
        reads: vec![],
        temporal_reads: vec![],
        emits: vec![],
        span: Span::new(0, 0, 0, 0),
    };

    let compiled = compiler.compile_execution(&execution).unwrap();
    let block = compiled.program.block(compiled.root).unwrap();
    assert_eq!(block.instructions.len(), 2);
    assert_eq!(block.instructions[0].kind, OpcodeKind::PushLiteral);
}
