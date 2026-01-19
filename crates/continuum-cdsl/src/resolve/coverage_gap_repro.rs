#[cfg(test)]
mod coverage_gap_tests {
    use super::*;
    use crate::ast::{Execution, ExecutionBody, ExprKind, Node, RoleData, TypedExpr};
    use crate::foundation::{KernelType, Path, Phase, Shape, Span, Type, Unit};

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

    #[test]
    fn test_reads_union_from_multiple_blocks() {
        let span = test_span();
        let path = Path::from_str("test.operator");
        let mut node = Node::new(path, span, RoleData::Operator, ());

        let path1 = Path::from_str("signal.a");
        let path2 = Path::from_str("signal.b");

        // Execution 1 reads signal.a
        let expr1 = TypedExpr::new(ExprKind::Signal(path1.clone()), scalar_type(), span);
        node.execution_blocks.push((
            "collect".to_string(),
            crate::ast::BlockBody::TypedExpression(expr1),
        ));

        // Execution 2 reads signal.b
        let expr2 = TypedExpr::new(ExprKind::Signal(path2.clone()), scalar_type(), span);
        node.execution_blocks.push((
            "resolve".to_string(),
            crate::ast::BlockBody::TypedExpression(expr2),
        ));

        // Compile
        compile_execution_blocks(&mut node).expect("Compilation failed");

        // Verify union and sorting
        assert_eq!(node.reads.len(), 2);
        assert_eq!(node.reads[0], path1);
        assert_eq!(node.reads[1], path2);
    }
}
