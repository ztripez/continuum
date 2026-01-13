//! Tests for SSA IR lowering and validation.

use continuum_foundation::SignalId;

use crate::{BinaryOpIr, CompiledExpr, UnaryOpIr};

use super::{BlockId, SsaInstruction, Terminator, lower_to_ssa, validate_ssa};

#[test]
fn test_lower_literal() {
    let expr = CompiledExpr::Literal(42.0, None);
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    assert_eq!(ssa.vreg_count, 1);

    let block = &ssa.blocks[0];
    assert_eq!(block.instructions.len(), 1);
    assert!(matches!(
        &block.instructions[0],
        SsaInstruction::LoadConst { value, .. } if *value == 42.0
    ));
    assert!(matches!(&block.terminator, Some(Terminator::Return(_))));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_prev() {
    let expr = CompiledExpr::Prev;
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[0],
        SsaInstruction::LoadPrev { .. }
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_dt_raw() {
    let expr = CompiledExpr::DtRaw;
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[0],
        SsaInstruction::LoadDt { .. }
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_signal() {
    let expr = CompiledExpr::Signal(SignalId::from("temperature"));
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[0],
        SsaInstruction::LoadSignal { signal, .. } if signal.to_string() == "temperature"
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_binary_add() {
    // prev + 1.0
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Add,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Literal(1.0, None)),
    };
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    assert_eq!(ssa.vreg_count, 3); // prev, literal, add result

    let block = &ssa.blocks[0];
    assert_eq!(block.instructions.len(), 3);

    // Check order: prev, const, add
    assert!(matches!(
        &block.instructions[0],
        SsaInstruction::LoadPrev { .. }
    ));
    assert!(matches!(
        &block.instructions[1],
        SsaInstruction::LoadConst { .. }
    ));
    assert!(matches!(
        &block.instructions[2],
        SsaInstruction::BinOp {
            op: BinaryOpIr::Add,
            ..
        }
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_unary_neg() {
    let expr = CompiledExpr::Unary {
        op: UnaryOpIr::Neg,
        operand: Box::new(CompiledExpr::Literal(5.0, None)),
    };
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    assert_eq!(ssa.vreg_count, 2); // literal, neg result

    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[1],
        SsaInstruction::UnaryOp {
            op: UnaryOpIr::Neg,
            ..
        }
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_nested_binary() {
    // (prev + 1.0) * 2.0
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Mul,
        left: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0, None)),
        }),
        right: Box::new(CompiledExpr::Literal(2.0, None)),
    };
    let ssa = lower_to_ssa(&expr);

    // Should have: prev, 1.0, add, 2.0, mul
    assert_eq!(ssa.vreg_count, 5);

    let block = &ssa.blocks[0];
    assert_eq!(block.instructions.len(), 5);

    // Verify the binary ops are in correct order
    assert!(matches!(
        &block.instructions[2],
        SsaInstruction::BinOp {
            op: BinaryOpIr::Add,
            ..
        }
    ));
    assert!(matches!(
        &block.instructions[4],
        SsaInstruction::BinOp {
            op: BinaryOpIr::Mul,
            ..
        }
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_if_expression() {
    // if prev > 0 then 1.0 else 0.0
    let expr = CompiledExpr::If {
        condition: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Gt,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(0.0, None)),
        }),
        then_branch: Box::new(CompiledExpr::Literal(1.0, None)),
        else_branch: Box::new(CompiledExpr::Literal(0.0, None)),
    };
    let ssa = lower_to_ssa(&expr);

    // Should have 4 blocks: entry, then, else, merge
    assert_eq!(ssa.blocks.len(), 4);

    // Entry block should have branch terminator
    let entry = &ssa.blocks[0];
    assert!(matches!(
        &entry.terminator,
        Some(Terminator::Branch { then_block, else_block, .. })
        if *then_block == BlockId(1) && *else_block == BlockId(2)
    ));

    // Then and else blocks should jump to merge
    assert!(matches!(
        &ssa.blocks[1].terminator,
        Some(Terminator::Jump(BlockId(3)))
    ));
    assert!(matches!(
        &ssa.blocks[2].terminator,
        Some(Terminator::Jump(BlockId(3)))
    ));

    // Merge block should have phi node
    let merge = &ssa.blocks[3];
    assert!(matches!(&merge.instructions[0], SsaInstruction::Phi { .. }));
    assert!(matches!(&merge.terminator, Some(Terminator::Return(_))));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_let_binding() {
    // let x = 1.0 in x + x
    let expr = CompiledExpr::Let {
        name: "x".to_string(),
        value: Box::new(CompiledExpr::Literal(1.0, None)),
        body: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Local("x".to_string())),
            right: Box::new(CompiledExpr::Local("x".to_string())),
        }),
    };
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    // Should reuse the same register for both references to x
    // So: const(1.0), add(reg0, reg0)
    assert_eq!(ssa.vreg_count, 2);

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_kernel_call() {
    // kernel.sqrt(prev)
    let expr = CompiledExpr::KernelCall {
        function: "sqrt".to_string(),
        args: vec![CompiledExpr::Prev],
    };
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[1],
        SsaInstruction::KernelCall { function, args, .. }
        if function == "sqrt" && args.len() == 1
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_kernel_call_integrate() {
    // kernel.integrate(prev, rate)
    let expr = CompiledExpr::KernelCall {
        function: "integrate".to_string(),
        args: vec![CompiledExpr::Prev, CompiledExpr::Literal(1.0, None)],
    };
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[2],
        SsaInstruction::KernelCall { function, .. } if function == "integrate"
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_lower_self_field() {
    let expr = CompiledExpr::SelfField("mass".to_string());
    let ssa = lower_to_ssa(&expr);

    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];
    assert!(matches!(
        &block.instructions[0],
        SsaInstruction::SelfField { field, .. } if field == "mass"
    ));

    assert!(validate_ssa(&ssa).is_ok());
}

#[test]
fn test_pretty_print() {
    // prev + signal.heat * 0.5
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Add,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Mul,
            left: Box::new(CompiledExpr::Signal(SignalId::from("heat"))),
            right: Box::new(CompiledExpr::Literal(0.5, None)),
        }),
    };
    let ssa = lower_to_ssa(&expr);
    let output = ssa.pretty_print();

    assert!(output.contains("block0:"));
    assert!(output.contains("LoadPrev"));
    assert!(output.contains("LoadSignal(heat)"));
    assert!(output.contains("LoadConst(0.5)"));
    assert!(output.contains("Mul"));
    assert!(output.contains("Add"));
    assert!(output.contains("Return"));
}

#[test]
fn test_complex_expression() {
    // clamp(prev + collected - signal.stress * dt, 0.0, 1.0)
    // Represented as: kernel.clamp(prev + collected - signal.stress * dt, 0.0, 1.0)
    let expr = CompiledExpr::KernelCall {
        function: "clamp".to_string(),
        args: vec![
            CompiledExpr::Binary {
                op: BinaryOpIr::Sub,
                left: Box::new(CompiledExpr::Binary {
                    op: BinaryOpIr::Add,
                    left: Box::new(CompiledExpr::Prev),
                    right: Box::new(CompiledExpr::Collected),
                }),
                right: Box::new(CompiledExpr::Binary {
                    op: BinaryOpIr::Mul,
                    left: Box::new(CompiledExpr::Signal(SignalId::from("stress"))),
                    right: Box::new(CompiledExpr::DtRaw),
                }),
            },
            CompiledExpr::Literal(0.0, None),
            CompiledExpr::Literal(1.0, None),
        ],
    };
    let ssa = lower_to_ssa(&expr);

    // Verify structure
    assert_eq!(ssa.blocks.len(), 1);
    let block = &ssa.blocks[0];

    // Count instruction types
    let load_prev_count = block
        .instructions
        .iter()
        .filter(|i| matches!(i, SsaInstruction::LoadPrev { .. }))
        .count();
    let load_collected_count = block
        .instructions
        .iter()
        .filter(|i| matches!(i, SsaInstruction::LoadCollected { .. }))
        .count();
    let load_signal_count = block
        .instructions
        .iter()
        .filter(|i| matches!(i, SsaInstruction::LoadSignal { .. }))
        .count();
    let binop_count = block
        .instructions
        .iter()
        .filter(|i| matches!(i, SsaInstruction::BinOp { .. }))
        .count();
    let kernel_call_count = block
        .instructions
        .iter()
        .filter(|i| matches!(i, SsaInstruction::KernelCall { .. }))
        .count();

    assert_eq!(load_prev_count, 1);
    assert_eq!(load_collected_count, 1);
    assert_eq!(load_signal_count, 1);
    assert_eq!(binop_count, 3); // add, mul, sub
    assert_eq!(kernel_call_count, 1);

    assert!(validate_ssa(&ssa).is_ok());
}
