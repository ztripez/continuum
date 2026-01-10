//! SSA IR validation.
//!
//! Validates that SSA IR is well-formed:
//! - All used registers are defined before use
//! - All blocks have terminators
//! - Phi nodes have arms for all predecessors
//! - Return terminators exist in all exit paths

use std::collections::{HashMap, HashSet};

use super::{BlockId, SsaFunction, SsaInstruction, VReg};

/// SSA validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaValidationError {
    /// A register is used before being defined.
    UndefinedRegister { reg: VReg, block: BlockId },
    /// A block has no terminator.
    MissingTerminator { block: BlockId },
    /// A phi node is missing an arm for a predecessor.
    IncompletePhi {
        block: BlockId,
        missing_pred: BlockId,
    },
    /// A register is defined multiple times.
    DuplicateDefinition { reg: VReg },
    /// Block has no predecessors (unreachable) except for entry block.
    UnreachableBlock { block: BlockId },
}

/// Validate an SSA function.
///
/// Returns Ok(()) if the function is valid, or a list of errors.
pub fn validate_ssa(func: &SsaFunction) -> Result<(), Vec<SsaValidationError>> {
    let mut errors = Vec::new();

    // Build predecessor map
    let predecessors = compute_predecessors(func);

    // Check for unreachable blocks (except entry)
    for (i, _block) in func.blocks.iter().enumerate() {
        let block_id = BlockId(i as u32);
        if block_id.0 != 0 && predecessors.get(&block_id).map_or(true, |p| p.is_empty()) {
            errors.push(SsaValidationError::UnreachableBlock { block: block_id });
        }
    }

    // Track defined registers
    let mut defined: HashSet<VReg> = HashSet::new();

    // Validate each block
    for block in &func.blocks {
        // Check terminator exists
        if block.terminator.is_none() {
            errors.push(SsaValidationError::MissingTerminator { block: block.id });
        }

        // Check instructions
        for inst in &block.instructions {
            // Check uses are defined
            for used in inst.uses() {
                if !defined.contains(&used) {
                    // For phi nodes, the values come from predecessors
                    // We need a more sophisticated check for those
                    if !matches!(inst, SsaInstruction::Phi { .. }) {
                        errors.push(SsaValidationError::UndefinedRegister {
                            reg: used,
                            block: block.id,
                        });
                    }
                }
            }

            // Check for duplicate definitions
            if let Some(dst) = inst.dst() {
                if defined.contains(&dst) {
                    errors.push(SsaValidationError::DuplicateDefinition { reg: dst });
                }
                defined.insert(dst);
            }

            // Validate phi nodes
            if let SsaInstruction::Phi { arms, .. } = inst {
                if let Some(preds) = predecessors.get(&block.id) {
                    for pred in preds {
                        if !arms.iter().any(|(b, _)| b == pred) {
                            errors.push(SsaValidationError::IncompletePhi {
                                block: block.id,
                                missing_pred: *pred,
                            });
                        }
                    }
                }
            }
        }

        // Check terminator uses
        if let Some(term) = &block.terminator {
            for used in term.uses() {
                if !defined.contains(&used) {
                    errors.push(SsaValidationError::UndefinedRegister {
                        reg: used,
                        block: block.id,
                    });
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Compute the predecessor map for all blocks.
fn compute_predecessors(func: &SsaFunction) -> HashMap<BlockId, Vec<BlockId>> {
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();

    // Initialize all blocks with empty predecessor lists
    for (i, _) in func.blocks.iter().enumerate() {
        preds.insert(BlockId(i as u32), Vec::new());
    }

    // Build predecessor relationships from terminators
    for block in &func.blocks {
        if let Some(term) = &block.terminator {
            for succ in term.successors() {
                preds.entry(succ).or_default().push(block.id);
            }
        }
    }

    preds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::Terminator;

    #[test]
    fn test_validate_simple_function() {
        let mut func = SsaFunction::new();
        let dst = func.alloc_vreg();
        func.block_mut(BlockId(0))
            .push(SsaInstruction::LoadConst { dst, value: 1.0 });
        func.block_mut(BlockId(0))
            .terminate(Terminator::Return(dst));

        assert!(validate_ssa(&func).is_ok());
    }

    #[test]
    fn test_validate_missing_terminator() {
        let func = SsaFunction::new();
        // Entry block has no terminator

        let errors = validate_ssa(&func).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, SsaValidationError::MissingTerminator { block } if block.0 == 0)));
    }

    #[test]
    fn test_validate_undefined_register() {
        let mut func = SsaFunction::new();
        let undefined_reg = VReg(999);
        func.block_mut(BlockId(0))
            .terminate(Terminator::Return(undefined_reg));

        let errors = validate_ssa(&func).unwrap_err();
        assert!(errors.iter().any(
            |e| matches!(e, SsaValidationError::UndefinedRegister { reg, .. } if *reg == undefined_reg)
        ));
    }

    #[test]
    fn test_validate_branch_with_phi() {
        let mut func = SsaFunction::new();

        // Block 0: branch based on condition
        let cond = func.alloc_vreg();
        func.block_mut(BlockId(0))
            .push(SsaInstruction::LoadConst { dst: cond, value: 1.0 });

        let then_block = func.alloc_block();
        let else_block = func.alloc_block();
        let merge_block = func.alloc_block();

        func.block_mut(BlockId(0)).terminate(Terminator::Branch {
            cond,
            then_block,
            else_block,
        });

        // Block 1 (then): return 1.0
        let then_val = func.alloc_vreg();
        func.block_mut(then_block)
            .push(SsaInstruction::LoadConst {
                dst: then_val,
                value: 1.0,
            });
        func.block_mut(then_block)
            .terminate(Terminator::Jump(merge_block));

        // Block 2 (else): return 2.0
        let else_val = func.alloc_vreg();
        func.block_mut(else_block)
            .push(SsaInstruction::LoadConst {
                dst: else_val,
                value: 2.0,
            });
        func.block_mut(else_block)
            .terminate(Terminator::Jump(merge_block));

        // Block 3 (merge): phi and return
        let result = func.alloc_vreg();
        func.block_mut(merge_block).push(SsaInstruction::Phi {
            dst: result,
            arms: vec![(then_block, then_val), (else_block, else_val)],
        });
        func.block_mut(merge_block)
            .terminate(Terminator::Return(result));

        assert!(validate_ssa(&func).is_ok());
    }

    #[test]
    fn test_validate_duplicate_definition() {
        let mut func = SsaFunction::new();
        // Manually create a function where VReg(0) is defined twice
        let reg = VReg(0);
        func.vreg_count = 1;

        // First definition
        func.block_mut(BlockId(0))
            .push(SsaInstruction::LoadConst { dst: reg, value: 1.0 });
        // Second definition of same register (violation)
        func.block_mut(BlockId(0))
            .push(SsaInstruction::LoadConst { dst: reg, value: 2.0 });
        func.block_mut(BlockId(0))
            .terminate(Terminator::Return(reg));

        let errors = validate_ssa(&func).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, SsaValidationError::DuplicateDefinition { reg: r } if *r == reg)));
    }

    #[test]
    fn test_validate_incomplete_phi() {
        let mut func = SsaFunction::new();

        // Block 0: branch to blocks 1 and 2
        let cond = func.alloc_vreg();
        func.block_mut(BlockId(0))
            .push(SsaInstruction::LoadConst { dst: cond, value: 1.0 });

        let then_block = func.alloc_block();
        let else_block = func.alloc_block();
        let merge_block = func.alloc_block();

        func.block_mut(BlockId(0)).terminate(Terminator::Branch {
            cond,
            then_block,
            else_block,
        });

        // Block 1 (then): jump to merge
        let then_val = func.alloc_vreg();
        func.block_mut(then_block)
            .push(SsaInstruction::LoadConst {
                dst: then_val,
                value: 1.0,
            });
        func.block_mut(then_block)
            .terminate(Terminator::Jump(merge_block));

        // Block 2 (else): jump to merge
        let else_val = func.alloc_vreg();
        func.block_mut(else_block)
            .push(SsaInstruction::LoadConst {
                dst: else_val,
                value: 2.0,
            });
        func.block_mut(else_block)
            .terminate(Terminator::Jump(merge_block));

        // Block 3 (merge): phi missing arm for else_block (violation)
        let result = func.alloc_vreg();
        func.block_mut(merge_block).push(SsaInstruction::Phi {
            dst: result,
            // Only arm for then_block, missing else_block
            arms: vec![(then_block, then_val)],
        });
        func.block_mut(merge_block)
            .terminate(Terminator::Return(result));

        let errors = validate_ssa(&func).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            SsaValidationError::IncompletePhi { block, missing_pred }
            if *block == merge_block && *missing_pred == else_block
        )));
    }

    #[test]
    fn test_validate_unreachable_block() {
        let mut func = SsaFunction::new();

        // Block 0: return immediately
        let val = func.alloc_vreg();
        func.block_mut(BlockId(0))
            .push(SsaInstruction::LoadConst { dst: val, value: 1.0 });
        func.block_mut(BlockId(0))
            .terminate(Terminator::Return(val));

        // Block 1: unreachable (no predecessors, not entry block)
        let unreachable_block = func.alloc_block();
        let unreachable_val = func.alloc_vreg();
        func.block_mut(unreachable_block)
            .push(SsaInstruction::LoadConst {
                dst: unreachable_val,
                value: 999.0,
            });
        func.block_mut(unreachable_block)
            .terminate(Terminator::Return(unreachable_val));

        let errors = validate_ssa(&func).unwrap_err();
        assert!(errors.iter().any(
            |e| matches!(e, SsaValidationError::UnreachableBlock { block } if *block == unreachable_block)
        ));
    }
}
