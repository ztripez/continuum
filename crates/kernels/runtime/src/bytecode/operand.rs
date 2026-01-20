//! Operand encoding for bytecode instructions.
//!
//! Operands represent the data that opcodes operate on: signal slots, entity indices,
//! local bindings, temporal markers, block references, etc.

use continuum_cdsl::ast::expr::AggregateOp;
use continuum_foundation::{EntityId, Path, Value};
use serde::{Deserialize, Serialize};

/// Slot identifier for stack/register-based execution.
///
/// Slots represent storage locations for values during bytecode execution:
/// - Signal values (current or previous tick)
/// - Local let bindings
/// - Temporary computation results
/// - Entity iteration state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Slot(
    /// Zero-based slot index within a bytecode block.
    pub u32,
);

impl Slot {
    /// Create a new slot
    ///
    /// Slots are zero-based indices into the block's slot table.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the slot ID
    ///
    /// The ID is a zero-based index into the slot table.
    pub fn id(self) -> u32 {
        self.0
    }
}

/// Block identifier for nested bytecode blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(
    /// Zero-based block index within a bytecode program.
    pub u32,
);

impl BlockId {
    /// Create a new block id
    ///
    /// Block ids are zero-based indices into the program's block list.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the block id
    ///
    /// The ID is a zero-based index into the program's block list.
    pub fn id(self) -> u32 {
        self.0
    }
}

/// Operand types for bytecode instructions.
///
/// Operands encode all data references needed by opcodes: where to read from,
/// where to write to, what entities to iterate over, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operand {
    /// Stack slot (local, temporary, or signal)
    Slot(Slot),

    /// Immediate literal value
    Literal(Value),

    /// Signal path reference (resolved at compile time to slot)
    Signal(Path),

    /// Field path reference (observer only)
    Field(Path),

    /// Entity ID for iteration
    Entity(EntityId),

    /// Config value reference
    Config(Path),

    /// Const value reference
    Const(Path),

    /// String operand (field names, labels)
    String(String),

    /// Block reference for nested execution
    Block(BlockId),

    /// Aggregate operation
    AggregateOp(AggregateOp),
}

/// Helper to expect a specific operand type.
fn expect_operand<T>(
    operand: &Operand,
    expected: &'static str,
    map: impl FnOnce(&Operand) -> Option<T>,
) -> Result<T, crate::bytecode::runtime::ExecutionError> {
    map(operand).ok_or_else(
        || crate::bytecode::runtime::ExecutionError::InvalidOperand {
            message: format!("Expected {expected} operand"),
        },
    )
}

/// Decode a slot operand.
pub fn operand_slot(operand: &Operand) -> Result<Slot, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Slot", |op| {
        if let Operand::Slot(slot) = op {
            Some(*slot)
        } else {
            None
        }
    })
}

/// Decode a block operand.
pub fn operand_block(
    operand: &Operand,
) -> Result<BlockId, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Block", |op| {
        if let Operand::Block(block) = op {
            Some(*block)
        } else {
            None
        }
    })
}

/// Decode a literal value operand.
pub fn operand_literal(
    operand: &Operand,
) -> Result<Value, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Literal", |op| {
        if let Operand::Literal(value) = op {
            Some(value.clone())
        } else {
            None
        }
    })
}

/// Decode a literal integer as usize.
pub fn operand_usize(operand: &Operand) -> Result<usize, crate::bytecode::runtime::ExecutionError> {
    match operand_literal(operand)? {
        Value::Integer(value) => usize::try_from(value).map_err(|_| {
            crate::bytecode::runtime::ExecutionError::InvalidOperand {
                message: format!("Invalid integer literal: {value}"),
            }
        }),
        Value::Scalar(value) => {
            if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
                return Err(crate::bytecode::runtime::ExecutionError::InvalidOperand {
                    message: format!("Invalid scalar integer literal: {value}"),
                });
            }
            if value > i64::MAX as f64 {
                return Err(crate::bytecode::runtime::ExecutionError::InvalidOperand {
                    message: format!("Invalid scalar integer literal: {value}"),
                });
            }
            let int_value = value as i64;
            usize::try_from(int_value).map_err(|_| {
                crate::bytecode::runtime::ExecutionError::InvalidOperand {
                    message: format!("Invalid scalar integer literal: {value}"),
                }
            })
        }
        other => Err(crate::bytecode::runtime::ExecutionError::InvalidOperand {
            message: format!("Expected integer literal, got {other:?}"),
        }),
    }
}

/// Decode a string operand.
pub fn operand_string(
    operand: &Operand,
) -> Result<String, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "String", |op| {
        if let Operand::String(value) = op {
            Some(value.clone())
        } else {
            None
        }
    })
}

/// Decode a signal path operand.
pub fn operand_signal_path(
    operand: &Operand,
) -> Result<Path, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Signal", |op| {
        if let Operand::Signal(path) = op {
            Some(path.clone())
        } else {
            None
        }
    })
}

/// Decode a field path operand.
pub fn operand_field_path(
    operand: &Operand,
) -> Result<Path, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Field", |op| {
        if let Operand::Field(path) = op {
            Some(path.clone())
        } else {
            None
        }
    })
}

/// Decode a config path operand.
pub fn operand_config_path(
    operand: &Operand,
) -> Result<Path, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Config", |op| {
        if let Operand::Config(path) = op {
            Some(path.clone())
        } else {
            None
        }
    })
}

/// Decode a const path operand.
pub fn operand_const_path(
    operand: &Operand,
) -> Result<Path, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Const", |op| {
        if let Operand::Const(path) = op {
            Some(path.clone())
        } else {
            None
        }
    })
}

/// Decode an entity ID operand.
pub fn operand_entity(
    operand: &Operand,
) -> Result<continuum_foundation::EntityId, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Entity", |op| {
        if let Operand::Entity(entity) = op {
            Some(entity.clone())
        } else {
            None
        }
    })
}

/// Decode an aggregate operation operand.
pub fn operand_aggregate_op(
    operand: &Operand,
) -> Result<AggregateOp, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "AggregateOp", |op| {
        if let Operand::AggregateOp(agg) = op {
            Some(*agg)
        } else {
            None
        }
    })
}

/// Helper for dynamic field access during execution.
pub fn field_access(
    object: &Value,
    field: &str,
) -> Result<Value, crate::bytecode::runtime::ExecutionError> {
    match object {
        Value::Map(fields) => fields
            .iter()
            .find(|(name, _)| name == field)
            .map(|(_, value)| value.clone())
            .ok_or_else(
                || crate::bytecode::runtime::ExecutionError::InvalidOperand {
                    message: format!("Unknown field {field}"),
                },
            ),
        other => other.component(field).map(Value::Scalar).ok_or_else(|| {
            crate::bytecode::runtime::ExecutionError::InvalidOperand {
                message: format!("Cannot access field {field} on {other:?}"),
            }
        }),
    }
}
