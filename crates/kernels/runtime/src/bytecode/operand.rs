//! Operand encoding for bytecode instructions.
//!
//! Operands represent the data that opcodes operate on: signal slots, entity indices,
//! local bindings, temporal markers, block references, etc.

use continuum_cdsl::ast::AggregateOp;
use continuum_foundation::{EntityId, Path, Value};
use serde::{Deserialize, Serialize};

/// Slot identifier for stack/register-based execution.
///
/// Slots represent storage locations for values during bytecode execution:
/// - Signal values (current or previous tick)
/// - Local let bindings
/// - Temporary computation results
/// - Entity iteration state
///
/// Each slot corresponds to a fixed memory location in the executor's slot table,
/// allocated during compilation to ensure deterministic access.
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
///
/// A `BlockId` refers to a specific [`super::program::BytecodeBlock`] within
/// a [`super::program::BytecodeProgram`]. It is used to implement control flow
/// operations like `Aggregate` and `Fold` where a sub-computation must be
/// executed for multiple instances.
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
/// Operands provide the static data required by opcodes. They encode references
/// to storage locations (slots), identifiers (paths, entity IDs), or immediate values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operand {
    /// A reference to a storage slot (local binding, signal, or temporary).
    Slot(Slot),

    /// An immediate constant value embedded in the instruction.
    Literal(Value),

    /// A path to a simulation signal.
    Signal(Path),

    /// A path to an observer field.
    Field(Path),

    /// An entity identifier used for spawning, destruction, or iteration.
    Entity(EntityId),

    /// A path to a configuration parameter.
    Config(Path),

    /// A path to a global simulation constant.
    Const(Path),

    /// A raw string used for field names or lookup keys.
    String(String),

    /// A reference to another bytecode block within the same program.
    Block(BlockId),

    /// An identifier for an aggregation operation (sum, min, max, etc.).
    AggregateOp(AggregateOp),
}

/// Helper to expect a specific operand type.
///
/// # Parameters
/// - `operand`: The operand to check.
/// - `expected`: Human-readable name of the expected operand variant (for errors).
/// - `map`: Closure that attempts to extract the desired type from the operand.
///
/// # Returns
/// The extracted value of type `T`.
///
/// # Errors
/// Returns [`ExecutionError::InvalidOperand`] if the mapping fails.
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
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Slot`].
pub fn operand_slot(operand: &Operand) -> Result<Slot, crate::bytecode::runtime::ExecutionError> {
    expect_operand(operand, "Slot", |op| {
        if let Operand::Slot(slot) = op {
            Some(*slot)
        } else {
            None
        }
    })
}

/// Decode a block identifier operand.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Block`].
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
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Literal`].
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

/// Decodes a literal integer value as a `usize`.
///
/// Supports both [`Value::Integer`] and [`Value::Scalar`] (if the scalar is a finite,
/// non-negative integer). Used for argument counts and indices.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the value is not a valid integer or
/// cannot fit into a `usize`.
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
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::String`].
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
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Signal`].
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
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Field`].
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

/// Decode a configuration path operand.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Config`].
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

/// Decode a simulation constant path operand.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Const`].
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

/// Decode an entity identifier operand.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not a [`Operand::Entity`].
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
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the operand is not an [`Operand::AggregateOp`].
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

/// Helper for performing dynamic field or component access during execution.
///
/// Supports [`Value::Map`] for named field lookup and other [`Value`] types
/// for component access (e.g., .x, .y on vectors).
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidOperand`] if the field does not exist or the
/// type does not support field access.
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
