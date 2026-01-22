//! Opcode registry linking metadata to handlers.

use std::sync::OnceLock;

use continuum_foundation::Phase;

use super::handlers::{
    handle_aggregate, handle_build_struct, handle_build_vector, handle_call_kernel, handle_destroy,
    handle_dup, handle_emit, handle_emit_field, handle_field_access, handle_filter, handle_fold,
    handle_load, handle_load_config, handle_load_const, handle_load_current, handle_load_dt,
    handle_load_entity, handle_load_field, handle_load_inputs, handle_load_other,
    handle_load_payload, handle_load_prev, handle_load_self, handle_load_signal, handle_nearest,
    handle_noop, handle_pop, handle_push_literal, handle_spawn, handle_store, handle_within,
    Handler,
};
use super::opcode::{OpcodeKind, OpcodeMetadata, OperandCount};

/// Metadata and handler specification for an opcode.
///
/// This structure links a specific [`OpcodeKind`] to its static metadata
/// (used for validation and compilation) and its runtime execution logic.
///
/// The opcode registry uses this specification to perform O(1) dispatch
/// during execution and to validate instruction correctness during compilation.
#[derive(Debug, Clone)]
pub struct OpcodeSpec {
    /// The specific opcode kind this specification covers.
    pub kind: OpcodeKind,
    /// Static metadata for validation and operand parsing.
    pub metadata: OpcodeMetadata,
    /// The runtime function responsible for executing this opcode.
    pub handler: Handler,
}

/// Retrieves the global list of all registered opcode specifications.
///
/// This is the primary source of truth for VM behavior. It is used by the
/// compiler to verify instruction shapes and by the executor to build its
/// jump table. The list is lazily initialized on the first call.
pub fn opcode_specs() -> &'static [OpcodeSpec] {
    static SPECS: OnceLock<Vec<OpcodeSpec>> = OnceLock::new();
    SPECS.get_or_init(|| build_specs())
}

/// Total number of opcodes in the system.
///
/// This constant must match the number of variants in [`OpcodeKind`]. It is used
/// to size the internal jump tables for O(1) metadata and handler lookups.
const OPCODE_COUNT: usize = 33;

/// Retrieves metadata for a specific opcode kind in O(1) time.
///
/// # Panics
///
/// Panics if the opcode kind has not been registered in the spec table.
pub fn metadata_for(kind: OpcodeKind) -> &'static OpcodeMetadata {
    static METADATA: OnceLock<[&'static OpcodeMetadata; OPCODE_COUNT]> = OnceLock::new();
    METADATA.get_or_init(|| {
        let mut table: [Option<&'static OpcodeMetadata>; OPCODE_COUNT] = [None; OPCODE_COUNT];
        for spec in opcode_specs() {
            table[spec.kind as usize] = unsafe { std::mem::transmute(&spec.metadata) };
        }
        std::array::from_fn(|index| {
            table[index]
                .unwrap_or_else(|| panic!("Missing opcode metadata for opcode index {}", index))
        })
    })[kind as usize]
}

/// Retrieves the execution handler for a specific opcode kind in O(1) time.
///
/// # Panics
///
/// Panics if the opcode kind has not been registered in the spec table.
pub fn handler_for(kind: OpcodeKind) -> Handler {
    static HANDLERS: OnceLock<[Handler; OPCODE_COUNT]> = OnceLock::new();
    HANDLERS.get_or_init(|| {
        let mut table: [Option<Handler>; OPCODE_COUNT] = [None; OPCODE_COUNT];
        for spec in opcode_specs() {
            table[spec.kind as usize] = Some(spec.handler);
        }
        std::array::from_fn(|index| {
            table[index]
                .unwrap_or_else(|| panic!("Missing opcode handler for opcode index {}", index))
        })
    })[kind as usize]
}

/// Master list of opcode specifications, linking kinds to metadata and handlers.
fn build_specs() -> Vec<OpcodeSpec> {
    use OpcodeKind::*;

    macro_rules! op {
        ($kind:ident, $count:expr, $handler:ident) => {
            OpcodeSpec {
                kind: $kind,
                metadata: OpcodeMetadata {
                    operand_count: $count,
                    has_effect: false,
                    allowed_phases: None,
                },
                handler: $handler,
            }
        };
        ($kind:ident, $count:expr, $effect:expr, $phases:expr, $handler:ident) => {
            OpcodeSpec {
                kind: $kind,
                metadata: OpcodeMetadata {
                    operand_count: $count,
                    has_effect: $effect,
                    allowed_phases: $phases,
                },
                handler: $handler,
            }
        };
    }

    vec![
        op!(PushLiteral, OperandCount::Fixed(1), handle_push_literal),
        op!(Load, OperandCount::Fixed(1), handle_load),
        op!(Store, OperandCount::Fixed(1), handle_store),
        op!(Dup, OperandCount::Fixed(0), handle_dup),
        op!(Pop, OperandCount::Fixed(0), handle_pop),
        op!(BuildVector, OperandCount::Fixed(1), handle_build_vector),
        op!(
            BuildStruct,
            OperandCount::Variable { min: 1, max: None },
            handle_build_struct
        ),
        op!(CallKernel, OperandCount::Fixed(1), handle_call_kernel),
        op!(Let, OperandCount::Fixed(1), handle_noop),
        op!(EndLet, OperandCount::Fixed(0), handle_noop),
        op!(LoadEntity, OperandCount::Fixed(1), handle_load_entity),
        op!(Filter, OperandCount::Fixed(2), handle_filter),
        op!(Nearest, OperandCount::Fixed(0), handle_nearest),
        op!(Within, OperandCount::Fixed(0), handle_within),
        op!(Aggregate, OperandCount::Fixed(3), handle_aggregate),
        op!(Fold, OperandCount::Fixed(3), handle_fold),
        op!(FieldAccess, OperandCount::Fixed(1), handle_field_access),
        op!(
            LoadSignal,
            OperandCount::Fixed(1),
            false,
            Some(&[
                Phase::Collect,
                Phase::Resolve,
                Phase::Fracture,
                Phase::Measure
            ]),
            handle_load_signal
        ),
        op!(LoadConfig, OperandCount::Fixed(1), handle_load_config),
        op!(LoadConst, OperandCount::Fixed(1), handle_load_const),
        op!(LoadPrev, OperandCount::Fixed(0), handle_load_prev),
        op!(
            LoadCurrent,
            OperandCount::Fixed(0),
            false,
            Some(&[Phase::Fracture, Phase::Measure]),
            handle_load_current
        ),
        op!(
            LoadInputs,
            OperandCount::Fixed(0),
            false,
            Some(&[Phase::Resolve]),
            handle_load_inputs
        ),
        op!(LoadDt, OperandCount::Fixed(0), handle_load_dt),
        op!(LoadSelf, OperandCount::Fixed(0), handle_load_self),
        op!(LoadOther, OperandCount::Fixed(0), handle_load_other),
        op!(LoadPayload, OperandCount::Fixed(0), handle_load_payload),
        op!(
            Emit,
            OperandCount::Fixed(1),
            true,
            Some(&[Phase::Collect]),
            handle_emit
        ),
        op!(
            EmitField,
            OperandCount::Fixed(1),
            true,
            Some(&[Phase::Measure]),
            handle_emit_field
        ),
        op!(
            Spawn,
            OperandCount::Fixed(1),
            true,
            Some(&[Phase::Fracture]),
            handle_spawn
        ),
        op!(
            Destroy,
            OperandCount::Fixed(1),
            true,
            Some(&[Phase::Fracture]),
            handle_destroy
        ),
        op!(
            LoadField,
            OperandCount::Fixed(1),
            false,
            Some(&[Phase::Measure]),
            handle_load_field
        ),
        op!(Return, OperandCount::Fixed(0), handle_noop),
    ]
}
