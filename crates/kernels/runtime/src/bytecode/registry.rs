//! Opcode registry linking metadata to handlers.

use std::sync::OnceLock;

use continuum_foundation::Phase;

use super::handlers::{
    handle_aggregate, handle_build_struct, handle_build_vector, handle_call_kernel, handle_destroy,
    handle_dup, handle_emit, handle_emit_field, handle_field_access, handle_fold, handle_load,
    handle_load_config, handle_load_const, handle_load_current, handle_load_dt, handle_load_field,
    handle_load_inputs, handle_load_other, handle_load_payload, handle_load_prev, handle_load_self,
    handle_load_signal, handle_noop, handle_pop, handle_push_literal, handle_spawn, handle_store,
    Handler,
};
use super::opcode::{OpcodeKind, OpcodeMetadata, OperandCount};

/// Opcode registry entry.
#[derive(Debug, Clone)]
pub struct OpcodeSpec {
    pub kind: OpcodeKind,
    pub metadata: OpcodeMetadata,
    pub handler: Handler,
}

/// Get all opcode specifications.
pub fn opcode_specs() -> &'static [OpcodeSpec] {
    static SPECS: OnceLock<Vec<OpcodeSpec>> = OnceLock::new();
    SPECS.get_or_init(|| build_specs())
}

/// Get metadata for an opcode kind.
pub fn metadata_for(kind: OpcodeKind) -> &'static OpcodeMetadata {
    opcode_specs()
        .iter()
        .find(|spec| spec.kind == kind)
        .map(|spec| &spec.metadata)
        .unwrap_or_else(|| panic!("Missing opcode metadata for {kind:?}"))
}

/// Get handler for an opcode kind.
pub fn handler_for(kind: OpcodeKind) -> Handler {
    opcode_specs()
        .iter()
        .find(|spec| spec.kind == kind)
        .map(|spec| spec.handler)
        .unwrap_or_else(|| panic!("Missing opcode handler for {kind:?}"))
}

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
        op!(Aggregate, OperandCount::Fixed(4), handle_aggregate),
        op!(Fold, OperandCount::Fixed(4), handle_fold),
        op!(FieldAccess, OperandCount::Fixed(1), handle_field_access),
        op!(LoadSignal, OperandCount::Fixed(1), handle_load_signal),
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
