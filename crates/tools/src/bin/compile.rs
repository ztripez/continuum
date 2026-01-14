//!
//! Compile a Continuum world into a bytecode bundle.
//!
//! Usage: `compile <world-dir> [--out-dir DIR] [--output FILE]`

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process;
use tracing::{error, info, warn};

use continuum_compiler::ir::compile_to_bytecode;
use continuum_vm::BytecodeChunk;

#[derive(Parser, Debug)]
#[command(name = "compile")]
#[command(about = "Compile a Continuum world into a bytecode bundle")]
struct Args {
    /// Path to the World root directory
    world_dir: PathBuf,

    /// Output directory for the bytecode bundle
    #[arg(long = "out-dir", default_value = "build")]
    out_dir: PathBuf,

    /// Explicit output file path
    #[arg(long = "output")]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BytecodeBundle {
    version: u32,
    world: String,
    entries: Vec<BytecodeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BytecodeEntry {
    kind: BytecodeKind,
    id: String,
    label: Option<String>,
    bytecode: BytecodeChunk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum BytecodeKind {
    SignalResolve,
    SignalResolveComponent,
    SignalWarmup,
    SignalAssertion,
    FieldMeasure,
    OperatorBody,
    OperatorAssertion,
    MemberInitial,
    MemberResolve,
    MemberAssertion,
    EraTransition,
    FractureCondition,
    FractureEmit,
    ChronicleCondition,
    ChronicleField,
}

fn main() {
    continuum_tools::init_logging();

    let args = Args::parse();

    let compile_result = continuum_compiler::compile_from_dir_result(&args.world_dir);
    if compile_result.has_errors() {
        error!("{}", compile_result.format_diagnostics().trim_end());
        process::exit(1);
    }

    if !compile_result.diagnostics.is_empty() {
        warn!("{}", compile_result.format_diagnostics().trim_end());
    }

    let world = compile_result.world.expect("no world despite no errors");
    let world_name = args
        .world_dir
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "world".to_string());

    let output_path = args.output.clone().unwrap_or_else(|| {
        let file_name = format!("{}.cvm", world_name);
        args.out_dir.join(file_name)
    });

    if let Some(parent) = output_path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            error!("Failed to create {}: {}", parent.display(), error);
            process::exit(1);
        }
    }

    let mut entries = Vec::new();

    for (signal_id, signal) in world.signals() {
        if let Some(ref expr) = signal.resolve {
            push_entry(
                &mut entries,
                BytecodeKind::SignalResolve,
                signal_id.to_string(),
                None,
                expr,
            );
        }

        if let Some(ref components) = signal.resolve_components {
            for (index, expr) in components.iter().enumerate() {
                push_entry(
                    &mut entries,
                    BytecodeKind::SignalResolveComponent,
                    signal_id.to_string(),
                    Some(format!("component:{}", index)),
                    expr,
                );
            }
        }

        if let Some(ref warmup) = signal.warmup {
            push_entry(
                &mut entries,
                BytecodeKind::SignalWarmup,
                signal_id.to_string(),
                None,
                &warmup.iterate,
            );
        }

        for (index, assertion) in signal.assertions.iter().enumerate() {
            push_entry(
                &mut entries,
                BytecodeKind::SignalAssertion,
                signal_id.to_string(),
                Some(format!("assertion:{}", index)),
                &assertion.condition,
            );
        }
    }

    for (field_id, field) in world.fields() {
        if let Some(ref expr) = field.measure {
            push_entry(
                &mut entries,
                BytecodeKind::FieldMeasure,
                field_id.to_string(),
                None,
                expr,
            );
        }
    }

    for (operator_id, operator) in world.operators() {
        if let Some(ref body) = operator.body {
            push_entry(
                &mut entries,
                BytecodeKind::OperatorBody,
                operator_id.to_string(),
                None,
                body,
            );
        }

        for (index, assertion) in operator.assertions.iter().enumerate() {
            push_entry(
                &mut entries,
                BytecodeKind::OperatorAssertion,
                operator_id.to_string(),
                Some(format!("assertion:{}", index)),
                &assertion.condition,
            );
        }
    }

    for (member_id, member) in world.members() {
        if let Some(ref expr) = member.initial {
            push_entry(
                &mut entries,
                BytecodeKind::MemberInitial,
                member_id.to_string(),
                None,
                expr,
            );
        }

        if let Some(ref expr) = member.resolve {
            push_entry(
                &mut entries,
                BytecodeKind::MemberResolve,
                member_id.to_string(),
                None,
                expr,
            );
        }

        for (index, assertion) in member.assertions.iter().enumerate() {
            push_entry(
                &mut entries,
                BytecodeKind::MemberAssertion,
                member_id.to_string(),
                Some(format!("assertion:{}", index)),
                &assertion.condition,
            );
        }
    }

    for (fracture_id, fracture) in world.fractures() {
        for (index, expr) in fracture.conditions.iter().enumerate() {
            push_entry(
                &mut entries,
                BytecodeKind::FractureCondition,
                fracture_id.to_string(),
                Some(format!("condition:{}", index)),
                expr,
            );
        }

        for emit in &fracture.emits {
            push_entry(
                &mut entries,
                BytecodeKind::FractureEmit,
                fracture_id.to_string(),
                Some(format!("emit:{}", emit.target)),
                &emit.value,
            );
        }
    }

    for (era_id, era) in world.eras() {
        for transition in &era.transitions {
            push_entry(
                &mut entries,
                BytecodeKind::EraTransition,
                era_id.to_string(),
                Some(format!("target:{}", transition.target_era)),
                &transition.condition,
            );
        }
    }

    for (chronicle_id, chronicle) in world.chronicles() {
        for handler in &chronicle.handlers {
            push_entry(
                &mut entries,
                BytecodeKind::ChronicleCondition,
                chronicle_id.to_string(),
                Some(format!("event:{}", handler.event_name)),
                &handler.condition,
            );

            for field in &handler.event_fields {
                push_entry(
                    &mut entries,
                    BytecodeKind::ChronicleField,
                    chronicle_id.to_string(),
                    Some(format!("event:{}:field:{}", handler.event_name, field.name)),
                    &field.value,
                );
            }
        }
    }

    let bundle = BytecodeBundle {
        version: 1,
        world: world_name,
        entries,
    };

    let encoded = match bincode::serialize(&bundle) {
        Ok(data) => data,
        Err(error) => {
            error!("Failed to encode bytecode bundle: {}", error);
            process::exit(1);
        }
    };

    if let Err(error) = fs::write(&output_path, encoded) {
        error!("Failed to write {}: {}", output_path.display(), error);
        process::exit(1);
    }

    info!("Wrote bytecode bundle to {}", output_path.display());
}

fn push_entry(
    entries: &mut Vec<BytecodeEntry>,
    kind: BytecodeKind,
    id: String,
    label: Option<String>,
    expr: &continuum_compiler::ir::CompiledExpr,
) {
    let bytecode = compile_to_bytecode(expr);
    entries.push(BytecodeEntry {
        kind,
        id,
        label,
        bytecode,
    });
}
