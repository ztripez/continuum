//! Continuum IR - Lowering Logic
//!
//! This module transforms DSL abstract syntax trees into the typed intermediate
//! representation (IR) used for DAG construction and runtime execution.

mod chronicles;
mod convert;
mod deps;
mod entities;
mod eras;
mod events;
mod expr;
mod members;
mod operators;
mod signals;

#[cfg(test)]
mod tests;

use indexmap::IndexMap;
use thiserror::Error;

use continuum_dsl::ast::{self, CompilationUnit, Item, Span};
use continuum_foundation::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, MemberId, OperatorId, Path,
    SignalId, StratumId, TypeId,
};

use crate::{
    CompiledChronicle, CompiledEntity, CompiledEra, CompiledField, CompiledFn, CompiledFracture,
    CompiledImpulse, CompiledMember, CompiledOperator, CompiledSignal, CompiledStratum,
    CompiledType, CompiledTypeField, CompiledWorld,
};

/// Errors that can occur during the lowering phase.
#[derive(Debug, Error)]
pub enum LowerError {
    #[error("undefined stratum: {name}")]
    UndefinedStratum {
        name: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error("undefined signal: {name}")]
    UndefinedSignal {
        name: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error("duplicate definition: {name}")]
    DuplicateDefinition {
        name: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error("invalid literal: {message}")]
    InvalidLiteral {
        message: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error("undeclared dt_raw usage in signal '{name}'")]
    UndeclaredDtRawUsage {
        name: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error("invalid expression: {message}")]
    InvalidExpression {
        message: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error(
        "signal '{signal}' has {constraint_kind} constraint but type is {actual_type}, not {expected_type}"
    )]
    MismatchedConstraint {
        signal: String,
        constraint_kind: String,
        actual_type: String,
        expected_type: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
    #[error("{message}")]
    Generic {
        message: String,
        file: Option<std::path::PathBuf>,
        span: Span,
    },
}

impl LowerError {
    pub fn span(&self) -> Span {
        match self {
            LowerError::UndefinedStratum { span, .. } => span.clone(),
            LowerError::UndefinedSignal { span, .. } => span.clone(),
            LowerError::DuplicateDefinition { span, .. } => span.clone(),
            LowerError::InvalidLiteral { span, .. } => span.clone(),
            LowerError::UndeclaredDtRawUsage { span, .. } => span.clone(),
            LowerError::InvalidExpression { span, .. } => span.clone(),
            LowerError::MismatchedConstraint { span, .. } => span.clone(),
            LowerError::Generic { span, .. } => span.clone(),
        }
    }

    pub fn file(&self) -> Option<std::path::PathBuf> {
        match self {
            LowerError::UndefinedStratum { file, .. } => file.clone(),
            LowerError::UndefinedSignal { file, .. } => file.clone(),
            LowerError::DuplicateDefinition { file, .. } => file.clone(),
            LowerError::InvalidLiteral { file, .. } => file.clone(),
            LowerError::UndeclaredDtRawUsage { file, .. } => file.clone(),
            LowerError::InvalidExpression { file, .. } => file.clone(),
            LowerError::MismatchedConstraint { file, .. } => file.clone(),
            LowerError::Generic { file, .. } => file.clone(),
        }
    }
}

pub fn lower(unit: &CompilationUnit) -> Result<CompiledWorld, LowerError> {
    lower_with_file(unit, None)
}

pub fn lower_multi(
    units: Vec<(std::path::PathBuf, &CompilationUnit)>,
) -> Result<CompiledWorld, LowerError> {
    let mut lowerer = Lowerer::new(None);

    // Pass 1: Types, Strata, Functions, Constants, Config
    for (path, unit) in &units {
        lowerer.file = Some(path.clone());
        for item in &unit.items {
            match &item.node {
                Item::ConstBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.to_string();
                        if lowerer.constants.contains_key(&key) {
                            return Err(LowerError::DuplicateDefinition {
                                name: format!("const.{}", key),
                                file: lowerer.file.clone(),
                                span: entry.path.span.clone(),
                            });
                        }
                        let value = lowerer.literal_to_f64(&entry.value.node, &entry.value.span)?;
                        lowerer.constants.insert(key, value);
                    }
                }
                Item::ConfigBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.to_string();
                        if lowerer.config.contains_key(&key) {
                            return Err(LowerError::DuplicateDefinition {
                                name: format!("config.{}", key),
                                file: lowerer.file.clone(),
                                span: entry.path.span.clone(),
                            });
                        }
                        let value = lowerer.literal_to_f64(&entry.value.node, &entry.value.span)?;
                        lowerer.config.insert(key, value);
                    }
                }
                Item::FnDef(def) => {
                    lowerer.lower_fn(def, item.span.clone())?;
                }
                Item::StrataDef(def) => {
                    lowerer.lower_stratum(def, item.span.clone())?;
                }
                Item::TypeDef(def) => {
                    lowerer.lower_type_def(def, item.span.clone())?;
                }
                _ => {}
            }
        }
    }

    // Pass 2: Eras (depends on strata)
    for (path, unit) in &units {
        lowerer.file = Some(path.clone());
        for item in &unit.items {
            if let Item::EraDef(def) = &item.node {
                lowerer.lower_era(def, item.span.clone())?;
            }
        }
    }

    // Pass 3: Signals, Fields, Operators, Impulses, Fractures, Entities, Members, Chronicles
    for (path, unit) in &units {
        lowerer.file = Some(path.clone());
        for item in &unit.items {
            match &item.node {
                Item::SignalDef(def) => lowerer.lower_signal(def, item.span.clone())?,
                Item::FieldDef(def) => lowerer.lower_field(def, item.span.clone())?,
                Item::OperatorDef(def) => lowerer.lower_operator(def, item.span.clone())?,
                Item::ImpulseDef(def) => lowerer.lower_impulse(def, item.span.clone())?,
                Item::FractureDef(def) => lowerer.lower_fracture(def, item.span.clone())?,
                Item::EntityDef(def) => lowerer.lower_entity(def, item.span.clone())?,
                Item::MemberDef(def) => lowerer.lower_member(def, item.span.clone())?,
                Item::ChronicleDef(def) => lowerer.lower_chronicle(def, item.span.clone())?,
                _ => {}
            }
        }
    }

    Ok(lowerer.finish())
}

pub fn lower_with_file(
    unit: &CompilationUnit,
    file: Option<std::path::PathBuf>,
) -> Result<CompiledWorld, LowerError> {
    let mut lowerer = Lowerer::new(file);
    lowerer.lower_unit(unit)?;
    Ok(lowerer.finish())
}

pub(crate) struct Lowerer {
    pub(crate) file: Option<std::path::PathBuf>,
    pub(crate) constants: IndexMap<String, f64>,
    pub(crate) config: IndexMap<String, f64>,
    pub(crate) functions: IndexMap<FnId, CompiledFn>,
    pub(crate) strata: IndexMap<StratumId, CompiledStratum>,
    pub(crate) eras: IndexMap<EraId, CompiledEra>,
    pub(crate) signals: IndexMap<SignalId, CompiledSignal>,
    pub(crate) fields: IndexMap<FieldId, CompiledField>,
    pub(crate) operators: IndexMap<OperatorId, CompiledOperator>,
    pub(crate) impulses: IndexMap<ImpulseId, CompiledImpulse>,
    pub(crate) fractures: IndexMap<FractureId, CompiledFracture>,
    pub(crate) entities: IndexMap<EntityId, CompiledEntity>,
    pub(crate) members: IndexMap<MemberId, CompiledMember>,
    pub(crate) chronicles: IndexMap<ChronicleId, CompiledChronicle>,
    pub(crate) types: IndexMap<TypeId, CompiledType>,
}

impl Lowerer {
    fn new(file: Option<std::path::PathBuf>) -> Self {
        Self {
            file,
            constants: IndexMap::new(),
            config: IndexMap::new(),
            functions: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
            entities: IndexMap::new(),
            members: IndexMap::new(),
            chronicles: IndexMap::new(),
            types: IndexMap::new(),
        }
    }

    pub(crate) fn validate_stratum(
        &self,
        stratum: &StratumId,
        span: &Span,
    ) -> Result<(), LowerError> {
        if stratum.to_string() == "default" {
            return Ok(());
        }
        if !self.strata.contains_key(stratum) {
            return Err(LowerError::UndefinedStratum {
                name: stratum.to_string(),
                file: self.file.clone(),
                span: span.clone(),
            });
        }
        Ok(())
    }

    fn lower_unit(&mut self, unit: &CompilationUnit) -> Result<(), LowerError> {
        for item in &unit.items {
            match &item.node {
                Item::ConstBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.to_string();
                        if self.constants.contains_key(&key) {
                            return Err(LowerError::DuplicateDefinition {
                                name: format!("const.{}", key),
                                file: self.file.clone(),
                                span: entry.path.span.clone(),
                            });
                        }
                        let value = self.literal_to_f64(&entry.value.node, &entry.value.span)?;
                        self.constants.insert(key, value);
                    }
                }
                Item::ConfigBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.to_string();
                        if self.config.contains_key(&key) {
                            return Err(LowerError::DuplicateDefinition {
                                name: format!("config.{}", key),
                                file: self.file.clone(),
                                span: entry.path.span.clone(),
                            });
                        }
                        let value = self.literal_to_f64(&entry.value.node, &entry.value.span)?;
                        self.config.insert(key, value);
                    }
                }
                Item::FnDef(def) => {
                    self.lower_fn(def, item.span.clone())?;
                }
                Item::StrataDef(def) => {
                    self.lower_stratum(def, item.span.clone())?;
                }
                Item::TypeDef(def) => {
                    self.lower_type_def(def, item.span.clone())?;
                }
                _ => {}
            }
        }

        for item in &unit.items {
            if let Item::EraDef(def) = &item.node {
                self.lower_era(def, item.span.clone())?;
            }
        }

        for item in &unit.items {
            match &item.node {
                Item::SignalDef(def) => self.lower_signal(def, item.span.clone())?,
                Item::FieldDef(def) => self.lower_field(def, item.span.clone())?,
                Item::OperatorDef(def) => self.lower_operator(def, item.span.clone())?,
                Item::ImpulseDef(def) => self.lower_impulse(def, item.span.clone())?,
                Item::FractureDef(def) => self.lower_fracture(def, item.span.clone())?,
                Item::EntityDef(def) => self.lower_entity(def, item.span.clone())?,
                Item::MemberDef(def) => self.lower_member(def, item.span.clone())?,
                Item::ChronicleDef(def) => self.lower_chronicle(def, item.span.clone())?,
                _ => {}
            }
        }

        Ok(())
    }

    fn lower_type_def(&mut self, def: &ast::TypeDef, span: Span) -> Result<(), LowerError> {
        let id = TypeId::from(Path::from_str(&def.name.node));
        if self.types.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("type.{}", id),
                file: self.file.clone(),
                span: def.name.span.clone(),
            });
        }

        let fields = def
            .fields
            .iter()
            .map(|f| CompiledTypeField {
                name: f.name.node.clone(),
                value_type: self.lower_type_expr(&f.ty.node),
            })
            .collect();

        let compiled_type = CompiledType {
            file: self.file.clone(),
            span,
            id: id.clone(),
            fields,
        };
        self.types.insert(id, compiled_type);
        Ok(())
    }

    fn lower_stratum(
        &mut self,
        def: &continuum_dsl::ast::StrataDef,
        span: Span,
    ) -> Result<(), LowerError> {
        let id = StratumId::from(def.path.node.clone());
        if self.strata.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("strata.{}", id),
                file: self.file.clone(),
                span: def.path.span.clone(),
            });
        }

        let stratum = CompiledStratum {
            file: self.file.clone(),
            span,
            id: id.clone(),
            title: def.title.as_ref().map(|s| s.node.clone()),
            symbol: def.symbol.as_ref().map(|s| s.node.clone()),
            default_stride: def.stride.as_ref().map(|s| s.node).unwrap_or(1),
        };
        self.strata.insert(id, stratum);
        Ok(())
    }

    fn finish(self) -> CompiledWorld {
        let mut nodes = IndexMap::new();

        for (id, signal) in &self.signals {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: signal.file.clone(),
                span: signal.span.clone(),
                stratum: Some(signal.stratum.clone()),
                reads: signal.reads.clone(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Signal(
                    crate::unified_nodes::SignalProperties {
                        title: signal.title.clone(),
                        symbol: signal.symbol.clone(),
                        value_type: signal.value_type.clone(),
                        uses_dt_raw: signal.uses_dt_raw,
                        resolve: signal.resolve.clone(),
                        resolve_components: signal.resolve_components.clone(),
                        warmup: signal.warmup.clone(),
                        assertions: signal.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, field) in &self.fields {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: field.file.clone(),
                span: field.span.clone(),
                stratum: Some(field.stratum.clone()),
                reads: field.reads.clone(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Field(
                    crate::unified_nodes::FieldProperties {
                        title: field.title.clone(),
                        topology: field.topology,
                        value_type: field.value_type.clone(),
                        measure: field.measure.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, operator) in &self.operators {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: operator.file.clone(),
                span: operator.span.clone(),
                stratum: Some(operator.stratum.clone()),
                reads: operator.reads.clone(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Operator(
                    crate::unified_nodes::OperatorProperties {
                        phase: operator.phase,
                        body: operator.body.clone(),
                        assertions: operator.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, impulse) in &self.impulses {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: impulse.file.clone(),
                span: impulse.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Impulse(
                    crate::unified_nodes::ImpulseProperties {
                        payload_type: impulse.payload_type.clone(),
                        apply: impulse.apply.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, fracture) in &self.fractures {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: fracture.file.clone(),
                span: fracture.span.clone(),
                stratum: Some(fracture.stratum.clone()),
                reads: fracture.reads.clone(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Fracture(
                    crate::unified_nodes::FractureProperties {
                        conditions: fracture.conditions.clone(),
                        emits: fracture.emits.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, entity) in &self.entities {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: entity.file.clone(),
                span: entity.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Entity(
                    crate::unified_nodes::EntityProperties {
                        count_source: entity.count_source.clone(),
                        count_bounds: entity.count_bounds,
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, member) in &self.members {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: member.file.clone(),
                span: member.span.clone(),
                stratum: Some(member.stratum.clone()),
                reads: member.reads.clone(),
                member_reads: member.member_reads.clone(),
                kind: crate::unified_nodes::NodeKind::Member(
                    crate::unified_nodes::MemberProperties {
                        entity_id: member.entity_id.clone(),
                        signal_name: member.signal_name.clone(),
                        title: member.title.clone(),
                        symbol: member.symbol.clone(),
                        value_type: member.value_type.clone(),
                        uses_dt_raw: member.uses_dt_raw,
                        initial: member.initial.clone(),
                        resolve: member.resolve.clone(),
                        assertions: member.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, chronicle) in &self.chronicles {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: chronicle.file.clone(),
                span: chronicle.span.clone(),
                stratum: None,
                reads: chronicle.reads.clone(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Chronicle(
                    crate::unified_nodes::ChronicleProperties {
                        handlers: chronicle.handlers.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, function) in &self.functions {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: function.file.clone(),
                span: function.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Function(
                    crate::unified_nodes::FunctionProperties {
                        params: function.params.clone(),
                        body: function.body.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, ty) in &self.types {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: ty.file.clone(),
                span: ty.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Type(crate::unified_nodes::TypeProperties {
                    fields: ty.fields.clone(),
                }),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, stratum) in &self.strata {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: stratum.file.clone(),
                span: stratum.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Stratum(
                    crate::unified_nodes::StratumProperties {
                        title: stratum.title.clone(),
                        symbol: stratum.symbol.clone(),
                        default_stride: stratum.default_stride,
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        for (id, era) in &self.eras {
            let node = crate::unified_nodes::CompiledNode {
                id: id.path().clone(),
                file: era.file.clone(),
                span: era.span.clone(),
                stratum: None,
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: crate::unified_nodes::NodeKind::Era(crate::unified_nodes::EraProperties {
                    is_initial: era.is_initial,
                    is_terminal: era.is_terminal,
                    title: era.title.clone(),
                    dt_seconds: era.dt_seconds,
                    strata_states: era.strata_states.clone(),
                    transitions: era.transitions.clone(),
                }),
            };
            nodes.insert(id.path().clone(), node);
        }

        CompiledWorld {
            constants: self.constants,
            config: self.config,
            nodes,
        }
    }
}
