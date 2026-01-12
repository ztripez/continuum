//! AST to IR Lowering
//!
//! This module transforms DSL abstract syntax trees into the typed intermediate
//! representation (IR) used for DAG construction and runtime execution.
//!
//! # Overview
//!
//! Lowering performs several key transformations:
//!
//! 1. **Multi-pass processing**: Items are processed in dependency order
//!    (constants/strata first, then eras, then signals/fields/entities)
//! 2. **Dependency extraction**: Signal reads are collected from expressions
//! 3. **Function inlining**: User-defined functions are expanded at call sites
//! 4. **Type resolution**: AST types become concrete IR value types
//! 5. **Expression simplification**: Complex AST nodes become simpler IR forms
//!
//! # Usage
//!
//! ```ignore
//! use continuum_dsl::parse;
//! use continuum_ir::lower;
//!
//! let source = r#"
//!     strata.terra {}
//!     era.main { : initial }
//!     signal.terra.temp {
//!         : strata(terra)
//!         resolve { prev + 1.0 }
//!     }
//! "#;
//!
//! let (unit, errors) = parse(source);
//! let world = lower(&unit.unwrap())?;
//! println!("Signals: {}", world.signals().len());
//! ```
//!
//! # Errors
//!
//! Lowering can fail with [`LowerError`] for issues like:
//! - Undefined strata or signal references
//! - Duplicate definitions
//! - Using `dt_raw` without declaring it
//!
//! # dt-robustness Checking
//!
//! Signals that use the raw time step (`dt_raw`) must explicitly declare this
//! with `: uses(dt_raw)`. This enables dt-robustness auditing to track signals
//! whose behavior depends on the time step size.

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

use continuum_dsl::ast::{CompilationUnit, Item, Span, TypeDef};
use continuum_foundation::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, MemberId, OperatorId,
    SignalId, StratumId, TypeId,
};

use crate::{
    CompiledChronicle, CompiledEntity, CompiledEra, CompiledField, CompiledFn, CompiledFracture,
    CompiledImpulse, CompiledMember, CompiledOperator, CompiledSignal, CompiledStratum,
    CompiledType, CompiledTypeField, CompiledWorld,
};

/// Errors that can occur during the lowering phase.
///
/// These represent semantic errors in the DSL that prevent successful
/// compilation to IR. Parse errors are handled separately by the parser.
#[derive(Debug, Error)]
pub enum LowerError {
    /// A stratum was referenced that has not been defined.
    ///
    /// Strata must be defined before they can be used in signal, field,
    /// or operator declarations.
    #[error("undefined stratum: {0}")]
    UndefinedStratum(String),

    /// A signal was referenced that has not been defined.
    ///
    /// This typically indicates a typo in a signal path or a missing
    /// signal definition.
    #[error("undefined signal: {0}")]
    UndefinedSignal(String),

    /// An identifier was defined more than once.
    ///
    /// Each signal, field, operator, etc. must have a unique path.
    #[error("duplicate definition: {0}")]
    DuplicateDefinition(String),

    /// A required field in a definition is missing.
    ///
    /// Some definitions require certain fields to be present (e.g.,
    /// signals need either a resolve block or an initial value).
    #[error("missing required field: {0}")]
    MissingRequiredField(String),

    /// An expression could not be lowered to a valid form.
    ///
    /// This may occur when an expression uses unsupported syntax or
    /// contains type mismatches.
    #[error("invalid expression: {0}")]
    InvalidExpression(String),

    /// A signal uses `dt_raw` without explicitly declaring it.
    ///
    /// For dt-robustness auditing, signals that depend on the raw time
    /// step must declare this with `: uses(dt_raw)`. This makes explicit
    /// which signals may behave differently with different time steps.
    #[error(
        "signal '{0}' uses dt_raw without explicit `: uses(dt_raw)` declaration - this is required for dt-robustness auditing"
    )]
    UndeclaredDtRawUsage(String),

    /// Type constraints were applied to an incompatible type.
    ///
    /// Tensor constraints (`:symmetric`, `:positive_definite`) can only be applied
    /// to Tensor types. Sequence constraints (`:each()`, `:sum()`) can only be
    /// applied to Seq types.
    #[error(
        "signal '{signal}' has {constraint_kind} constraint but type is {actual_type}, not {expected_type}"
    )]
    MismatchedConstraint {
        /// The signal path where the mismatch occurred.
        signal: String,
        /// The kind of constraint (e.g., "tensor" or "sequence").
        constraint_kind: String,
        /// The actual type found.
        actual_type: String,
        /// The expected type for this constraint.
        expected_type: String,
    },

    /// Generic error message.
    #[error("{0}")]
    Generic(String),
}

/// Lower a parsed compilation unit to the typed intermediate representation.
///
/// This is the main entry point for the lowering phase. It transforms the AST
/// produced by the parser into [`CompiledWorld`], resolving types, collecting
/// dependencies, and inlining user-defined functions.
///
/// # Processing Order
///
/// Items are processed in multiple passes to handle dependencies:
///
/// 1. **First pass**: Constants, config values, user-defined functions, and strata
/// 2. **Second pass**: Era definitions (which reference strata)
/// 3. **Third pass**: Signals, fields, operators, impulses, fractures, entities
///
/// # Errors
///
/// Returns [`LowerError`] if the AST contains semantic errors such as:
/// - References to undefined symbols
/// - Duplicate definitions
/// - Using `dt_raw` without explicit declaration
///
/// # Example
///
/// ```ignore
/// let (unit, _) = continuum_dsl::parse(source);
/// let world = lower(&unit.unwrap())?;
/// assert!(!world.signals().is_empty());
/// ```
pub fn lower(unit: &CompilationUnit) -> Result<CompiledWorld, LowerError> {
    let mut lowerer = Lowerer::new();
    lowerer.lower_unit(unit)?;
    Ok(lowerer.finish())
}

pub(crate) struct Lowerer {
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
    fn new() -> Self {
        Self {
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

    /// Validate that a stratum exists, returning UndefinedStratum error if not.
    pub(crate) fn validate_stratum(&self, stratum: &StratumId) -> Result<(), LowerError> {
        // "default" stratum is always valid (implicit)
        if stratum.to_string() == "default" {
            return Ok(());
        }
        if !self.strata.contains_key(stratum) {
            return Err(LowerError::UndefinedStratum(stratum.to_string()));
        }
        Ok(())
    }

    /// Lower a custom type definition to CompiledType.
    fn lower_type_def(&mut self, def: &TypeDef, span: Span) -> Result<(), LowerError> {
        let id = TypeId::from(def.name.node.clone());
        if self.types.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("type.{}", id)));
        }

        let fields: Vec<CompiledTypeField> = def
            .fields
            .iter()
            .map(|field| CompiledTypeField {
                name: field.name.node.clone(),
                value_type: self.lower_type_expr(&field.ty.node),
            })
            .collect();

        let compiled_type = CompiledType {
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
            return Err(LowerError::DuplicateDefinition(format!("strata.{}", id)));
        }

        let stratum = CompiledStratum {
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
        // Build unified nodes from legacy collections
        let mut nodes = IndexMap::new();

        // Convert signals to unified nodes
        for (id, signal) in &self.signals {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: signal.span.clone(),
                stratum: Some(signal.stratum.clone()),
                reads: signal.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Signal(
                    super::unified_nodes::SignalProperties {
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

        // Convert fields to unified nodes
        for (id, field) in &self.fields {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: field.span.clone(),
                stratum: Some(field.stratum.clone()),
                reads: field.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Field(
                    super::unified_nodes::FieldProperties {
                        title: field.title.clone(),
                        topology: field.topology,
                        value_type: field.value_type.clone(),
                        measure: field.measure.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert operators to unified nodes
        for (id, operator) in &self.operators {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: operator.span.clone(),
                stratum: Some(operator.stratum.clone()),
                reads: operator.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Operator(
                    super::unified_nodes::OperatorProperties {
                        phase: operator.phase,
                        body: operator.body.clone(),
                        assertions: operator.assertions.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert impulses to unified nodes
        for (id, impulse) in &self.impulses {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: impulse.span.clone(),
                stratum: None, // Impulses don't belong to specific strata
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Impulse(
                    super::unified_nodes::ImpulseProperties {
                        payload_type: impulse.payload_type.clone(),
                        apply: impulse.apply.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert fractures to unified nodes
        for (id, fracture) in &self.fractures {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: fracture.span.clone(),
                stratum: Some(fracture.stratum.clone()),
                reads: fracture.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Fracture(
                    super::unified_nodes::FractureProperties {
                        conditions: fracture.conditions.clone(),
                        emits: fracture.emits.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert entities to unified nodes
        for (id, entity) in &self.entities {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: entity.span.clone(),
                stratum: None, // Entities are pure identity, no execution
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Entity(
                    super::unified_nodes::EntityProperties {
                        count_source: entity.count_source.clone(),
                        count_bounds: entity.count_bounds,
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert members to unified nodes
        for (id, member) in &self.members {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: member.span.clone(),
                stratum: Some(member.stratum.clone()),
                reads: member.reads.clone(),
                member_reads: member.member_reads.clone(),
                kind: super::unified_nodes::NodeKind::Member(
                    super::unified_nodes::MemberProperties {
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

        // Convert chronicles to unified nodes
        for (id, chronicle) in &self.chronicles {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: chronicle.span.clone(),
                stratum: None, // Chronicles are observer-only
                reads: chronicle.reads.clone(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Chronicle(
                    super::unified_nodes::ChronicleProperties {
                        handlers: chronicle.handlers.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert functions to unified nodes
        for (id, function) in &self.functions {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: function.span.clone(),
                stratum: None, // Functions are inlined, don't execute independently
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Function(
                    super::unified_nodes::FunctionProperties {
                        params: function.params.clone(),
                        body: function.body.clone(),
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert types to unified nodes
        for (id, type_def) in &self.types {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: type_def.span.clone(),
                stratum: None, // Types are compile-time only
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Type(super::unified_nodes::TypeProperties {
                    fields: type_def.fields.clone(),
                }),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert strata to unified nodes
        for (id, stratum) in &self.strata {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: stratum.span.clone(),
                stratum: None, // Strata define execution structure, don't execute themselves
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Stratum(
                    super::unified_nodes::StratumProperties {
                        title: stratum.title.clone(),
                        symbol: stratum.symbol.clone(),
                        default_stride: stratum.default_stride,
                    },
                ),
            };
            nodes.insert(id.path().clone(), node);
        }

        // Convert eras to unified nodes
        for (id, era) in &self.eras {
            let node = super::unified_nodes::CompiledNode {
                id: id.path().clone(),
                span: era.span.clone(),
                stratum: None, // Eras define time phases, don't execute themselves
                reads: Vec::new(),
                member_reads: Vec::new(),
                kind: super::unified_nodes::NodeKind::Era(super::unified_nodes::EraProperties {
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

    pub(crate) fn lower_unit(&mut self, unit: &CompilationUnit) -> Result<(), LowerError> {
        // First pass: collect constants, config, functions, and strata
        for item in &unit.items {
            match &item.node {
                Item::ConstBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.to_string();
                        if self.constants.contains_key(&key) {
                            return Err(LowerError::DuplicateDefinition(format!("const.{}", key)));
                        }
                        let value = self.literal_to_f64(&entry.value.node)?;
                        self.constants.insert(key, value);
                    }
                }
                Item::ConfigBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.to_string();
                        if self.config.contains_key(&key) {
                            return Err(LowerError::DuplicateDefinition(format!("config.{}", key)));
                        }
                        let value = self.literal_to_f64(&entry.value.node)?;
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

        // Second pass: eras (need strata)
        for item in &unit.items {
            if let Item::EraDef(def) = &item.node {
                self.lower_era(def, item.span.clone())?;
            }
        }

        // Third pass: signals, fields, operators, impulses, fractures, entities, members, chronicles
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
}
