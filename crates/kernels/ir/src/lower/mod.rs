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
//! println!("Signals: {}", world.signals.len());
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

mod convert;
mod deps;
mod entities;
mod eras;
mod events;
mod expr;
mod operators;
mod signals;

#[cfg(test)]
mod tests;

use indexmap::IndexMap;
use thiserror::Error;

use continuum_dsl::ast::{CompilationUnit, Item};
use continuum_foundation::{
    EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, OperatorId, SignalId, StratumId,
};

use crate::{
    CompiledEntity, CompiledEra, CompiledField, CompiledFn, CompiledFracture, CompiledImpulse,
    CompiledOperator, CompiledSignal, CompiledStratum, CompiledWorld,
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
    #[error("signal '{0}' uses dt_raw without explicit `: uses(dt_raw)` declaration - this is required for dt-robustness auditing")]
    UndeclaredDtRawUsage(String),
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
/// assert!(!world.signals.is_empty());
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
        }
    }

    fn finish(self) -> CompiledWorld {
        CompiledWorld {
            constants: self.constants,
            config: self.config,
            functions: self.functions,
            strata: self.strata,
            eras: self.eras,
            signals: self.signals,
            fields: self.fields,
            operators: self.operators,
            impulses: self.impulses,
            fractures: self.fractures,
            entities: self.entities,
        }
    }

    pub(crate) fn lower_unit(&mut self, unit: &CompilationUnit) -> Result<(), LowerError> {
        // First pass: collect constants, config, functions, and strata
        for item in &unit.items {
            match &item.node {
                Item::ConstBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.join(".");
                        let value = self.literal_to_f64(&entry.value.node)?;
                        self.constants.insert(key, value);
                    }
                }
                Item::ConfigBlock(block) => {
                    for entry in &block.entries {
                        let key = entry.path.node.join(".");
                        let value = self.literal_to_f64(&entry.value.node)?;
                        self.config.insert(key, value);
                    }
                }
                Item::FnDef(def) => {
                    self.lower_fn(def)?;
                }
                Item::StrataDef(def) => {
                    let id = StratumId::from(def.path.node.join(".").as_str());
                    let stratum = CompiledStratum {
                        id: id.clone(),
                        title: def.title.as_ref().map(|s| s.node.clone()),
                        symbol: def.symbol.as_ref().map(|s| s.node.clone()),
                        default_stride: def.stride.as_ref().map(|s| s.node).unwrap_or(1),
                    };
                    self.strata.insert(id, stratum);
                }
                _ => {}
            }
        }

        // Second pass: eras (need strata)
        for item in &unit.items {
            if let Item::EraDef(def) = &item.node {
                self.lower_era(def)?;
            }
        }

        // Third pass: signals, fields, operators, impulses, fractures, entities
        for item in &unit.items {
            match &item.node {
                Item::SignalDef(def) => self.lower_signal(def)?,
                Item::FieldDef(def) => self.lower_field(def)?,
                Item::OperatorDef(def) => self.lower_operator(def)?,
                Item::ImpulseDef(def) => self.lower_impulse(def)?,
                Item::FractureDef(def) => self.lower_fracture(def)?,
                Item::EntityDef(def) => self.lower_entity(def)?,
                _ => {}
            }
        }

        Ok(())
    }
}
