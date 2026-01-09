//! Item parsers for top-level DSL constructs.
//!
//! This module parses all top-level declarations in the Continuum DSL.
//! Each declaration type has a dedicated parser function that produces
//! the corresponding AST node.
//!
//! # Supported Items
//!
//! ## Configuration
//! - [`ConstBlock`](crate::ast::ConstBlock) - `const { ... }` compile-time constants
//! - [`ConfigBlock`](crate::ast::ConfigBlock) - `config { ... }` runtime parameters
//!
//! ## Types and Functions
//! - [`TypeDef`](crate::ast::TypeDef) - `type.name { ... }` custom type definitions
//! - [`FnDef`](crate::ast::FnDef) - `fn.name(...) -> Type { ... }` pure functions
//!
//! ## Simulation Structure
//! - [`StrataDef`](crate::ast::StrataDef) - `strata.name { ... }` time strata
//! - [`EraDef`](crate::ast::EraDef) - `era.name { ... }` simulation eras
//!
//! ## Core Simulation
//! - [`SignalDef`](crate::ast::SignalDef) - `signal.name { ... }` authoritative state
//! - [`FieldDef`](crate::ast::FieldDef) - `field.name { ... }` observer data
//! - [`OperatorDef`](crate::ast::OperatorDef) - `operator.name { ... }` phase logic
//!
//! ## Events
//! - [`ImpulseDef`](crate::ast::ImpulseDef) - `impulse.name { ... }` external inputs
//! - [`FractureDef`](crate::ast::FractureDef) - `fracture.name { ... }` tension detectors
//! - [`ChronicleDef`](crate::ast::ChronicleDef) - `chronicle.name { ... }` observers
//!
//! ## Collections
//! - [`EntityDef`](crate::ast::EntityDef) - `entity.name { ... }` indexed state

mod common;
mod config;
mod entity;
mod events;
mod signals;
mod time;
mod types;

use chumsky::prelude::*;

use crate::ast::Item;

use super::ParseError;

// Re-export public parsers
pub use config::{config_block, const_block};
pub use entity::entity_def;
pub use events::{chronicle_def, fracture_def, impulse_def};
pub use signals::{field_def, operator_def, signal_def};
pub use time::{era_def, strata_def};
pub use types::{fn_def, type_def};

/// Main entry point for parsing top-level items.
pub fn item<'src>() -> impl Parser<'src, &'src str, Item, extra::Err<ParseError<'src>>> {
    choice((
        const_block().map(Item::ConstBlock),
        config_block().map(Item::ConfigBlock),
        type_def().map(Item::TypeDef),
        fn_def().map(Item::FnDef),
        strata_def().map(Item::StrataDef),
        era_def().map(Item::EraDef),
        signal_def().map(Item::SignalDef),
        field_def().map(Item::FieldDef),
        operator_def().map(Item::OperatorDef),
        impulse_def().map(Item::ImpulseDef),
        fracture_def().map(Item::FractureDef),
        chronicle_def().map(Item::ChronicleDef),
        entity_def().map(Item::EntityDef),
    ))
}
