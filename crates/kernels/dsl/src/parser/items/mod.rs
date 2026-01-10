//! Item parsers for top-level DSL constructs.
//!
//! This module parses all top-level declarations in the Continuum DSL.
//! Each declaration type has a dedicated parser function that produces
//! the corresponding AST node.
//!
//! # Supported Items
//!
//! ## World Manifest
//! - [`WorldDef`](crate::ast::WorldDef) - `world.name { ... }` world manifest and policy
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
//! - [`EntityDef`](crate::ast::EntityDef) - `entity.name { ... }` index spaces
//! - [`MemberDef`](crate::ast::MemberDef) - `member.entity.field { ... }` per-entity state

mod common;
mod config;
mod entity;
mod events;
mod member;
mod signals;
mod time;
mod types;
mod world;

use chumsky::prelude::*;

use crate::ast::Item;

use super::primitives::doc_comment;
use super::ParseError;

// Re-export public parsers
pub use config::{config_block, const_block};
pub use entity::entity_def;
pub use events::{chronicle_def, fracture_def, impulse_def};
pub use member::member_def;
pub use signals::{field_def, operator_def, signal_def};
pub use time::{era_def, strata_def};
pub use types::{fn_def, type_def};
pub use world::world_def;

/// Main entry point for parsing top-level items.
///
/// Doc comments (`///`) are captured before items that support them
/// and stored in the item's `doc` field.
pub fn item<'src>() -> impl Parser<'src, &'src str, Item, extra::Err<ParseError<'src>>> {
    choice((
        // World definition (no doc comments - it's a manifest)
        world_def().map(Item::WorldDef),
        // Config blocks don't have doc comments (they're structural, not semantic items)
        const_block().map(Item::ConstBlock),
        config_block().map(Item::ConfigBlock),
        // All other items support doc comments
        doc_comment().then(type_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::TypeDef(def)
        }),
        doc_comment().then(fn_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::FnDef(def)
        }),
        doc_comment().then(strata_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::StrataDef(def)
        }),
        doc_comment().then(era_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::EraDef(def)
        }),
        doc_comment().then(signal_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::SignalDef(def)
        }),
        doc_comment().then(field_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::FieldDef(def)
        }),
        doc_comment().then(operator_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::OperatorDef(def)
        }),
        doc_comment().then(impulse_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::ImpulseDef(def)
        }),
        doc_comment().then(fracture_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::FractureDef(def)
        }),
        doc_comment().then(chronicle_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::ChronicleDef(def)
        }),
        doc_comment().then(entity_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::EntityDef(def)
        }),
        doc_comment().then(member_def()).map(|(doc, mut def)| {
            def.doc = doc;
            Item::MemberDef(def)
        }),
    ))
}
