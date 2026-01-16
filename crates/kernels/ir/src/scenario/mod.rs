//! Scenario System
//!
//! A Scenario configures how a World is instantiated for execution.
//! It does not define causal structure or execution semantics.
//!
//! # Design
//!
//! A Scenario may configure:
//! - Initial values for signals (applied before the first tick)
//! - Parameter/config overrides (constants used by signals or operators)
//! - Seed for deterministic random number generation
//!
//! A Scenario must NOT:
//! - Define new signals
//! - Define new operators
//! - Alter dependencies
//! - Change execution phases
//! - Change time semantics
//! - Bypass validation or assertions
//!
//! # Identity
//!
//! Two runs with:
//! - The same World
//! - The same Scenario
//! - The same seed
//!
//! Must produce identical causal history.
//!
//! # File Format
//!
//! Scenarios are stored as YAML files:
//!
//! ```yaml
//! apiVersion: continuum/v1
//! kind: Scenario
//!
//! metadata:
//!   name: early_earth
//!   title: "Early Earth Configuration"
//!   description: "Earth parameters for 4.5 billion years ago"
//!
//! # Override configuration values
//! config:
//!   core.initial_temp: 6000.0
//!   mantle.initial_temp: 2000.0
//!   crust.initial_thickness: 10000.0
//!
//! # Set initial signal values (applied before first tick)
//! initial:
//!   core.temp: 6000.0
//!   crust.thickness: 10000.0
//!
//! # Execution parameters
//! seed: 42
//! ```

mod types;

#[cfg(test)]
mod tests;

pub use types::*;
