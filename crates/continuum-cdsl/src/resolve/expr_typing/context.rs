//! Typing context for expression resolution.
//!
//! This module defines the [`TypingContext`] which carries all necessary
//! registries, local bindings, and execution context state required to
//! resolve the type of an expression.

use crate::ast::KernelRegistry;
use crate::foundation::{Path, Type};
use crate::resolve::types::TypeTable;
use continuum_foundation::Phase;
use std::collections::HashMap;

/// Context for expression typing.
///
/// Provides access to type registries and tracks local bindings and execution
/// context during typing. The context is designed to be forked (cloned) when
/// introducing new scopes (e.g., in let-bindings or aggregates).
#[derive(Clone)]
pub struct TypingContext<'a> {
    /// User-defined type definitions for struct construction and field access.
    pub type_table: &'a TypeTable,

    /// Kernel signatures used to resolve function calls and derive return types.
    pub kernel_registry: &'a KernelRegistry,

    /// Mapping from signal path to its authoritative resolved type.
    pub signal_types: &'a HashMap<Path, Type>,

    /// Mapping from field path to its authoritative observation type.
    pub field_types: &'a HashMap<Path, Type>,

    /// Mapping from world configuration path to its value type.
    pub config_types: &'a HashMap<Path, Type>,

    /// Mapping from CDSL constant path to its declared type.
    pub const_types: &'a HashMap<Path, Type>,

    /// Currently in-scope local variable bindings (name → type).
    pub local_bindings: HashMap<String, Type>,

    /// The type of the `self` entity instance in the current execution block.
    pub self_type: Option<Type>,

    /// The type of the `other` entity instance (used in n-body interaction blocks).
    pub other_type: Option<Type>,

    /// The output type of the node being typed (for `prev`, `current`, and aggregates).
    ///
    /// When typing a signal or field's execution block, this is set to the node's
    /// own output type.
    pub node_output: Option<Type>,

    /// The type of the inputs struct (for the `inputs` expression).
    pub inputs_type: Option<Type>,

    /// The type of the impulse payload (for `payload` expressions in impulse handlers).
    pub payload_type: Option<Type>,

    /// The execution phase for phase-based boundary enforcement.
    pub phase: Option<Phase>,
}

impl<'a> TypingContext<'a> {
    /// Create a new typing context with the provided registries.
    ///
    /// # Parameters
    /// - `type_table`: Registry of user-defined struct types.
    /// - `kernel_registry`: Registry of built-in kernel signatures.
    /// - `signal_types`: Map of declared signal paths to their types.
    /// - `field_types`: Map of declared field paths to their types.
    /// - `config_types`: Map of world configuration paths to their types.
    /// - `const_types`: Map of global constant paths to their types.
    pub fn new(
        type_table: &'a TypeTable,
        kernel_registry: &'a KernelRegistry,
        signal_types: &'a HashMap<Path, Type>,
        field_types: &'a HashMap<Path, Type>,
        config_types: &'a HashMap<Path, Type>,
        const_types: &'a HashMap<Path, Type>,
    ) -> Self {
        Self {
            type_table,
            kernel_registry,
            signal_types,
            field_types,
            config_types,
            const_types,
            local_bindings: HashMap::new(),
            self_type: None,
            other_type: None,
            node_output: None,
            inputs_type: None,
            payload_type: None,
            phase: None,
        }
    }

    /// Fork context with an additional local variable binding.
    ///
    /// # Parameters
    /// - `name`: The name of the new local variable.
    /// - `ty`: The type of the value being bound.
    pub fn with_binding(&self, name: String, ty: Type) -> Self {
        let mut ctx = self.clone();
        ctx.local_bindings.insert(name, ty);
        ctx
    }

    /// Set execution context for a specific node.
    ///
    /// This configures the types available via special context expressions
    /// like `self`, `other`, `inputs`, and `payload`.
    ///
    /// # Parameters
    /// - `self_type`: Type of the current entity instance.
    /// - `other_type`: Type of the interaction partner (if any).
    /// - `node_output`: Output type of the current node (for `prev`/`current`).
    /// - `inputs_type`: Type of the accumulated inputs struct.
    /// - `payload_type`: Type of the impulse payload (for impulse handlers).
    pub fn with_execution_context(
        &self,
        self_type: Option<Type>,
        other_type: Option<Type>,
        node_output: Option<Type>,
        inputs_type: Option<Type>,
        payload_type: Option<Type>,
    ) -> Self {
        let mut ctx = self.clone();
        ctx.self_type = self_type;
        ctx.other_type = other_type;
        ctx.node_output = node_output;
        ctx.inputs_type = inputs_type;
        ctx.payload_type = payload_type;
        ctx
    }

    /// Set phase context for boundary enforcement.
    ///
    /// # Parameters
    /// - `phase`: The execution phase to enforce.
    pub fn with_phase(&self, phase: Phase) -> Self {
        let mut ctx = self.clone();
        ctx.phase = Some(phase);
        ctx
    }
}
