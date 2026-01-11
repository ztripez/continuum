//! Signal and field lowering.
//!
//! This module handles lowering signal and field definitions from AST to IR,
//! including dependency extraction and dt-robustness validation.

use continuum_dsl::ast;
use continuum_foundation::{FieldId, SignalId, StratumId};

use crate::{CompiledExpr, CompiledField, CompiledSignal, CompiledWarmup, TopologyIr, ValueType};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_signal(&mut self, def: &ast::SignalDef) -> Result<(), LowerError> {
        let id = SignalId::from(def.path.node.join(".").as_str());
        let signal_path = def.path.node.join(".");

        // Check for duplicate signal definition
        if self.signals.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("signal.{}", id.0)));
        }

        // Determine stratum
        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Validate stratum exists
        self.validate_stratum(&stratum)?;

        // Process local const blocks - add to global constants with signal-prefixed keys
        for entry in &def.local_consts {
            let local_key = entry.path.node.join(".");
            // Add with full signal path prefix: signal.path.local_key
            let full_key = format!("{}.{}", signal_path, local_key);
            let value = self.literal_to_f64(&entry.value.node)?;
            self.constants.insert(full_key, value);
        }

        // Process local config blocks - add to global config with signal-prefixed keys
        for entry in &def.local_config {
            let local_key = entry.path.node.join(".");
            // Add with full signal path prefix: signal.path.local_key
            let full_key = format!("{}.{}", signal_path, local_key);
            let value = self.literal_to_f64(&entry.value.node)?;
            self.config.insert(full_key, value);
        }

        // Collect signal dependencies from resolve expression
        let mut reads = Vec::new();
        if let Some(resolve) = &def.resolve {
            self.collect_signal_refs(&resolve.body.node, &mut reads);
        }

        // Validate dt_raw usage: if resolve uses dt_raw, signal must declare it
        if let Some(resolve) = &def.resolve {
            if !def.dt_raw && self.expr_uses_dt_raw(&resolve.body.node) {
                return Err(LowerError::UndeclaredDtRawUsage(signal_path));
            }
        }

        // Validate type constraints match the declared type
        self.validate_type_constraints(def, &signal_path)?;

        // Lower the type FIRST, so we can pass it to expression lowering
        // This enables type-aware expansion of vector/tensor operations
        let value_type = self.lower_signal_type(def);

        // Lower warmup if present (warmup iterate uses signal's type context)
        let warmup = def.warmup.as_ref().map(|w| CompiledWarmup {
            iterations: w.iterations.node,
            convergence: w.convergence.as_ref().map(|c| c.node),
            iterate: self.lower_expr_typed(&w.iterate.node, &value_type),
        });

        // Lower resolve expression with type context for vector/tensor expansion
        let resolve = def
            .resolve
            .as_ref()
            .map(|r| self.lower_expr_typed(&r.body.node, &value_type));

        // Lower assertions
        let assertions = def
            .assertions
            .as_ref()
            .map(|a| self.lower_assert_block(a))
            .unwrap_or_default();

        // For vector types, expand the resolve expression into per-component expressions
        let (resolve, resolve_components) = self.expand_resolve_for_type(resolve, &value_type);

        let signal = CompiledSignal {
            id: id.clone(),
            stratum,
            title: def.title.as_ref().map(|s| s.node.clone()),
            symbol: def.symbol.as_ref().map(|s| s.node.clone()),
            value_type,
            uses_dt_raw: def.dt_raw,
            reads,
            resolve,
            resolve_components,
            warmup,
            assertions,
        };

        self.signals.insert(id, signal);
        Ok(())
    }

    pub(crate) fn lower_field(&mut self, def: &ast::FieldDef) -> Result<(), LowerError> {
        let id = FieldId::from(def.path.node.join(".").as_str());

        // Check for duplicate field definition
        if self.fields.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("field.{}", id.0)));
        }

        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Validate stratum exists
        self.validate_stratum(&stratum)?;

        let mut reads = Vec::new();
        if let Some(measure) = &def.measure {
            self.collect_signal_refs(&measure.body.node, &mut reads);
        }

        let field = CompiledField {
            id: id.clone(),
            stratum,
            title: def.title.as_ref().map(|s| s.node.clone()),
            topology: def
                .topology
                .as_ref()
                .map(|t| self.lower_topology(&t.node))
                .unwrap_or(TopologyIr::SphereSurface),
            value_type: def
                .ty
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar {
                    unit: None,
                    dimension: None,
                    range: None,
                }),
            reads,
            measure: def.measure.as_ref().map(|m| self.lower_expr(&m.body.node)),
        };

        self.fields.insert(id, field);
        Ok(())
    }

    /// Validates that type constraints on a signal match the declared type.
    ///
    /// Tensor constraints (`:symmetric`, `:positive_definite`) require a Tensor type.
    /// Sequence constraints (`:each()`, `:sum()`) require a Seq type.
    fn validate_type_constraints(
        &self,
        def: &ast::SignalDef,
        signal_path: &str,
    ) -> Result<(), LowerError> {
        // Check tensor constraints
        if !def.tensor_constraints.is_empty() {
            let is_tensor = def
                .ty
                .as_ref()
                .is_some_and(|t| matches!(t.node, ast::TypeExpr::Tensor { .. }));
            if !is_tensor {
                let actual_type = def
                    .ty
                    .as_ref()
                    .map(|t| self.type_expr_name(&t.node))
                    .unwrap_or_else(|| "unspecified".to_string());
                return Err(LowerError::MismatchedConstraint {
                    signal: signal_path.to_string(),
                    constraint_kind: "tensor".to_string(),
                    actual_type,
                    expected_type: "Tensor".to_string(),
                });
            }
        }

        // Check sequence constraints
        if !def.seq_constraints.is_empty() {
            let is_seq = def
                .ty
                .as_ref()
                .is_some_and(|t| matches!(t.node, ast::TypeExpr::Seq { .. }));
            if !is_seq {
                let actual_type = def
                    .ty
                    .as_ref()
                    .map(|t| self.type_expr_name(&t.node))
                    .unwrap_or_else(|| "unspecified".to_string());
                return Err(LowerError::MismatchedConstraint {
                    signal: signal_path.to_string(),
                    constraint_kind: "sequence".to_string(),
                    actual_type,
                    expected_type: "Seq".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Returns a human-readable name for a TypeExpr.
    fn type_expr_name(&self, ty: &ast::TypeExpr) -> String {
        match ty {
            ast::TypeExpr::Scalar { unit, .. } => {
                if unit.is_empty() {
                    "Scalar".to_string()
                } else {
                    format!("Scalar<{}>", unit)
                }
            }
            ast::TypeExpr::Vector { dim, unit, .. } => format!("Vec{}<{}>", dim, unit),
            ast::TypeExpr::Tensor {
                rows, cols, unit, ..
            } => {
                format!("Tensor<{},{},{}>", rows, cols, unit)
            }
            ast::TypeExpr::Grid { width, height, .. } => format!("Grid<{},{}>", width, height),
            ast::TypeExpr::Seq { .. } => "Seq<...>".to_string(),
            ast::TypeExpr::Named(name) => name.clone(),
        }
    }

    /// Lowers a signal's type, applying any constraints from the signal definition.
    fn lower_signal_type(&self, def: &ast::SignalDef) -> ValueType {
        match &def.ty {
            None => ValueType::Scalar {
                unit: None,
                dimension: None,
                range: None,
            },
            Some(spanned_ty) => {
                // Clone the type and apply constraints before lowering
                let mut ty = spanned_ty.node.clone();

                // Apply tensor constraints
                if let ast::TypeExpr::Tensor {
                    ref mut constraints,
                    ..
                } = ty
                {
                    *constraints = def.tensor_constraints.clone();
                }

                // Apply sequence constraints
                if let ast::TypeExpr::Seq {
                    ref mut constraints,
                    ..
                } = ty
                {
                    *constraints = def.seq_constraints.clone();
                }

                self.lower_type_expr(&ty)
            }
        }
    }

    /// Expands a resolve expression for vector types into per-component expressions.
    ///
    /// For scalar types, returns the expression unchanged.
    /// For vector types (Vec2, Vec3, Vec4), expands the expression into N scalar
    /// expressions, one per component (x, y, z, w).
    ///
    /// # Returns
    ///
    /// A tuple of (scalar_resolve, component_resolves):
    /// - For scalars: (Some(expr), None)
    /// - For vectors: (None, Some([expr_x, expr_y, ...]))
    fn expand_resolve_for_type(
        &self,
        resolve: Option<CompiledExpr>,
        value_type: &ValueType,
    ) -> (Option<CompiledExpr>, Option<Vec<CompiledExpr>>) {
        let Some(expr) = resolve else {
            return (None, None);
        };

        // Determine component count based on type
        let components: Option<&[&str]> = match value_type {
            ValueType::Vec2 { .. } => Some(&["x", "y"]),
            ValueType::Vec3 { .. } => Some(&["x", "y", "z"]),
            ValueType::Vec4 { .. } => Some(&["x", "y", "z", "w"]),
            _ => None,
        };

        match components {
            None => {
                // Scalar or unsupported type - keep as-is
                (Some(expr), None)
            }
            Some(component_names) => {
                // Vector type - expand to per-component expressions
                let expanded: Vec<CompiledExpr> = component_names
                    .iter()
                    .map(|&comp| self.expand_expr_for_component(&expr, comp))
                    .collect();
                (None, Some(expanded))
            }
        }
    }

    /// Recursively expands an expression for a specific vector component.
    ///
    /// Transforms:
    /// - `prev` → `prev.{component}`
    /// - `collected` → `collected.{component}`
    /// - Binary/unary ops → component-wise application
    /// - Calls → passed through (functions operate on scalars)
    fn expand_expr_for_component(&self, expr: &CompiledExpr, component: &str) -> CompiledExpr {
        use crate::CompiledExpr::*;

        match expr {
            // prev becomes prev.x, prev.y, etc.
            Prev => FieldAccess {
                object: Box::new(Prev),
                field: component.to_string(),
            },

            // collected becomes collected.x, collected.y, etc.
            Collected => FieldAccess {
                object: Box::new(Collected),
                field: component.to_string(),
            },

            // Literals stay as-is (scalar broadcast)
            Literal(v) => Literal(*v),
            DtRaw => DtRaw,
            SimTime => SimTime,
            Const(name) => Const(name.clone()),
            Config(name) => Config(name.clone()),
            Local(name) => Local(name.clone()),

            // Signals: if accessing a vector signal without explicit component,
            // add the component accessor
            Signal(id) => FieldAccess {
                object: Box::new(Signal(id.clone())),
                field: component.to_string(),
            },

            // Binary ops: expand both sides for the same component
            Binary { op, left, right } => Binary {
                op: *op,
                left: Box::new(self.expand_expr_for_component(left, component)),
                right: Box::new(self.expand_expr_for_component(right, component)),
            },

            // Unary ops: expand operand
            Unary { op, operand } => Unary {
                op: *op,
                operand: Box::new(self.expand_expr_for_component(operand, component)),
            },

            // If: expand all branches for the same component
            If {
                condition,
                then_branch,
                else_branch,
            } => If {
                condition: Box::new(self.expand_expr_for_component(condition, component)),
                then_branch: Box::new(self.expand_expr_for_component(then_branch, component)),
                else_branch: Box::new(self.expand_expr_for_component(else_branch, component)),
            },

            // Let: expand value and body
            Let { name, value, body } => Let {
                name: name.clone(),
                value: Box::new(self.expand_expr_for_component(value, component)),
                body: Box::new(self.expand_expr_for_component(body, component)),
            },

            // Function calls: handle vector constructors specially
            Call { function, args } => {
                // Check if this is a vector constructor
                let component_idx = match function.as_str() {
                    "vec2" => match component {
                        "x" => Some(0),
                        "y" => Some(1),
                        _ => None,
                    },
                    "vec3" => match component {
                        "x" => Some(0),
                        "y" => Some(1),
                        "z" => Some(2),
                        _ => None,
                    },
                    "vec4" => match component {
                        "x" => Some(0),
                        "y" => Some(1),
                        "z" => Some(2),
                        "w" => Some(3),
                        _ => None,
                    },
                    _ => None,
                };

                if let Some(idx) = component_idx {
                    // Extract the component directly from the constructor arguments
                    if idx < args.len() {
                        self.expand_expr_for_component(&args[idx], component)
                    } else {
                        panic!(
                            "Vector constructor {} has {} args but component {} needs index {}",
                            function,
                            args.len(),
                            component,
                            idx
                        )
                    }
                } else {
                    // Regular function call - expand all arguments
                    Call {
                        function: function.clone(),
                        args: args
                            .iter()
                            .map(|a| self.expand_expr_for_component(a, component))
                            .collect(),
                    }
                }
            }

            // Kernel calls: expand all arguments
            KernelCall { function, args } => KernelCall {
                function: function.clone(),
                args: args
                    .iter()
                    .map(|a| self.expand_expr_for_component(a, component))
                    .collect(),
            },

            // DtRobust calls: expand all arguments
            DtRobustCall {
                operator,
                args,
                method,
            } => DtRobustCall {
                operator: *operator,
                args: args
                    .iter()
                    .map(|a| self.expand_expr_for_component(a, component))
                    .collect(),
                method: *method,
            },

            // FieldAccess: if it's already a component access (like prev.x), keep it
            // otherwise expand the object
            FieldAccess { object, field } => {
                // Check if this is already a component access
                if matches!(field.as_str(), "x" | "y" | "z" | "w") {
                    // Already accessing a specific component - keep as-is
                    FieldAccess {
                        object: object.clone(),
                        field: field.clone(),
                    }
                } else {
                    // Accessing a non-component field - expand the object
                    FieldAccess {
                        object: Box::new(self.expand_expr_for_component(object, component)),
                        field: field.clone(),
                    }
                }
            }

            // Entity expressions: not supported in vector context
            SelfField(_)
            | EntityAccess { .. }
            | Aggregate { .. }
            | Other { .. }
            | Pairs { .. }
            | Filter { .. }
            | First { .. }
            | Nearest { .. }
            | Within { .. } => {
                panic!(
                    "Entity expressions cannot be used in vector signal resolve blocks: {:?}",
                    expr
                )
            }
        }
    }
}
