//! Type and operator conversion utilities for lowering.
//!
//! This module contains functions that convert AST types to IR types,
//! including operators, type expressions, and literal values.

use continuum_dsl::ast::{
    self, AggregateOp, AssertBlock, AssertSeverity, BinaryOp, Expr, Literal, OperatorPhase, Span,
    Topology, TypeExpr, UnaryOp,
};

use crate::{
    AggregateOpIr, AssertionSeverity, BinaryOpIr, CompiledAssertion, OperatorPhaseIr, TopologyIr,
    UnaryOpIr, ValueRange, ValueType,
};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_binary_op(&self, op: BinaryOp) -> BinaryOpIr {
        match op {
            BinaryOp::Add => BinaryOpIr::Add,
            BinaryOp::Sub => BinaryOpIr::Sub,
            BinaryOp::Mul => BinaryOpIr::Mul,
            BinaryOp::Div => BinaryOpIr::Div,
            BinaryOp::Pow => BinaryOpIr::Pow,
            BinaryOp::Eq => BinaryOpIr::Eq,
            BinaryOp::Ne => BinaryOpIr::Ne,
            BinaryOp::Lt => BinaryOpIr::Lt,
            BinaryOp::Le => BinaryOpIr::Le,
            BinaryOp::Gt => BinaryOpIr::Gt,
            BinaryOp::Ge => BinaryOpIr::Ge,
            BinaryOp::And => BinaryOpIr::And,
            BinaryOp::Or => BinaryOpIr::Or,
        }
    }

    pub(crate) fn lower_unary_op(&self, op: UnaryOp) -> UnaryOpIr {
        match op {
            UnaryOp::Neg => UnaryOpIr::Neg,
            UnaryOp::Not => UnaryOpIr::Not,
        }
    }

    pub(crate) fn lower_aggregate_op(&self, op: AggregateOp) -> AggregateOpIr {
        match op {
            AggregateOp::Sum => AggregateOpIr::Sum,
            AggregateOp::Product => AggregateOpIr::Product,
            AggregateOp::Min => AggregateOpIr::Min,
            AggregateOp::Max => AggregateOpIr::Max,
            AggregateOp::Mean => AggregateOpIr::Mean,
            AggregateOp::Count => AggregateOpIr::Count,
            AggregateOp::Any => AggregateOpIr::Any,
            AggregateOp::All => AggregateOpIr::All,
            AggregateOp::None => AggregateOpIr::None,
        }
    }

    pub(crate) fn lower_type_expr(&self, ty: &TypeExpr) -> ValueType {
        match ty {
            TypeExpr::Scalar { unit, range } => {
                let (unit_str, dimension) = self.parse_unit_with_dimension(unit);
                ValueType::Scalar {
                    unit: unit_str,
                    dimension,
                    range: range.as_ref().map(|r| ValueRange {
                        min: r.min,
                        max: r.max,
                    }),
                }
            }
            TypeExpr::Vector {
                dim,
                unit,
                magnitude,
            } => {
                let (unit_str, dimension) = self.parse_unit_with_dimension(unit);
                match dim {
                    2 => ValueType::Vec2 {
                        unit: unit_str,
                        dimension,
                        magnitude: magnitude.as_ref().map(|r| ValueRange {
                            min: r.min,
                            max: r.max,
                        }),
                    },
                    3 => ValueType::Vec3 {
                        unit: unit_str,
                        dimension,
                        magnitude: magnitude.as_ref().map(|r| ValueRange {
                            min: r.min,
                            max: r.max,
                        }),
                    },
                    4 => ValueType::Vec4 {
                        unit: unit_str,
                        dimension,
                        magnitude: magnitude.as_ref().map(|r| ValueRange {
                            min: r.min,
                            max: r.max,
                        }),
                    },
                    _ => ValueType::Scalar {
                        unit: unit_str,
                        dimension,
                        range: None,
                    },
                }
            }
            TypeExpr::Tensor {
                rows,
                cols,
                unit,
                constraints,
            } => {
                let (unit_str, dimension) = self.parse_unit_with_dimension(unit);
                ValueType::Tensor {
                    rows: *rows,
                    cols: *cols,
                    unit: unit_str,
                    dimension,
                    constraints: constraints
                        .iter()
                        .map(|c| self.lower_tensor_constraint(*c))
                        .collect(),
                }
            }
            TypeExpr::Grid {
                width,
                height,
                element_type,
            } => ValueType::Grid {
                width: *width,
                height: *height,
                element_type: Box::new(self.lower_type_expr(element_type)),
            },
            TypeExpr::Seq {
                element_type,
                constraints,
            } => ValueType::Seq {
                element_type: Box::new(self.lower_type_expr(element_type)),
                constraints: constraints
                    .iter()
                    .map(|c| self.lower_seq_constraint(c))
                    .collect(),
            },
            TypeExpr::Named(_) => ValueType::Scalar {
                unit: None,
                dimension: None,
                range: None,
            }, // resolve named types later
        }
    }

    pub(crate) fn lower_tensor_constraint(
        &self,
        c: ast::TensorConstraint,
    ) -> crate::TensorConstraintIr {
        match c {
            ast::TensorConstraint::Symmetric => crate::TensorConstraintIr::Symmetric,
            ast::TensorConstraint::PositiveDefinite => crate::TensorConstraintIr::PositiveDefinite,
        }
    }

    pub(crate) fn lower_seq_constraint(&self, c: &ast::SeqConstraint) -> crate::SeqConstraintIr {
        match c {
            ast::SeqConstraint::Each(r) => crate::SeqConstraintIr::Each(ValueRange {
                min: r.min,
                max: r.max,
            }),
            ast::SeqConstraint::Sum(r) => crate::SeqConstraintIr::Sum(ValueRange {
                min: r.min,
                max: r.max,
            }),
        }
    }

    /// Parses a unit string and returns both the string representation and
    /// the structured dimensional representation.
    fn parse_unit_with_dimension(
        &self,
        unit: &str,
    ) -> (Option<String>, Option<crate::units::Unit>) {
        if unit.is_empty() {
            (None, None)
        } else {
            let dimension = crate::units::Unit::parse(unit);
            (Some(unit.to_string()), dimension)
        }
    }

    pub(crate) fn lower_topology(&self, topo: &Topology) -> TopologyIr {
        match topo {
            Topology::SphereSurface => TopologyIr::SphereSurface,
            Topology::PointCloud => TopologyIr::PointCloud,
            Topology::Volume => TopologyIr::Volume,
        }
    }

    pub(crate) fn lower_operator_phase(&self, phase: &OperatorPhase) -> OperatorPhaseIr {
        match phase {
            OperatorPhase::Warmup => OperatorPhaseIr::Warmup,
            OperatorPhase::Collect => OperatorPhaseIr::Collect,
            OperatorPhase::Measure => OperatorPhaseIr::Measure,
        }
    }

    pub(crate) fn lower_assert_block(&self, block: &AssertBlock) -> Vec<CompiledAssertion> {
        block
            .assertions
            .iter()
            .map(|a| CompiledAssertion {
                condition: self.lower_expr(&a.condition.node),
                severity: self.lower_assert_severity(a.severity),
                message: a.message.as_ref().map(|m| m.node.clone()),
            })
            .collect()
    }

    pub(crate) fn lower_assert_severity(&self, severity: AssertSeverity) -> AssertionSeverity {
        match severity {
            AssertSeverity::Warn => AssertionSeverity::Warn,
            AssertSeverity::Error => AssertionSeverity::Error,
            AssertSeverity::Fatal => AssertionSeverity::Fatal,
        }
    }

    /// Check if an AST expression contains dt_raw usage
    pub(crate) fn expr_uses_dt_raw(&self, expr: &Expr) -> bool {
        match expr {
            Expr::DtRaw => true,
            Expr::Binary { left, right, .. } => {
                self.expr_uses_dt_raw(&left.node) || self.expr_uses_dt_raw(&right.node)
            }
            Expr::Unary { operand, .. } => self.expr_uses_dt_raw(&operand.node),
            Expr::Call { function, args } => {
                self.expr_uses_dt_raw(&function.node)
                    || args.iter().any(|a| self.expr_uses_dt_raw(&a.value.node))
            }
            Expr::MethodCall { object, args, .. } => {
                self.expr_uses_dt_raw(&object.node)
                    || args.iter().any(|a| self.expr_uses_dt_raw(&a.value.node))
            }
            Expr::Let { value, body, .. } => {
                self.expr_uses_dt_raw(&value.node) || self.expr_uses_dt_raw(&body.node)
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.expr_uses_dt_raw(&condition.node)
                    || self.expr_uses_dt_raw(&then_branch.node)
                    || else_branch
                        .as_ref()
                        .map(|e| self.expr_uses_dt_raw(&e.node))
                        .unwrap_or(false)
            }
            Expr::FieldAccess { object, .. } => self.expr_uses_dt_raw(&object.node),
            Expr::Block(exprs) => exprs.iter().any(|e| self.expr_uses_dt_raw(&e.node)),
            Expr::For { iter, body, .. } => {
                self.expr_uses_dt_raw(&iter.node) || self.expr_uses_dt_raw(&body.node)
            }
            Expr::EmitSignal { value, .. } => self.expr_uses_dt_raw(&value.node),
            Expr::EmitField {
                position, value, ..
            } => self.expr_uses_dt_raw(&position.node) || self.expr_uses_dt_raw(&value.node),
            Expr::Struct(fields) => fields.iter().any(|(_, e)| self.expr_uses_dt_raw(&e.node)),
            Expr::Map { sequence, function } => {
                self.expr_uses_dt_raw(&sequence.node) || self.expr_uses_dt_raw(&function.node)
            }
            Expr::Fold {
                sequence,
                init,
                function,
            } => {
                self.expr_uses_dt_raw(&sequence.node)
                    || self.expr_uses_dt_raw(&init.node)
                    || self.expr_uses_dt_raw(&function.node)
            }
            // Entity expressions
            Expr::SelfField(_) | Expr::EntityRef(_) | Expr::Other(_) | Expr::Pairs(_) => false,
            Expr::EntityAccess { instance, .. } => self.expr_uses_dt_raw(&instance.node),
            Expr::Aggregate { body, .. } => self.expr_uses_dt_raw(&body.node),
            Expr::Filter { predicate, .. } => self.expr_uses_dt_raw(&predicate.node),
            Expr::First { predicate, .. } => self.expr_uses_dt_raw(&predicate.node),
            Expr::Nearest { position, .. } => self.expr_uses_dt_raw(&position.node),
            Expr::Within {
                position, radius, ..
            } => self.expr_uses_dt_raw(&position.node) || self.expr_uses_dt_raw(&radius.node),
            // These don't contain dt_raw
            Expr::Literal(_)
            | Expr::LiteralWithUnit { .. }
            | Expr::Path(_)
            | Expr::Prev
            | Expr::PrevField(_)
            | Expr::Payload
            | Expr::PayloadField(_)
            | Expr::SignalRef(_)
            | Expr::ConstRef(_)
            | Expr::ConfigRef(_)
            | Expr::FieldRef(_)
            | Expr::Collected
            | Expr::SimTime
            | Expr::MathConst(_) => false,
        }
    }

    pub(crate) fn literal_to_f64(&self, lit: &Literal, span: &Span) -> Result<f64, LowerError> {
        match lit {
            Literal::Integer(i) => Ok(*i as f64),
            Literal::Float(f) => Ok(*f),
            Literal::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            Literal::String(_) => Err(LowerError::InvalidExpression {
                message: "string cannot be converted to f64".to_string(),
                file: self.file.clone(),
                span: span.clone(),
            }),
        }
    }

    pub(crate) fn literal_to_f64_unchecked(&self, lit: &Literal) -> f64 {
        match lit {
            Literal::Integer(i) => *i as f64,
            Literal::Float(f) => *f,
            Literal::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Literal::String(_) => 0.0,
        }
    }

    pub(crate) fn value_with_unit_to_seconds(&self, vwu: &ast::ValueWithUnit) -> f64 {
        let base = self.literal_to_f64_unchecked(&vwu.value);
        // Convert common time units to seconds
        match vwu.unit.as_str() {
            "s" | "sec" | "second" | "seconds" => base,
            "ms" | "millisecond" | "milliseconds" => base / 1000.0,
            "us" | "microsecond" | "microseconds" => base / 1_000_000.0,
            "ns" | "nanosecond" | "nanoseconds" => base / 1_000_000_000.0,
            "min" | "minute" | "minutes" => base * 60.0,
            "h" | "hr" | "hour" | "hours" => base * 3600.0,
            "d" | "day" | "days" => base * 86400.0,
            "yr" | "year" | "years" => base * 31_557_600.0, // Julian year
            "kyr" => base * 31_557_600_000.0,
            "myr" | "Myr" | "Ma" => base * 31_557_600_000_000.0,
            "byr" | "Ga" => base * 31_557_600_000_000_000.0,
            _ => base, // assume seconds for unknown units
        }
    }
}
