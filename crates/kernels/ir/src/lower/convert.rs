//! Type and operator conversion utilities for lowering.
//!
//! This module contains functions that convert AST types to IR types,
//! including operators, type expressions, and literal values.

use continuum_dsl::ast::{
    self, AggregateOp, AssertBlock, AssertSeverity, BinaryOp, Expr, Literal, OperatorPhase,
    PrimitiveParamValue, PrimitiveTypeExpr, Span, Topology, TypeExpr, UnaryOp,
};

use continuum_foundation::{PrimitiveParamKind, primitive_type_by_name};

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
            TypeExpr::Primitive(primitive) => self.lower_primitive_type(primitive),
            TypeExpr::Named(_) => ValueType::scalar(None, None, None), // resolve named types later
        }
    }

    fn lower_primitive_type(&self, primitive: &PrimitiveTypeExpr) -> ValueType {
        let def = primitive_type_by_name(primitive.id.name())
            .expect("primitive type missing from registry");
        match def.shape {
            continuum_foundation::PrimitiveShape::Scalar => {
                let unit = self.param_unit(primitive);
                let (unit_str, dimension) = self.parse_unit_with_dimension(&unit);
                ValueType::scalar(
                    unit_str,
                    dimension,
                    self.param_range(primitive, PrimitiveParamKind::Range),
                )
            }
            continuum_foundation::PrimitiveShape::Vector { dim } => {
                let unit = self.param_unit(primitive);
                let (unit_str, dimension) = self.parse_unit_with_dimension(&unit);
                let magnitude = self.param_range(primitive, PrimitiveParamKind::Magnitude);
                if primitive.id.name() == "Quat" {
                    return ValueType::quat(magnitude);
                }
                match dim {
                    2 | 3 | 4 => ValueType::vector(dim, unit_str, dimension, magnitude),
                    _ => ValueType::scalar(unit_str, dimension, None),
                }
            }
            continuum_foundation::PrimitiveShape::Matrix { rows, cols } => {
                let unit = self.param_unit(primitive);
                let (unit_str, dimension) = self.parse_unit_with_dimension(&unit);
                ValueType::matrix(rows, cols, unit_str, dimension)
            }
            continuum_foundation::PrimitiveShape::Tensor => {
                let rows = self.param_u8(primitive, PrimitiveParamKind::Rows);
                let cols = self.param_u8(primitive, PrimitiveParamKind::Cols);
                let unit = self.param_unit(primitive);
                let (unit_str, dimension) = self.parse_unit_with_dimension(&unit);
                ValueType::tensor(
                    rows,
                    cols,
                    unit_str,
                    dimension,
                    primitive
                        .constraints
                        .iter()
                        .map(|c| match c {
                            ast::TensorConstraint::Symmetric => {
                                crate::TensorConstraintIr::Symmetric
                            }
                            ast::TensorConstraint::PositiveDefinite => {
                                crate::TensorConstraintIr::PositiveDefinite
                            }
                        })
                        .collect(),
                )
            }
            continuum_foundation::PrimitiveShape::Grid => {
                let width = self.param_u32(primitive, PrimitiveParamKind::Width);
                let height = self.param_u32(primitive, PrimitiveParamKind::Height);
                let element_type = self.param_element_type(primitive);
                ValueType::grid(width, height, self.lower_type_expr(element_type))
            }
            continuum_foundation::PrimitiveShape::Seq => {
                let element_type = self.param_element_type(primitive);
                ValueType::seq(
                    self.lower_type_expr(element_type),
                    primitive
                        .seq_constraints
                        .iter()
                        .map(|c| self.lower_seq_constraint(c))
                        .collect(),
                )
            }
        }
    }

    fn param_unit(&self, primitive: &PrimitiveTypeExpr) -> String {
        match self.param_value(primitive, PrimitiveParamKind::Unit) {
            Some(PrimitiveParamValue::Unit(unit)) => unit.clone(),
            _ => "".to_string(),
        }
    }

    fn param_range(
        &self,
        primitive: &PrimitiveTypeExpr,
        kind: PrimitiveParamKind,
    ) -> Option<ValueRange> {
        match self.param_value(primitive, kind) {
            Some(PrimitiveParamValue::Range(range)) => Some(ValueRange {
                min: range.min,
                max: range.max,
            }),
            Some(PrimitiveParamValue::Magnitude(range)) => Some(ValueRange {
                min: range.min,
                max: range.max,
            }),
            _ => None,
        }
    }

    fn param_u8(&self, primitive: &PrimitiveTypeExpr, kind: PrimitiveParamKind) -> u8 {
        match self.param_value(primitive, kind) {
            Some(PrimitiveParamValue::Rows(value)) => *value,
            Some(PrimitiveParamValue::Cols(value)) => *value,
            _ => panic!("missing required integer parameter"),
        }
    }

    fn param_u32(&self, primitive: &PrimitiveTypeExpr, kind: PrimitiveParamKind) -> u32 {
        match self.param_value(primitive, kind) {
            Some(PrimitiveParamValue::Width(value)) => *value,
            Some(PrimitiveParamValue::Height(value)) => *value,
            _ => panic!("missing required integer parameter"),
        }
    }

    fn param_element_type<'a>(&self, primitive: &'a PrimitiveTypeExpr) -> &'a TypeExpr {
        match self.param_value(primitive, PrimitiveParamKind::ElementType) {
            Some(PrimitiveParamValue::ElementType(inner)) => inner.as_ref(),
            _ => {
                panic!("missing element_type parameter");
            }
        }
    }

    fn param_value<'a>(
        &self,
        primitive: &'a PrimitiveTypeExpr,
        kind: PrimitiveParamKind,
    ) -> Option<&'a PrimitiveParamValue> {
        primitive.params.iter().find(|param| param.kind() == kind)
    }

    #[allow(dead_code)]
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
            Expr::FieldAccess { object, field } => {
                // Detect dt.raw pattern
                if let Expr::Path(path) = &object.node {
                    if path.segments.len() == 1 && path.segments[0] == "dt" && field == "raw" {
                        return true;
                    }
                }
                self.expr_uses_dt_raw(&object.node)
            }
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
            // Vector literal check
            Expr::Vector(elems) => elems.iter().any(|e| self.expr_uses_dt_raw(&e.node)),
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
