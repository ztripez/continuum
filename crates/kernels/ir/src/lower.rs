//! AST to IR lowering
//!
//! Transforms the DSL AST into a typed IR suitable for DAG construction.

use indexmap::IndexMap;
use thiserror::Error;

use continuum_dsl::ast::{
    self, BinaryOp, CompilationUnit, Expr, Item, Literal, OperatorBody, OperatorPhase,
    StrataStateKind, Topology, TypeExpr, UnaryOp,
};
use continuum_foundation::{
    EraId, FieldId, FractureId, ImpulseId, OperatorId, SignalId, StratumId,
};

use crate::{
    BinaryOpIr, CompiledEmit, CompiledEra, CompiledExpr, CompiledField, CompiledFracture,
    CompiledImpulse, CompiledOperator, CompiledSignal, CompiledStratum, CompiledTransition,
    CompiledWarmup, CompiledWorld, OperatorPhaseIr, StratumStateIr, TopologyIr, UnaryOpIr,
    ValueType,
};

/// Errors that can occur during lowering
#[derive(Debug, Error)]
pub enum LowerError {
    #[error("undefined stratum: {0}")]
    UndefinedStratum(String),

    #[error("undefined signal: {0}")]
    UndefinedSignal(String),

    #[error("duplicate definition: {0}")]
    DuplicateDefinition(String),

    #[error("missing required field: {0}")]
    MissingRequiredField(String),

    #[error("invalid expression: {0}")]
    InvalidExpression(String),
}

/// Lower a compilation unit to IR
pub fn lower(unit: &CompilationUnit) -> Result<CompiledWorld, LowerError> {
    let mut lowerer = Lowerer::new();
    lowerer.lower_unit(unit)?;
    Ok(lowerer.finish())
}

struct Lowerer {
    constants: IndexMap<String, f64>,
    config: IndexMap<String, f64>,
    strata: IndexMap<StratumId, CompiledStratum>,
    eras: IndexMap<EraId, CompiledEra>,
    signals: IndexMap<SignalId, CompiledSignal>,
    fields: IndexMap<FieldId, CompiledField>,
    operators: IndexMap<OperatorId, CompiledOperator>,
    impulses: IndexMap<ImpulseId, CompiledImpulse>,
    fractures: IndexMap<FractureId, CompiledFracture>,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            constants: IndexMap::new(),
            config: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
        }
    }

    fn finish(self) -> CompiledWorld {
        CompiledWorld {
            constants: self.constants,
            config: self.config,
            strata: self.strata,
            eras: self.eras,
            signals: self.signals,
            fields: self.fields,
            operators: self.operators,
            impulses: self.impulses,
            fractures: self.fractures,
        }
    }

    fn lower_unit(&mut self, unit: &CompilationUnit) -> Result<(), LowerError> {
        // First pass: collect constants, config, and strata
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

        // Third pass: signals, fields, operators, impulses, fractures
        for item in &unit.items {
            match &item.node {
                Item::SignalDef(def) => self.lower_signal(def)?,
                Item::FieldDef(def) => self.lower_field(def)?,
                Item::OperatorDef(def) => self.lower_operator(def)?,
                Item::ImpulseDef(def) => self.lower_impulse(def)?,
                Item::FractureDef(def) => self.lower_fracture(def)?,
                _ => {}
            }
        }

        Ok(())
    }

    fn lower_era(&mut self, def: &ast::EraDef) -> Result<(), LowerError> {
        let id = EraId::from(def.name.node.as_str());

        // Convert dt to seconds
        let dt_seconds = def
            .dt
            .as_ref()
            .map(|dt| self.value_with_unit_to_seconds(&dt.node))
            .unwrap_or(1.0);

        // Convert strata states
        let mut strata_states = IndexMap::new();
        for state in &def.strata_states {
            let stratum_id = StratumId::from(state.strata.node.join(".").as_str());
            let ir_state = match &state.state {
                StrataStateKind::Active => StratumStateIr::Active,
                StrataStateKind::ActiveWithStride(s) => StratumStateIr::ActiveWithStride(*s),
                StrataStateKind::Gated => StratumStateIr::Gated,
            };
            strata_states.insert(stratum_id, ir_state);
        }

        // Convert transitions
        let transitions = def
            .transitions
            .iter()
            .map(|t| {
                // Combine all conditions with AND
                let condition = if t.conditions.is_empty() {
                    CompiledExpr::Literal(1.0) // always true
                } else if t.conditions.len() == 1 {
                    self.lower_expr(&t.conditions[0].node)
                } else {
                    t.conditions.iter().skip(1).fold(
                        self.lower_expr(&t.conditions[0].node),
                        |acc, cond| CompiledExpr::Binary {
                            op: BinaryOpIr::And,
                            left: Box::new(acc),
                            right: Box::new(self.lower_expr(&cond.node)),
                        },
                    )
                };

                CompiledTransition {
                    target_era: EraId::from(t.target.node.join(".").as_str()),
                    condition,
                }
            })
            .collect();

        let era = CompiledEra {
            id: id.clone(),
            is_initial: def.is_initial,
            is_terminal: def.is_terminal,
            title: def.title.as_ref().map(|s| s.node.clone()),
            dt_seconds,
            strata_states,
            transitions,
        };

        self.eras.insert(id, era);
        Ok(())
    }

    fn lower_signal(&mut self, def: &ast::SignalDef) -> Result<(), LowerError> {
        let id = SignalId::from(def.path.node.join(".").as_str());

        // Determine stratum
        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Collect signal dependencies from resolve expression
        let mut reads = Vec::new();
        if let Some(resolve) = &def.resolve {
            self.collect_signal_refs(&resolve.body.node, &mut reads);
        }

        // Lower warmup if present
        let warmup = def.warmup.as_ref().map(|w| CompiledWarmup {
            iterations: w.iterations.node,
            convergence: w.convergence.as_ref().map(|c| c.node),
            iterate: self.lower_expr(&w.iterate.node),
        });

        // Lower resolve expression
        let resolve = def.resolve.as_ref().map(|r| self.lower_expr(&r.body.node));

        let signal = CompiledSignal {
            id: id.clone(),
            stratum,
            title: def.title.as_ref().map(|s| s.node.clone()),
            symbol: def.symbol.as_ref().map(|s| s.node.clone()),
            value_type: def
                .ty
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar),
            uses_dt_raw: def.dt_raw,
            reads,
            resolve,
            warmup,
        };

        self.signals.insert(id, signal);
        Ok(())
    }

    fn lower_field(&mut self, def: &ast::FieldDef) -> Result<(), LowerError> {
        let id = FieldId::from(def.path.node.join(".").as_str());

        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

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
                .unwrap_or(ValueType::Scalar),
            reads,
            measure: def.measure.as_ref().map(|m| self.lower_expr(&m.body.node)),
        };

        self.fields.insert(id, field);
        Ok(())
    }

    fn lower_operator(&mut self, def: &ast::OperatorDef) -> Result<(), LowerError> {
        let id = OperatorId::from(def.path.node.join(".").as_str());

        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        let phase = def
            .phase
            .as_ref()
            .map(|p| self.lower_operator_phase(&p.node))
            .or_else(|| {
                def.body.as_ref().map(|b| match b {
                    OperatorBody::Warmup(_) => OperatorPhaseIr::Warmup,
                    OperatorBody::Collect(_) => OperatorPhaseIr::Collect,
                    OperatorBody::Measure(_) => OperatorPhaseIr::Measure,
                })
            })
            .unwrap_or(OperatorPhaseIr::Collect);

        let body_expr = def.body.as_ref().map(|b| match b {
            OperatorBody::Warmup(e) | OperatorBody::Collect(e) | OperatorBody::Measure(e) => {
                &e.node
            }
        });

        let mut reads = Vec::new();
        if let Some(expr) = body_expr {
            self.collect_signal_refs(expr, &mut reads);
        }

        let operator = CompiledOperator {
            id: id.clone(),
            stratum,
            phase,
            reads,
            body: body_expr.map(|e| self.lower_expr(e)),
        };

        self.operators.insert(id, operator);
        Ok(())
    }

    fn lower_impulse(&mut self, def: &ast::ImpulseDef) -> Result<(), LowerError> {
        let id = ImpulseId::from(def.path.node.join(".").as_str());

        let impulse = CompiledImpulse {
            id: id.clone(),
            payload_type: def
                .payload_type
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar),
            apply: def.apply.as_ref().map(|a| self.lower_expr(&a.body.node)),
        };

        self.impulses.insert(id, impulse);
        Ok(())
    }

    fn lower_fracture(&mut self, def: &ast::FractureDef) -> Result<(), LowerError> {
        let id = FractureId::from(def.path.node.join(".").as_str());

        let mut reads = Vec::new();
        for cond in &def.conditions {
            self.collect_signal_refs(&cond.node, &mut reads);
        }
        for emit in &def.emit {
            self.collect_signal_refs(&emit.value.node, &mut reads);
        }

        let fracture = CompiledFracture {
            id: id.clone(),
            reads,
            conditions: def.conditions.iter().map(|c| self.lower_expr(&c.node)).collect(),
            emits: def
                .emit
                .iter()
                .map(|e| CompiledEmit {
                    target: SignalId::from(e.target.node.join(".").as_str()),
                    value: self.lower_expr(&e.value.node),
                })
                .collect(),
        };

        self.fractures.insert(id, fracture);
        Ok(())
    }

    fn lower_expr(&self, expr: &Expr) -> CompiledExpr {
        match expr {
            Expr::Literal(lit) => CompiledExpr::Literal(self.literal_to_f64_unchecked(lit)),
            Expr::LiteralWithUnit { value, .. } => {
                CompiledExpr::Literal(self.literal_to_f64_unchecked(value))
            }
            Expr::Prev | Expr::PrevField(_) => CompiledExpr::Prev,
            Expr::DtRaw => CompiledExpr::DtRaw,
            Expr::SumInputs => CompiledExpr::SumInputs,
            Expr::Path(path) => {
                // Could be signal, const, or config reference
                let joined = path.join(".");
                if self.constants.contains_key(&joined) {
                    CompiledExpr::Const(joined)
                } else if self.config.contains_key(&joined) {
                    CompiledExpr::Config(joined)
                } else {
                    CompiledExpr::Signal(SignalId::from(joined.as_str()))
                }
            }
            Expr::SignalRef(path) => {
                CompiledExpr::Signal(SignalId::from(path.join(".").as_str()))
            }
            Expr::ConstRef(path) => CompiledExpr::Const(path.join(".")),
            Expr::ConfigRef(path) => CompiledExpr::Config(path.join(".")),
            Expr::Binary { op, left, right } => CompiledExpr::Binary {
                op: self.lower_binary_op(*op),
                left: Box::new(self.lower_expr(&left.node)),
                right: Box::new(self.lower_expr(&right.node)),
            },
            Expr::Unary { op, operand } => CompiledExpr::Unary {
                op: self.lower_unary_op(*op),
                operand: Box::new(self.lower_expr(&operand.node)),
            },
            Expr::Call { function, args } => {
                let func_name = self.expr_to_function_name(&function.node);
                CompiledExpr::Call {
                    function: func_name,
                    args: args.iter().map(|a| self.lower_expr(&a.node)).collect(),
                }
            }
            Expr::FieldAccess { object, field } => CompiledExpr::FieldAccess {
                object: Box::new(self.lower_expr(&object.node)),
                field: field.clone(),
            },
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => CompiledExpr::If {
                condition: Box::new(self.lower_expr(&condition.node)),
                then_branch: Box::new(self.lower_expr(&then_branch.node)),
                else_branch: Box::new(
                    else_branch
                        .as_ref()
                        .map(|e| self.lower_expr(&e.node))
                        .unwrap_or(CompiledExpr::Literal(0.0)),
                ),
            },
            Expr::Let { name, value, body } => CompiledExpr::Let {
                name: name.clone(),
                value: Box::new(self.lower_expr(&value.node)),
                body: Box::new(self.lower_expr(&body.node)),
            },
            Expr::MathConst(c) => {
                let val = match c {
                    ast::MathConst::Pi => std::f64::consts::PI,
                    ast::MathConst::Tau => std::f64::consts::TAU,
                    ast::MathConst::E => std::f64::consts::E,
                    ast::MathConst::Phi => 1.618_033_988_749_895,
                    ast::MathConst::I => 0.0, // imaginary unit not directly representable
                };
                CompiledExpr::Literal(val)
            }
            // Block, For, Map, Fold, Struct, EmitSignal, EmitField, FieldRef, Payload, PayloadField
            // These require more complex lowering or are handled specially
            Expr::Block(exprs) => {
                if exprs.is_empty() {
                    CompiledExpr::Literal(0.0)
                } else {
                    // For now, just evaluate to the last expression
                    self.lower_expr(&exprs.last().unwrap().node)
                }
            }
            _ => CompiledExpr::Literal(0.0), // placeholder for complex expressions
        }
    }

    fn expr_to_function_name(&self, expr: &Expr) -> String {
        match expr {
            Expr::Path(path) => path.join("."),
            Expr::FieldAccess { object, field } => {
                format!("{}.{}", self.expr_to_function_name(&object.node), field)
            }
            _ => "unknown".to_string(),
        }
    }

    fn collect_signal_refs(&self, expr: &Expr, refs: &mut Vec<SignalId>) {
        match expr {
            Expr::SignalRef(path) => {
                let id = SignalId::from(path.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
            }
            Expr::Path(path) => {
                let joined = path.join(".");
                if !self.constants.contains_key(&joined) && !self.config.contains_key(&joined) {
                    let id = SignalId::from(joined.as_str());
                    if !refs.contains(&id) {
                        refs.push(id);
                    }
                }
            }
            Expr::Binary { left, right, .. } => {
                self.collect_signal_refs(&left.node, refs);
                self.collect_signal_refs(&right.node, refs);
            }
            Expr::Unary { operand, .. } => {
                self.collect_signal_refs(&operand.node, refs);
            }
            Expr::Call { function, args } => {
                self.collect_signal_refs(&function.node, refs);
                for arg in args {
                    self.collect_signal_refs(&arg.node, refs);
                }
            }
            Expr::FieldAccess { object, .. } => {
                self.collect_signal_refs(&object.node, refs);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_signal_refs(&condition.node, refs);
                self.collect_signal_refs(&then_branch.node, refs);
                if let Some(eb) = else_branch {
                    self.collect_signal_refs(&eb.node, refs);
                }
            }
            Expr::Let { value, body, .. } => {
                self.collect_signal_refs(&value.node, refs);
                self.collect_signal_refs(&body.node, refs);
            }
            Expr::Block(exprs) => {
                for e in exprs {
                    self.collect_signal_refs(&e.node, refs);
                }
            }
            Expr::For { iter, body, .. } => {
                self.collect_signal_refs(&iter.node, refs);
                self.collect_signal_refs(&body.node, refs);
            }
            Expr::Map { sequence, function } | Expr::Fold { sequence, function, .. } => {
                self.collect_signal_refs(&sequence.node, refs);
                self.collect_signal_refs(&function.node, refs);
            }
            Expr::EmitSignal { value, .. } | Expr::EmitField { value, .. } => {
                self.collect_signal_refs(&value.node, refs);
            }
            Expr::Struct(fields) => {
                for (_, v) in fields {
                    self.collect_signal_refs(&v.node, refs);
                }
            }
            _ => {}
        }
    }

    fn lower_binary_op(&self, op: BinaryOp) -> BinaryOpIr {
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

    fn lower_unary_op(&self, op: UnaryOp) -> UnaryOpIr {
        match op {
            UnaryOp::Neg => UnaryOpIr::Neg,
            UnaryOp::Not => UnaryOpIr::Not,
        }
    }

    fn lower_type_expr(&self, ty: &TypeExpr) -> ValueType {
        match ty {
            TypeExpr::Scalar { .. } => ValueType::Scalar,
            TypeExpr::Vector { dim, .. } => match dim {
                2 => ValueType::Vec2,
                3 => ValueType::Vec3,
                4 => ValueType::Vec4,
                _ => ValueType::Scalar,
            },
            TypeExpr::Named(_) => ValueType::Scalar, // resolve named types later
        }
    }

    fn lower_topology(&self, topo: &Topology) -> TopologyIr {
        match topo {
            Topology::SphereSurface => TopologyIr::SphereSurface,
            Topology::PointCloud => TopologyIr::PointCloud,
            Topology::Volume => TopologyIr::Volume,
        }
    }

    fn lower_operator_phase(&self, phase: &OperatorPhase) -> OperatorPhaseIr {
        match phase {
            OperatorPhase::Warmup => OperatorPhaseIr::Warmup,
            OperatorPhase::Collect => OperatorPhaseIr::Collect,
            OperatorPhase::Measure => OperatorPhaseIr::Measure,
        }
    }

    fn literal_to_f64(&self, lit: &Literal) -> Result<f64, LowerError> {
        match lit {
            Literal::Integer(i) => Ok(*i as f64),
            Literal::Float(f) => Ok(*f),
            Literal::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            Literal::String(_) => Err(LowerError::InvalidExpression(
                "string cannot be converted to f64".to_string(),
            )),
        }
    }

    fn literal_to_f64_unchecked(&self, lit: &Literal) -> f64 {
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

    fn value_with_unit_to_seconds(&self, vwu: &ast::ValueWithUnit) -> f64 {
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
            "myr" | "Ma" => base * 31_557_600_000_000.0,
            "byr" | "Ga" => base * 31_557_600_000_000_000.0,
            _ => base, // assume seconds for unknown units
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_dsl::parse;

    #[test]
    fn test_lower_empty() {
        let unit = CompilationUnit::default();
        let world = lower(&unit).unwrap();
        assert!(world.signals.is_empty());
        assert!(world.strata.is_empty());
    }

    #[test]
    fn test_lower_const() {
        let src = r#"
            const {
                physics.gravity: 9.81
            }
        "#;
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        let unit = unit.unwrap();
        let world = lower(&unit).unwrap();
        assert_eq!(world.constants.get("physics.gravity"), Some(&9.81));
    }

    #[test]
    fn test_lower_strata() {
        let src = r#"
            strata.terra {
                : title("Terra")
                : stride(10)
            }
        "#;
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        let unit = unit.unwrap();
        let world = lower(&unit).unwrap();
        let terra = world.strata.get(&StratumId::from("terra")).unwrap();
        assert_eq!(terra.title, Some("Terra".to_string()));
        assert_eq!(terra.default_stride, 10);
    }
}
