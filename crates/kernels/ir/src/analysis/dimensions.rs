use crate::units::{DimensionError, Unit};
use crate::{BinaryOpIr, CompiledExpr, CompiledWorld, DtRobustOperator, UnaryOpIr, ValueType};
use continuum_foundation::Path;
use std::collections::HashMap;

/// A dimensional analysis error with source context.
#[derive(Debug, Clone)]
pub struct DimensionalDiagnostic {
    pub message: String,
    pub path: Path,
    pub error: DimensionError,
}

/// Infer dimensions for all signals and check expression consistency.
pub fn analyze_dimensions(world: &CompiledWorld) -> Vec<DimensionalDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut symbol_units = HashMap::new();

    // 1. Collect declared units for signals and members
    let signals = world.signals();
    for (id, signal) in &signals {
        if let Some(unit) = get_unit_from_type(&signal.value_type) {
            symbol_units.insert(id.to_string(), unit);
        }
    }

    let members = world.members();
    for (id, member) in &members {
        if let Some(unit) = get_unit_from_type(&member.value_type) {
            symbol_units.insert(id.to_string(), unit);
        }
    }

    // 2. Check expressions
    for (id, signal) in &signals {
        if let Some(resolve) = &signal.resolve {
            // Check internal consistency
            if let Err(e) = check_expr(resolve, world, &symbol_units, Some(id.to_string())) {
                diagnostics.push(DimensionalDiagnostic {
                    message: format!("dimensional error in signal '{}': {}", id, e),
                    path: id.to_string().into(),
                    error: e,
                });
            } else {
                // Verify resolve expression matches signal's declared unit
                if let Ok(inferred) =
                    infer_unit(resolve, world, &symbol_units, Some(id.to_string()))
                {
                    if let Some(declared) = symbol_units.get(&id.to_string()) {
                        if inferred != *declared {
                            let err = DimensionError::IncompatibleUnits {
                                expected: *declared,
                                found: inferred,
                                operation: format!("signal '{}' resolution", id),
                            };
                            diagnostics.push(DimensionalDiagnostic {
                                message: format!("{}", err),
                                path: id.to_string().into(),
                                error: err,
                            });
                        }
                    }
                }
            }
        }
    }

    diagnostics
}

fn get_unit_from_type(ty: &ValueType) -> Option<Unit> {
    match ty {
        ValueType::Scalar { dimension, .. } => *dimension,
        ValueType::Vec2 { dimension, .. } => *dimension,
        ValueType::Vec3 { dimension, .. } => *dimension,
        ValueType::Vec4 { dimension, .. } => *dimension,
        ValueType::Tensor { dimension, .. } => *dimension,
        _ => None,
    }
}

fn infer_unit(
    expr: &CompiledExpr,
    world: &CompiledWorld,
    symbol_units: &HashMap<String, Unit>,
    current_signal: Option<String>,
) -> Result<Unit, DimensionError> {
    match expr {
        CompiledExpr::Literal(_) => Ok(Unit::dimensionless()),
        CompiledExpr::Prev => {
            if let Some(ref id) = current_signal {
                Ok(symbol_units.get(id).cloned().unwrap_or_default())
            } else {
                Ok(Unit::dimensionless())
            }
        }
        CompiledExpr::DtRaw | CompiledExpr::SimTime => Ok(Unit::parse("s").unwrap()),
        CompiledExpr::Collected => Ok(Unit::dimensionless()), // depends on impulses

        CompiledExpr::Signal(id) => Ok(symbol_units
            .get(&id.to_string())
            .cloned()
            .unwrap_or_default()),
        CompiledExpr::Const(_) | CompiledExpr::Config(_) => Ok(Unit::dimensionless()),

        CompiledExpr::Binary { op, left, right } => {
            let u_left = infer_unit(left, world, symbol_units, current_signal.clone())?;
            let u_right = infer_unit(right, world, symbol_units, current_signal.clone())?;

            match op {
                BinaryOpIr::Add | BinaryOpIr::Sub => {
                    // Polymorphic Literals: If one side is a unitless literal, it adopts the other side's unit.
                    if u_left.is_dimensionless() && !u_right.is_dimensionless() {
                        return Ok(u_right);
                    }
                    if u_right.is_dimensionless() && !u_left.is_dimensionless() {
                        return Ok(u_left);
                    }

                    if u_left == u_right {
                        Ok(u_left)
                    } else {
                        Err(DimensionError::IncompatibleUnits {
                            expected: u_left,
                            found: u_right,
                            operation: format!("{:?}", op),
                        })
                    }
                }
                BinaryOpIr::Mul => Ok(u_left.multiply(&u_right)),
                BinaryOpIr::Div => Ok(u_left.divide(&u_right)),
                BinaryOpIr::Pow => {
                    // Only support integer powers for dimensional analysis for now
                    if let CompiledExpr::Literal(val) = &**right {
                        if val.fract() == 0.0 {
                            Ok(u_left.power(*val as i8))
                        } else {
                            Ok(Unit::dimensionless()) // Fractional powers lose unit safety for now
                        }
                    } else {
                        Ok(Unit::dimensionless())
                    }
                }
                _ => Ok(Unit::dimensionless()), // Comparisons etc return dimensionless (boolean)
            }
        }

        CompiledExpr::Unary { op, operand } => {
            let u = infer_unit(operand, world, symbol_units, current_signal)?;
            match op {
                UnaryOpIr::Neg => Ok(u),
                UnaryOpIr::Not => Ok(Unit::dimensionless()),
            }
        }

        CompiledExpr::KernelCall { function, args } => {
            match function.as_str() {
                "sin" | "cos" | "tan" => {
                    let u = infer_unit(&args[0], world, symbol_units, current_signal)?;
                    if u.is_angle() || u.is_dimensionless() {
                        Ok(Unit::dimensionless())
                    } else {
                        Err(DimensionError::RequiresAngle {
                            function: function.clone(),
                            found: u,
                        })
                    }
                }
                "abs" | "min" | "max" | "clamp" => {
                    // All args should match
                    let u0 = infer_unit(&args[0], world, symbol_units, current_signal.clone())?;
                    for arg in args.iter().skip(1) {
                        let ui = infer_unit(arg, world, symbol_units, current_signal.clone())?;
                        if ui != u0 {
                            return Err(DimensionError::IncompatibleUnits {
                                expected: u0,
                                found: ui,
                                operation: format!("kernel call '{}'", function),
                            });
                        }
                    }
                    Ok(u0)
                }
                "sqrt" => {
                    let u = infer_unit(&args[0], world, symbol_units, current_signal)?;
                    u.sqrt().ok_or(DimensionError::InvalidSqrt { unit: u })
                }
                _ => Ok(Unit::dimensionless()),
            }
        }

        CompiledExpr::DtRobustCall { operator, args, .. } => {
            match operator {
                DtRobustOperator::Integrate => {
                    // integrate(prev, rate) -> prev + rate * dt
                    let u_prev = infer_unit(&args[0], world, symbol_units, current_signal.clone())?;
                    let u_rate = infer_unit(&args[1], world, symbol_units, current_signal)?;
                    let u_dt = Unit::parse("s").unwrap();
                    let u_integrated = u_rate.multiply(&u_dt);

                    if u_prev == u_integrated {
                        Ok(u_prev)
                    } else {
                        Err(DimensionError::IncompatibleUnits {
                            expected: u_prev,
                            found: u_integrated,
                            operation: "integrate".to_string(),
                        })
                    }
                }
                DtRobustOperator::Decay => {
                    // decay(value, halflife) -> value * 0.5^(dt/halflife)
                    let u_val = infer_unit(&args[0], world, symbol_units, current_signal.clone())?;
                    let u_half = infer_unit(&args[1], world, symbol_units, current_signal)?;
                    let u_dt = Unit::parse("s").unwrap();
                    if u_half == u_dt || u_half.is_dimensionless() {
                        Ok(u_val)
                    } else {
                        Err(DimensionError::IncompatibleUnits {
                            expected: u_dt,
                            found: u_half,
                            operation: "decay halflife".to_string(),
                        })
                    }
                }
                _ => Ok(Unit::dimensionless()),
            }
        }

        CompiledExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let u_then = infer_unit(then_branch, world, symbol_units, current_signal.clone())?;
            let u_else = infer_unit(else_branch, world, symbol_units, current_signal)?;
            if u_then == u_else {
                Ok(u_then)
            } else {
                Err(DimensionError::IncompatibleUnits {
                    expected: u_then,
                    found: u_else,
                    operation: "if-else branches".to_string(),
                })
            }
        }

        _ => Ok(Unit::dimensionless()),
    }
}

fn check_expr(
    expr: &CompiledExpr,
    world: &CompiledWorld,
    symbol_units: &HashMap<String, Unit>,
    current_signal: Option<String>,
) -> Result<(), DimensionError> {
    // Just try inferring it, which runs all recursive checks
    infer_unit(expr, world, symbol_units, current_signal)?;
    Ok(())
}
