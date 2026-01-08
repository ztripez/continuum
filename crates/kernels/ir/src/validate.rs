//! IR validation
//!
//! Validates the compiled IR and emits warnings for potential issues.

use tracing::warn;

use crate::{CompiledWorld, ValueType};

/// A compilation warning
#[derive(Debug, Clone)]
pub struct CompileWarning {
    /// Warning code for filtering/identification
    pub code: WarningCode,
    /// Human-readable message
    pub message: String,
    /// The entity this warning relates to (signal path, operator path, etc.)
    pub entity: String,
}

/// Warning codes for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningCode {
    /// Signal has range constraint but no assertions to validate it
    MissingRangeAssertion,
}

/// Validate a compiled world and return any warnings
pub fn validate(world: &CompiledWorld) -> Vec<CompileWarning> {
    let mut warnings = Vec::new();

    check_range_assertions(world, &mut warnings);

    // Log warnings
    for warning in &warnings {
        warn!(
            code = ?warning.code,
            entity = %warning.entity,
            "{}",
            warning.message
        );
    }

    warnings
}

/// Check that signals with range types have assertions
fn check_range_assertions(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    for (signal_id, signal) in &world.signals {
        // Check if the signal has a range constraint
        let has_range = matches!(&signal.value_type, ValueType::Scalar { range: Some(_) });

        if has_range && signal.assertions.is_empty() {
            warnings.push(CompileWarning {
                code: WarningCode::MissingRangeAssertion,
                message: format!(
                    "signal '{}' has a range constraint but no assertions to validate it at runtime",
                    signal_id.0
                ),
                entity: signal_id.0.clone(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower;
    use continuum_dsl::parse;

    fn parse_and_lower(src: &str) -> CompiledWorld {
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        lower(&unit.unwrap()).unwrap()
    }

    #[test]
    fn test_signal_with_range_no_assertion_warns() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K, 100..10000>
                : strata(terra)
                resolve { prev }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, WarningCode::MissingRangeAssertion);
        assert!(warnings[0].entity.contains("terra.temp"));
    }

    #[test]
    fn test_signal_with_range_and_assertion_no_warning() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K, 100..10000>
                : strata(terra)
                resolve { prev }
                assert {
                    prev > 100
                }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert!(warnings.is_empty());
    }

    #[test]
    fn test_signal_without_range_no_warning() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { prev }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert!(warnings.is_empty());
    }
}
