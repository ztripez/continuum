//! Declaration round-trip tests.
//!
//! These tests verify that all 13 declaration types can be parsed correctly
//! by checking the resulting AST structure.
//!
//! ## Declaration Types Tested
//!
//! 1. Signal (global node)
//! 2. Field (global node)
//! 3. Operator (global node)
//! 4. Impulse (global node)
//! 5. Fracture (global node)
//! 6. Chronicle (global node)
//! 7. Member (per-entity signal/field)
//! 8. Entity
//! 9. Stratum
//! 10. Era
//! 11. Type
//! 12. World
//! 13. Const/Config blocks

use continuum_cdsl_ast::{BlockBody, Declaration, RoleData, Stmt};
use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::parse_declarations;
use logos::Logos;

/// Helper to parse declarations from source with real byte spans.
fn parse(source: &str) -> Vec<Declaration> {
    let tokens: Vec<_> = Token::lexer(source)
        .spanned()
        .filter_map(|(r, s)| r.ok().map(|t| (t, s)))
        .collect();
    parse_declarations(&tokens, 0).expect("Parse failed")
}

// =============================================================================
// Global Primitives (Node declarations)
// =============================================================================

#[test]
fn test_signal_declaration() {
    let source = r#"
        signal temperature {
            : title("Temperature")
            
            resolve { 300.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "temperature");
            assert!(matches!(node.role, RoleData::Signal));
            assert!(!node.attributes.is_empty());
            assert_eq!(node.execution_blocks.len(), 1);
            assert_eq!(node.execution_blocks[0].0, "resolve");
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_field_declaration() {
    let source = r#"
        field velocity {
            : title("Velocity")
            
            measure { [0.0, 0.0, 0.0] }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "velocity");
            assert!(matches!(node.role, RoleData::Field { .. }));
            assert_eq!(node.execution_blocks.len(), 1);
            assert_eq!(node.execution_blocks[0].0, "measure");
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_operator_declaration() {
    let source = r#"
        operator physics.update {
            : strata(physics)
            
            resolve { 1.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "physics.update");
            assert!(matches!(node.role, RoleData::Operator));
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_impulse_declaration() {
    let source = r#"
        impulse spawn_entity {
            : strata(genesis)
            
            resolve { 1.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "spawn_entity");
            assert!(matches!(node.role, RoleData::Impulse { .. }));
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_fracture_declaration() {
    let source = r#"
        fracture stress_limit {
            : strata(tectonics)
            
            resolve { 1.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "stress_limit");
            assert!(matches!(node.role, RoleData::Fracture));
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_chronicle_declaration() {
    let source = r#"
        chronicle formation_events {
            : strata(genesis)
            
            measure { 1.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "formation_events");
            assert!(matches!(node.role, RoleData::Chronicle));
        }
        _ => panic!("Expected Node declaration"),
    }
}

// =============================================================================
// Entity Declaration (includes nested members)
// =============================================================================
//
// NOTE: Standalone member declarations (e.g., "member plate.velocity { ... }")
// were deprecated and removed. Members are now declared inside entity blocks
// using nested syntax. See test_entity_declaration below for examples.

#[test]
fn test_entity_declaration() {
    let source = r#"
        entity plate {}
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Entity(entity) => {
            assert_eq!(entity.path.to_string(), "plate");
            assert_eq!(entity.id.to_string(), "plate");
        }
        _ => panic!("Expected Entity declaration"),
    }
}

// =============================================================================
// Stratum Declaration
// =============================================================================

#[test]
fn test_stratum_declaration() {
    let source = r#"
        strata tectonics {
            : stride(1)
            : title("Plate Tectonics")
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Stratum(stratum) => {
            assert_eq!(stratum.path.to_string(), "tectonics");
            assert_eq!(stratum.id.to_string(), "tectonics");
            assert!(stratum.attributes.len() >= 2);
        }
        _ => panic!("Expected Stratum declaration"),
    }
}

// =============================================================================
// Era Declaration
// =============================================================================

#[test]
fn test_era_declaration() {
    let source = r#"
        era formation {
            : initial
            : title("Formation")
            : dt(1.0)
            
            strata {
                genesis: active
                rotation: active
            }
            
            transition cooling when { time > 100.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Era(era) => {
            assert_eq!(era.path.to_string(), "formation");
            assert!(era.attributes.len() >= 2); // At least initial and title
            assert_eq!(era.strata_policy.len(), 2);
            assert_eq!(era.transitions.len(), 1);
            assert!(era.dt.is_some());
        }
        _ => panic!("Expected Era declaration"),
    }
}

// =============================================================================
// Type Declaration
// =============================================================================

#[test]
fn test_type_declaration() {
    let source = r#"
        type PlateData {
            mass: Scalar
            area: Scalar
            age: Scalar
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Type(type_decl) => {
            assert_eq!(type_decl.name, "PlateData");
            assert_eq!(type_decl.fields.len(), 3);
            assert_eq!(type_decl.fields[0].name, "mass");
            assert_eq!(type_decl.fields[1].name, "area");
            assert_eq!(type_decl.fields[2].name, "age");
        }
        _ => panic!("Expected Type declaration"),
    }
}

// =============================================================================
// World Declaration
// =============================================================================

#[test]
fn test_world_declaration() {
    let source = r#"
        world terra {
            : title("Earth Simulation")
            : version("1.0.0")
            
            warmup {
                : iterations(100)
            }
            
            policy {
                determinism: strict;
                faults: fatal;
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::World(world) => {
            assert_eq!(world.path.to_string(), "terra");
            assert_eq!(world.title, Some("Earth Simulation".to_string()));
            assert_eq!(world.version, Some("1.0.0".to_string()));
            assert!(world.warmup.is_some());
        }
        _ => panic!("Expected World declaration"),
    }
}

// =============================================================================
// Const Block
// =============================================================================

#[test]
fn test_const_block() {
    let source = r#"
        const {
            G: Scalar = 6.674e-11
            c: Scalar = 299792458.0
            earth.radius: Scalar = 6371000.0
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Const(entries) => {
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].path.to_string(), "G");
            assert_eq!(entries[1].path.to_string(), "c");
            assert_eq!(entries[2].path.to_string(), "earth.radius");
        }
        _ => panic!("Expected Const declaration"),
    }
}

// =============================================================================
// Config Block
// =============================================================================

#[test]
fn test_config_block() {
    let source = r#"
        config {
            simulation.timestep: Scalar = 1.0
            output.frequency: Scalar = 100.0
            debug.enabled: Bool
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Config(entries) => {
            assert_eq!(entries.len(), 3);
            assert_eq!(entries[0].path.to_string(), "simulation.timestep");
            assert_eq!(entries[1].path.to_string(), "output.frequency");
            assert_eq!(entries[2].path.to_string(), "debug.enabled");
            assert!(entries[0].default.is_some());
            assert!(entries[1].default.is_some());
            assert!(entries[2].default.is_none());
        }
        _ => panic!("Expected Config declaration"),
    }
}

// =============================================================================
// Type Inference for Const/Config
// =============================================================================

#[test]
fn test_const_type_inference() {
    let source = r#"
        const {
            physics.g: 9.81
            math.pi: 3.14159
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Const(entries) => {
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].path.to_string(), "physics.g");
            assert_eq!(entries[1].path.to_string(), "math.pi");
        }
        _ => panic!("Expected Const declaration"),
    }
}

#[test]
fn test_config_type_inference() {
    let source = r#"
        config {
            sim.timestep: 1.0
            sim.iterations: 100.0
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Config(entries) => {
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].path.to_string(), "sim.timestep");
            assert_eq!(entries[1].path.to_string(), "sim.iterations");
        }
        _ => panic!("Expected Config declaration"),
    }
}

// =============================================================================
// Multiple Declarations
// =============================================================================

#[test]
fn test_multiple_declarations() {
    let source = r#"
        entity plate {}
        
        strata tectonics {
            : stride(1)
        }
        
        signal temperature {
            resolve { 300.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 3);

    assert!(matches!(decls[0], Declaration::Entity(_)));
    assert!(matches!(decls[1], Declaration::Stratum(_)));
    assert!(matches!(decls[2], Declaration::Node(_)));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_blocks() {
    let source = r#"
        entity empty_entity {}
        
        const {}
        
        config {}
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 3);

    match &decls[1] {
        Declaration::Const(entries) => assert_eq!(entries.len(), 0),
        _ => panic!("Expected Const"),
    }

    match &decls[2] {
        Declaration::Config(entries) => assert_eq!(entries.len(), 0),
        _ => panic!("Expected Config"),
    }
}

#[test]
fn test_nested_paths() {
    let source = r#"
        signal earth.atmosphere.temperature {
            resolve { 288.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "earth.atmosphere.temperature");
        }
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_multiple_execution_blocks() {
    let source = r#"
        signal counter {
            resolve { 1.0 }
            collect { 2.0 }
            assert { 3.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.execution_blocks.len(), 3);
            assert_eq!(node.execution_blocks[0].0, "resolve");
            assert_eq!(node.execution_blocks[1].0, "collect");
            assert_eq!(node.execution_blocks[2].0, "assert");
        }
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_assertion_with_severity() {
    let source = r#"
        signal temp {
            assert {
                prev > 0 : fatal
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.execution_blocks.len(), 1);
            assert_eq!(node.execution_blocks[0].0, "assert");

            // Verify it's a statement block
            match &node.execution_blocks[0].1 {
                BlockBody::Statements(stmts) => {
                    assert_eq!(stmts.len(), 1);
                    match &stmts[0] {
                        Stmt::Assert {
                            severity, message, ..
                        } => {
                            assert_eq!(severity.as_deref(), Some("fatal"));
                            assert_eq!(message.as_deref(), None);
                        }
                        _ => panic!("Expected Stmt::Assert"),
                    }
                }
                _ => panic!("Expected BlockBody::Statements for assert block"),
            }
        }
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_assertion_with_message() {
    let source = r#"
        signal temp {
            assert {
                prev > 0 : "temperature must be positive"
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => match &node.execution_blocks[0].1 {
            BlockBody::Statements(stmts) => match &stmts[0] {
                Stmt::Assert {
                    severity, message, ..
                } => {
                    assert_eq!(severity.as_deref(), None);
                    assert_eq!(message.as_deref(), Some("temperature must be positive"));
                }
                _ => panic!("Expected Stmt::Assert"),
            },
            _ => panic!("Expected BlockBody::Statements"),
        },
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_assertion_with_severity_and_message() {
    let source = r#"
        signal temp {
            assert {
                prev > 0 : fatal, "temperature must be positive"
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => match &node.execution_blocks[0].1 {
            BlockBody::Statements(stmts) => match &stmts[0] {
                Stmt::Assert {
                    severity, message, ..
                } => {
                    assert_eq!(severity.as_deref(), Some("fatal"));
                    assert_eq!(message.as_deref(), Some("temperature must be positive"));
                }
                _ => panic!("Expected Stmt::Assert"),
            },
            _ => panic!("Expected BlockBody::Statements"),
        },
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_multiple_assertions_with_mixed_metadata() {
    let source = r#"
        signal temp {
            assert {
                prev > 0;
                prev < 10000 : fatal;
                prev != 273.15 : "not exactly freezing";
                prev >= 100 : warn, "high temperature"
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            match &node.execution_blocks[0].1 {
                BlockBody::Statements(stmts) => {
                    assert_eq!(stmts.len(), 4);

                    // First: no metadata
                    match &stmts[0] {
                        Stmt::Assert {
                            severity, message, ..
                        } => {
                            assert_eq!(severity.as_deref(), None);
                            assert_eq!(message.as_deref(), None);
                        }
                        _ => panic!("Expected Stmt::Assert"),
                    }

                    // Second: severity only
                    match &stmts[1] {
                        Stmt::Assert {
                            severity, message, ..
                        } => {
                            assert_eq!(severity.as_deref(), Some("fatal"));
                            assert_eq!(message.as_deref(), None);
                        }
                        _ => panic!("Expected Stmt::Assert"),
                    }

                    // Third: message only
                    match &stmts[2] {
                        Stmt::Assert {
                            severity, message, ..
                        } => {
                            assert_eq!(severity.as_deref(), None);
                            assert_eq!(message.as_deref(), Some("not exactly freezing"));
                        }
                        _ => panic!("Expected Stmt::Assert"),
                    }

                    // Fourth: both
                    match &stmts[3] {
                        Stmt::Assert {
                            severity, message, ..
                        } => {
                            assert_eq!(severity.as_deref(), Some("warn"));
                            assert_eq!(message.as_deref(), Some("high temperature"));
                        }
                        _ => panic!("Expected Stmt::Assert"),
                    }
                }
                _ => panic!("Expected BlockBody::Statements"),
            }
        }
        _ => panic!("Expected Node"),
    }
}

// =============================================================================
// Edge Case Tests for Assertion Metadata and Span Tracking
// =============================================================================

#[test]
fn test_empty_assertion_block() {
    // Empty assert block is valid (parsed as empty statement list)
    let source = r#"
        signal temp {
            assert {}
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => match &node.execution_blocks[0].1 {
            BlockBody::Statements(stmts) => {
                // Empty assert block results in empty statement list
                assert_eq!(stmts.len(), 0, "Expected empty statement list");
            }
            _ => panic!("Expected BlockBody::Statements"),
        },
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_assertion_condition_ast_verification() {
    // Verify that assertion conditions are parsed as proper expressions
    let source = r#"
        signal velocity {
            assert {
                prev > 0 && prev < 100 : "velocity in valid range"
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => match &node.execution_blocks[0].1 {
            BlockBody::Statements(stmts) => match &stmts[0] {
                Stmt::Assert { condition, .. } => {
                    // Verify condition is parsed as a binary expression AST
                    // (not just a literal or identifier)
                    use continuum_cdsl_ast::UntypedKind;
                    match &condition.kind {
                        UntypedKind::Binary { .. } => {
                            // Successfully parsed as binary operation.
                            // Operator precedence determines exact tree structure,
                            // but we've verified it's not a trivial expression.
                        }
                        _ => panic!(
                            "Expected Binary expression for complex condition, got: {:?}",
                            condition.kind
                        ),
                    }
                }
                _ => panic!("Expected Stmt::Assert"),
            },
            _ => panic!("Expected BlockBody::Statements"),
        },
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_assertion_with_unicode_in_message() {
    // Unicode characters should not break span tracking (multi-byte chars)
    let source = r#"
        signal temp {
            assert {
                prev > 0 : "温度必须为正 (temperature must be positive) 🌡️"
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => match &node.execution_blocks[0].1 {
            BlockBody::Statements(stmts) => match &stmts[0] {
                Stmt::Assert { message, .. } => {
                    assert!(message
                        .as_ref()
                        .expect("test: assert message should be present")
                        .contains("温度必须为正 (temperature must be positive) 🌡️"));
                }
                _ => panic!("Expected Stmt::Assert"),
            },
            _ => panic!("Expected BlockBody::Statements"),
        },
        _ => panic!("Expected Node"),
    }
}

#[test]
fn test_assertion_with_empty_string_message() {
    // Empty string message should be preserved
    let source = r#"
        signal temp {
            assert {
                prev > 0 : ""
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => match &node.execution_blocks[0].1 {
            BlockBody::Statements(stmts) => match &stmts[0] {
                Stmt::Assert { message, .. } => {
                    assert_eq!(
                        message.as_deref(),
                        Some(""),
                        "Empty string message should be preserved"
                    );
                }
                _ => panic!("Expected Stmt::Assert"),
            },
            _ => panic!("Expected BlockBody::Statements"),
        },
        _ => panic!("Expected Node"),
    }
}

// =============================================================================
// Entity-First Architecture (Wave 1)
// =============================================================================

#[test]
fn test_entity_with_nested_members() {
    let source = r#"
        entity plate {
            signal velocity {
                resolve { 0.0 }
            }
            field stress {
                measure { 0.0 }
            }
            operator apply_friction {
                resolve { 0.0 }
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Entity(entity) => {
            assert_eq!(entity.path.to_string(), "plate");
            assert_eq!(entity.members.len(), 3);
            assert!(matches!(entity.members[0].role, RoleData::Signal));
            assert!(matches!(entity.members[1].role, RoleData::Field { .. }));
            assert!(matches!(entity.members[2].role, RoleData::Operator));
        }
        _ => panic!("Expected Entity declaration"),
    }
}

#[test]
fn test_entity_with_nested_child_entity() {
    let source = r#"
        entity plate {
            signal age {
                resolve { 0.0 }
            }
            entity boundary {
                signal stress {
                    resolve { 0.0 }
                }
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Entity(entity) => {
            assert_eq!(entity.path.to_string(), "plate");
            assert_eq!(entity.members.len(), 1);
            assert_eq!(entity.children.len(), 1);

            let child = entity.children.values().next().expect("should have child");
            assert_eq!(child.path.to_string(), "boundary");
            assert_eq!(child.members.len(), 1);
        }
        _ => panic!("Expected Entity declaration"),
    }
}

#[test]
fn test_entity_with_deeply_nested_children() {
    let source = r#"
        entity planet {
            entity continent {
                entity region {
                    signal population {
                        resolve { 0.0 }
                    }
                }
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Entity(entity) => {
            assert_eq!(entity.path.to_string(), "planet");
            assert_eq!(entity.children.len(), 1);

            let continent = entity.children.values().next().expect("continent");
            assert_eq!(continent.path.to_string(), "continent");
            assert_eq!(continent.children.len(), 1);

            let region = continent.children.values().next().expect("region");
            assert_eq!(region.path.to_string(), "region");
            assert_eq!(region.members.len(), 1);
        }
        _ => panic!("Expected Entity declaration"),
    }
}

#[test]
fn test_world_with_body_declarations() {
    let source = r#"
        world terra {
            : title("Terra")

            signal temperature {
                resolve { 300.0 }
            }

            entity plate {
                signal velocity {
                    resolve { 0.0 }
                }
            }

            strata tectonics {
                : stride(1)
            }
        }
    "#;

    let decls = parse(source);

    // World metadata + signal + entity + stratum = 4 declarations
    assert_eq!(decls.len(), 4);
    assert!(matches!(decls[0], Declaration::World(_)));
    assert!(matches!(decls[1], Declaration::Node(_)));
    assert!(matches!(decls[2], Declaration::Entity(_)));
    assert!(matches!(decls[3], Declaration::Stratum(_)));

    if let Declaration::World(world) = &decls[0] {
        assert_eq!(world.title.as_deref(), Some("Terra"));
    }
}

#[test]
fn test_world_with_era_in_body() {
    let source = r#"
        world terra {
            era formation {
                : initial
                : dt(1000000.0)
            }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 2);
    assert!(matches!(decls[0], Declaration::World(_)));
    assert!(matches!(decls[1], Declaration::Era(_)));
}

#[test]
fn test_standalone_member_keyword() {
    let source = r#"
        member plate.velocity {
            resolve { 0.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Member(node) => {
            // Member path should be just "velocity" (entity prefix stripped)
            assert_eq!(node.path.to_string(), "velocity");
            // Entity should be "plate"
            assert_eq!(
                node.entity
                    .as_ref()
                    .expect("should have entity")
                    .to_string(),
                "plate"
            );
            assert!(matches!(node.role, RoleData::Signal));
        }
        _ => panic!("Expected Member declaration"),
    }
}

#[test]
fn test_standalone_member_with_dotted_path() {
    let source = r#"
        member plate.physics.velocity {
            resolve { 0.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Member(node) => {
            // Member path should be "physics.velocity" (first segment is entity)
            assert_eq!(node.path.to_string(), "physics.velocity");
            assert_eq!(
                node.entity
                    .as_ref()
                    .expect("should have entity")
                    .to_string(),
                "plate"
            );
        }
        _ => panic!("Expected Member declaration"),
    }
}

// =============================================================================
// Namespace header tests
// =============================================================================

#[test]
fn test_namespace_prefixes_signal() {
    let source = r#"
        namespace terra.atmosphere

        signal surface_temp {
            resolve { 300.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.atmosphere.surface_temp");
            assert!(matches!(node.role, RoleData::Signal));
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_namespace_prefixes_multiple_declarations() {
    let source = r#"
        namespace terra.atmosphere

        signal surface_temp {
            resolve { 300.0 }
        }

        field temperature_map {
            measure { 0.0 }
        }

        operator thermal_cycle {
            collect { 0.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 3);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.atmosphere.surface_temp");
        }
        _ => panic!("Expected Node (signal)"),
    }

    match &decls[1] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.atmosphere.temperature_map");
        }
        _ => panic!("Expected Node (field)"),
    }

    match &decls[2] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.atmosphere.thermal_cycle");
        }
        _ => panic!("Expected Node (operator)"),
    }
}

#[test]
fn test_namespace_prefixes_entity() {
    let source = r#"
        namespace terra

        entity plate {
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Entity(entity) => {
            assert_eq!(entity.path.to_string(), "terra.plate");
        }
        _ => panic!("Expected Entity declaration"),
    }
}

#[test]
fn test_namespace_prefixes_stratum_and_era() {
    let source = r#"
        namespace terra

        strata physics {
        }

        era formation {
            : initial
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 2);

    match &decls[0] {
        Declaration::Stratum(s) => {
            assert_eq!(s.path.to_string(), "terra.physics");
        }
        _ => panic!("Expected Stratum declaration"),
    }

    match &decls[1] {
        Declaration::Era(e) => {
            assert_eq!(e.path.to_string(), "terra.formation");
        }
        _ => panic!("Expected Era declaration"),
    }
}

#[test]
fn test_namespace_does_not_affect_const_config() {
    let source = r#"
        namespace terra.atmosphere

        const {
            base_temp: 288.0
        }

        config {
            resolution: 100
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 2);

    // Const/config paths should NOT be prefixed
    match &decls[0] {
        Declaration::Const(entries) => {
            assert_eq!(entries[0].path.to_string(), "base_temp");
        }
        _ => panic!("Expected Const declaration"),
    }

    match &decls[1] {
        Declaration::Config(entries) => {
            assert_eq!(entries[0].path.to_string(), "resolution");
        }
        _ => panic!("Expected Config declaration"),
    }
}

#[test]
fn test_no_namespace_leaves_paths_unchanged() {
    let source = r#"
        signal terra.atmosphere.surface_temp {
            resolve { 300.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.atmosphere.surface_temp");
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_namespace_single_segment() {
    let source = r#"
        namespace terra

        signal elevation {
            resolve { 0.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.elevation");
        }
        _ => panic!("Expected Node declaration"),
    }
}

#[test]
fn test_namespace_with_doc_comment_before() {
    let source = r#"
        /// Atmosphere module
        namespace terra.atmosphere

        signal surface_temp {
            resolve { 300.0 }
        }
    "#;

    let decls = parse(source);
    assert_eq!(decls.len(), 1);

    match &decls[0] {
        Declaration::Node(node) => {
            assert_eq!(node.path.to_string(), "terra.atmosphere.surface_temp");
        }
        _ => panic!("Expected Node declaration"),
    }
}
