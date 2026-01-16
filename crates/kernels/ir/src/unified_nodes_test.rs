//! Tests for unified node architecture

use crate::lower;
use continuum_dsl::parse;

#[test]
fn test_unified_nodes_populated_from_signals() {
    let src = r#"
        strata test {}
        
        signal temperature {
            : Scalar<K>
            : strata(test)
            resolve { 300.0 }
        }
        
        signal pressure {
            : Scalar<Pa>
            : strata(test)
            resolve { signal.temperature * 10.0 }
        }
    "#;

    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);

    let world = lower(&unit.unwrap()).unwrap();

    // Check that unified nodes contain our signals
    assert_eq!(world.nodes.len(), 3); // 1 stratum + 2 signals

    // Verify temperature signal node
    let temp_path = continuum_foundation::Path::from("temperature");
    let temp_node = world
        .nodes
        .get(&temp_path)
        .expect("temperature node should exist");

    assert_eq!(temp_node.id, temp_path);
    assert!(temp_node.stratum.is_some());
    assert!(temp_node.reads.is_empty());

    match &temp_node.kind {
        crate::unified_nodes::NodeKind::Signal(props) => {
            assert!(props.title.is_none());
            assert_eq!(props.uses_dt_raw, false);
            assert!(props.resolve.is_some());
        }
        _ => panic!("Expected signal node"),
    }

    // Verify pressure signal node
    let pressure_path = continuum_foundation::Path::from("pressure");
    let pressure_node = world
        .nodes
        .get(&pressure_path)
        .expect("pressure node should exist");

    match &pressure_node.kind {
        crate::unified_nodes::NodeKind::Signal(props) => {
            // Should read temperature signal
            assert_eq!(pressure_node.reads.len(), 1);
            assert_eq!(
                pressure_node.reads[0].path(),
                &continuum_foundation::Path::from("temperature")
            );
        }
        _ => panic!("Expected signal node"),
    }
}

// TODO: Add comprehensive test for all node types once syntax is figured out
