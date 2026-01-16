//! Tests for the scenario system.

use super::*;

#[test]
fn test_scenario_from_yaml() {
    let yaml = r#"
apiVersion: continuum/v1
kind: Scenario

metadata:
  name: test_scenario
  title: "Test Scenario"
  description: "A test scenario for unit testing"

config:
  core.initial_temp: 6000.0
  mantle.initial_temp: 2000.0

initial:
  core.temp: 5500.0
  crust.thickness: 35000.0

seed: 42
"#;

    let scenario = Scenario::from_yaml(yaml).unwrap();
    assert_eq!(scenario.metadata.name, "test_scenario");
    assert_eq!(scenario.metadata.title, Some("Test Scenario".to_string()));
    assert_eq!(scenario.seed, 42);
    assert_eq!(scenario.config.get("core.initial_temp"), Some(&6000.0));
    assert_eq!(scenario.config.get("mantle.initial_temp"), Some(&2000.0));

    let initial = scenario.initial.get("core.temp").unwrap();
    assert!(matches!(initial, ScenarioValue::Scalar(v) if *v == 5500.0));
}

#[test]
fn test_scenario_builder() {
    let scenario = Scenario::new("builder_test")
        .with_title("Builder Test")
        .with_description("Testing builder pattern")
        .with_config("core.temp", 5000.0)
        .with_initial("signal.test", 42.0)
        .with_seed(123);

    assert_eq!(scenario.metadata.name, "builder_test");
    assert_eq!(scenario.metadata.title, Some("Builder Test".to_string()));
    assert_eq!(scenario.seed, 123);
    assert_eq!(scenario.config.get("core.temp"), Some(&5000.0));
}

#[test]
fn test_scenario_vector_initial() {
    let yaml = r#"
apiVersion: continuum/v1
kind: Scenario

metadata:
  name: vector_test

initial:
  position: [1.0, 2.0, 3.0]
  velocity: [0.1, 0.2]
  orientation: [1.0, 0.0, 0.0, 0.0]
"#;

    let scenario = Scenario::from_yaml(yaml).unwrap();

    let pos = scenario.initial.get("position").unwrap();
    assert!(matches!(pos, ScenarioValue::Vec3([1.0, 2.0, 3.0])));

    let vel = scenario.initial.get("velocity").unwrap();
    assert!(matches!(vel, ScenarioValue::Vec2([0.1, 0.2])));

    let ori = scenario.initial.get("orientation").unwrap();
    assert!(matches!(ori, ScenarioValue::Vec4([1.0, 0.0, 0.0, 0.0])));
}

#[test]
fn test_invalid_api_version() {
    let yaml = r#"
apiVersion: continuum/v2
kind: Scenario
metadata:
  name: test
"#;

    let result = Scenario::from_yaml(yaml);
    assert!(matches!(result, Err(ScenarioError::InvalidApiVersion(_))));
}

#[test]
fn test_invalid_kind() {
    let yaml = r#"
apiVersion: continuum/v1
kind: World
metadata:
  name: test
"#;

    let result = Scenario::from_yaml(yaml);
    assert!(matches!(result, Err(ScenarioError::InvalidKind(_))));
}

#[test]
fn test_missing_name() {
    // When metadata.name is empty (or not provided), validation should fail.
    let yaml = r#"
apiVersion: continuum/v1
kind: Scenario
metadata:
  name: ""
  title: "No Name"
"#;

    let result = Scenario::from_yaml(yaml);
    assert!(matches!(result, Err(ScenarioError::MissingField(_))));
}

#[test]
fn test_omitted_name() {
    // When metadata.name is entirely omitted, serde_yaml requires it.
    let yaml = r#"
apiVersion: continuum/v1
kind: Scenario
metadata:
  title: "No Name"
"#;

    let result = Scenario::from_yaml(yaml);
    // serde_yaml reports a missing field error during parsing.
    assert!(
        matches!(result, Err(ScenarioError::YamlError(_))),
        "Expected YamlError for missing name field, got: {:?}",
        result
    );
}

#[test]
fn test_scenario_value_conversion() {
    let scalar = ScenarioValue::Scalar(42.0);
    let value = scalar.to_value();
    assert!(matches!(value, continuum_foundation::Value::Scalar(42.0)));

    let vec3 = ScenarioValue::Vec3([1.0, 2.0, 3.0]);
    let value = vec3.to_value();
    assert!(matches!(
        value,
        continuum_foundation::Value::Vec3([1.0, 2.0, 3.0])
    ));
}

#[test]
fn test_scenario_defaults() {
    let yaml = r#"
metadata:
  name: minimal
"#;

    let scenario = Scenario::from_yaml(yaml).unwrap();
    assert_eq!(scenario.api_version, "continuum/v1");
    assert_eq!(scenario.kind, "Scenario");
    assert_eq!(scenario.seed, 0);
    assert!(scenario.config.is_empty());
    assert!(scenario.initial.is_empty());
}
