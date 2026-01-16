//! Scenario type definitions and loading.

use std::collections::HashMap;
use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::CompiledWorld;

/// Errors that can occur when loading or validating a scenario.
#[derive(Debug, Error)]
pub enum ScenarioError {
    /// Failed to read the scenario file.
    #[error("failed to read scenario file: {0}")]
    IoError(#[from] std::io::Error),

    /// Failed to parse the scenario YAML.
    #[error("failed to parse scenario YAML: {0}")]
    YamlError(#[from] serde_yaml::Error),

    /// Invalid API version.
    #[error("invalid apiVersion: expected 'continuum/v1', got '{0}'")]
    InvalidApiVersion(String),

    /// Invalid kind.
    #[error("invalid kind: expected 'Scenario', got '{0}'")]
    InvalidKind(String),

    /// Missing required field.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// Config key not found in world.
    #[error("config key '{key}' not found in world")]
    UnknownConfigKey { key: String },

    /// Signal not found in world.
    #[error("signal '{signal}' not found in world")]
    UnknownSignal { signal: String },
}

/// Result type for scenario operations.
pub type ScenarioResult<T> = Result<T, ScenarioError>;

/// A scenario configures how a World is instantiated.
///
/// Scenarios provide:
/// - Metadata (name, title, description)
/// - Configuration overrides
/// - Initial signal values
/// - Random seed for determinism
///
/// Scenarios are loaded from YAML files and applied to a compiled world
/// before runtime construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Scenario {
    /// API version for compatibility checking.
    #[serde(default = "default_api_version")]
    pub api_version: String,

    /// Kind must be "Scenario".
    #[serde(default = "default_kind")]
    pub kind: String,

    /// Scenario metadata.
    #[serde(default)]
    pub metadata: ScenarioMetadata,

    /// Configuration value overrides.
    /// Keys must match config values defined in the world.
    #[serde(default)]
    pub config: IndexMap<String, f64>,

    /// Initial signal values.
    /// Keys must match signal IDs defined in the world.
    /// These are applied before the first tick.
    #[serde(default)]
    pub initial: IndexMap<String, ScenarioValue>,

    /// Random seed for deterministic execution.
    /// If not specified, a default seed of 0 is used.
    #[serde(default)]
    pub seed: u64,
}

fn default_api_version() -> String {
    "continuum/v1".to_string()
}

fn default_kind() -> String {
    "Scenario".to_string()
}

/// Metadata for a scenario.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScenarioMetadata {
    /// Machine identifier for this scenario (lowercase, no spaces).
    pub name: String,

    /// Human-readable title.
    #[serde(default)]
    pub title: Option<String>,

    /// Description of what this scenario configures.
    #[serde(default)]
    pub description: Option<String>,
}

/// A value that can be set in a scenario.
///
/// Supports scalars, vectors, and quaternions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioValue {
    /// A scalar value.
    Scalar(f64),

    /// A 2D vector [x, y].
    Vec2([f64; 2]),

    /// A 3D vector [x, y, z].
    Vec3([f64; 3]),

    /// A 4D vector or quaternion [x, y, z, w] or [w, x, y, z].
    Vec4([f64; 4]),
}

impl ScenarioValue {
    /// Convert to a runtime Value.
    pub fn to_value(&self) -> continuum_foundation::Value {
        match self {
            ScenarioValue::Scalar(v) => continuum_foundation::Value::Scalar(*v),
            ScenarioValue::Vec2(v) => continuum_foundation::Value::Vec2(*v),
            ScenarioValue::Vec3(v) => continuum_foundation::Value::Vec3(*v),
            ScenarioValue::Vec4(v) => continuum_foundation::Value::Vec4(*v),
        }
    }

    /// Get scalar value if this is a scalar.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            ScenarioValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

impl From<f64> for ScenarioValue {
    fn from(v: f64) -> Self {
        ScenarioValue::Scalar(v)
    }
}

impl From<[f64; 2]> for ScenarioValue {
    fn from(v: [f64; 2]) -> Self {
        ScenarioValue::Vec2(v)
    }
}

impl From<[f64; 3]> for ScenarioValue {
    fn from(v: [f64; 3]) -> Self {
        ScenarioValue::Vec3(v)
    }
}

impl From<[f64; 4]> for ScenarioValue {
    fn from(v: [f64; 4]) -> Self {
        ScenarioValue::Vec4(v)
    }
}

impl Scenario {
    /// Create an empty scenario with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            api_version: default_api_version(),
            kind: default_kind(),
            metadata: ScenarioMetadata {
                name: name.into(),
                title: None,
                description: None,
            },
            config: IndexMap::new(),
            initial: IndexMap::new(),
            seed: 0,
        }
    }

    /// Load a scenario from a YAML file.
    pub fn load(path: impl AsRef<Path>) -> ScenarioResult<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse a scenario from a YAML string.
    pub fn from_yaml(yaml: &str) -> ScenarioResult<Self> {
        let scenario: Scenario = serde_yaml::from_str(yaml)?;
        scenario.validate_schema()?;
        Ok(scenario)
    }

    /// Validate the scenario schema (API version, kind).
    fn validate_schema(&self) -> ScenarioResult<()> {
        if self.api_version != "continuum/v1" {
            return Err(ScenarioError::InvalidApiVersion(self.api_version.clone()));
        }
        if self.kind != "Scenario" {
            return Err(ScenarioError::InvalidKind(self.kind.clone()));
        }
        if self.metadata.name.is_empty() {
            return Err(ScenarioError::MissingField("metadata.name".to_string()));
        }
        Ok(())
    }

    /// Validate that all scenario keys exist in the world.
    ///
    /// This checks:
    /// - All config keys exist in world.config
    /// - All initial signal keys exist as signals
    pub fn validate_against_world(&self, world: &CompiledWorld) -> ScenarioResult<()> {
        // Check config keys
        for key in self.config.keys() {
            if !world.config.contains_key(key) {
                return Err(ScenarioError::UnknownConfigKey { key: key.clone() });
            }
        }

        // Check initial signal keys
        let signals = world.signals();
        for key in self.initial.keys() {
            let signal_id = continuum_foundation::SignalId::from(key.as_str());
            if !signals.contains_key(&signal_id) {
                return Err(ScenarioError::UnknownSignal {
                    signal: key.clone(),
                });
            }
        }

        Ok(())
    }

    /// Apply this scenario's configuration overrides to a compiled world.
    ///
    /// This modifies the world's config values in place.
    pub fn apply_config(&self, world: &mut CompiledWorld) {
        for (key, value) in &self.config {
            if let Some(entry) = world.config.get_mut(key) {
                entry.0 = *value;
            }
        }
    }

    /// Get initial signal values as a map.
    ///
    /// These should be applied after runtime construction but before
    /// the first simulation tick.
    pub fn initial_values(&self) -> &IndexMap<String, ScenarioValue> {
        &self.initial
    }

    /// Builder method: set a config override.
    pub fn with_config(mut self, key: impl Into<String>, value: f64) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Builder method: set an initial signal value.
    pub fn with_initial(
        mut self,
        signal: impl Into<String>,
        value: impl Into<ScenarioValue>,
    ) -> Self {
        self.initial.insert(signal.into(), value.into());
        self
    }

    /// Builder method: set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder method: set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.metadata.title = Some(title.into());
        self
    }

    /// Builder method: set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.metadata.description = Some(description.into());
        self
    }
}

/// Find scenario files in a world directory.
///
/// Scenarios are stored in a `scenarios/` subdirectory of the world root.
/// Each `.yaml` file in that directory is treated as a scenario.
pub fn find_scenarios(world_dir: impl AsRef<Path>) -> Vec<std::path::PathBuf> {
    let scenarios_dir = world_dir.as_ref().join("scenarios");
    if !scenarios_dir.exists() {
        return Vec::new();
    }

    let mut scenarios = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&scenarios_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                scenarios.push(path);
            }
        }
    }
    scenarios.sort();
    scenarios
}

/// Load all scenarios from a world directory.
pub fn load_scenarios(world_dir: impl AsRef<Path>) -> HashMap<String, Scenario> {
    let mut scenarios = HashMap::new();
    for path in find_scenarios(world_dir) {
        match Scenario::load(&path) {
            Ok(scenario) => {
                scenarios.insert(scenario.metadata.name.clone(), scenario);
            }
            Err(e) => {
                tracing::warn!("Failed to load scenario from {:?}: {}", path, e);
            }
        }
    }
    scenarios
}
