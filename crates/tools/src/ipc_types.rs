//! Data transfer types for IPC responses.
//!
//! Serializable structs that describe signals, fields, impulses, and assertions
//! in the shape expected by inspector clients. These are boundary-only adapters
//! over the canonical engine types.

use serde::{Deserialize, Serialize};

/// Signal metadata exposed through the IPC API.
#[derive(Serialize, Deserialize)]
pub struct SignalInfo {
    pub id: String,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub doc: Option<String>,
    pub value_type: Option<String>,
    pub unit: Option<String>,
    pub range: Option<[f64; 2]>,
    pub stratum: Option<String>,
}

/// Field metadata exposed through the IPC API.
#[derive(Serialize, Deserialize)]
pub struct FieldInfo {
    pub id: String,
    pub title: Option<String>,
    pub symbol: Option<String>,
    pub doc: Option<String>,
    pub topology: Option<String>,
    pub value_type: Option<String>,
    pub unit: Option<String>,
    pub range: Option<[f64; 2]>,
}

/// Impulse metadata exposed through the IPC API.
#[derive(Serialize, Deserialize)]
pub struct ImpulseInfo {
    pub path: String,
    pub doc: Option<String>,
    pub payload_type: Option<String>,
}

/// Assertion metadata exposed through the IPC API.
#[derive(Serialize, Deserialize)]
pub struct AssertionInfo {
    pub signal_id: String,
    pub severity: String,
    pub message: Option<String>,
}

/// A single field sample entry serialized for IPC transport.
///
/// Flattens the engine's `FieldSample` (which uses `Value` enum) into a
/// JSON-friendly shape with a typed value representation.
#[derive(Serialize, Deserialize)]
pub struct FieldSampleEntry {
    /// Position in the field's coordinate space [x, y, z].
    pub position: [f64; 3],
    /// Scalar value at this position (`null` for non-scalar values).
    pub scalar: Option<f64>,
    /// Full value representation (for non-scalar types like vectors).
    pub value: serde_json::Value,
}

/// Response payload for `field.samples` requests.
///
/// Contains the latest field sample data for a specific field,
/// tagged with the tick and simulation time when they were captured.
#[derive(Serialize, Deserialize)]
pub struct FieldSampleData {
    /// Field identifier (dotted path).
    pub field_id: String,
    /// Tick at which samples were captured.
    pub tick: u64,
    /// Simulation time at which samples were captured.
    pub sim_time: f64,
    /// Sample entries for this field.
    pub samples: Vec<FieldSampleEntry>,
}

/// A single entry in a signal's time-series history.
#[derive(Serialize, Deserialize)]
pub struct SignalHistoryEntry {
    /// Tick number when this value was recorded.
    pub tick: u64,
    /// Simulation time when this value was recorded.
    pub sim_time: f64,
    /// Scalar value (`null` for non-scalar signals).
    pub scalar: Option<f64>,
    /// Full value representation.
    pub value: serde_json::Value,
}

/// Member metadata exposed through the IPC API.
///
/// Describes a per-entity member signal, including which entity it belongs to,
/// its role (signal, field, fracture, etc.), and type information.
#[derive(Serialize, Deserialize)]
pub struct MemberInfo {
    /// Fully qualified path (e.g., "terra.plate.velocity").
    pub id: String,
    /// Short title for UI display.
    pub title: Option<String>,
    /// Documentation comment from source.
    pub doc: Option<String>,
    /// Role of this member (signal, field, operator, fracture, chronicle).
    pub role: String,
    /// Output type description (e.g., "Scalar<K>", "Vec3<m/s>").
    pub value_type: Option<String>,
    /// Stratum assignment.
    pub stratum: Option<String>,
    /// Entity this member belongs to.
    pub entity_id: String,
}

/// A single member instance value entry serialized for IPC transport.
#[derive(Serialize, Deserialize)]
pub struct MemberValueEntry {
    /// Instance index within the entity.
    pub instance: usize,
    /// Scalar value at this instance (`null` for non-scalar values).
    pub scalar: Option<f64>,
    /// Full value representation.
    pub value: serde_json::Value,
}

/// Response payload for `member.values` requests.
///
/// Contains the current values of a specific member signal across all entity
/// instances, tagged with the tick and simulation time when they were read.
#[derive(Serialize, Deserialize)]
pub struct MemberValueData {
    /// Member signal identifier (dotted path).
    pub member_id: String,
    /// Entity this member belongs to.
    pub entity_id: String,
    /// Number of entity instances.
    pub instance_count: usize,
    /// Tick at which values were read.
    pub tick: u64,
    /// Simulation time at which values were read.
    pub sim_time: f64,
    /// Per-instance values.
    pub values: Vec<MemberValueEntry>,
}

/// Response payload for `signal.history` requests.
///
/// Contains a time-series of signal values from a ring buffer,
/// ordered oldest-to-newest.
#[derive(Serialize, Deserialize)]
pub struct SignalHistoryData {
    /// Signal identifier (dotted path).
    pub signal_id: String,
    /// History entries, ordered oldest to newest.
    pub entries: Vec<SignalHistoryEntry>,
}

/// A node in the world tree, sent to inspector clients for tree-based navigation.
///
/// Represents entities, signals, fields, operators, fractures, chronicles,
/// and synthetic namespace groups in a recursive hierarchy.
#[derive(Serialize, Deserialize, Clone)]
pub struct TreeNode {
    /// Unique identifier (entity path, node path, or synthetic namespace key).
    pub id: String,
    /// Display label (last path segment or entity name).
    pub label: String,
    /// Node kind: `"world"`, `"entity"`, `"signal"`, `"field"`,
    /// `"operator"`, `"fracture"`, `"chronicle"`, `"impulse"`, `"namespace"`.
    pub kind: String,
    /// Child nodes (empty for leaf nodes).
    pub children: Vec<TreeNode>,
    /// Stratum assignment, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stratum: Option<String>,
    /// Output type description (e.g. `"Scalar<K>"`), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
}
