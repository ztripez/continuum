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
