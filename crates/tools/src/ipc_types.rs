use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize)]
pub struct ImpulseInfo {
    pub path: String,
    pub doc: Option<String>,
    pub payload_type: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AssertionInfo {
    pub signal_id: String,
    pub severity: String,
    pub message: Option<String>,
}
