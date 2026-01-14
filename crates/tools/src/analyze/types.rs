//! Core types for the analyze tool.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use continuum_runtime::storage::FieldSample;
use continuum_runtime::types::Value;

// ============================================================================
// Run Manifest Types
// ============================================================================

/// Run manifest structure (matches run output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub created_at: String,
    pub seed: u64,
    pub steps: u64,
    pub stride: u64,
    pub signals: Vec<String>,
    pub fields: Vec<String>,
}

// ============================================================================
// Snapshot Types
// ============================================================================

/// Tick snapshot from JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickSnapshot {
    pub tick: u64,
    pub time_seconds: f64,
    pub signals: HashMap<String, Value>,
    pub fields: HashMap<String, Vec<FieldSample>>,
}

// ============================================================================
// Snapshot Run
// ============================================================================

/// Loaded snapshot run with all field snapshots.
pub struct SnapshotRun {
    pub manifest: RunManifest,
    /// Path to the run directory
    pub run_dir: PathBuf,
    /// Tick -> snapshot
    pub snapshots: HashMap<u64, TickSnapshot>,
}

impl SnapshotRun {
    /// Load a snapshot run from a directory.
    pub fn load(run_dir: PathBuf) -> Result<Self, String> {
        // Load manifest
        let manifest_path = run_dir.join("run.json");
        let manifest_str = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read {}: {}", manifest_path.display(), e))?;
        let manifest: RunManifest = serde_json::from_str(&manifest_str)
            .map_err(|e| format!("Failed to parse manifest: {}", e))?;

        // Load tick snapshots
        let mut snapshots: HashMap<u64, TickSnapshot> = HashMap::new();

        if run_dir.exists() {
            for entry in fs::read_dir(&run_dir)
                .map_err(|e| format!("Failed to read {}: {}", run_dir.display(), e))?
            {
                let entry = entry.map_err(|e| format!("Dir entry error: {}", e))?;
                let path = entry.path();

                if path.extension().is_some_and(|ext| ext == "json")
                    && path
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .starts_with("tick_")
                {
                    let content = fs::read_to_string(&path)
                        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
                    let snapshot: TickSnapshot = serde_json::from_str(&content)
                        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

                    snapshots.insert(snapshot.tick, snapshot);
                }
            }
        }

        Ok(Self {
            manifest,
            run_dir,
            snapshots,
        })
    }

    /// Get all ticks, sorted.
    pub fn ticks(&self) -> Vec<u64> {
        let mut ticks: Vec<u64> = self.snapshots.keys().copied().collect();
        ticks.sort();
        ticks
    }

    /// Get snapshot at a specific tick.
    pub fn get_snapshot(&self, tick: u64) -> Option<&TickSnapshot> {
        self.snapshots.get(&tick)
    }

    /// Get all values for a field at a tick (as f64 scalars).
    pub fn get_field_values(&self, field: &str, tick: u64) -> Vec<f64> {
        self.get_snapshot(tick)
            .and_then(|s| s.fields.get(field))
            .map(|samples| samples.iter().filter_map(|s| s.value.as_scalar()).collect())
            .unwrap_or_default()
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Basic statistics for a set of values.
#[derive(Debug, Clone, Serialize)]
pub struct Statistics {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub percentiles: Percentiles,
}

/// Percentile values.
#[derive(Debug, Clone, Serialize)]
pub struct Percentiles {
    pub p5: f64,
    pub p25: f64,
    pub p75: f64,
    pub p95: f64,
}

impl Statistics {
    /// Compute statistics from a slice of f64 values.
    pub fn compute(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted.len();
        let min = sorted[0];
        let max = sorted[count - 1];
        let sum: f64 = sorted.iter().sum();
        let mean = sum / count as f64;

        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let variance: f64 = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let percentile = |p: f64| -> f64 {
            let idx = (p / 100.0 * (count - 1) as f64) as usize;
            sorted[idx.min(count - 1)]
        };

        Some(Self {
            count,
            min,
            max,
            mean,
            median,
            std_dev,
            variance,
            percentiles: Percentiles {
                p5: percentile(5.0),
                p25: percentile(25.0),
                p75: percentile(75.0),
                p95: percentile(95.0),
            },
        })
    }
}
