use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use clap::Subcommand;
use serde::{Deserialize, Serialize};

use crate::analyze::helpers::compute_samples_hash;
use crate::analyze::types::{SnapshotRun, Statistics};

#[derive(Subcommand, Debug)]
pub enum BaselineCommand {
    /// Record field statistics as a baseline.
    Record {
        /// Path to snapshot run directory
        snapshot_dir: PathBuf,

        /// Output baseline file path
        #[arg(short, long)]
        output: PathBuf,

        /// Fields to include (comma-separated, default: all)
        #[arg(short, long)]
        fields: Option<String>,

        /// Include sample SHA256 for exact match verification
        #[arg(long)]
        include_samples_hash: bool,
    },

    /// Compare snapshot against a recorded baseline.
    Compare {
        /// Path to snapshot run directory
        snapshot_dir: PathBuf,

        /// Path to baseline file
        #[arg(short, long)]
        baseline: PathBuf,

        /// Tolerance for relative deviation (e.g., 0.05 for 5%)
        #[arg(short, long, default_value = "0.05")]
        tolerance: f64,

        /// Fields to compare (comma-separated, default: all in baseline)
        #[arg(short, long)]
        fields: Option<String>,
    },
}

/// Baseline file format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub schema: String,
    pub created_at: String,
    pub run_id: String,
    pub fields: BTreeMap<String, FieldBaseline>,
}

/// Baseline data for a single field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldBaseline {
    pub stats: BaselineStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples_sha256: Option<String>,
    pub tick: u64,
    pub sample_count: usize,
}

/// Statistics stored in baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
}

impl From<&Statistics> for BaselineStats {
    fn from(stats: &Statistics) -> Self {
        Self {
            min: stats.min,
            max: stats.max,
            mean: stats.mean,
            median: stats.median,
            std_dev: stats.std_dev,
            variance: stats.variance,
        }
    }
}

pub fn run(command: BaselineCommand) -> Result<(), String> {
    match command {
        BaselineCommand::Record {
            snapshot_dir,
            output,
            fields,
            include_samples_hash,
        } => record(snapshot_dir, output, fields, include_samples_hash),
        BaselineCommand::Compare {
            snapshot_dir,
            baseline,
            tolerance,
            fields,
        } => compare(snapshot_dir, baseline, tolerance, fields),
    }
}

fn record(
    snapshot_dir: PathBuf,
    output: PathBuf,
    fields_filter: Option<String>,
    include_samples_hash: bool,
) -> Result<(), String> {
    let run = SnapshotRun::load(snapshot_dir)?;

    let field_names: Vec<String> = if let Some(filter) = fields_filter {
        filter.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        run.manifest.fields.clone()
    };

    let mut field_baselines = BTreeMap::new();

    // Use the last tick available
    let last_tick = run.ticks().last().copied().ok_or("No snapshots found")?;

    for field_name in &field_names {
        let values = run.get_field_values(field_name, last_tick);
        if values.is_empty() {
            continue;
        }

        let stats = Statistics::compute(&values).ok_or("Failed to compute statistics")?;
        let samples_sha256 = if include_samples_hash {
            Some(compute_samples_hash(&values))
        } else {
            None
        };

        field_baselines.insert(
            field_name.clone(),
            FieldBaseline {
                stats: BaselineStats::from(&stats),
                samples_sha256,
                tick: last_tick,
                sample_count: values.len(),
            },
        );
    }

    let baseline = Baseline {
        schema: "continuum.baseline/v1".to_string(),
        created_at: chrono::Local::now().to_rfc3339(),
        run_id: run.manifest.run_id,
        fields: field_baselines,
    };

    let json = serde_json::to_string_pretty(&baseline)
        .map_err(|e| format!("Failed to serialize baseline: {}", e))?;

    let mut file = fs::File::create(&output)
        .map_err(|e| format!("Failed to create {}: {}", output.display(), e))?;
    file.write_all(json.as_bytes())
        .map_err(|e| format!("Failed to write baseline: {}", e))?;

    println!("Baseline recorded to: {}", output.display());
    Ok(())
}

fn compare(
    snapshot_dir: PathBuf,
    baseline_path: PathBuf,
    tolerance: f64,
    fields_filter: Option<String>,
) -> Result<(), String> {
    let run = SnapshotRun::load(snapshot_dir)?;

    let baseline_str = fs::read_to_string(&baseline_path)
        .map_err(|e| format!("Failed to read baseline: {}", e))?;
    let baseline: Baseline = serde_json::from_str(&baseline_str)
        .map_err(|e| format!("Failed to parse baseline: {}", e))?;

    let field_names: Vec<String> = if let Some(filter) = fields_filter {
        filter.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        baseline.fields.keys().cloned().collect()
    };

    let mut failures = Vec::new();

    for field_name in &field_names {
        let baseline_field = baseline.fields.get(field_name).ok_or_else(|| {
            format!("Field not found in baseline: {}", field_name)
        })?;

        // Try to find the exact tick, otherwise use last
        let tick = if run.get_snapshot(baseline_field.tick).is_some() {
            baseline_field.tick
        } else {
            run.ticks().last().copied().ok_or("No snapshots found")?
        };

        let values = run.get_field_values(field_name, tick);
        if values.is_empty() {
            failures.push(format!("No values for field {}", field_name));
            continue;
        }

        let stats = Statistics::compute(&values).ok_or("Failed to compute stats")?;
        let b_stats = &baseline_field.stats;

        // Check mean
        if !is_within_tolerance(stats.mean, b_stats.mean, tolerance) {
            failures.push(format!(
                "{}: mean mismatch (actual {:.4}, expected {:.4})",
                field_name, stats.mean, b_stats.mean
            ));
        }

        // Check std dev
        if !is_within_tolerance(stats.std_dev, b_stats.std_dev, tolerance) {
            failures.push(format!(
                "{}: std_dev mismatch (actual {:.4}, expected {:.4})",
                field_name, stats.std_dev, b_stats.std_dev
            ));
        }

        // Check hash if present
        if let Some(ref expected_hash) = baseline_field.samples_sha256 {
            let actual_hash = compute_samples_hash(&values);
            if &actual_hash != expected_hash {
                failures.push(format!("{}: hash mismatch", field_name));
            }
        }
    }

    if failures.is_empty() {
        println!("Baseline check passed.");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {}", f);
        }
        Err("Baseline check failed".to_string())
    }
}

fn is_within_tolerance(actual: f64, expected: f64, tol: f64) -> bool {
    if expected.abs() < 1e-9 {
        actual.abs() < 1e-9 // absolute zero check
    } else {
        ((actual - expected) / expected).abs() <= tol
    }
}
