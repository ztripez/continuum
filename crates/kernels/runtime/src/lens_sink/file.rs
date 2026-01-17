//! File-based lens sink (JSON format)
//!
//! Writes field data to JSON files for analysis and debugging.

use serde::{Deserialize, Serialize};
use std::fs::{self};
use std::path::PathBuf;

use super::{LensData, LensSink, LensSinkError, Result};
use crate::storage::FieldSample;
use crate::types::FieldId;

/// Manifest describing a lens snapshot run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensManifest {
    /// Unique run identifier (timestamp-based)
    pub run_id: String,

    /// Creation timestamp
    pub created_at: String,

    /// Simulation seed
    pub seed: u64,

    /// Total steps executed
    pub steps: u64,

    /// Stride (ticks between emissions)
    pub stride: u64,

    /// List of field IDs captured
    pub fields: Vec<String>,
}

/// Single tick of field data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    /// Simulation tick
    pub tick: u64,

    /// Simulation time in seconds
    pub time_seconds: f64,

    /// Field samples (field_id → samples)
    pub fields: std::collections::HashMap<String, Vec<FieldSample>>,
}

/// File-based sink configuration
#[derive(Debug, Clone)]
pub struct FileSinkConfig {
    /// Base output directory
    pub output_dir: PathBuf,

    /// Simulation seed (for manifest)
    pub seed: u64,

    /// Total steps (for manifest)
    pub steps: u64,

    /// Stride (for manifest)
    pub stride: u64,

    /// Field filter (empty = all fields)
    pub field_filter: Vec<FieldId>,
}

/// File-based lens sink (JSON format)
pub struct FileSink {
    config: FileSinkConfig,
    run_dir: PathBuf,
    manifest_path: PathBuf,
    captured_fields: Vec<String>,
    is_closed: bool,
}

impl FileSink {
    /// Create a new file sink
    pub fn new(config: FileSinkConfig) -> Result<Self> {
        // Create run directory with timestamp
        let run_id = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let run_dir = config.output_dir.join(&run_id);

        fs::create_dir_all(&run_dir).map_err(|e| {
            LensSinkError::Config(format!("Failed to create output directory: {}", e))
        })?;

        let manifest_path = run_dir.join("manifest.json");

        Ok(Self {
            config,
            run_dir,
            manifest_path,
            captured_fields: Vec::new(),
            is_closed: false,
        })
    }

    fn check_not_closed(&self) -> Result<()> {
        if self.is_closed {
            Err(LensSinkError::AlreadyClosed)
        } else {
            Ok(())
        }
    }
}

impl LensSink for FileSink {
    fn emit_tick(&mut self, tick: u64, time_seconds: f64, data: LensData) -> Result<()> {
        self.check_not_closed()?;

        // Track captured fields
        for field_id in data.fields.keys() {
            let field_str = field_id.to_string();
            if !self.captured_fields.contains(&field_str) {
                self.captured_fields.push(field_str);
            }
        }

        // Convert to serializable format
        let tick_data = TickData {
            tick,
            time_seconds,
            fields: data
                .fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        };

        // Write tick file
        let filename = format!("tick_{:06}.json", tick);
        let file_path = self.run_dir.join(filename);
        let json = serde_json::to_string_pretty(&tick_data)
            .map_err(|e| LensSinkError::Serialization(e.to_string()))?;

        fs::write(&file_path, json)?;

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.check_not_closed()?;
        // File writes are synchronous, nothing to flush
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if self.is_closed {
            return Ok(());
        }

        // Write manifest
        let manifest = LensManifest {
            run_id: self
                .run_dir
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            created_at: chrono::Local::now().to_rfc3339(),
            seed: self.config.seed,
            steps: self.config.steps,
            stride: self.config.stride,
            fields: self.captured_fields.clone(),
        };

        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| LensSinkError::Serialization(e.to_string()))?;

        fs::write(&self.manifest_path, manifest_json)?;

        self.is_closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn test_file_sink_create() {
        let temp_dir = std::env::temp_dir().join("continuum_test_file_sink");
        let _ = fs::remove_dir_all(&temp_dir); // Clean up from previous runs

        let config = FileSinkConfig {
            output_dir: temp_dir.clone(),
            seed: 42,
            steps: 100,
            stride: 10,
            field_filter: Vec::new(),
        };

        let sink = FileSink::new(config);
        assert!(sink.is_ok());

        // Check run directory was created
        assert!(temp_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_file_sink_emit_and_close() {
        let temp_dir = std::env::temp_dir().join("continuum_test_file_sink_emit");
        let _ = fs::remove_dir_all(&temp_dir);

        let config = FileSinkConfig {
            output_dir: temp_dir.clone(),
            seed: 42,
            steps: 10,
            stride: 1,
            field_filter: Vec::new(),
        };

        let mut sink = FileSink::new(config).unwrap();

        // Emit a tick
        let field_id = FieldId::new("test.field");
        let mut fields = IndexMap::new();
        fields.insert(
            field_id,
            vec![FieldSample {
                position: [0.0, 0.0, 0.0],
                value: crate::types::Value::Scalar(42.0),
            }],
        );

        let data = LensData { fields };

        sink.emit_tick(0, 0.0, data).unwrap();
        sink.close().unwrap();

        // Check manifest was written
        let manifest_path = sink.manifest_path;
        assert!(manifest_path.exists());

        let manifest_content = fs::read_to_string(&manifest_path).unwrap();
        let manifest: LensManifest = serde_json::from_str(&manifest_content).unwrap();

        assert_eq!(manifest.seed, 42);
        assert_eq!(manifest.steps, 10);
        assert_eq!(manifest.stride, 1);
        assert_eq!(manifest.fields, vec!["test.field"]);

        // Check tick file was written
        let tick_files: Vec<_> = fs::read_dir(&sink.run_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|s| s == "json").unwrap_or(false))
            .filter(|e| e.file_name().to_string_lossy().starts_with("tick_"))
            .collect();

        assert_eq!(tick_files.len(), 1);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
