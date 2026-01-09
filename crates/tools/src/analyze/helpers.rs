use sha2::{Digest, Sha256};

/// Format a float value for display.
pub fn fmt_value(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() < 0.01 || v.abs() > 10000.0 {
        format!("{:.4e}", v)
    } else {
        format!("{:.4}", v)
    }
}

/// Compute SHA256 hash of a slice of values (for determinism checks).
pub fn compute_samples_hash(values: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}
