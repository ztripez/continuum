---
name: validate-world
description: Validate any Continuum world simulation using CDSL analyzers and baseline tests
license: MIT
compatibility: opencode
metadata:
  domain: simulation
  category: validation
---

# Validate World Simulation

This skill provides a general framework for validating Continuum world simulations using CDSL analyzers.

## Validation Workflow

### 1. Discover Available Analyzers

```bash
# List all analyzers defined in a world
continuum analyze list <world_path>

# Example output:
# terra.hypsometric_integral  - Land/ocean ratio
# terra.ocean_analysis        - Ocean depth zones
# terra.isostasy_balance      - Isostatic equilibrium
```

### 2. Run Simulation with Field Capture

```bash
# Run simulation and capture fields for analysis
continuum run <world_path> \
  --steps <num_steps> \
  --dt <timestep> \
  --samples <sample_count> \
  --snapshot-dir <output_dir>
```

### 3. Execute Analyzers

```bash
# Run a specific analyzer
continuum analyze run <analyzer_name> <snapshot_dir> [--tick N]

# Run all validations (checks with severity levels)
continuum analyze validate <snapshot_dir>
```

### 4. Interpret Results

Analyzer output structure:
```json
{
  "analyzer": "<name>",
  "tick": 1000,
  "result": {
    // Computed values from : compute { } block
  },
  "validations": [
    {
      "passed": true|false,
      "severity": "error|warning|info",
      "message": "Human-readable result"
    }
  ]
}
```

## Writing CDSL Analyzers

### Analyzer Structure

```cdsl
analyzer <namespace>.<name> {
    : doc "<description>"
    : requires(fields: [<field1>, <field2>])
    
    : compute {
        let samples = field.samples(<field_id>)
        // Analysis logic
        emit {
            result_key: computed_value,
            // ...
        }
    }
    
    : validate {
        check <condition>
            : severity(error|warning|info)
            : message("<template with {variables}>")
    }
}
```

### Field Sampling API

```cdsl
// Get all samples for a field
let samples = field.samples(field_id)

// Sample properties
samples.count()       // Number of samples
samples.sum()         // Sum of values
samples.min()         // Minimum value
samples.max()         // Maximum value

// Filter samples
let filtered = samples.filter(|s| s.value > threshold)

// Transform samples
let transformed = samples.map(|s| s.value * factor)

// Access sample properties in closures
// s.value    - The sample value
// s.position - [x, y, z] position on sphere
```

### Statistics Functions

```cdsl
stats.count(samples)              // Count
stats.sum(samples)                // Sum
stats.mean(samples)               // Arithmetic mean
stats.median(samples)             // Median
stats.std_dev(samples)            // Standard deviation
stats.variance(samples)           // Variance
stats.percentile(samples, p)      // p-th percentile
stats.correlation(samples1, samples2)  // Pearson correlation
stats.histogram(samples, bins)    // Histogram bin counts
stats.compute(samples)            // All stats as object
```

### Validation Checks

```cdsl
: validate {
    // Range check
    check value >= 0.2 && value <= 0.4
        : severity(warning)
        : message("Value {value:.2} in range [0.2, 0.4]")
    
    // Threshold check
    check correlation < -0.5
        : severity(error)
        : message("Correlation {correlation:.3} (expected < -0.5)")
    
    // Positive correlation
    check r > 0.5
        : severity(warning)
        : message("Correlation {r:.3} (expected > 0.5)")
}
```

## Severity Levels

| Level | CI Behavior | Use Case |
|-------|-------------|----------|
| `error` | Fails build | Critical physics violations |
| `warning` | Logged only | Recommended thresholds |
| `info` | Informational | Diagnostic output |

## Creating Baseline Tests

### Baseline File Structure (YAML)

```yaml
schema: "continuum.baselines/v1"
description: "<world> domain baselines"

validations:
  - name: <analyzer_name>
    field: <field_id>
    check: <output_key>
    expected: "<expression>"
    severity: error|warning|info
    description: "<what this checks>"

field_baselines:
  <field_id>:
    description: "<field description>"
    statistics:
      min: "<range>"
      max: "<range>"
      mean: "<range>"
```

### Expression Syntax

| Format | Meaning |
|--------|---------|
| `"0.2..0.4"` | Value in range [0.2, 0.4] |
| `"> 0.5"` | Value greater than 0.5 |
| `"< -0.5"` | Value less than -0.5 |
| `"~0.29 +/- 0.1"` | Within tolerance |
| `"== true"` | Boolean check |

## World-Specific Skills

For domain-specific validation:
- Terra (planetary): Use `validate-terra` skill
- Other worlds: Create `validate-<world>` skill following this pattern

## Troubleshooting

### Analyzer Not Found
```
Error: Analyzer '<name>' not found in world
```
- Check analyzer is defined in `*.cdsl` files
- Verify world path is correct
- Run `continuum analyze list` to see available analyzers

### Field Not Captured
```
Error: Field '<id>' not found in snapshot
```
- Ensure simulation captured the field
- Check field name matches CDSL definition
- Verify snapshot directory contains field data

### Empty Samples
```
Error: No samples for field
```
- Check simulation ran long enough
- Verify field has data at requested tick
- Use `--tick N` to specify different tick

## CI Integration

```bash
#!/bin/bash
# validate.sh - CI validation script

WORLD="examples/terra"
SNAPSHOT_DIR="./snapshots"

# Run simulation
continuum run $WORLD --steps 1000 --snapshot-dir $SNAPSHOT_DIR

# Validate (exit 1 on error-level failures)
continuum analyze validate $SNAPSHOT_DIR --severity error
exit $?
```

Exit codes:
- `0`: All validations passed
- `1`: Error-level validations failed
- `2`: Analyzer or field not found
- `3`: Snapshot not found
