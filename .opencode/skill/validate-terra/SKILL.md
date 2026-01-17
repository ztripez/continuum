---
name: validate-terra
description: Run the terra test suite to validate simulation produces Earth-like results
license: MIT
compatibility: opencode
metadata:
  domain: simulation
  world: terra
---

# Validate Terra Simulation

Use the existing test suite to validate terra simulation output.

## Test Suite Location

- **Analyzers**: `examples/terra/analyzers.cdsl`
- **Baselines**: `examples/terra/tests/terra_baselines.yaml`

## How to Validate

### Step 1: Build the analyze tool

```bash
cargo build -p continuum-tools --bin analyze
```

### Step 2: Run baseline comparison

```bash
# Record a baseline from a known-good run
./target/debug/analyze baseline record <snapshot_dir> -o baseline.json

# Compare new run against baseline
./target/debug/analyze baseline compare <snapshot_dir> -b baseline.json -t 0.05
```

### Step 3: Run CDSL analyzers

```bash
# List available analyzers
./target/debug/analyze analyzer list <world.json>

# Run specific analyzer
./target/debug/analyze analyzer run terra.hypsometric_integral <world.json> <snapshot_dir>
```

## What the Test Suite Checks

Read `examples/terra/tests/terra_baselines.yaml` for expected values:

| Check | Expected | Severity |
|-------|----------|----------|
| `terra.water_elevation_check` | r < -0.5 | **error** |
| `terra.hypsometric_integral` | 0.20-0.40 | warning |
| `terra.isostasy_balance` | r > 0.5 | warning |

## When Validation Fails

1. Read the error message
2. Check `terra_baselines.yaml` for the expected value
3. Look at the analyzer in `analyzers.cdsl`
4. Fix the simulation code

## Reference

- Old Rust implementation: `continuum-alpha/continuum-prime/crates/tools/src/analyze/domain/terra.rs`
- Field mappings: `examples/terra/AGENTS.md`
