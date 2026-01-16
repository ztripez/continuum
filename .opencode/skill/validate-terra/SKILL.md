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

All validation definitions are in:
- **Analyzers**: `examples/terra/analyzers.cdsl`
- **Baselines**: `examples/terra/tests/terra_baselines.yaml`

## How to Validate

### 1. Run the Test Suite

```bash
# Run all terra validations against a snapshot
continuum analyze validate <snapshot_dir> --baselines examples/terra/tests/terra_baselines.yaml
```

### 2. Or Run Individual Analyzers

```bash
continuum analyze run terra.hypsometric_integral <snapshot_dir>
continuum analyze run terra.water_elevation_check <snapshot_dir>
continuum analyze run terra.isostasy_balance <snapshot_dir>
```

## What the Test Suite Checks

Read `examples/terra/tests/terra_baselines.yaml` for full details. Key checks:

| Check | Expected | Severity |
|-------|----------|----------|
| `terra.water_elevation_check` | r < -0.5 | **error** |
| `terra.hypsometric_integral` | 0.20-0.40 | warning |
| `terra.isostasy_balance` | r > 0.5 | warning |

## When Validation Fails

1. Read the error message from the analyzer
2. Check `terra_baselines.yaml` for the expected value
3. Look at the analyzer definition in `analyzers.cdsl`
4. Fix the simulation code that produces the failing metric

## Reference

- Old implementation: `/home/ztripez/Documents/code/sides/continuum-alpha/continuum-prime/crates/tools/src/analyze/domain/terra.rs`
- Field mappings in `examples/terra/AGENTS.md`
