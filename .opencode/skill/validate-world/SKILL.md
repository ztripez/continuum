---
name: validate-world
description: Run test suites to validate Continuum world simulations
license: MIT
compatibility: opencode
metadata:
  domain: simulation
  category: validation
---

# Validate World Simulation

Every world should have a test suite. Use it to validate simulations.

## Test Suite Convention

```
examples/<world>/
  analyzers.cdsl              # CDSL analyzer definitions
  tests/
    <world>_baselines.yaml    # Expected values and thresholds
```

## CLI Tools

Build the tools:
```bash
cargo build -p continuum-tools
```

Available binaries in `target/debug/`:
- `analyze` - Run analyzers and baseline comparisons
- `compile` - Compile CDSL to IR
- `run` - Run simulations

## Baseline Validation

```bash
# Record baseline from known-good run
./target/debug/analyze baseline record <snapshot_dir> -o baseline.json

# Compare new run against baseline (5% tolerance)
./target/debug/analyze baseline compare <snapshot_dir> -b baseline.json -t 0.05
```

## CDSL Analyzer Validation

```bash
# List analyzers in compiled world
./target/debug/analyze analyzer list <world.json>

# Run specific analyzer
./target/debug/analyze analyzer run <name> <world.json> <snapshot_dir>
```

## When Validation Fails

1. Read the error message
2. Check baseline YAML for expected values
3. Check analyzer CDSL for computation logic
4. Fix simulation code

## World-Specific Skills

- Terra: Use `validate-terra` skill
- Other worlds: Create `validate-<world>` following same pattern

## Reference

See `examples/terra/` for complete example.
