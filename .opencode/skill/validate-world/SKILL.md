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

Each world follows this structure:
```
examples/<world>/
  analyzers.cdsl              # Analyzer definitions
  tests/
    <world>_baselines.yaml    # Expected values and thresholds
```

## How to Validate

### 1. Find the Test Suite

```bash
# Check what analyzers exist
continuum analyze list examples/<world>

# Find the baselines file
ls examples/<world>/tests/
```

### 2. Run Validation

```bash
# Run all validations against baselines
continuum analyze validate <snapshot_dir> --baselines examples/<world>/tests/<world>_baselines.yaml
```

### 3. Check Results

- Exit code 0 = all passed
- Exit code 1 = error-level checks failed
- Read output for specific failures

## When Validation Fails

1. **Read the error** - which analyzer failed?
2. **Check the baseline** - what was expected?
3. **Check the analyzer** - what does it compute?
4. **Fix the simulation** - correct the physics/logic

## World-Specific Skills

- Terra: Use `validate-terra` skill
- Other worlds: Follow same pattern

## Creating Test Suites for New Worlds

1. Define analyzers in `<world>/analyzers.cdsl`
2. Create `<world>/tests/<world>_baselines.yaml`
3. Document expected values and why

See `examples/terra/` for reference implementation.
