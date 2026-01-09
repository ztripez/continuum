# Continuum Analyze Tool

This document defines the **analyze** tool, used for inspecting simulation outputs, verifying determinism, and debugging causal history.

It consumes artifacts produced by `continuum run`.

---

## Usage

```bash
continuum analyze <COMMAND> [OPTIONS]
```

## Commands

### `inspect`
Read and display signal or field values from a run output.

```bash
continuum analyze inspect --run <OUTPUT_DIR> --signal "terra.surface.temp" --tick 100
```

### `verify-determinism`
Compare two run outputs to ensure they are causally identical.

```bash
continuum analyze verify-determinism <RUN_A_DIR> <RUN_B_DIR>
```

- Checks signal values bit-for-bit.
- Checks era transition timing.
- Ignores observer-only differences (fields, logs) unless `--strict` is used.

### `trace-causality`
Trace the inputs that contributed to a signal's value at a specific tick.

```bash
continuum analyze trace-causality --run <OUTPUT_DIR> --signal "terra.fracture.quake" --tick 5050
```

### `lens`
Start a standalone Lens server for an existing run output.

```bash
continuum analyze lens --run <OUTPUT_DIR> --port 8080
```

Allows interactive exploration of a completed run.

## Analysis Modes

### Static Analysis
Some commands operate on the compiled World IR without a run:

```bash
continuum analyze deps --world <WORLD_DIR> --signal "terra.core.temp"
```
Prints the dependency graph for a signal.

### Replay Analysis
The tool can re-run a specific tick to extract detailed debug info:

```bash
continuum analyze debug-tick --world <WORLD_DIR> --scenario <SCENARIO> --seed <SEED> --tick 42
```

## Examples

Check if two runs are identical:
```bash
continuum analyze verify-determinism ./run1 ./run2
```

Export a signal's history to CSV:
```bash
continuum analyze export --run ./output --signal "terra.global_temp" --format csv > temp.csv
```
