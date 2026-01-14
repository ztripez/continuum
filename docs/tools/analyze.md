# Continuum Analyze Tool

This document defines the **analyze** tool, used for validating and comparing simulation runs.

It consumes snapshot artifacts produced by `run --save`, and must access
field data through **Lens semantics** (observer-only). `FieldSnapshot` is
internal transport and must not be treated as a public API for end programs.

When batch queries are used, Lens may auto-select a GPU backend if enabled.

## Usage

```bash
cargo run --bin analyze -- <COMMAND> [OPTIONS]
```

## Commands

### `baseline`
Record and compare reference baselines for regression testing.

#### `record`
Create a baseline JSON file from a snapshot run.

```bash
cargo run --bin analyze -- baseline record <SNAPSHOT_DIR> --output <BASELINE_FILE>
```

Options:
- `--fields <LIST>`: Comma-separated list of fields to include (default: all).
- `--include-samples-hash`: Compute SHA256 of all samples for exact determinism checks.

#### `compare`
Validate a snapshot run against a recorded baseline.

```bash
cargo run --bin analyze -- baseline compare <SNAPSHOT_DIR> --baseline <BASELINE_FILE>
```

Options:
- `--tolerance <FLOAT>`: Allowed relative deviation (default: 0.05 for 5%).
- `--fields <LIST>`: Specific fields to compare.

## Examples

**1. Generate a reference run:**
```bash
cargo run --bin run -- examples/terra --save ./output/ref --steps 50
```

**2. Record baseline:**
```bash
cargo run --bin analyze -- baseline record ./output/ref --output terra_baseline.json
```

**3. Run a test:**
```bash
cargo run --bin run -- examples/terra --save ./output/test --steps 50
```

**4. Verify against baseline:**
```bash
cargo run --bin analyze -- baseline compare ./output/test --baseline terra_baseline.json
```
