# Continuum Run Tool

This document defines the **run** tool, the primary interface for executing Continuum simulations.

The run tool implements the lifecycle defined in `@execution/lifecycle.md`.

---

## Usage

```bash
continuum run [OPTIONS] <WORLD_DIR>
```

## Arguments

- `<WORLD_DIR>`: Path to the World root directory (containing `world.yaml`).

## Options

### Scenario Selection
- `--scenario <FILE>`: Path to a scenario YAML file.
- `--scenario-name <NAME>`: Name of a scenario if bundled in the world.

### Execution Control
- `--ticks <N>`: Run for exactly N causal ticks.
- `--until <TIME>`: Run until simulated time reaches value (e.g., "1000 yr").
- `--seed <INT>`: Explicit seed for deterministic execution.

### Observer Output
- `--output <DIR>`: Directory to write observer artifacts (fields, logs).
- `--lens-port <PORT>`: Enable live Lens server on specified port.
- `--quiet`: Suppress stdout logging.

## Execution Flow

The tool performs the following steps:

1. **Load World**: Discovers and compiles `world.yaml` and `*.cdsl` from `<WORLD_DIR>`.
2. **Load Scenario**: Applies the specified scenario configuration.
3. **Warmup**: Executes the pre-causal warmup phase.
4. **Loop**:
   - Checks stop conditions (ticks, time, terminal era).
   - Executes one tick (Configure -> Collect -> Resolve -> Fracture -> Measure).
   - Notifies attached observers (Lens, disk writers).
5. **Shutdown**: Flushes pending writes and exits.

## Exit Codes

- `0`: Run completed successfully (reached limit or terminal era).
- `1`: Configuration or compilation error.
- `2`: Runtime fault (assertion failure, divergence).
- `3`: Determinism violation detected.

---

## Examples

Run the `terra` world with the `genesis` scenario for 1 million years:

```bash
continuum run examples/terra --scenario scenarios/genesis.yaml --until "1 Myr"
```

Run with a fixed seed and live visualization:

```bash
continuum run examples/terra --seed 12345 --lens-port 8080
```
