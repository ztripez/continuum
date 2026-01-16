# Continuum Run Tool

This document defines the **run** tool, the primary interface for executing Continuum simulations.

The run tool implements the lifecycle defined in `@execution/lifecycle.md`.

## Usage

```bash
cargo run --bin run -- [OPTIONS] <WORLD_DIR>
```

## Arguments

- `<WORLD_DIR>`: Path to the World root directory (containing `world.yaml`).

## Options

### Execution Control
- `--steps <N>` (or `--ticks <N>`): Run for exactly N causal ticks (default: 10).
- `--dt <SECONDS>`: Override the timestep (default: determined by Era).

### Snapshot Output
- `--save <DIR>` (or `--snapshot-dir <DIR>`): Directory to write snapshot artifacts (manifest and JSON states).
- `--stride <N>` (or `--snapshot-stride <N>`): Interval for capturing snapshots (default: 10).

### Checkpoint & Resume
- `--checkpoint-dir <PATH>`: Enable checkpointing and set storage directory. Creates `{run_id}/` subdirectories.
- `--checkpoint-stride <N>`: Save checkpoint every N ticks (default: 1000).
- `--checkpoint-interval <SECONDS>`: Wall-clock throttle - max one checkpoint per T seconds.
- `--keep-checkpoints <N>`: Prune old checkpoints, keep only last N.
- `--resume`: Resume from latest checkpoint in `--checkpoint-dir`.
- `--force-resume`: Skip world IR validation on resume (dangerous - may crash or produce incorrect results).

## Execution Flow

The tool performs the following steps:

1. **Load World**: Discovers and compiles `world.yaml` and `*.cdsl` from `<WORLD_DIR>`.
2. **Lower & Validate**: Lowers AST to IR and validates constraints.
3. **Compile**: Builds execution DAGs for all (Era, Phase, Stratum) combinations.
4. **Runtime**: Initializes the runtime with an initial Era.
5. **Loop**:
   - Executes tick phases: Configure -> Collect -> Resolve -> Fracture -> Measure.
   - Captures snapshots if enabled (`--save` specified) and stride is met.
   - Advances tick and processes era transitions.

## Snapshot Output

When `--save` is used, the tool creates a subdirectory with a timestamp (e.g., `20260109_120000`) containing:

- `run.json`: A manifest with run metadata (seed, steps, stride, signal list).
- `tick_XXXXXX.json`: The full state of signals and fields for that tick.

## Examples

Run the `terra` world for 100 ticks, saving snapshots every 10 ticks:

```bash
cargo run --bin run -- examples/terra --steps 100 --save ./snapshots --stride 10
```

Run with a custom timestep:

```bash
cargo run --bin run -- examples/terra --dt 3.15e7
```

Run with checkpoints every 1000 ticks:

```bash
cargo run --bin run -- examples/terra --steps 100000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-stride 1000 \
  --keep-checkpoints 10
```

Resume from latest checkpoint:

```bash
cargo run --bin run -- examples/terra --resume \
  --checkpoint-dir ./checkpoints \
  --steps 50000
```

## Checkpoint vs Snapshot

**Checkpoints** (`--checkpoint-dir`):
- Store **causal state** (signals, entities, member signals)
- Enable crash recovery and resume
- Binary format (bincode + zstd compression)
- Tied to world IR version

**Snapshots** (`--save`):
- Store **observer data** (fields, measurements)
- For analysis and visualization
- JSON format (human-readable)
- Not resumable

See `@docs/persistence.md` for detailed comparison.