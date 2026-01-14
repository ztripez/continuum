# Continuum Tools

CLI tools for interacting with the Continuum engine.

This crate provides the binary entry points for running simulations, analyzing outputs, and linting DSL code.

## Logging

All tools use `tracing` for structured logging. You can control the log level using the `RUST_LOG` environment variable.

- **Default**: `info` level for continuum crates, `warn` for dependencies.
- **Example**: `RUST_LOG=debug cargo run --bin world-run -- examples/terra`

## Binaries

### `world-run`
Executes a simulation world and optionally captures state snapshots.

```bash
cargo run --bin world-run -- <WORLD_DIR> [--steps N] [--save DIR]
```

### `analyze`
Validates simulation outputs against baselines.

```bash
cargo run --bin analyze -- baseline record <SNAPSHOT_DIR> --output baseline.json
cargo run --bin analyze -- baseline compare <SNAPSHOT_DIR> --baseline baseline.json
```

### `dsl-lint`
Parses and validates DSL files, reporting syntax and semantic errors.

```bash
cargo run --bin dsl-lint -- <FILE_OR_DIR>
```

### `world-load`
Test utility to verify that a full world directory can be loaded and compiled to IR without error.

```bash
cargo run --bin world-load -- <WORLD_DIR>
```

### `world-ipc`
Runs a world as a long-lived IPC server over a Unix socket.

```bash
cargo run --bin world-ipc -- examples/terra --socket /tmp/continuum.sock
```

Commands:
- `status`
- `step [n]`
- `run [n]` (omit `n` to run indefinitely)
- `stop`
- `quit`

While running, the server broadcasts lines like:
- `tick <n> era=<era> sim_time=<seconds>`

### `world-ipc-web`
Serves a small WebSocket proxy that forwards frames to a Unix IPC socket, plus a basic frontend.

```bash
cargo run --bin world-ipc-web -- --socket /tmp/continuum.sock --bind 0.0.0.0:8080
```
