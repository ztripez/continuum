# Continuum Tools

CLI tools for interacting with the Continuum engine.

This crate provides the binary entry points for running simulations, analyzing outputs, and linting DSL code.

## Logging

All tools use `tracing` for structured logging. You can control the log level using the `RUST_LOG` environment variable.

- **Default**: `info` level for continuum crates, `warn` for dependencies.
- **Example**: `RUST_LOG=debug cargo run --bin run -- examples/terra`

### `run`
Executes a simulation world and optionally captures state snapshots.

```bash
cargo run --bin run -- <WORLD_DIR> [--steps N] [--save DIR]
```

### `compile`
Compiles a world into a bytecode bundle for later execution.

```bash
cargo run --bin compile -- <WORLD_DIR> [--out-dir build]
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

### `check`
Test utility to verify that a full world directory can be loaded and compiled to IR without error.

```bash
cargo run --bin check -- <WORLD_DIR>
```

### `world-ipc`
Runs a world as a long-lived IPC server over a Unix socket. Uses a binary framing protocol (Bincode).

```bash
cargo run --bin world-ipc -- examples/terra --socket /tmp/continuum.sock
```

### `world-ipc-web`
Serves a small WebSocket proxy that translates between JSON (WebSocket) and the binary Sim Server protocol, plus a basic frontend.

```bash
cargo run --bin world-ipc-web -- --socket /tmp/continuum.sock --bind 0.0.0.0:8080
```

#### JSON Protocol (UI Clients)

Requests: `{"id": 1, "type": "command", "payload": { ... }}`
Responses: `{"id": 1, "ok": true, "payload": { ... }}`
Events: `{"type": "tick|chronicle.event", "payload": { ... }}`

Commands:
- `status`: Returns current tick, era, time, and running state.
- `step { "count": n }`: Executes `n` ticks.
- `run { "count": n }`: Runs `n` ticks (or indefinitely if `count` is null) at maximum speed.
- `stop`: Stops a running simulation.
- `field.list`: Returns all available fields.
- `field.query { "field_id": "...", "position": [x,y,z] }`: Returns interpolated value at simulation time.
- `field.history { "field_id": "..." }`: Returns available historical ticks for a field.
- `impulse.list`: Returns all registered impulses and their payload types.
- `impulse.emit { "impulse_id": "...", "payload": { ... } }`: Injects an impulse for the next tick.
- `playback.set { "lag_ticks": n, "speed": m }`: Configures the observer playback clock.
