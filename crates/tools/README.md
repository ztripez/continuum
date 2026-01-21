# Continuum Tools

Library surface for interacting with the Continuum engine.

This crate provides the canonical API for running simulations, and interacting with world execution via a transport-agnostic API.

## Core API

### `RunWorldIntent`
The single authoritative intent for executing a world. It supports loading from source directories, compiled `.cvm` bundles, or JSON.

```rust
use continuum_tools::run_world_intent::{RunWorldIntent, WorldSource};

let source = WorldSource::from_path("path/to/world".into())?;
let intent = RunWorldIntent::new(source, 100);
let report = intent.execute()?;
```

### `WorldApi`
A transport-agnostic API for interacting with a running world. Provides a unified request/response/event model for observers and controllers.

See `src/world_api.rs` for the message schema.

### `WorldIpcServer`
A helper to run a `RunWorldIntent` as an IPC server over a Unix socket, providing a `WorldApi` interface.

## Legacy Code Removal
The following CLI binaries have been removed in favor of the library API:
- `run`
- `compile`
- `analyze`
- `dsl-lint`
- `check`
- `world-ipc`
- `world-ipc-web` (replaced by `continuum-inspector`)
