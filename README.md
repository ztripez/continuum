# Continuum Prime

> **Systems evolve. Structure is a consequence.**
> **Structure emerges from tension.**

Continuum is a deterministic, causality-first simulation engine. It models systems where behavior emerges from declared signals, operators, constraints, and coupling, compiled into a deterministic execution graph.

## Core Principles

- **Causality is Primary:** Simulation behavior must arise from declared causes.
- **Graph-First Execution:** Logic is compiled to a deterministic execution DAG.
- **Determinism is Mandatory:** All ordering is explicit; randomness is derived.
- **Signals are Authority:** State is expressed as authoritative resolved values.
- **Observer Boundary is Absolute:** Observation (fields, chronicles) never influences causality.

## Architecture

Continuum is built as a modular workspace of Rust crates:

### Core Kernels
- **`continuum-foundation`**: Core stable identifiers, hashing, and primitives.
- **`continuum-dsl`**: Parser, AST, and validation for the Continuum DSL (`*.cdsl`).
- **`continuum-ir`**: Typed Intermediate Representation and lowering logic.
- **`continuum-runtime`**: The execution engine, DAG scheduler, and state storage.
- **`continuum-vm`**: Stack-based bytecode virtual machine for expression evaluation.

### Kernel System
- **`continuum-kernel-registry`**: Distributed registry for engine-provided functions.
- **`continuum-kernel-macros`**: Proc-macros for defining kernels (`#[kernel_fn]`).
- **`continuum-functions`**: Standard library of mathematical and physical kernels.

### Tooling
- **`continuum-tools`**: Library API for running simulations, and transport-agnostic IPC server.
- **`continuum-inspector`**: Web-based inspector UI for interacting with worlds.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Philosophy:** [`@docs/manifesto.md`](docs/manifesto.md), [`@docs/principles.md`](docs/principles.md)
- **DSL:** [`@docs/dsl/language.md`](docs/dsl/language.md), [`@docs/dsl/syntax.md`](docs/dsl/syntax.md)
- **Execution:** [`@docs/execution/lifecycle.md`](docs/execution/lifecycle.md), [`@docs/execution/dag.md`](docs/execution/dag.md)
- **Concepts:** [`@docs/signals.md`](docs/signals.md), [`@docs/time.md`](docs/time.md), [`@docs/eras.md`](docs/eras.md)

## Usage

To run a simulation world with the web inspector:

```bash
cargo run -p continuum_inspector -- examples/terra
```

To execute a world programmatically via the library:

```rust
use continuum_tools::run_world_intent::{RunWorldIntent, WorldSource};

let source = WorldSource::from_path("examples/terra".into())?;
let intent = RunWorldIntent::new(source, 100);
let report = intent.execute()?;
```
