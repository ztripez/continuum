# Continuum Runtime

The core execution engine for Continuum.

This crate manages the lifecycle of a simulation run. It constructs the execution DAG, manages state storage, and orchestrates the tick loop through the five execution phases.

## Responsibilities

- **DAG Construction**: Builds `(Phase × Stratum × Era)` dependency graphs from the IR.
- **State Storage**: Manages double-buffered `SignalStorage`, input channels, and field buffers.
- **Scheduler**: Executes DAG nodes in topological levels, enabling safe parallelism.
- **Phase Orchestration**: Enforces the `Configure` → `Collect` → `Resolve` → `Fracture` → `Measure` cycle.
- **Determinism**: Ensures all ordering and execution is strictly reproducible.

## Key Modules

- **`dag`**: Graph data structures and topological sorting.
- **`storage`**: Signal and entity value management.
- **`executor`**: The tick loop and phase barriers.
- **`types`**: Runtime value types (`Value`, `Phase`, `StratumState`).
