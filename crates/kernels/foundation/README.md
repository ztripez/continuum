# Continuum Foundation

Core primitives and stable identifiers for the Continuum engine.

This crate provides the fundamental types used across the entire system, ensuring consistent identity, hashing, and basic value representation.

## Key Types

- **`SignalId`, `OperatorId`, `FieldId`, ...**: Strongly-typed, string-interned identifiers for simulation entities.
- **`StableHash`**: Deterministic hashing trait used for graph construction and replay identity.
- **`Dt`**: Type-safe timestep wrapper.

This crate has minimal dependencies and serves as the root of the dependency graph.
