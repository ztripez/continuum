# Runtime Crate

Tick-based execution engine for Continuum simulations.

## Purpose

This crate executes compiled DAGs through deterministic tick phases:
1. **Configure** - Prepare tick context
2. **Collect** - Accumulate inputs from operators
3. **Resolve** - Compute signal values (parallelized within levels)
4. **Fracture** - Detect tension and queue outputs for next tick
5. **Measure** - Emit fields for observation (TODO)

## Rules

### Determinism is Sacred
- Execution order must be stable and reproducible
- Use `IndexMap` instead of `HashMap` for ordered iteration
- Parallel execution (rayon) only within DAG levels where order doesn't matter
- Results are applied sequentially after parallel computation

### Signal Storage
- All signals (global and per-entity) are stored in `MemberSignalBuffer` (SoA layout)
- Global signals use the root entity with instance_count=1
- `current` holds values being resolved this tick
- `previous` holds last tick's resolved values
- `advance_tick()` preserves gated signal values before swapping
- Never lose signal state for gated strata

### Phase Boundaries
- Collect operators write to `InputChannels`
- Resolve reads from `InputChannels` and writes to `MemberSignalBuffer`
- Fracture outputs queue for next tick (not current)
- Phases must not cross-contaminate

### Error Handling
- NaN and Infinite results are errors, not silent failures
- Missing signals/eras are explicit errors
- Use `thiserror` for error types

### ID Types
- Import from `continuum_foundation`, not defined locally
- Re-export through `types.rs` for convenience

## Structure

```
src/
├── lib.rs              # Module exports
├── types.rs            # Runtime types + re-exported IDs
├── error.rs            # Error types
├── storage.rs          # InputChannels, FractureQueue, EntityStorage
├── soa_storage.rs      # MemberSignalBuffer (SoA signal storage)
├── unified_storage.rs  # UnifiedStorage (member_signals + entities)
├── dag/                # DAG structures and topological leveling
└── executor/           # Runtime, phases, bytecode VM, tick execution
```

## Key Types

- `Runtime` - Main execution engine
- `MemberSignalBuffer` - SoA double-buffered signal storage (global + per-entity)
- `UnifiedStorage` - Combines MemberSignalBuffer + EntityStorage
- `ExecutableDag` - Leveled DAG for a (phase, stratum) pair
- `ResolverFn`, `CollectFn`, `FractureFn` - Registered execution functions

## Testing

- Tests use simple DAGs with counters to verify phase execution
- Stratum stride test verifies gated signals preserve values
- Run `cargo test -p continuum-runtime` before changes
