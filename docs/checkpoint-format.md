# Checkpoint File Format Specification

**Status**: Implemented  
**Version**: 1  
**File Extension**: `.ckpt`

---

## Purpose

Checkpoints serialize complete **causal state** of a Continuum simulation to enable:
- Crash recovery
- Long-running simulations (pause/resume)
- Experimentation from saved states
- Portable simulation transfer between machines

**Critical distinction**: Checkpoints store **causal state** (signals, entities, member signals). 
Observer data (fields, chronicles) is stored separately in **Lens snapshots**.

---

## File Format

```
checkpoint_{tick:010}.ckpt
│
├─ [Compression Layer: zstd]
│   └─ [Serialization Layer: bincode]
│       ├─ CheckpointHeader
│       └─ CheckpointState
```

### Compression

- **Format**: zstd (Zstandard)
- **Level**: 3 (default, balances speed/size)
- **Typical ratio**: 5-10x compression for simulation data

### Serialization

- **Format**: bincode (Rust binary serialization)
- **Endianness**: Little-endian (portable across x86_64/ARM64)
- **Schema versioning**: Via `CheckpointHeader.version`

---

## Data Structure

### CheckpointHeader

Metadata for validation and resume logic.

```rust
pub struct CheckpointHeader {
    /// Checkpoint format version (currently 1)
    pub version: u32,
    
    /// Blake3 hash of CompiledWorld IR (for validation)
    pub world_ir_hash: [u8; 32],
    
    /// Simulation tick at checkpoint
    pub tick: u64,
    
    /// Simulation time in seconds
    pub sim_time: f64,
    
    /// Random seed (for determinism)
    pub seed: u64,
    
    /// Current era at checkpoint
    pub current_era: EraId,
    
    /// Wall-clock timestamp when checkpoint was created
    pub created_at: SystemTime,
    
    /// Git commit hash of world directory (if available)
    pub world_git_hash: Option<String>,
}
```

**Validation on load**:
- `world_ir_hash` must match current world (unless `--force-resume`)
- `version` must be supported by runtime

### CheckpointState

Complete causal state for resume.

```rust
pub struct CheckpointState {
    /// Global signals (SignalId → Value)
    pub signals: HashMap<SignalId, Value>,
    
    /// Entity instance counts (EntityId → EntityInstances)
    pub entities: HashMap<EntityId, EntityInstances>,
    
    /// Member signal data (extracted from SoA buffers)
    pub member_signals: MemberSignalData,
    
    /// Era configurations (for validation)
    pub era_configs: IndexMap<EraId, EraConfigSnapshot>,
    
    /// Stratum execution state (cadence counters)
    pub stratum_states: HashMap<StratumId, StratumState>,
}
```

### MemberSignalData

Portable representation of SoA buffers.

```rust
pub struct MemberSignalData {
    /// Signal name → [(instance_idx, value), ...]
    pub signals: HashMap<String, Vec<(usize, Value)>>,
    
    /// Entity type → instance count
    pub entity_instance_counts: HashMap<String, usize>,
    
    /// Total instance count across all entities
    pub total_instance_count: usize,
}
```

**Why not serialize SoA buffers directly?**
- SoA buffers use custom SIMD-aligned allocators
- Pointer addresses are machine-specific
- `MemberSignalData` is a portable, relocatable representation

**Extraction process**:
1. Iterate all registered member signals
2. For each signal, extract values for all instances
3. Store as `(instance_idx, Value)` pairs
4. Preserve entity instance count metadata

**Restoration process**:
1. Reconstruct SoA buffers with correct signal registration
2. Initialize with correct instance counts per entity
3. Restore values via `buffer.set_current(signal, idx, value)`

---

## File Naming Convention

```
checkpoint_{tick:010}.ckpt
```

Examples:
- `checkpoint_0000000000.ckpt` - Initial state (tick 0)
- `checkpoint_0000001000.ckpt` - After 1000 ticks
- `checkpoint_0000050000.ckpt` - After 50000 ticks

**Zero-padding**: Ensures lexicographic sort matches chronological order.

---

## Directory Structure

```
{checkpoint_dir}/
  {run_id}/
    manifest.json              # Run metadata
    checkpoint_0000001000.ckpt
    checkpoint_0000002000.ckpt
    checkpoint_0000003000.ckpt
    latest -> checkpoint_0000003000.ckpt  # Symlink (Unix)
```

### manifest.json

```json
{
  "run_id": "20260116_094530",
  "created_at": "2026-01-16T09:45:30Z",
  "seed": 42,
  "steps": 100000,
  "stride": 1000,
  "signals": ["terra.temperature", "terra.pressure", ...],
  "fields": ["terra.elevation", "terra.wind", ...]
}
```

### 'latest' Symlink

- **Platform**: Unix-like systems (Linux, macOS)
- **Purpose**: Quick resume without searching for newest checkpoint
- **Fallback**: If symlink doesn't exist, find newest by filename

---

## Resume Validation

When loading a checkpoint, the runtime validates:

### 1. World IR Hash
```rust
if checkpoint.header.world_ir_hash != compute_world_ir_hash(world) {
    return Err("World IR has changed - checkpoint incompatible");
}
```

**Why it matters**: World IR changing means operators, signals, or eras changed. 
Resume would produce incorrect results or crash.

**Override**: Use `--force-resume` to skip validation (dangerous).

### 2. Era Configuration
- Era `dt` must match
- Stratum cadences must match
- Era count must match

### 3. Signal Schema
- All checkpointed signals must exist in current world
- Signal types must match (Scalar, Vec3, etc.)

---

## Non-Blocking I/O

**Philosophy**: Lost checkpoint > blocked simulation

Checkpoints use a **bounded queue** with non-blocking writes:

```
Main Thread                Background Writer Thread
───────────────           ────────────────────────
tick()                    
  ...                     
  request_checkpoint() ──→ Queue (depth: 3)
  (returns immediately)     │
tick()                      │
  ...                       └→ write to disk
tick()                         (blocking I/O here)
  ...
```

**If queue is full**:
- Checkpoint request is **dropped**
- Warning logged: `"Checkpoint request failed: queue full"`
- Simulation continues without blocking

**Rationale**: Blocking on I/O would break determinism and performance guarantees.

---

## Size Estimates

Typical checkpoint sizes (zstd compressed):

| Simulation | Signals | Entities | Member Signals | Size (compressed) |
|-----------|---------|----------|----------------|-------------------|
| POC       | 5       | 0        | 0              | ~1 KB             |
| Terra     | 25      | 3 types  | 45 members     | ~500 KB - 5 MB    |
| Terra (long) | 25   | 3 types  | 45 members     | ~5 MB - 50 MB     |

**Factors affecting size**:
- Entity instance counts (more plates/cells = larger)
- Member signal value distributions (randomness compresses poorly)
- Number of signals/members

---

## Compatibility

### Portable Across
- ✅ CPU architectures (x86_64, ARM64)
- ✅ Operating systems (Linux, macOS, Windows)
- ✅ Machines with different RAM/disk

### NOT Portable Across
- ❌ Different Continuum versions (may change IR format)
- ❌ Different world versions (world code changes)
- ❌ Different DSL (*.cdsl changes)

**Rule**: Checkpoint is tied to exact world IR. Changing a single operator 
invalidates all checkpoints (detected via hash mismatch).

---

## Error Handling

### Write Errors
- Background writer logs error
- Partial file is deleted
- Simulation continues

### Load Errors

| Error | Behavior |
|-------|----------|
| File not found | Clear error: "Checkpoint not found at {path}" |
| Decompression failed | "Checkpoint file corrupted (decompression failed)" |
| Deserialization failed | "Checkpoint format incompatible (deserialize failed)" |
| World IR mismatch | "World has changed - checkpoint invalid (hash: ...)" |
| Missing signals | "Checkpoint references unknown signal: {name}" |

All errors are **fail-hard** (no silent fallbacks).

---

## Implementation Files

- `crates/kernels/runtime/src/checkpoint.rs` - Core format and I/O (503 lines)
- `crates/kernels/runtime/src/executor/mod.rs` - Runtime integration
- `crates/tools/src/bin/run.rs` - CLI support

---

## Future Enhancements

### Possible Improvements
- **Delta checkpoints**: Store only changed values since last checkpoint
- **Parallel compression**: zstd supports multi-threaded compression
- **Checkpoint metadata queries**: Inspect checkpoint without full load
- **Checkpoint diffs**: Show what changed between two checkpoints

### Not Planned
- ❌ JSON export (too large, defeats compression)
- ❌ Backwards compatibility (world IR is versioned independently)
- ❌ Cloud storage (user can sync checkpoint dir manually)

---

## See Also

- `docs/persistence.md` - Overview of checkpoint vs lens snapshots
- `docs/checkpoint-usage.md` - User guide with CLI examples
- `tools/run.md` - CLI reference for checkpoint flags
