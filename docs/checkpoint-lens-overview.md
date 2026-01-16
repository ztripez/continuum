# Checkpoint and Lens Snapshot System - Implementation Overview

This document provides a high-level overview of the two persistence systems being implemented for Continuum.

---

## Quick Reference

| System | Purpose | Contains | Frequency | Use Case |
|--------|---------|----------|-----------|----------|
| **Checkpoints** | Resume/recovery | Causal state | Infrequent (1000s ticks) | Crash recovery, long runs |
| **Lens Snapshots** | Analysis/viz | Observer data | Frequent (10s-100s ticks) | Post-hoc analysis, export |

---

## 1. Checkpoints (Resume/Recovery)

**Epic**: `continuum-mblt`

### Purpose
Enable simulations to be checkpointed during execution and resumed from saved state.

### What Gets Saved
- Signal values (current + previous)
- Entity instances
- Member signal buffers (SoA)
- Era and stratum state
- World IR hash for validation

### Key Features
- **Non-blocking I/O**: Bounded queue, drop-on-full (lost checkpoint > blocked simulation)
- **Deterministic resume**: Same world + checkpoint → identical continuation
- **World IR validation**: Fail if world has changed (unless --force-resume)
- **Portable**: bincode + zstd compression

### CLI Usage
```bash
# Enable checkpointing
continuum run ./terra --steps 100000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-stride 1000 \
  --keep-checkpoints 5

# Resume from latest
continuum run ./terra --resume --steps 50000
```

### Implementation Phases
1. **Phase 1**: Core format and serialization (continuum-tly7)
2. **Phase 2**: CLI support (continuum-5s7u, continuum-4wu0)
3. **Phase 3**: Validation and determinism tests (continuum-16hw)
4. **Phase 4**: Compression and optimization (continuum-6grl)
5. **Phase 5**: IPC and utilities (continuum-xvhj, continuum-ak6s)

---

## 2. Lens Snapshots (Analysis/Visualization)

**Epic**: `continuum-d6wi`

### Purpose
Enable field frames to be saved during simulation and loaded later for analysis and visualization.

### What Gets Saved
- Field samples per tick (FieldFrame history)
- Field metadata (topology, reconstruction hints)
- Tick → field mapping
- Optionally: signal values (for correlation analysis)

### Key Features
- **Chunked storage**: Large runs split into tick-range chunks (e.g., 50k ticks)
- **Field filtering**: Capture only specified fields (reduces size)
- **Selective loading**: Load only chunks of interest
- **Export**: Parquet, HDF5, or other formats for external tools

### CLI Usage
```bash
# Enable lens snapshots
continuum run ./terra --steps 100000 \
  --lens-dir ./lens \
  --lens-stride 100 \
  --lens-fields "atmosphere.temperature,plates.velocity"

# Query field data
continuum lens query atmosphere.temperature \
  --tick 5000 \
  --position 0,0,1

# Export to Parquet
continuum lens export ./lens/snapshot.lens \
  --format parquet \
  --output ./data/terra.parquet
```

### Implementation Phases
1. **Phase 1**: Serialization for FieldLens types (continuum-cjw8)
2. **Phase 2**: Chunked snapshot format (continuum-1x9p)
3. **Phase 3**: CLI support for capture (continuum-xh8p)
4. **Phase 4**: Loading and query API (continuum-1ff0)
5. **Phase 5**: Integration with checkpoints (continuum-fmmc)

---

## 3. Using Both Together

Checkpoints and lens snapshots are **complementary** and can run simultaneously:

```bash
continuum run ./terra --steps 1000000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-stride 10000 \
  --lens-dir ./lens \
  --lens-stride 100 \
  --lens-chunk-size 50000
```

**Result**:
- Checkpoints every 10,000 ticks (for resume)
- Lens snapshots every 100 ticks (for analysis)
- Can resume from checkpoint and analyze with lens chunks

---

## 4. Architecture Decisions

### Non-Blocking I/O (Checkpoints)
**Philosophy**: Lost checkpoint > blocked simulation

- Bounded channel with fixed capacity (e.g., 2-3 slots)
- Background writer thread handles compression + serialization
- If queue full, drop checkpoint request and log warning
- Main simulation thread never blocks

### Chunked Storage (Lens)
**Philosophy**: Enable selective loading for large datasets

- Default chunk size: 50,000 ticks
- Each chunk is independent (separate file)
- Manifest file lists all chunks
- Can load only ticks of interest (e.g., ticks 100k-150k)

### World IR Validation (Checkpoints)
**Philosophy**: Fail loudly on world mismatch

- Compute blake3 hash of CompiledWorld IR
- Store in checkpoint header
- On resume, validate hash matches
- Fail with clear error if mismatch (unless --force-resume)

### Observer Boundary (Lens)
**Philosophy**: Lens is observer-only

- Field frames do not contain causal state
- Cannot resume simulation from lens snapshot
- Can be pruned/cleared during run without affecting causality
- Separate from checkpoints (which ARE causal state)

---

## 5. File Formats

### Checkpoint Format
```
checkpoint_{tick:010}.ckpt (bincode + zstd)
├─ header
│  ├─ version: u32
│  ├─ world_ir_hash: [u8; 32]
│  ├─ tick: u64
│  ├─ sim_time: f64
│  ├─ current_era: EraId
│  └─ created_at: SystemTime
└─ state
   ├─ signals: SignalStorage
   ├─ entities: EntityStorage
   ├─ member_signals: MemberSignalBuffer
   ├─ era_configs: IndexMap<EraId, EraConfig>
   └─ stratum_states: HashMap<StratumId, StratumState>
```

### Lens Snapshot Format
```
lens_snapshot_{start}-{end}.lens (bincode + zstd)
├─ header
│  ├─ version: u32
│  ├─ run_id: String
│  ├─ tick_range: (u64, u64)
│  ├─ field_list: Vec<FieldId>
│  └─ created_at: SystemTime
└─ frames: Vec<FrameRecord>
   ├─ tick: u64
   ├─ sim_time: f64
   └─ fields: HashMap<FieldId, Vec<FieldSample>>
```

---

## 6. Directory Structures

### Checkpoints
```
./checkpoints/{run_id}/
├─ manifest.json
├─ checkpoint_0000001000.ckpt
├─ checkpoint_0000002000.ckpt
├─ checkpoint_0000003000.ckpt
└─ latest -> checkpoint_0000003000.ckpt  (symlink)
```

### Lens Snapshots
```
./lens/{run_id}/
├─ manifest.json
├─ lens_snapshot_0000000000-0000050000.lens
├─ lens_snapshot_0000050000-0000100000.lens
└─ lens_snapshot_0000100000-0000150000.lens
```

---

## 7. Related Documentation

- **`docs/persistence.md`** - Detailed comparison of checkpoint vs lens systems
- **`docs/checkpoint-format.md`** - Checkpoint file format specification (to be written)
- **`docs/checkpoint-usage.md`** - Checkpoint usage guide (to be written)
- **`tools/run.md`** - CLI reference (to be updated)
- **`tools/lens.md`** - Lens CLI tool reference (to be written)

---

## 8. Beads Issues

### Checkpoint Epic
- **`continuum-mblt`** - Checkpoint and Resume System (epic)
  - `continuum-tly7` - Define checkpoint file format
  - `continuum-5s7u` - Add checkpoint stride and throttling
  - `continuum-4wu0` - CLI support for checkpoint/resume
  - `continuum-16hw` - Deterministic resume validation
  - `continuum-6grl` - Compression and optimization
  - `continuum-xvhj` - IPC commands
  - `continuum-ak6s` - Management utilities
  - `continuum-nbyb` - Documentation

### Lens Snapshot Epic
- **`continuum-d6wi`** - Lens Snapshot Persistence System (epic)
  - `continuum-cjw8` - Add serialization to FieldLens
  - `continuum-1x9p` - Chunked snapshot format
  - `continuum-xh8p` - CLI support for lens capture
  - `continuum-1ff0` - Lens loading and query API
  - `continuum-fmmc` - Integration with checkpoints

---

## 9. Next Steps

1. Begin implementation with **checkpoint format** (continuum-tly7)
2. Parallel track: **lens serialization** (continuum-cjw8)
3. CLI integration for both systems
4. Validation and testing
5. Utilities and documentation

**Priority**: Checkpoints first (enables crash recovery), then lens snapshots (enables analysis).
