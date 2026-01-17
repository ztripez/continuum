# Persistence in Continuum

This document defines the **two persistence systems** in Continuum: **Checkpoints** and **Lens Snapshots**.

These systems serve different purposes and have different guarantees.

---

## Overview

Continuum provides two types of persistence:

1. **Checkpoints** - Resume and recovery
2. **Lens Snapshots** - Analysis and visualization

These are **separate, complementary systems**:
- Checkpoints save causal state for resume
- Lens snapshots save observer data for analysis
- Both can be enabled simultaneously
- Neither influences execution

---

## 1. Checkpoints (Resume/Recovery)

### Purpose

Checkpoints enable:
- **Crash recovery** - Resume after unexpected failures
- **Long-running simulations** - Checkpoint worlds that run for days/weeks
- **Experimentation** - Save state and resume with different parameters
- **Portability** - Checkpoint on one machine, resume on another

### What Gets Checkpointed

**Causal state only:**
- Signal values (current + previous tick)
- Entity instances
- Member signal buffers (SoA storage)
- Era and stratum state
- Tick, sim_time, current era
- World IR hash (for validation)

**Not checkpointed (reconstructed on resume):**
- Execution DAGs (rebuilt from IR)
- Registered resolvers/operators (re-registered)
- Warmup state (re-executed if needed)

**Not checkpointed (transient):**
- Input channels (cleared between ticks)
- Field buffers (observer data - goes in lens snapshots instead)
- Event buffers (observer data)
- Fracture queue (applied before checkpoint)

### File Format

```
checkpoint_0001000.ckpt (bincode + zstd)
  - header:
      - version: u32
      - world_ir_hash: [u8; 32]
      - tick: u64
      - sim_time: f64
      - current_era: EraId
      - created_at: SystemTime
  - state:
      - signals: SignalStorage
      - entities: EntityStorage
      - member_signals: MemberSignalBuffer
      - era_configs: IndexMap<EraId, EraConfig>
      - stratum_states: HashMap<StratumId, StratumState>
```

### Directory Structure

```
./checkpoints/{run_id}/
  manifest.json                     # Run metadata
  checkpoint_0000001000.ckpt
  checkpoint_0000002000.ckpt
  checkpoint_0000003000.ckpt
  latest -> checkpoint_0000003000.ckpt  # Symlink to latest
```

### Usage

```bash
# Enable checkpointing
continuum run ./terra --steps 100000 \
  --checkpoint-dir ./terra-checkpoints \
  --checkpoint-stride 1000 \
  --checkpoint-interval 3600 \
  --keep-checkpoints 5

# Resume from latest checkpoint
continuum run ./terra --resume --steps 50000

# Resume with validation override
continuum run ./terra --resume --force-resume
```

### Determinism Guarantee

> Same world IR + same checkpoint → **identical continuation**

Checkpoints preserve complete determinism:
- World IR hash validated on resume
- All ordering is stable
- Seed stored in checkpoint
- No ambient randomness

### Implementation Details

**Non-blocking I/O:**
- Bounded queue with drop-on-full semantics
- Philosophy: **lost checkpoint > blocked simulation**
- Background writer thread handles compression + serialization
- Main simulation never blocks on checkpoint I/O

**Validation on resume:**
- World IR hash must match (unless `--force-resume`)
- Era configs must be compatible
- Stratum definitions must match

**See also:**
- `@docs/checkpoint-format.md` - Detailed format specification
- `@docs/checkpoint-usage.md` - Usage guide and best practices
- `@tools/run.md` - CLI reference

---

## 2. Lens Snapshots (Analysis/Visualization)

### Purpose

Lens snapshots enable:
- **Post-hoc analysis** - Analyze field evolution after run completes
- **Visualization** - Render fields at any captured tick
- **Data export** - Export field samples for external tools
- **Replay** - Play back field history without re-running simulation

### What Gets Saved

**Observer data only:**
- Field samples per tick (FieldFrame history)
- Field metadata (topology, reconstruction hints)
- Tick → field mapping
- Signal values (for correlation analysis)

**Not saved:**
- Causal state (use checkpoints for that)
- Execution DAGs
- Transient buffers

### File Format

```
lens_snapshot_{run_id}.lens (bincode + zstd)
  - header:
      - version: u32
      - run_id: String
      - world_name: String
      - seed: u64
      - created_at: SystemTime
      - tick_range: (u64, u64)  # First and last tick
  - config:
      - captured_fields: Vec<FieldId>
      - field_configs: HashMap<FieldId, FieldConfig>
      - topology: TopologySpec
  - frames:
      - frames: Vec<FrameRecord>
        - tick: u64
        - sim_time: f64
        - fields: HashMap<FieldId, Vec<FieldSample>>
        - signals: HashMap<SignalId, Value>  # Optional
```

### Directory Structure

```
./lens/{run_id}/
  manifest.json                    # Lens metadata
  lens_snapshot_full.lens          # Full snapshot (all frames)
  lens_snapshot_0000000000-0000010000.lens  # Chunked (ticks 0-10000)
  lens_snapshot_0000010000-0000020000.lens  # Chunked (ticks 10000-20000)
```

### Usage

```bash
# Enable lens snapshot capture during run
continuum run ./terra --steps 100000 \
  --lens-dir ./terra-lens \
  --lens-stride 10 \
  --lens-fields "atmosphere.temperature,plates.velocity" \
  --lens-chunk-size 10000

# Load lens snapshot for analysis
continuum lens load ./terra-lens/lens_snapshot_full.lens

# Query field at specific tick
continuum lens query atmosphere.temperature --tick 5000 --position 0,0,1

# Export to external format
continuum lens export ./terra-lens/lens_snapshot_full.lens \
  --format parquet \
  --output ./data/terra_fields.parquet
```

### Lens-Specific Features

**Chunk-based storage:**
- Large runs split into tick-range chunks
- Enables selective loading (only load ticks of interest)
- Reduces memory footprint for analysis

**Field filtering:**
- Capture only specified fields (not all)
- Reduces snapshot size
- `--lens-fields` accepts comma-separated list or "all"

**Reconstruction caching:**
- Snapshots include reconstruction hints
- Enables fast nearest-neighbor queries
- GPU-accelerated reconstruction (optional)

**Topology metadata:**
- Cubed sphere topology stored with snapshot
- Enables correct spatial queries on replay
- Tile hierarchy preserved

### Integration with FieldLens

Lens snapshots are **serialized FieldLens state**:

```rust
// During run: FieldLens accumulates frames in memory
let mut lens = FieldLens::new(config)?;
lens.record_many(tick, fields);

// Periodic snapshot: serialize current lens state
lens.save_snapshot("./lens/snapshot_000001000.lens")?;

// After run: load snapshot for analysis
let lens = FieldLens::load_snapshot("./lens/snapshot_000001000.lens")?;
let recon = lens.at(&field_id, tick)?;
let value = recon.sample(&position)?;
```

**Key difference from checkpoints:**
- Lens can be cleared/pruned during run (bounded history)
- Checkpoints must preserve **all** state for resume
- Lens snapshots are **lossy** (can drop old frames)
- Checkpoints are **lossless** (complete state)

---

## 3. Using Both Together

Checkpoints and lens snapshots are **complementary**:

```bash
# Long-running simulation with both systems enabled
continuum run ./terra --steps 1000000 \
  --checkpoint-dir ./terra-checkpoints \
  --checkpoint-stride 10000 \
  --lens-dir ./terra-lens \
  --lens-stride 100 \
  --lens-chunk-size 50000
```

**Result:**
```
./terra-checkpoints/20260116_095430/
  checkpoint_0000010000.ckpt
  checkpoint_0000020000.ckpt
  latest -> checkpoint_0000020000.ckpt

./terra-lens/20260116_095430/
  lens_snapshot_0000000000-0000050000.lens
  lens_snapshot_0000050000-0000100000.lens
  manifest.json
```

**Benefits:**
- **Crash recovery**: Resume from checkpoint, continue simulation
- **Analysis**: Load lens chunks, analyze field evolution
- **Efficiency**: Checkpoint stride can be large (10k ticks), lens stride small (100 ticks)
- **Storage**: Lens chunks can be pruned/archived, checkpoints kept for resume

---

## 4. Implementation Strategy

### Phase 1: Checkpoints (Priority 1)
1. Define checkpoint format and serialization
2. Add CLI support for checkpoint/resume
3. Implement validation and determinism tests

### Phase 2: Lens Snapshots (Priority 2)
1. Add serialization to FieldLens and FieldStorage
2. Implement chunked snapshot format
3. Add CLI support for lens capture
4. Add lens loading and query API

### Phase 3: Integration (Priority 2)
1. Enable simultaneous checkpoint + lens capture
2. Add IPC commands for both systems
3. Integrate with continuum-inspector UI

### Phase 4: Utilities (Priority 3)
1. Checkpoint management tools
2. Lens analysis tools
3. Export to external formats (Parquet, HDF5, etc.)

---

## 5. Comparison Table

| Feature | Checkpoints | Lens Snapshots |
|---------|-------------|----------------|
| **Purpose** | Resume/recovery | Analysis/visualization |
| **Contains** | Causal state | Observer data |
| **Determinism** | Must be exact | Not required |
| **Portability** | World IR must match | Portable across worlds |
| **Size** | Larger (complete state) | Smaller (observer data only) |
| **Frequency** | Infrequent (1000s of ticks) | Frequent (10s-100s of ticks) |
| **I/O strategy** | Non-blocking, drop-on-full | Background writer, can batch |
| **Pruning** | Keep last N | Chunk-based, prune old chunks |
| **Resume** | Can resume simulation | Cannot resume (observer-only) |
| **Analysis** | Not designed for analysis | Designed for post-hoc analysis |

---

## Summary

- **Two systems, two purposes**
- Checkpoints = causal state for resume
- Lens snapshots = observer data for analysis
- Both can run simultaneously
- Neither influences execution
- Checkpoints are lossless, lens snapshots are lossy
- Use checkpoints for recovery, lens for visualization

When in doubt:

> **If you need to resume: checkpoint**  
> **If you need to analyze: lens snapshot**  
> **If you need both: enable both**
