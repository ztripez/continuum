# Checkpoint Usage Guide

**For**: Continuum users running long simulations, experimenting with resume, or recovering from crashes.

---

## Quick Start

### Enable Checkpoints

```bash
continuum run ./my-world \
  --steps 100000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-stride 1000
```

This will:
- Save a checkpoint every 1000 ticks
- Store checkpoints in `./checkpoints/{run_id}/`
- Create a `latest` symlink to newest checkpoint

### Resume from Latest Checkpoint

```bash
continuum run ./my-world \
  --resume \
  --checkpoint-dir ./checkpoints \
  --steps 50000
```

This will:
- Find the latest checkpoint in `./checkpoints/`
- Validate world compatibility
- Resume from that tick and run 50000 more

---

## CLI Flags

### Checkpointing Flags

```bash
--checkpoint-dir <PATH>
```
**Purpose**: Enable checkpointing and set storage directory  
**Default**: None (checkpointing disabled)  
**Example**: `--checkpoint-dir ./terra-checkpoints`

**Directory structure created**:
```
./terra-checkpoints/
  20260116_094530/              # Run ID (timestamp)
    manifest.json               # Run metadata
    checkpoint_0000001000.ckpt
    checkpoint_0000002000.ckpt
    latest -> checkpoint_0000002000.ckpt  # Symlink
```

---

```bash
--checkpoint-stride <N>
```
**Purpose**: Save checkpoint every N ticks  
**Default**: 1000  
**Example**: `--checkpoint-stride 500` (checkpoint every 500 ticks)

**Use cases**:
- Frequent: `--checkpoint-stride 100` (more checkpoints, less lost work)
- Moderate: `--checkpoint-stride 1000` (default, balances disk/time)
- Sparse: `--checkpoint-stride 10000` (fewer checkpoints, more disk space)

---

```bash
--checkpoint-interval <SECONDS>
```
**Purpose**: Wall-clock throttle (max once per T seconds)  
**Default**: None (no throttling)  
**Example**: `--checkpoint-interval 3600` (max 1 checkpoint per hour)

**Why use this?**
- Prevent disk hammering in fast simulations
- Limit checkpoint overhead for I/O-constrained systems
- Combine with stride: checkpoint every N ticks OR T seconds (whichever comes first)

**Example: stride + interval**:
```bash
--checkpoint-stride 1000 --checkpoint-interval 300
```
This checkpoints when **either**:
- 1000 ticks elapse, OR
- 5 minutes (300s) of wall-clock time passes

---

```bash
--keep-checkpoints <N>
```
**Purpose**: Prune old checkpoints, keep only last N  
**Default**: None (keep all checkpoints)  
**Example**: `--keep-checkpoints 5` (delete all but 5 newest)

**Use case**: Long-running simulations where disk space is limited.

---

### Resume Flags

```bash
--resume
```
**Purpose**: Resume from latest checkpoint  
**Default**: false (start from tick 0)

**Discovery order**:
1. Look for `latest` symlink in `--checkpoint-dir`
2. If not found, scan for newest checkpoint by filename
3. Load and validate

---

```bash
--force-resume
```
**Purpose**: Skip world IR validation (dangerous)  
**Default**: false (validate world hash)

**When to use**: 
- World DSL changed but you want to resume anyway
- **Risk**: Simulation may crash or produce incorrect results

**Error without `--force-resume`**:
```
Error: Cannot resume - world IR has changed
  Checkpoint world hash: a1b2c3d4...
  Current world hash:    e5f6g7h8...
  
  Either:
    1. Use the exact same world version
    2. Use --force-resume to skip validation (may crash)
```

---

## Examples

### Long-Running Terra Simulation

Run Terra for 1 million ticks with checkpoints every 10,000 ticks, keep only last 10:

```bash
continuum run ./examples/terra \
  --steps 1000000 \
  --checkpoint-dir ./terra-run-001 \
  --checkpoint-stride 10000 \
  --keep-checkpoints 10
```

**Disk usage**: ~10 × 5MB = 50MB (Terra checkpoint size)

---

### Resume After Crash

Simulation crashed at tick 450,000. Resume:

```bash
continuum run ./examples/terra \
  --resume \
  --checkpoint-dir ./terra-run-001 \
  --steps 550000  # Continue to 1 million
```

**Output**:
```
Loading checkpoint from ./terra-run-001/20260116_094530/checkpoint_0000450000.ckpt
Checkpoint: tick=450000, sim_time=450000.0s, era=main
Resuming simulation...
Tick 450001 [era: main] [dt: 1.0s]
...
```

---

### Experimentation: Multiple Branches

Save checkpoint at tick 100,000, then try different parameters:

**Branch A: High temperature**
```bash
continuum run ./my-world \
  --resume \
  --checkpoint-dir ./base-run \
  --steps 50000 \
  --checkpoint-dir ./branch-a-hot \
  --checkpoint-stride 5000
  # (modify world to increase temperature)
```

**Branch B: Low temperature**
```bash
continuum run ./my-world \
  --resume \
  --checkpoint-dir ./base-run \
  --steps 50000 \
  --checkpoint-dir ./branch-b-cold \
  --checkpoint-stride 5000
  # (modify world to decrease temperature)
```

**Note**: This requires world changes to be compatible with checkpoint (signal schema unchanged).

---

### Throttled Checkpoints

Slow simulation (1 tick/second). Checkpoint every 10 minutes of wall-clock time:

```bash
continuum run ./my-world \
  --steps 1000000 \
  --checkpoint-dir ./checkpoints \
  --checkpoint-interval 600  # 10 minutes
```

This avoids checkpointing too frequently when ticks are slow.

---

### Combined: Stride + Interval + Pruning

Practical production setup:

```bash
continuum run ./terra \
  --steps 10000000 \
  --checkpoint-dir ./terra-production \
  --checkpoint-stride 5000 \
  --checkpoint-interval 1800 \  # Max 1 checkpoint per 30 min
  --keep-checkpoints 20
```

**Behavior**:
- Checkpoints when 5000 ticks OR 30 minutes elapse
- Keeps rolling window of 20 checkpoints
- Disk usage: 20 × 5MB = 100MB

---

## Non-Blocking I/O

Checkpoint writes **never block** the simulation.

### How It Works

1. Simulation reaches checkpoint tick
2. State is cloned and sent to background writer queue
3. Simulation continues immediately (non-blocking)
4. Background thread writes checkpoint to disk

### If Queue Is Full

- **Queue depth**: 3 (bounded)
- **If full**: Checkpoint is **dropped** and warning logged
- **Simulation**: Continues without blocking

**Warning log**:
```
WARN: Checkpoint request failed: queue full
```

**Why drop instead of block?**  
Blocking on I/O would break determinism guarantees and performance. Better to lose a checkpoint than stall the simulation.

---

## Validation on Resume

When loading a checkpoint, Continuum validates:

### 1. World IR Hash

Checkpoint stores Blake3 hash of compiled world IR. On resume, it must match current world.

**Fails if**:
- Signals added/removed
- Operators changed
- Members added/removed
- Eras or strata modified

**Override**: Use `--force-resume` (dangerous)

### 2. Signal Schema

All checkpointed signals must exist in current world with same types.

**Fails if**:
- Signal renamed
- Signal type changed (Scalar → Vec3)
- Signal removed

### 3. Era Configuration

Era `dt` values and stratum cadences must match.

**Fails if**:
- Era `dt` changed
- Stratum cadence changed

---

## Checkpoint Inspection

To inspect a checkpoint without loading it:

```bash
continuum checkpoint inspect ./checkpoints/.../checkpoint_0001000.ckpt
```

**Output**:
```
Checkpoint: checkpoint_0001000.ckpt
Format version: 1
Tick: 1000
Simulation time: 1000.0s
Era: main
World IR hash: a1b2c3d4e5f6...
Seed: 42
Signals: 25
Entities: 3 types, 5,432 instances
Member signals: 45 signals, 16,296 total instances
File size: 4.2 MB (compressed), 28.5 MB (uncompressed)
Compression ratio: 6.8x
Created: 2026-01-16 09:45:30
Age: 2 hours
```

**Note**: `continuum checkpoint` tool not yet implemented (future work).

---

## Determinism Guarantees

**Property**: Same world + checkpoint → identical continuation

**Test**:
```bash
# Run 2000 ticks straight
continuum run ./world --steps 2000 --seed 42

# Run 1000, checkpoint, resume 1000
continuum run ./world --steps 1000 --seed 42 \
  --checkpoint-dir ./test --checkpoint-stride 1000
continuum run ./world --steps 1000 --seed 42 \
  --resume --checkpoint-dir ./test
```

**Result**: Final state must be **bitwise identical**.

**Why it matters**: Checkpoints don't introduce non-determinism.

---

## Troubleshooting

### "Checkpoint not found"

```
Error: Checkpoint not found at ./checkpoints
```

**Cause**: `--checkpoint-dir` directory doesn't exist or has no checkpoints  
**Fix**: Check path, ensure checkpoints were created

---

### "World IR has changed"

```
Error: Cannot resume - world IR has changed
  Checkpoint world hash: a1b2c3d4...
  Current world hash:    e5f6g7h8...
```

**Cause**: World DSL (*.cdsl) changed since checkpoint was created  
**Fix**: 
1. Use exact same world version, OR
2. Use `--force-resume` (may crash)

---

### "Checkpoint file corrupted"

```
Error: Checkpoint file corrupted (decompression failed)
```

**Cause**: File was truncated or damaged (partial write, disk error)  
**Fix**: Use an older checkpoint

---

### Checkpoint writes are slow

**Symptom**: Checkpoint warnings about full queue  
**Cause**: Disk I/O is slower than checkpoint generation rate

**Fixes**:
1. Increase `--checkpoint-stride` (fewer checkpoints)
2. Add `--checkpoint-interval` throttle
3. Use faster disk (SSD instead of HDD)
4. Reduce checkpoint size (fewer entities/signals)

---

### Resume continues from wrong tick

**Symptom**: `--resume` loads checkpoint from tick 5000 but you expected 10000

**Cause**: `latest` symlink points to older checkpoint

**Fix**: Manually specify checkpoint:
```bash
continuum run ./world --resume \
  --checkpoint-path ./checkpoints/.../checkpoint_0010000.ckpt
```

**Note**: `--checkpoint-path` not yet implemented (use symlink workaround).

---

## Performance Impact

### Checkpoint Overhead

| Operation | Time (typical) | Blocking? |
|-----------|---------------|-----------|
| State clone | ~1-10ms | Yes (minimal) |
| Queue push | <1μs | No |
| Disk write | ~50-500ms | No (background) |

**Total impact on tick time**: <10ms (negligible)

### Disk Usage

| World | Checkpoint Size (compressed) |
|-------|------------------------------|
| POC | ~1 KB |
| Terra (small) | ~500 KB - 2 MB |
| Terra (large) | ~5 MB - 20 MB |

**Estimate**: `checkpoint_size × keep_checkpoints = disk usage`

---

## Best Practices

### Checkpoint Strategy

**Development**:
- Frequent checkpoints: `--checkpoint-stride 100`
- Keep all: omit `--keep-checkpoints`

**Production**:
- Moderate checkpoints: `--checkpoint-stride 1000`
- Pruning: `--keep-checkpoints 10-20`
- Throttling: `--checkpoint-interval 1800` (30 min)

### Backup Strategy

Checkpoints are **not backups**. They're tied to world version.

**For backups**:
1. Version control world DSL (git)
2. Archive checkpoint dir with world hash tag
3. Store separate from live checkpoints

### Resume Testing

Before long runs, test resume works:

```bash
# Short run with checkpoint
continuum run ./world --steps 100 \
  --checkpoint-dir ./test --checkpoint-stride 50

# Resume
continuum run ./world --steps 50 \
  --resume --checkpoint-dir ./test

# Verify no errors
```

---

## See Also

- `docs/checkpoint-format.md` - Technical specification
- `docs/persistence.md` - Checkpoint vs Lens snapshots
- `tools/run.md` - Full CLI reference
