# Inspector Process Management Testing Guide

## Overview

Phase 5 implementation allows the Inspector to spawn and kill `continuum-run` processes, enabling dynamic world loading without manual restarts.

## Features Implemented

1. **Process Spawning** - Inspector spawns `continuum-run` as child process
2. **Process Killing** - Clean shutdown with socket cleanup
3. **REST API Endpoints** - `/api/sim/load`, `/api/sim/stop`, `/api/sim/restart`, `/api/sim/status`
4. **Graceful Shutdown** - SIGTERM/SIGINT handler kills child process
5. **Auto-launch** - If `--world` provided at startup, automatically spawns simulation

## API Endpoints

### `POST /api/sim/load`
Load a new world, killing any existing simulation.

**Request:**
```json
{
  "world_path": "examples/terra",
  "scenario": "default"  // optional
}
```

**Response:**
```json
{
  "success": true,
  "message": "Loaded world: examples/terra"
}
```

### `POST /api/sim/stop`
Stop the current simulation.

**Response:**
```json
{
  "success": true,
  "message": "Simulation stopped"
}
```

### `POST /api/sim/restart`
Restart the current world (must have previously loaded one).

**Response:**
```json
{
  "success": true,
  "message": "Simulation restarted"
}
```

### `GET /api/sim/status`
Check if simulation is running and which world is loaded.

**Response:**
```json
{
  "running": true,
  "world_path": "examples/terra"
}
```

## Manual Testing

### Test 1: Auto-launch world at startup

```bash
cargo run --bin continuum_inspector -- examples/terra --bind 127.0.0.1:8080 --socket /tmp/test.sock
```

**Expected:**
- Inspector starts on http://127.0.0.1:8080
- Terra world auto-spawns
- Socket appears at `/tmp/test.sock`
- WebSocket connects successfully

### Test 2: Load world via API

```bash
# Terminal 1: Start inspector without world
cargo run --bin continuum_inspector -- --bind 127.0.0.1:8080 --socket /tmp/test.sock

# Terminal 2: Load world via API
curl -X POST http://localhost:8080/api/sim/load \
  -H "Content-Type: application/json" \
  -d '{"world_path": "examples/terra"}'
```

**Expected:**
- API returns `{"success": true, ...}`
- Socket file `/tmp/test.sock` appears
- WebSocket connects
- Simulation starts ticking

### Test 3: Load world via UI

1. Start inspector: `cargo run --bin continuum_inspector`
2. Open http://localhost:8080
3. Click "Sim Controls" button
4. Enter world path: `examples/terra`
5. Click "Load World"

**Expected:**
- Success message appears
- Simulation starts
- Tick count increases

### Test 4: Stop simulation

```bash
curl -X POST http://localhost:8080/api/sim/stop
```

**Expected:**
- Simulation process terminates
- Socket file removed
- WebSocket disconnects
- API returns success

### Test 5: Restart simulation

```bash
# After loading a world
curl -X POST http://localhost:8080/api/sim/restart
```

**Expected:**
- Old process killed
- New process spawned
- Socket recreated
- WebSocket reconnects
- Tick count resets to 0

### Test 6: Load different world

```bash
# Load terra
curl -X POST http://localhost:8080/api/sim/load \
  -H "Content-Type: application/json" \
  -d '{"world_path": "examples/terra"}'

# Switch to poc
curl -X POST http://localhost:8080/api/sim/load \
  -H "Content-Type: application/json" \
  -d '{"world_path": "examples/poc"}'
```

**Expected:**
- Terra process killed
- POC process spawned
- No socket leaks
- WebSocket reconnects

### Test 7: Graceful shutdown

```bash
# Terminal 1: Start with world
cargo run --bin continuum_inspector -- examples/terra --socket /tmp/test.sock

# Terminal 2: Send SIGTERM
pkill -SIGTERM continuum_inspector

# Or use Ctrl+C in Terminal 1
```

**Expected:**
- Inspector receives signal
- Child `continuum-run` process killed
- Socket file removed
- Clean exit

### Test 8: Socket timeout handling

```bash
# Manually create a bad continuum-run that doesn't create socket
cat > /tmp/fake-continuum-run << 'EOF'
#!/bin/bash
sleep 100
EOF
chmod +x /tmp/fake-continuum-run

# Temporarily replace continuum-run in PATH
export PATH="/tmp:$PATH"

# Try to load - should timeout after 5 seconds
curl -X POST http://localhost:8080/api/sim/load \
  -H "Content-Type: application/json" \
  -d '{"world_path": "examples/terra"}'
```

**Expected:**
- Request times out after ~5 seconds
- Error message: "Timeout waiting for socket to be created"
- Process is killed
- No orphaned processes

## Common Issues

### Socket already exists
**Symptom:** "Failed to bind socket: Address already in use"

**Fix:**
```bash
rm /tmp/continuum-inspector.sock
# Or use different socket path
cargo run --bin continuum_inspector -- --socket /tmp/my-socket.sock
```

### Process not found
**Symptom:** "Failed to spawn continuum-run: No such file"

**Fix:** Ensure `continuum-run` is built and in PATH:
```bash
cargo build --bin continuum-run
export PATH="$PWD/target/debug:$PATH"
```

### Orphaned processes
**Symptom:** Multiple `continuum-run` processes running

**Fix:**
```bash
pkill continuum-run
rm /tmp/*.sock
```

## Verification Checklist

- [ ] Inspector starts successfully
- [ ] World auto-launches if provided at startup
- [ ] Load world via API works
- [ ] Load world via UI works
- [ ] Stop simulation cleans up process and socket
- [ ] Restart simulation works
- [ ] Loading different world kills previous one
- [ ] Graceful shutdown (Ctrl+C) kills child process
- [ ] Socket timeout prevents hanging
- [ ] No orphaned processes after shutdown
- [ ] Multiple load/stop/restart cycles work without issues

## Next Steps (Phase 6)

Frontend updates needed:
1. Update `TickEvent` type to include `execution_state`, `tick_rate`, `last_error`
2. Wire up run/pause/resume handlers in Header
3. Add tick rate slider control
4. Show error state banner with resume button
5. Handle execution state changes (running/paused/error)
