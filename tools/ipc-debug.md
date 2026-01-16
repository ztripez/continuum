# ipc-debug - Combined IPC Server + Web UI

A convenience tool that starts both the IPC server and web proxy in a single process for easy debugging and inspection of Continuum worlds.

## Usage

```bash
cargo run --bin ipc-debug -- <world-path>
```

### Example

```bash
cargo run --bin ipc-debug -- examples/terra
```

Then open your browser to: **http://localhost:8080**

## What It Does

The `ipc-debug` tool:

1. **Starts the IPC server** (`world-ipc`) with your world
2. **Starts the web proxy** (`world-ipc-web`) connected to the IPC server
3. **Monitors both processes** and keeps them running
4. **Handles Ctrl+C** gracefully, stopping both services cleanly
5. **Cleans up** the Unix socket on shutdown

## Options

```
USAGE:
    ipc-debug [OPTIONS] <WORLD>

ARGS:
    <WORLD>    Path to the world directory

OPTIONS:
    --socket <SOCKET>        Unix socket path [default: /tmp/continuum-debug.sock]
    --bind <BIND>            Web server bind address [default: 0.0.0.0:8080]
    --static-dir <DIR>       Static assets directory [default: crates/tools/assets/ipc-web]
    -h, --help               Print help information
```

## Features

- **Single command** - No need to manage two separate processes
- **Process monitoring** - Automatically detects if either service crashes
- **Clean shutdown** - Ctrl+C stops both services gracefully
- **Auto cleanup** - Removes Unix socket on exit
- **Clear logging** - Shows startup progress and ready status

## Requirements

The tool expects to find compiled binaries in the same directory:
- `world-ipc`
- `world-ipc-web`

Build them first:

```bash
cargo build --bin world-ipc --release
cargo build --bin world-ipc-web --release
cargo build --bin ipc-debug --release
```

Or build everything at once:

```bash
cargo build --release
```

## Output Example

```
Starting Continuum IPC Debug Server
World: examples/terra
Socket: /tmp/continuum-debug.sock
Web UI: http://0.0.0.0:8080

Press Ctrl+C to stop

Starting IPC server...
IPC server started successfully
Starting web proxy...

✓ IPC Debug Server Ready
✓ Open your browser to: http://0.0.0.0:8080
```

## Troubleshooting

### "binary not found" error

Make sure you've built the required binaries:

```bash
cargo build --bin world-ipc --bin world-ipc-web --release
```

### Port already in use

If port 8080 is already in use, specify a different port:

```bash
cargo run --bin ipc-debug -- --bind 0.0.0.0:8081 examples/terra
```

### Socket already exists

The tool automatically cleans up the socket on startup. If you see issues, manually remove it:

```bash
rm /tmp/continuum-debug.sock
```

## See Also

- [world-ipc](./world-ipc.md) - The underlying IPC server
- [Web UI Guide](/tmp/WEB_UI_STATUS.md) - Web inspector features and usage
