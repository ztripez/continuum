//! IPC simulation server.
//!
//! Runs a world and listens for simple text commands over a Unix socket.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::OwnedWriteHalf;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast};
use tokio::task::yield_now;
use tokio::time::{Duration, sleep};
use tracing::{error, info, warn};

use continuum_compiler::ir::{RuntimeBuildOptions, build_runtime, compile};
use continuum_runtime::executor::Runtime;

#[derive(Parser, Debug)]
#[command(name = "world-ipc")]
struct Cli {
    /// World directory to load.
    #[arg(value_name = "WORLD_DIR")]
    world_dir: PathBuf,

    /// Unix socket path to listen on.
    #[arg(long)]
    socket: PathBuf,

    /// Override dt for all eras.
    #[arg(long)]
    dt: Option<f64>,

    /// Sleep duration (ms) between ticks when running.
    #[arg(long, default_value = "0")]
    tick_delay_ms: u64,
}

struct ServerState {
    runtime: Runtime,
    sim_time: f64,
    running: bool,
    tick_delay: Duration,
    events: broadcast::Sender<String>,
}

#[tokio::main]
async fn main() {
    continuum_tools::init_logging();

    let cli = Cli::parse();

    info!("Loading world from: {}", cli.world_dir.display());

    let compile_result = continuum_compiler::compile_from_dir_result(&cli.world_dir);

    if compile_result.has_errors() {
        error!("{}", compile_result.format_diagnostics().trim_end());
        std::process::exit(1);
    }

    if !compile_result.diagnostics.is_empty() {
        warn!("{}", compile_result.format_diagnostics().trim_end());
    }

    let world = compile_result.world.expect("no world despite no errors");

    info!("Compiling to DAGs...");
    let compilation = match compile(&world) {
        Ok(compilation) => compilation,
        Err(err) => {
            error!("Compilation error: {}", err);
            std::process::exit(1);
        }
    };

    info!("Building runtime...");
    let (runtime, report) = match build_runtime(
        &world,
        compilation,
        RuntimeBuildOptions {
            dt_override: cli.dt,
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            error!("Runtime build error: {}", err);
            std::process::exit(1);
        }
    };

    if report.resolver_count > 0 || report.aggregate_count > 0 {
        info!(
            "  Total: {} resolvers, {} aggregate resolvers",
            report.resolver_count, report.aggregate_count
        );
    }
    if report.assertion_count > 0 {
        info!("  Registered {} assertions", report.assertion_count);
    }
    if report.field_count > 0 {
        info!("  Registered {} field measures", report.field_count);
    }
    if report.skipped_fields > 0 {
        info!(
            "  Skipped {} fields with entity expressions (EntityExecutor not yet implemented)",
            report.skipped_fields
        );
    }

    let (events, _rx) = broadcast::channel(1024);

    let state = Arc::new(Mutex::new(ServerState {
        runtime,
        sim_time: 0.0,
        running: false,
        tick_delay: Duration::from_millis(cli.tick_delay_ms),
        events,
    }));

    if cli.socket.exists() {
        if let Err(err) = fs::remove_file(&cli.socket).await {
            error!("Failed to remove socket {}: {err}", cli.socket.display());
            std::process::exit(1);
        }
    }

    let listener = match UnixListener::bind(&cli.socket) {
        Ok(listener) => listener,
        Err(err) => {
            error!("Failed to bind socket {}: {err}", cli.socket.display());
            std::process::exit(1);
        }
    };

    info!("Listening on socket {}", cli.socket.display());

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    if let Err(err) = handle_client(stream, state).await {
                        warn!("client error: {err}");
                    }
                });
            }
            Err(err) => {
                error!("Accept error: {err}");
                break;
            }
        }
    }
}

async fn handle_client(stream: UnixStream, state: Arc<Mutex<ServerState>>) -> anyhow::Result<()> {
    let (reader, writer) = stream.into_split();
    let writer = Arc::new(Mutex::new(writer));
    let mut lines = BufReader::new(reader).lines();

    let mut event_rx = {
        let state = state.lock().await;
        state.events.subscribe()
    };

    let event_writer = Arc::clone(&writer);
    tokio::spawn(async move {
        while let Ok(message) = event_rx.recv().await {
            let mut writer = event_writer.lock().await;
            if writer.write_all(message.as_bytes()).await.is_err() {
                break;
            }
        }
    });

    write_line(&writer, "ok world-ipc ready\n").await?;

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let command = parts.next().unwrap_or("");

        match command {
            "help" => {
                write_line(
                    &writer,
                    "ok commands: status | step [n] | run [n] | stop | quit\n",
                )
                .await?;
            }
            "status" => {
                let state = state.lock().await;
                let ctx = state.runtime.tick_context();
                let response = format!(
                    "ok tick={} era={} sim_time={:.6} running={}\n",
                    ctx.tick, ctx.era, state.sim_time, state.running
                );
                write_line(&writer, &response).await?;
            }
            "step" => {
                let count = parts
                    .next()
                    .map(|value| value.parse::<u64>())
                    .transpose()
                    .map_err(|_| anyhow::anyhow!("invalid step count"))?
                    .unwrap_or(1);

                let mut state = state.lock().await;
                if state.running {
                    write_line(&writer, "err running (stop first)\n").await?;

                    continue;
                }
                for _ in 0..count {
                    let ctx = state.runtime.execute_tick()?;
                    state.sim_time += ctx.dt.seconds();
                }
                let ctx = state.runtime.tick_context();
                let response = format!(
                    "ok tick={} era={} sim_time={:.6}\n",
                    ctx.tick, ctx.era, state.sim_time
                );
                write_line(&writer, &response).await?;
            }
            "run" => {
                let count = parts
                    .next()
                    .map(|value| value.parse::<u64>())
                    .transpose()
                    .map_err(|_| anyhow::anyhow!("invalid step count"))?;

                if let Some(count) = count {
                    let mut state = state.lock().await;
                    if state.running {
                        write_line(&writer, "err running (stop first)\n").await?;

                        continue;
                    }
                    for _ in 0..count {
                        let ctx = state.runtime.execute_tick()?;
                        state.sim_time += ctx.dt.seconds();
                    }
                    let ctx = state.runtime.tick_context();
                    let response = format!(
                        "ok tick={} era={} sim_time={:.6}\n",
                        ctx.tick, ctx.era, state.sim_time
                    );
                    write_line(&writer, &response).await?;
                } else {
                    let mut state_guard = state.lock().await;
                    if state_guard.running {
                        write_line(&writer, "err already running\n").await?;
                        continue;
                    }
                    state_guard.running = true;
                    let tick_delay = state_guard.tick_delay;
                    let state_clone = Arc::clone(&state);
                    tokio::spawn(async move {
                        run_loop(state_clone, tick_delay).await;
                    });
                    write_line(&writer, "ok running\n").await?;
                }
            }
            "stop" => {
                let mut state = state.lock().await;
                if state.running {
                    state.running = false;
                    write_line(&writer, "ok stopped\n").await?;
                } else {
                    write_line(&writer, "ok not running\n").await?;
                }
            }
            "quit" | "exit" => {
                write_line(&writer, "ok bye\n").await?;
                break;
            }
            _ => {
                write_line(&writer, "err unknown command (try 'help')\n").await?;
            }
        }
    }

    Ok(())
}

async fn run_loop(state: Arc<Mutex<ServerState>>, tick_delay: Duration) {
    loop {
        let mut state_guard = state.lock().await;
        if !state_guard.running {
            break;
        }

        let tick_result = state_guard.runtime.execute_tick();
        match tick_result {
            Ok(ctx) => {
                state_guard.sim_time += ctx.dt.seconds();
                let message = format!(
                    "tick {} era={} sim_time={:.6}\n",
                    ctx.tick, ctx.era, state_guard.sim_time
                );
                let _ = state_guard.events.send(message);
            }
            Err(err) => {
                warn!("run loop error: {err}");
                state_guard.running = false;
                break;
            }
        }
        drop(state_guard);

        if tick_delay > Duration::ZERO {
            sleep(tick_delay).await;
        } else {
            yield_now().await;
        }
    }
}

async fn write_line(writer: &Arc<Mutex<OwnedWriteHalf>>, line: &str) -> anyhow::Result<()> {
    let mut writer = writer.lock().await;
    writer.write_all(line.as_bytes()).await?;
    Ok(())
}
