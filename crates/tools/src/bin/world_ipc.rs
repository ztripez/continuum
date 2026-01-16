//! IPC simulation server.
//!
//! Runs a world and listens for binary commands over a Unix socket.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use clap::Parser;
use tokio::fs;
use tokio::io::BufReader;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast};
use tokio::task::yield_now;
use tokio::time::{Duration, sleep};
use tracing::{debug, error, info, warn};

use continuum_compiler::ir::{RuntimeBuildOptions, build_runtime, compile};
use continuum_ir::CompiledWorld;
use continuum_lens::{FieldLens, FieldLensConfig, PlaybackClock};
use continuum_runtime::executor::Runtime;
use continuum_tools::ipc_protocol::{
    AssertionEvent, ChronicleEvent, ChroniclePollPayload, EntityInfo, EntityListPayload, EraInfo,
    EraListPayload, FieldHistoryPayload, FieldInfo, FieldLatestPayload, FieldListPayload,
    FieldQueryBatchPayload, FieldQueryPayload, ImpulseEmitPayload, ImpulseInfo, ImpulseListPayload,
    IpcCommand, IpcEvent, IpcFrame, IpcRequest, IpcResponse, IpcResponsePayload, JsonValue,
    PlaybackPayload, SignalInfo, SignalListPayload, StatusPayload, StratumInfo, StratumListPayload,
    TickEvent, WorldInfo, read_frame, write_frame,
};

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
    world: Arc<CompiledWorld>,
    runtime: Runtime,
    lens: FieldLens,
    playback: PlaybackClock,
    sim_time: f64,
    running: bool,
    tick_delay: Duration,
    events: broadcast::Sender<IpcEvent>,
    signals: Vec<String>,
    fields: Vec<String>,
    impulse_handlers: HashMap<String, usize>,
    impulses: Vec<ImpulseInfo>,
    impulse_seq: u64,
}

struct ClientState {
    chronicle_events: Arc<StdMutex<VecDeque<ChronicleEvent>>>,
}

#[tokio::main]
async fn main() {
    continuum_tools::init_logging();

    let cli = Cli::parse();

    debug!("Loading world from: {}", cli.world_dir.display());

    let compile_result = continuum_compiler::compile_from_dir_result(&cli.world_dir);

    if compile_result.has_errors() {
        for diag in &compile_result.diagnostics {
            error!("{}", compile_result.format_diagnostic(diag));
        }
        std::process::exit(1);
    }

    if !compile_result.diagnostics.is_empty() {
        for diag in &compile_result.diagnostics {
            warn!("{}", compile_result.format_diagnostic(diag));
        }
    }

    let world = compile_result.world.expect("no world despite no errors");

    debug!("Compiling to DAGs...");
    let compilation = match compile(&world) {
        Ok(compilation) => compilation,
        Err(err) => {
            error!("Compilation error: {}", err);
            std::process::exit(1);
        }
    };

    debug!("Building runtime...");
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

    let lens = match FieldLens::new(FieldLensConfig::default()) {
        Ok(lens) => lens,
        Err(err) => {
            error!("Failed to initialize lens: {err}");
            std::process::exit(1);
        }
    };
    let playback = PlaybackClock::new(0.0);

    let signals = world
        .signals()
        .keys()
        .map(|id| id.to_string())
        .collect::<Vec<_>>();

    let fields = world
        .fields()
        .keys()
        .map(|id| id.to_string())
        .collect::<Vec<_>>();

    let impulses = world
        .impulses()
        .iter()
        .map(|(id, impulse)| {
            let unit = impulse
                .payload_type
                .param_value(continuum_foundation::PrimitiveParamKind::Unit)
                .and_then(|p| match p {
                    continuum_ir::ValueTypeParamValue::Unit(u) => Some(u.clone()),
                    _ => None,
                });

            let range = impulse
                .payload_type
                .param_value(continuum_foundation::PrimitiveParamKind::Range)
                .and_then(|p| match p {
                    continuum_ir::ValueTypeParamValue::Range(r) => Some((r.min, r.max)),
                    _ => None,
                });

            ImpulseInfo {
                id: id.to_string(),
                doc: impulse.doc.clone(),
                title: impulse.title.clone(),
                symbol: impulse.symbol.clone(),
                payload_type: impulse.payload_type.primitive_id().name().to_string(),
                unit,
                range,
            }
        })
        .collect::<Vec<_>>();

    let impulse_handlers = report
        .impulse_indices
        .iter()
        .map(|(id, idx)| (id.to_string(), *idx))
        .collect();

    let (events, _rx) = broadcast::channel(1024);

    let state = Arc::new(Mutex::new(ServerState {
        world: Arc::new(world),
        runtime,
        lens,
        playback,
        sim_time: 0.0,
        running: false,
        tick_delay: Duration::from_millis(cli.tick_delay_ms),
        events,
        signals,
        fields,
        impulse_handlers,
        impulses,
        impulse_seq: 0,
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

    debug!("Listening on socket {}", cli.socket.display());

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
    let mut reader = BufReader::new(reader);
    let writer = Arc::new(Mutex::new(writer));

    let client_state = Arc::new(ClientState {
        chronicle_events: Arc::new(StdMutex::new(VecDeque::new())),
    });

    // Spawn event listener for this client
    let mut event_rx = {
        let state = state.lock().await;
        state.events.subscribe()
    };

    let event_writer = Arc::clone(&writer);
    let chronicle_events = Arc::clone(&client_state.chronicle_events);

    tokio::spawn(async move {
        while let Ok(event) = event_rx.recv().await {
            // Buffer chronicle events
            if let IpcEvent::Chronicle(ref chronicle_event) = event {
                let mut buffer = chronicle_events
                    .lock()
                    .expect("chronicle event lock poisoned");
                buffer.push_back(chronicle_event.clone());
            }

            // Forward all events (Tick, Chronicle) to client
            let frame = IpcFrame::Event(event);
            let mut writer = event_writer.lock().await;
            if write_frame(&mut *writer, &frame).await.is_err() {
                break;
            }
        }
    });

    // Read loop
    loop {
        let request: IpcRequest = match read_frame(&mut reader).await {
            Ok(req) => req,
            Err(err) => {
                // Only warn if it's not a normal EOF
                if !err.to_string().contains("read frame length") {
                    warn!("read frame error: {err}");
                }
                break;
            }
        };

        let request_id = request.id;
        let response =
            match handle_command(request, Arc::clone(&state), Arc::clone(&client_state)).await {
                Ok(resp) => resp,
                Err(err) => IpcResponse {
                    id: request_id,
                    ok: false,
                    payload: None,
                    error: Some(err.to_string()),
                },
            };

        let frame = IpcFrame::Response(response);
        let mut writer_guard = writer.lock().await;
        if let Err(err) = write_frame(&mut *writer_guard, &frame).await {
            warn!("write frame error: {err}");
            break;
        }
    }

    Ok(())
}

async fn handle_command(
    request: IpcRequest,
    state: Arc<Mutex<ServerState>>,
    client: Arc<ClientState>,
) -> anyhow::Result<IpcResponse> {
    let id = request.id;
    match request.command {
        IpcCommand::Status => {
            let state = state.lock().await;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::Status(status_payload(&state))),
                error: None,
            })
        }
        IpcCommand::Step { count } => {
            let mut state = state.lock().await;
            if state.running {
                anyhow::bail!("running (stop first)");
            }
            for _ in 0..count {
                execute_tick(&mut state)?;
            }
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::Status(status_payload(&state))),
                error: None,
            })
        }
        IpcCommand::Run { count } => {
            let mut state_guard = state.lock().await;
            if let Some(count) = count {
                if state_guard.running {
                    anyhow::bail!("running (stop first)");
                }
                for _ in 0..count {
                    execute_tick(&mut state_guard)?;
                }
                Ok(IpcResponse {
                    id,
                    ok: true,
                    payload: Some(IpcResponsePayload::Status(status_payload(&state_guard))),
                    error: None,
                })
            } else {
                if state_guard.running {
                    anyhow::bail!("already running");
                }
                state_guard.running = true;
                let tick_delay = state_guard.tick_delay;
                let state_clone = Arc::clone(&state);
                tokio::spawn(async move {
                    run_loop(state_clone, tick_delay).await;
                });
                Ok(IpcResponse {
                    id,
                    ok: true,
                    payload: Some(IpcResponsePayload::Status(status_payload(&state_guard))),
                    error: None,
                })
            }
        }
        IpcCommand::Stop => {
            let mut state = state.lock().await;
            state.running = false;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::Status(status_payload(&state))),
                error: None,
            })
        }
        IpcCommand::WorldInfo => {
            let state = state.lock().await;

            let strata = state
                .world
                .strata()
                .values()
                .map(|s| StratumInfo {
                    id: s.id.to_string(),
                    doc: s.doc.clone(),
                    title: s.title.clone(),
                    symbol: s.symbol.clone(),
                    default_stride: s.default_stride,
                })
                .collect();

            let eras = state
                .world
                .eras()
                .values()
                .map(|e| EraInfo {
                    id: e.id.to_string(),
                    doc: e.doc.clone(),
                    title: e.title.clone(),
                    is_initial: e.is_initial,
                    is_terminal: e.is_terminal,
                    dt_seconds: e.dt_seconds,
                })
                .collect();

            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::WorldInfo(WorldInfo { strata, eras })),
                error: None,
            })
        }
        IpcCommand::SignalList => {
            let state = state.lock().await;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::SignalList(SignalListPayload {
                    signals: state.signals.clone(),
                })),
                error: None,
            })
        }
        IpcCommand::SignalDescribe { signal_id } => {
            let state = state.lock().await;
            let sid = continuum_foundation::SignalId::from(signal_id.as_str());

            // Get signal from compiled world
            let signals = state.world.signals();
            let signal = signals
                .get(&sid)
                .ok_or_else(|| anyhow::anyhow!("signal '{}' not found", signal_id))?;

            // Extract unit and range from value_type
            let unit = signal
                .value_type
                .param_value(continuum_foundation::PrimitiveParamKind::Unit)
                .and_then(|p| match p {
                    continuum_ir::ValueTypeParamValue::Unit(u) => Some(u.clone()),
                    _ => None,
                });

            let range = signal
                .value_type
                .param_value(continuum_foundation::PrimitiveParamKind::Range)
                .and_then(|p| match p {
                    continuum_ir::ValueTypeParamValue::Range(r) => Some((r.min, r.max)),
                    _ => None,
                });

            let signal_info = SignalInfo {
                id: signal_id.clone(),
                doc: signal.doc.clone(),
                title: signal.title.clone(),
                symbol: signal.symbol.clone(),
                value_type: signal.value_type.primitive_id().name().to_string(),
                unit,
                range,
                stratum: signal.stratum.to_string(),
            };

            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::SignalDescribe(signal_info)),
                error: None,
            })
        }
        IpcCommand::FieldList => {
            let state = state.lock().await;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldList(FieldListPayload {
                    fields: state.fields.clone(),
                })),
                error: None,
            })
        }
        IpcCommand::FieldDescribe { field_id } => {
            let state = state.lock().await;
            let fid = continuum_foundation::FieldId::from(field_id.as_str());

            // Get field from compiled world
            let fields = state.world.fields();
            let field = fields
                .get(&fid)
                .ok_or_else(|| anyhow::anyhow!("field '{}' not found", field_id))?;

            // Extract unit and range from value_type
            let unit = field
                .value_type
                .param_value(continuum_foundation::PrimitiveParamKind::Unit)
                .and_then(|p| match p {
                    continuum_ir::ValueTypeParamValue::Unit(u) => Some(u.clone()),
                    _ => None,
                });

            let range = field
                .value_type
                .param_value(continuum_foundation::PrimitiveParamKind::Range)
                .and_then(|p| match p {
                    continuum_ir::ValueTypeParamValue::Range(r) => Some((r.min, r.max)),
                    _ => None,
                });

            let field_info = FieldInfo {
                id: field_id.clone(),
                doc: field.doc.clone(),
                title: field.title.clone(),
                symbol: field.symbol.clone(),
                value_type: field.value_type.primitive_id().name().to_string(),
                unit,
                range,
                topology: format!("{:?}", field.topology),
                stratum: field.stratum.to_string(),
            };

            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldDescribe(field_info)),
                error: None,
            })
        }
        IpcCommand::FieldHistory { field_id } => {
            let state = state.lock().await;
            let fid = continuum_foundation::FieldId::from(field_id.as_str());
            let ticks = state
                .lens
                .history_ticks(&fid)
                .ok_or_else(|| anyhow::anyhow!("field '{}' has no history", field_id))?;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldHistory(FieldHistoryPayload {
                    field_id,
                    ticks,
                })),
                error: None,
            })
        }
        IpcCommand::FieldQuery {
            field_id,
            position,
            time,
        } => {
            let mut state = state.lock().await;
            let t = time.unwrap_or_else(|| state.playback.current_time());
            let value = query_field_value(&mut state, &field_id, position, t)?;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldQuery(FieldQueryPayload {
                    field_id,
                    value,
                })),
                error: None,
            })
        }
        IpcCommand::FieldQueryBatch {
            field_id,
            positions,
            tick,
        } => {
            let mut state = state.lock().await;
            let values = query_field_batch(&mut state, &field_id, positions, tick)?;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldQueryBatch(
                    FieldQueryBatchPayload { field_id, values },
                )),
                error: None,
            })
        }
        IpcCommand::FieldLatest { field_id, position } => {
            let mut state = state.lock().await;
            let (tick, value) = query_field_latest(&mut state, &field_id, position)?;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldLatest(FieldLatestPayload {
                    field_id,
                    tick,
                    value,
                })),
                error: None,
            })
        }
        IpcCommand::FieldTile {
            field_id,
            tile: _,
            tick,
            positions,
        } => {
            let mut state = state.lock().await;
            let values = query_field_batch(&mut state, &field_id, positions, Some(tick))?;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldQueryBatch(
                    FieldQueryBatchPayload { field_id, values },
                )),
                error: None,
            })
        }
        IpcCommand::PlaybackSet { lag_ticks, speed } => {
            let mut state = state.lock().await;
            state.playback = PlaybackClock::new(lag_ticks);
            state.playback.set_speed(speed);
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::Playback(PlaybackPayload {
                    time: state.playback.current_time(),
                })),
                error: None,
            })
        }
        IpcCommand::PlaybackSeek { time } => {
            let mut state = state.lock().await;
            state.playback.seek(time);
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::Playback(PlaybackPayload {
                    time: state.playback.current_time(),
                })),
                error: None,
            })
        }
        IpcCommand::PlaybackQuery { field_id, position } => {
            let mut state = state.lock().await;
            let time = state.playback.current_time();
            let value = query_field_value(&mut state, &field_id, position, time)?;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::FieldQuery(FieldQueryPayload {
                    field_id,
                    value,
                })),
                error: None,
            })
        }
        IpcCommand::ChroniclePoll => {
            let mut buffer = client
                .chronicle_events
                .lock()
                .expect("chronicle event lock poisoned");
            let events: Vec<_> = buffer.drain(..).collect();
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::ChroniclePoll(ChroniclePollPayload {
                    events,
                })),
                error: None,
            })
        }
        IpcCommand::ImpulseList => {
            let state = state.lock().await;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::ImpulseList(ImpulseListPayload {
                    impulses: state.impulses.clone(),
                })),
                error: None,
            })
        }
        IpcCommand::ImpulseEmit {
            impulse_id,
            payload,
        } => {
            let mut state = state.lock().await;
            let handler_idx = state
                .impulse_handlers
                .get(&impulse_id)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("unknown impulse '{}'", impulse_id))?;
            let seq = state.impulse_seq;
            state.impulse_seq += 1;
            let applied_tick = state.runtime.tick() + 1;
            state.runtime.inject_impulse(handler_idx, payload);
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::ImpulseEmit(ImpulseEmitPayload {
                    seq,
                    applied_tick,
                })),
                error: None,
            })
        }
        IpcCommand::StratumList => {
            let state = state.lock().await;
            let strata = state
                .world
                .strata()
                .keys()
                .map(|id| id.to_string())
                .collect();
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::StratumList(StratumListPayload {
                    strata,
                })),
                error: None,
            })
        }
        IpcCommand::StratumDescribe { stratum_id } => {
            let state = state.lock().await;
            let sid = continuum_foundation::StratumId::from(stratum_id.as_str());
            let strata = state.world.strata();
            let stratum = strata
                .get(&sid)
                .ok_or_else(|| anyhow::anyhow!("stratum '{}' not found", stratum_id))?;

            let info = StratumInfo {
                id: stratum_id,
                doc: stratum.doc.clone(),
                title: stratum.title.clone(),
                symbol: stratum.symbol.clone(),
                default_stride: stratum.default_stride,
            };

            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::StratumDescribe(info)),
                error: None,
            })
        }
        IpcCommand::EraList => {
            let state = state.lock().await;
            let eras = state.world.eras().keys().map(|id| id.to_string()).collect();
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::EraList(EraListPayload { eras })),
                error: None,
            })
        }
        IpcCommand::EraDescribe { era_id } => {
            let state = state.lock().await;
            let eid = continuum_foundation::EraId::from(era_id.as_str());
            let eras = state.world.eras();
            let era = eras
                .get(&eid)
                .ok_or_else(|| anyhow::anyhow!("era '{}' not found", era_id))?;

            let info = EraInfo {
                id: era_id,
                doc: era.doc.clone(),
                title: era.title.clone(),
                is_initial: era.is_initial,
                is_terminal: era.is_terminal,
                dt_seconds: era.dt_seconds,
            };

            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::EraDescribe(info)),
                error: None,
            })
        }
        IpcCommand::EntityList => {
            let state = state.lock().await;
            let entities = state
                .world
                .entities()
                .keys()
                .map(|id| id.to_string())
                .collect();
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::EntityList(EntityListPayload {
                    entities,
                })),
                error: None,
            })
        }
        IpcCommand::EntityDescribe { entity_id } => {
            let state = state.lock().await;
            let eid = continuum_foundation::EntityId::from(entity_id.as_str());
            let entities = state.world.entities();
            let entity = entities
                .get(&eid)
                .ok_or_else(|| anyhow::anyhow!("entity '{}' not found", entity_id))?;

            let info = EntityInfo {
                id: entity_id,
                doc: entity.doc.clone(),
                count_bounds: entity.count_bounds,
            };

            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::EntityDescribe(info)),
                error: None,
            })
        }
        IpcCommand::AssertionList => {
            let state = state.lock().await;
            let assertions = state
                .runtime
                .assertion_checker()
                .assertions()
                .iter()
                .map(|a| {
                    use continuum_tools::ipc_protocol::AssertionInfo;
                    AssertionInfo {
                        signal_id: a.signal.to_string(),
                        severity: match a.severity {
                            continuum_runtime::executor::AssertionSeverity::Warn => "warn",
                            continuum_runtime::executor::AssertionSeverity::Error => "error",
                            continuum_runtime::executor::AssertionSeverity::Fatal => "fatal",
                        }
                        .to_string(),
                        message: a.message.clone(),
                    }
                })
                .collect();

            use continuum_tools::ipc_protocol::AssertionListPayload;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::AssertionList(AssertionListPayload {
                    assertions,
                })),
                error: None,
            })
        }
        IpcCommand::AssertionFailures { signal_id } => {
            let state = state.lock().await;
            let all_failures = state.runtime.assertion_checker().failures();
            let filtered_failures: Vec<_> = if let Some(ref sig_id) = signal_id {
                let target = continuum_runtime::types::SignalId::from(sig_id.as_str());
                all_failures.iter().filter(|f| f.signal == target).collect()
            } else {
                all_failures.iter().collect()
            };

            let failures = filtered_failures
                .iter()
                .map(|f| {
                    use continuum_tools::ipc_protocol::AssertionFailure;
                    AssertionFailure {
                        signal_id: f.signal.to_string(),
                        severity: match f.severity {
                            continuum_runtime::executor::AssertionSeverity::Warn => "warn",
                            continuum_runtime::executor::AssertionSeverity::Error => "error",
                            continuum_runtime::executor::AssertionSeverity::Fatal => "fatal",
                        }
                        .to_string(),
                        message: f.message.clone(),
                        tick: f.tick,
                        era: f.era.clone(),
                        sim_time: f.sim_time,
                    }
                })
                .collect();

            use continuum_tools::ipc_protocol::AssertionFailuresPayload;
            Ok(IpcResponse {
                id,
                ok: true,
                payload: Some(IpcResponsePayload::AssertionFailures(
                    AssertionFailuresPayload { failures },
                )),
                error: None,
            })
        }
    }
}

/// Helper to broadcast any assertion failures that have been recorded
fn broadcast_assertion_failures(state: &mut ServerState) {
    let assertion_failures = state.runtime.assertion_checker_mut().drain_failures();
    for failure in &assertion_failures {
        let event = AssertionEvent {
            signal_id: failure.signal.to_string(),
            severity: match failure.severity {
                continuum_runtime::executor::AssertionSeverity::Warn => "warn".to_string(),
                continuum_runtime::executor::AssertionSeverity::Error => "error".to_string(),
                continuum_runtime::executor::AssertionSeverity::Fatal => "fatal".to_string(),
            },
            message: failure.message.clone(),
            tick: failure.tick,
            era: failure.era.clone(),
            sim_time: failure.sim_time,
        };
        let _ = state.events.send(IpcEvent::Assertion(event));
    }
}

fn execute_tick(state: &mut ServerState) -> anyhow::Result<()> {
    // Run warmup if not complete
    if !state.runtime.is_warmup_complete() {
        info!("Executing warmup...");
        state.runtime.execute_warmup()?;
        info!("Warmup complete");
    }

    // Execute tick - may fail due to assertion errors
    let tick_result = state.runtime.execute_tick();

    // Always broadcast assertion failures, even if tick failed
    broadcast_assertion_failures(state);

    // Now handle the tick result
    let ctx = tick_result?;
    state.sim_time += ctx.dt.seconds();
    state.playback.advance(state.runtime.tick());

    // Ingest fields into Lens
    let fields = state.runtime.drain_fields();
    state.lens.record_many(state.runtime.tick(), fields);

    // Process chronicle events
    let events = state.runtime.drain_events();
    let mut chronicle_events = Vec::new();
    for event in events {
        let fields = event
            .fields
            .into_iter()
            .map(|(k, v)| (k, JsonValue::from_value(&v)))
            .collect();
        chronicle_events.push(ChronicleEvent {
            chronicle_id: event.chronicle_id,
            name: event.name,
            fields,
            tick: ctx.tick,
            era: ctx.era.to_string(),
            sim_time: state.sim_time,
        });
    }

    // Broadcast tick event
    let tick_event = TickEvent {
        tick: ctx.tick,
        era: ctx.era.to_string(),
        sim_time: state.sim_time,
        field_count: state.fields.len(),
        event_count: chronicle_events.len(),
    };
    let _ = state.events.send(IpcEvent::Tick(tick_event));

    // Broadcast chronicle events
    for event in chronicle_events {
        let _ = state.events.send(IpcEvent::Chronicle(event));
    }

    Ok(())
}

async fn run_loop(state: Arc<Mutex<ServerState>>, tick_delay: Duration) {
    loop {
        let mut state_guard = state.lock().await;
        if !state_guard.running {
            break;
        }

        if let Err(err) = execute_tick(&mut state_guard) {
            warn!("run loop error: {err}");
            state_guard.running = false;
            break;
        }
        drop(state_guard);

        if tick_delay > Duration::ZERO {
            sleep(tick_delay).await;
        } else {
            yield_now().await;
        }
    }
}

fn status_payload(state: &ServerState) -> StatusPayload {
    let ctx = state.runtime.tick_context();
    StatusPayload {
        tick: ctx.tick,
        era: ctx.era.to_string(),
        sim_time: state.sim_time,
        dt: ctx.dt.seconds(),
        phase: format!("{:?}", state.runtime.phase()),
        running: state.running,
        warmup_complete: state.runtime.is_warmup_complete(),
    }
}

fn query_field_value(
    state: &mut ServerState,
    field_id: &str,
    position: [f64; 3],
    time: f64,
) -> anyhow::Result<JsonValue> {
    let field_id = continuum_foundation::FieldId::from(field_id);
    let value = state.lens.query(&field_id, position, time)?;
    Ok(JsonValue::Scalar(value))
}

fn query_field_batch(
    state: &mut ServerState,
    field_id: &str,
    positions: Vec<[f64; 3]>,
    tick: Option<u64>,
) -> anyhow::Result<Vec<f64>> {
    let field_id = continuum_foundation::FieldId::from(field_id);
    let tick = tick.unwrap_or_else(|| state.runtime.tick());
    let values = state.lens.query_batch(&field_id, &positions, tick)?;
    Ok(values)
}

fn query_field_latest(
    state: &mut ServerState,
    field_id: &str,
    position: Option<[f64; 3]>,
) -> anyhow::Result<(u64, Option<JsonValue>)> {
    let field_id = continuum_foundation::FieldId::from(field_id);
    let tick = state.runtime.tick();
    let value = if let Some(pos) = position {
        let v = state.lens.query(&field_id, pos, tick as f64)?;
        Some(JsonValue::Scalar(v))
    } else {
        None
    };
    Ok((tick, value))
}
