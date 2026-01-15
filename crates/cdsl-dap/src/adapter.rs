use continuum_compiler::ir::{build_runtime, compile, CompiledWorld, RuntimeBuildOptions};
use continuum_runtime::types::WarmupConfig;
use continuum_runtime::Runtime;
use dap::events::{Event, StoppedEventBody};
use dap::prelude::*;
use dap::types::{Capabilities, Message, Scope, StackFrame, StoppedEventReason, Thread, Variable};

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info};

pub struct ContinuumDebugAdapter {
    session: Arc<Mutex<Option<DebugSession>>>,
    event_tx: Arc<Mutex<Option<mpsc::Sender<Event>>>>,
    should_pause: Arc<AtomicBool>,
}

pub struct DebugSession {
    pub world: CompiledWorld,
    pub runtime: Runtime,
    pub sources: HashMap<PathBuf, String>,
    pub breakpoints: HashMap<PathBuf, HashSet<usize>>, // File -> Line numbers
    pub status: SessionStatus,
    pub current_halt_signal: Option<continuum_foundation::SignalId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Starting,
    Running,
    Paused,
}

fn offset_to_line_col(text: &str, offset: usize) -> (u32, u32) {
    let mut line = 0;
    let mut col = 0;
    let mut current_byte = 0;
    for c in text.chars() {
        if current_byte >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
        current_byte += c.len_utf8();
    }
    (line, col)
}

impl ContinuumDebugAdapter {
    pub fn new(event_tx: mpsc::Sender<Event>) -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            event_tx: Arc::new(Mutex::new(Some(event_tx))),
            should_pause: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn set_event_sender(&self, tx: mpsc::Sender<Event>) {
        let mut event_tx = self.event_tx.lock().await;
        *event_tx = Some(tx);
    }

    async fn send_event(&self, event: Event) {
        let event_tx = self.event_tx.lock().await;
        if let Some(ref tx) = *event_tx {
            let _ = tx.send(event).await;
        }
    }

    fn clone_for_task(&self) -> Self {
        Self {
            session: self.session.clone(),
            event_tx: self.event_tx.clone(),
            should_pause: self.should_pause.clone(),
        }
    }

    pub async fn handle_request(&self, request: Request) -> Response {
        info!("Handling request: {:?}", request.command);
        let body = match request.command {
            Command::Initialize(_) => {
                // Send Initialized event after returning response
                self.send_event(Event::Initialized).await;
                ResponseBody::Initialize(Capabilities {
                    supports_configuration_done_request: Some(true),
                    supports_step_back: Some(false),
                    supports_terminate_request: Some(true),
                    ..Default::default()
                })
            }
            Command::Launch(ref args) => {
                let world_path = if let Some(world_dir) = args
                    .additional_data
                    .as_ref()
                    .and_then(|data| data.get("world_dir"))
                    .and_then(|v| v.as_str())
                {
                    PathBuf::from(world_dir)
                } else {
                    PathBuf::from(".")
                };

                info!(world_path = ?world_path, "Launching simulation");

                let compile_result = continuum_compiler::compile_from_dir_result(&world_path);
                let diagnostics = compile_result.format_diagnostics();
                let sources = compile_result.sources.clone();
                let world = match compile_result.success() {
                    Ok(w) => w,
                    Err(_) => {
                        error!("Compilation failed: {}", diagnostics);
                        return self.make_error_response(
                            &request,
                            format!("Compilation failed: {}", diagnostics),
                        );
                    }
                };

                let compilation = match compile(&world) {
                    Ok(c) => c,
                    Err(e) => {
                        return self.make_error_response(
                            &request,
                            format!("DAG compilation failed: {}", e),
                        );
                    }
                };

                let (runtime, _report) =
                    match build_runtime(&world, compilation, RuntimeBuildOptions::default()) {
                        Ok(result) => result,
                        Err(error) => {
                            return self.make_error_response(
                                &request,
                                format!("Runtime build failed: {}", error),
                            );
                        }
                    };

                let mut session = self.session.lock().await;
                *session = Some(DebugSession {
                    world,
                    runtime,
                    sources,
                    breakpoints: HashMap::new(),
                    status: SessionStatus::Paused,
                    current_halt_signal: None,
                });

                self.send_event(Event::Stopped(StoppedEventBody {
                    reason: StoppedEventReason::String("entry".to_string()),
                    thread_id: Some(1),
                    all_threads_stopped: Some(true),
                    text: None,
                    description: None,
                    preserve_focus_hint: None,
                    hit_breakpoint_ids: None,
                }))
                .await;

                ResponseBody::Launch
            }
            Command::SetBreakpoints(ref args) => {
                let mut verified_breakpoints = vec![];
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    if let Some(ref source) = args.source.path {
                        let path = PathBuf::from(source);
                        let requested_lines: Vec<usize> = args
                            .breakpoints
                            .as_ref()
                            .map(|bs| bs.iter().map(|b| b.line as usize).collect())
                            .unwrap_or_default();

                        let lines: HashSet<usize> = requested_lines.iter().copied().collect();
                        session.breakpoints.insert(path.clone(), lines.clone());

                        // Update runtime breakpoints and track which ones were set
                        session.runtime.clear_breakpoints();
                        let mut set_breakpoints: HashSet<usize> = HashSet::new();

                        for (bp_path, bp_lines) in &session.breakpoints {
                            for (id, node) in session.world.nodes.iter() {
                                if let Some(ref node_file) = node.file {
                                    if node_file == bp_path {
                                        if let Some(source) = session.sources.get(node_file) {
                                            let (line, _) =
                                                offset_to_line_col(source, node.span.start);
                                            let line_1based = line as usize + 1;
                                            if bp_lines.contains(&line_1based) {
                                                let signal_id =
                                                    continuum_foundation::SignalId::from(
                                                        id.to_string(),
                                                    );
                                                session.runtime.add_breakpoint(signal_id);
                                                set_breakpoints.insert(line_1based);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Create breakpoint responses for each requested line
                        for line in requested_lines {
                            let verified = set_breakpoints.contains(&line);
                            verified_breakpoints.push(dap::types::Breakpoint {
                                id: Some(line as i64),
                                verified,
                                message: if !verified {
                                    Some("No signal found at this line".to_string())
                                } else {
                                    None
                                },
                                source: Some(args.source.clone()),
                                line: Some(line as i64),
                                column: None,
                                end_line: None,
                                end_column: None,
                                instruction_reference: None,
                                offset: None,
                            });
                        }
                    }
                }
                ResponseBody::SetBreakpoints(dap::responses::SetBreakpointsResponse {
                    breakpoints: verified_breakpoints,
                })
            }
            Command::ConfigurationDone => ResponseBody::ConfigurationDone,
            Command::Threads => ResponseBody::Threads(dap::responses::ThreadsResponse {
                threads: vec![Thread {
                    id: 1,
                    name: "Main Simulation".to_string(),
                }],
            }),
            Command::StackTrace(_) => {
                let session_opt = self.session.lock().await;
                let frames = if let Some(ref session) = *session_opt {
                    let mut frames = vec![];

                    // If we are at a breakpoint, add a frame for the signal
                    if let Some(ref signal_id) = session.current_halt_signal {
                        let path = continuum_foundation::Path::from(signal_id.to_string());
                        if let Some(node) = session.world.nodes.get(&path) {
                            let mut frame = StackFrame {
                                id: 2,
                                name: format!("Signal: {}", signal_id),
                                line: 0,
                                column: 0,
                                ..Default::default()
                            };

                            if let Some(ref file) = node.file {
                                frame.source = Some(dap::types::Source {
                                    name: Some(
                                        file.file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .to_string(),
                                    ),
                                    path: Some(file.to_string_lossy().to_string()),
                                    ..Default::default()
                                });

                                if let Some(source) = session.sources.get(file) {
                                    let (line, col) = offset_to_line_col(source, node.span.start);
                                    frame.line = line as i64 + 1;
                                    frame.column = col as i64 + 1;
                                }
                            }
                            frames.push(frame);
                        }
                    }

                    frames.push(StackFrame {
                        id: 1,
                        name: format!(
                            "Tick {} - {:?}",
                            session.runtime.tick(),
                            session.runtime.phase()
                        ),
                        source: None,
                        line: 0,
                        column: 0,
                        ..Default::default()
                    });
                    frames
                } else {
                    vec![]
                };
                let num_frames = frames.len() as i64;
                ResponseBody::StackTrace(dap::responses::StackTraceResponse {
                    stack_frames: frames,
                    total_frames: Some(num_frames),
                })
            }
            Command::Scopes(_) => ResponseBody::Scopes(dap::responses::ScopesResponse {
                scopes: vec![
                    Scope {
                        name: "Signals".to_string(),
                        variables_reference: 1,
                        expensive: false,
                        ..Default::default()
                    },
                    Scope {
                        name: "Entities".to_string(),
                        variables_reference: 3,
                        expensive: true,
                        ..Default::default()
                    },
                    Scope {
                        name: "Configuration".to_string(),
                        variables_reference: 2,
                        expensive: false,
                        ..Default::default()
                    },
                ],
            }),

            Command::Variables(ref args) => {
                let session_opt = self.session.lock().await;
                let mut variables = vec![];
                if let Some(ref session) = *session_opt {
                    if args.variables_reference == 1 {
                        // Scope: Signals
                        for (id, _signal) in &session.world.signals() {
                            if let Some(val) = session.runtime.get_signal(id) {
                                variables.push(Variable {
                                    name: id.to_string(),
                                    value: format!("{}", val),
                                    variables_reference: 0,
                                    ..Default::default()
                                });
                            }
                        }
                    } else if args.variables_reference == 2 {
                        // Scope: Configuration
                        for (name, (val, unit)) in &session.world.config {
                            variables.push(Variable {
                                name: name.clone(),
                                value: format!("{}{:?}", val, unit),
                                variables_reference: 0,
                                ..Default::default()
                            });
                        }
                    } else if args.variables_reference == 3 {
                        // Scope: Entities
                        for (i, (id, instances)) in session.runtime.entities().iter().enumerate() {
                            variables.push(Variable {
                                name: id.to_string(),
                                value: format!("{} instances", instances.count()),
                                variables_reference: 1000 + i as i64,
                                ..Default::default()
                            });
                        }
                    } else if args.variables_reference >= 1000 && args.variables_reference < 2000 {
                        // Entity Type -> List Instances
                        let entity_idx = (args.variables_reference - 1000) as usize;
                        if let Some((_id, instances)) =
                            session.runtime.entities().iter().nth(entity_idx)
                        {
                            // Show first 100 instances to avoid IDE lag
                            for (j, (inst_id, _data)) in instances.iter().enumerate().take(100) {
                                variables.push(Variable {
                                    name: format!("[{}] {}", j, inst_id),
                                    value: "Instance Data".to_string(),
                                    variables_reference: 10000
                                        + (entity_idx as i64 * 1000)
                                        + j as i64,
                                    ..Default::default()
                                });
                            }
                            if instances.count() > 100 {
                                variables.push(Variable {
                                    name: "...".to_string(),
                                    value: format!("{} more instances", instances.count() - 100),
                                    variables_reference: 0,
                                    ..Default::default()
                                });
                            }
                        }
                    } else if args.variables_reference >= 10000 {
                        // Instance -> List Fields
                        let ref_id = args.variables_reference - 10000;
                        let entity_idx = (ref_id / 1000) as usize;
                        let inst_idx = (ref_id % 1000) as usize;

                        if let Some((_id, instances)) =
                            session.runtime.entities().iter().nth(entity_idx)
                        {
                            if let Some((_inst_id, data)) = instances.iter().nth(inst_idx) {
                                for (field_name, val) in &data.fields {
                                    variables.push(Variable {
                                        name: field_name.clone(),
                                        value: format!("{}", val),
                                        variables_reference: 0,
                                        ..Default::default()
                                    });
                                }
                            }
                        }
                    }
                }
                ResponseBody::Variables(dap::responses::VariablesResponse { variables })
            }

            Command::Continue(_) => {
                // Clear pause flag before starting
                self.should_pause.store(false, Ordering::Relaxed);

                let session_arc = self.session.clone();
                let should_pause = self.should_pause.clone();
                let adapter = self.clone_for_task();

                // Spawn execution in background task so we don't block the DAP server
                tokio::spawn(async move {
                    info!("Continue task spawned");

                    // Initialize warmup once
                    {
                        let mut session_opt = session_arc.lock().await;
                        if let Some(ref mut session) = *session_opt {
                            session.status = SessionStatus::Running;
                            if !session.runtime.is_warmup_complete() {
                                info!("Executing warmup...");
                                let _ = session.runtime.execute_warmup();
                            }
                            info!("Starting execution loop");
                        }
                    }

                    // Run continuously until breakpoint or pause requested
                    let mut tick_count = 0;
                    loop {
                        // Check pause flag BEFORE acquiring lock
                        if should_pause.load(Ordering::Relaxed) {
                            let mut session_opt = session_arc.lock().await;
                            if let Some(ref mut session) = *session_opt {
                                session.status = SessionStatus::Paused;
                            }
                            let body = StoppedEventBody {
                                reason: StoppedEventReason::Pause,
                                thread_id: Some(1),
                                all_threads_stopped: Some(true),
                                text: Some("Paused by user".to_string()),
                                description: None,
                                preserve_focus_hint: None,
                                hit_breakpoint_ids: None,
                            };
                            adapter.send_event(Event::Stopped(body)).await;
                            info!("Paused by user request after {} ticks", tick_count);
                            break;
                        }

                        // Acquire lock, execute one tick, release lock
                        let result = {
                            let mut session_opt = session_arc.lock().await;
                            match *session_opt {
                                Some(ref mut session) => session.runtime.execute_until_breakpoint(),
                                None => break,
                            }
                        }; // Lock released here!

                        tick_count += 1;
                        if tick_count % 10 == 0 {
                            info!("Executed {} ticks", tick_count);
                        }

                        match result {
                            Ok(continuum_runtime::types::StepResult::Breakpoint { signal }) => {
                                // Hit a breakpoint - stop and notify
                                let mut session_opt = session_arc.lock().await;
                                if let Some(ref mut session) = *session_opt {
                                    session.status = SessionStatus::Paused;
                                    session.current_halt_signal = Some(signal.clone());
                                }
                                let body = StoppedEventBody {
                                    reason: StoppedEventReason::Breakpoint,
                                    thread_id: Some(1),
                                    all_threads_stopped: Some(true),
                                    text: Some(format!("Paused on signal: {}", signal)),
                                    description: None,
                                    preserve_focus_hint: None,
                                    hit_breakpoint_ids: None,
                                };
                                adapter.send_event(Event::Stopped(body)).await;
                                info!(
                                    "Stopped at breakpoint: {} after {} ticks",
                                    signal, tick_count
                                );
                                break;
                            }
                            Ok(continuum_runtime::types::StepResult::TickCompleted(_)) => {
                                // Tick completed, yield to allow other tasks to run
                                tokio::task::yield_now().await;
                            }
                            Ok(_) => {}
                            Err(e) => {
                                error!("Execution failed: {}", e);
                                let mut session_opt = session_arc.lock().await;
                                if let Some(ref mut session) = *session_opt {
                                    session.status = SessionStatus::Paused;
                                }
                                break;
                            }
                        }
                    }
                    info!("Continue task finished after {} ticks", tick_count);
                });

                ResponseBody::Continue(dap::responses::ContinueResponse {
                    all_threads_continued: Some(true),
                })
            }
            Command::Next(_) => {
                // Step over - execute one phase (or until breakpoint)
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    if !session.runtime.is_warmup_complete() {
                        let _ = session.runtime.execute_warmup();
                    }
                    match session.runtime.execute_step() {
                        Ok(result) => {
                            let body = match result {
                                continuum_runtime::types::StepResult::Breakpoint { signal } => {
                                    session.current_halt_signal = Some(signal.clone());
                                    StoppedEventBody {
                                        reason: StoppedEventReason::Breakpoint,
                                        thread_id: Some(1),
                                        all_threads_stopped: Some(true),
                                        text: Some(format!("Paused on signal: {}", signal)),
                                        description: None,
                                        preserve_focus_hint: None,
                                        hit_breakpoint_ids: None,
                                    }
                                }
                                continuum_runtime::types::StepResult::TickCompleted(ctx) => {
                                    StoppedEventBody {
                                        reason: StoppedEventReason::Step,
                                        thread_id: Some(1),
                                        all_threads_stopped: Some(true),
                                        text: Some(format!("Tick {} completed", ctx.tick)),
                                        description: None,
                                        preserve_focus_hint: None,
                                        hit_breakpoint_ids: None,
                                    }
                                }
                                _ => StoppedEventBody {
                                    reason: StoppedEventReason::Step,
                                    thread_id: Some(1),
                                    all_threads_stopped: Some(true),
                                    text: Some(format!("Phase: {:?}", session.runtime.phase())),
                                    description: None,
                                    preserve_focus_hint: None,
                                    hit_breakpoint_ids: None,
                                },
                            };
                            self.send_event(Event::Stopped(body)).await;
                        }
                        Err(e) => {
                            return self
                                .make_error_response(&request, format!("Step failed: {}", e));
                        }
                    }
                }
                ResponseBody::Next
            }
            Command::StepIn(_) => {
                // Step into - same as step over for now (execute one phase)
                // In the future, could step into individual signal resolutions
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    if !session.runtime.is_warmup_complete() {
                        let _ = session.runtime.execute_warmup();
                    }
                    match session.runtime.execute_step() {
                        Ok(result) => {
                            let body = match result {
                                continuum_runtime::types::StepResult::Breakpoint { signal } => {
                                    session.current_halt_signal = Some(signal.clone());
                                    StoppedEventBody {
                                        reason: StoppedEventReason::Breakpoint,
                                        thread_id: Some(1),
                                        all_threads_stopped: Some(true),
                                        text: Some(format!("Paused on signal: {}", signal)),
                                        description: None,
                                        preserve_focus_hint: None,
                                        hit_breakpoint_ids: None,
                                    }
                                }
                                continuum_runtime::types::StepResult::TickCompleted(ctx) => {
                                    StoppedEventBody {
                                        reason: StoppedEventReason::Step,
                                        thread_id: Some(1),
                                        all_threads_stopped: Some(true),
                                        text: Some(format!("Tick {} completed", ctx.tick)),
                                        description: None,
                                        preserve_focus_hint: None,
                                        hit_breakpoint_ids: None,
                                    }
                                }
                                _ => StoppedEventBody {
                                    reason: StoppedEventReason::Step,
                                    thread_id: Some(1),
                                    all_threads_stopped: Some(true),
                                    text: Some(format!("Phase: {:?}", session.runtime.phase())),
                                    description: None,
                                    preserve_focus_hint: None,
                                    hit_breakpoint_ids: None,
                                },
                            };
                            self.send_event(Event::Stopped(body)).await;
                        }
                        Err(e) => {
                            return self
                                .make_error_response(&request, format!("Step failed: {}", e));
                        }
                    }
                }
                ResponseBody::StepIn
            }
            Command::StepOut(_) => {
                // Step out - execute until current tick completes
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    if !session.runtime.is_warmup_complete() {
                        let _ = session.runtime.execute_warmup();
                    }

                    // Keep stepping until tick completes
                    loop {
                        match session.runtime.execute_step() {
                            Ok(result) => match result {
                                continuum_runtime::types::StepResult::Breakpoint { signal } => {
                                    session.current_halt_signal = Some(signal.clone());
                                    let body = StoppedEventBody {
                                        reason: StoppedEventReason::Breakpoint,
                                        thread_id: Some(1),
                                        all_threads_stopped: Some(true),
                                        text: Some(format!("Paused on signal: {}", signal)),
                                        description: None,
                                        preserve_focus_hint: None,
                                        hit_breakpoint_ids: None,
                                    };
                                    self.send_event(Event::Stopped(body)).await;
                                    break;
                                }
                                continuum_runtime::types::StepResult::TickCompleted(ctx) => {
                                    let body = StoppedEventBody {
                                        reason: StoppedEventReason::Step,
                                        thread_id: Some(1),
                                        all_threads_stopped: Some(true),
                                        text: Some(format!("Tick {} completed", ctx.tick)),
                                        description: None,
                                        preserve_focus_hint: None,
                                        hit_breakpoint_ids: None,
                                    };
                                    self.send_event(Event::Stopped(body)).await;
                                    break;
                                }
                                _ => continue,
                            },
                            Err(e) => {
                                return self.make_error_response(
                                    &request,
                                    format!("Step out failed: {}", e),
                                );
                            }
                        }
                    }
                }
                ResponseBody::StepOut
            }
            Command::Pause(_) => {
                // Signal the running execution to pause
                self.should_pause.store(true, Ordering::Relaxed);
                ResponseBody::Pause
            }
            Command::Terminate(_) | Command::Disconnect(_) => {
                let mut session = self.session.lock().await;
                *session = None;
                ResponseBody::Disconnect
            }
            _ => return self.make_error_response(&request, "Not implemented".to_string()),
        };

        Response {
            request_seq: request.seq,
            success: true,
            body: Some(body),
            error: None,
            message: None,
        }
    }

    fn make_error_response(&self, request: &Request, message: String) -> Response {
        Response {
            request_seq: request.seq,
            success: false,
            body: None,
            error: Some(Message {
                id: 0,
                format: message.clone(),
                variables: HashMap::new(),
                send_telemetry: None,
                show_user: None,
                url: None,
                url_label: None,
            }),
            message: Some(dap::responses::ResponseMessage::Error(message)),
        }
    }
}
