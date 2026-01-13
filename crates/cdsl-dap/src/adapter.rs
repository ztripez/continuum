use continuum_compiler::ir::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_signal_resolver,
    build_warmup_fn, compile, convert_assertion_severity, get_initial_signal_value, CompiledWorld,
};
use continuum_runtime::types::WarmupConfig;
use continuum_runtime::EraId;
use continuum_runtime::Runtime;
use dap::events::{Event, StoppedEventBody};
use dap::prelude::*;
use dap::types::{Capabilities, Message, Scope, StackFrame, StoppedEventReason, Thread, Variable};

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info};

pub struct ContinuumDebugAdapter {
    session: Arc<Mutex<Option<DebugSession>>>,
    event_tx: Arc<Mutex<Option<mpsc::Sender<Event>>>>,
}

pub struct DebugSession {
    pub world: CompiledWorld,
    pub runtime: Runtime,
    pub sources: HashMap<PathBuf, String>,
    pub breakpoints: HashMap<PathBuf, HashSet<usize>>, // File -> Line numbers
    pub signal_breakpoints: HashSet<String>,           // Signal names
    pub status: SessionStatus,
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

    pub async fn handle_request(&self, request: Request) -> Response {
        let body = match request.command {
            Command::Initialize(_) => ResponseBody::Initialize(Capabilities {
                supports_configuration_done_request: Some(true),
                supports_step_back: Some(false),
                supports_terminate_request: Some(true),
                ..Default::default()
            }),
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

                let initial_era = world
                    .eras()
                    .iter()
                    .find(|(_, era)| era.is_initial)
                    .map(|(id, _)| id.clone())
                    .unwrap_or_else(|| {
                        world
                            .eras()
                            .keys()
                            .next()
                            .cloned()
                            .unwrap_or_else(|| EraId::from("default"))
                    });

                let era_configs = build_era_configs(&world);
                let mut runtime = Runtime::new(initial_era, era_configs, compilation.dags);

                for (id, signal) in &world.signals() {
                    if let Some(resolver) = build_signal_resolver(signal, &world) {
                        runtime.register_resolver(resolver);
                    }

                    // Register warmup if present
                    if let Some(ref warmup) = signal.warmup {
                        let warmup_fn =
                            build_warmup_fn(&warmup.iterate, &world.constants, &world.config);
                        let config = WarmupConfig {
                            max_iterations: warmup.iterations,
                            convergence_epsilon: warmup.convergence,
                        };
                        runtime.register_warmup(id.clone(), warmup_fn, config);
                    }
                }

                for (id, signal) in &world.signals() {
                    for assertion in &signal.assertions {
                        runtime.register_assertion(
                            id.clone(),
                            build_assertion(&assertion.condition, &world),
                            convert_assertion_severity(assertion.severity),
                            assertion.message.clone(),
                        );
                    }
                }

                for (id, field) in &world.fields() {
                    if let Some(ref expr) = field.measure {
                        if let Some(measure_fn) = build_field_measure(id, expr, &world) {
                            runtime.register_measure_op(measure_fn);
                        }
                    }
                }

                for (_id, fracture) in &world.fractures() {
                    runtime.register_fracture(build_fracture(fracture, &world));
                }

                for (id, _signal) in &world.signals() {
                    runtime.init_signal(id.clone(), get_initial_signal_value(&world, id));
                }

                let mut session = self.session.lock().await;
                *session = Some(DebugSession {
                    world,
                    runtime,
                    sources,
                    breakpoints: HashMap::new(),
                    signal_breakpoints: HashSet::new(),
                    status: SessionStatus::Paused,
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
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    if let Some(ref source) = args.source.path {
                        let path = PathBuf::from(source);
                        let lines: HashSet<usize> = args
                            .breakpoints
                            .as_ref()
                            .map(|bs| bs.iter().map(|b| b.line as usize).collect())
                            .unwrap_or_default();
                        session.breakpoints.insert(path.clone(), lines.clone());

                        // Update runtime breakpoints
                        session.runtime.clear_breakpoints();
                        for (path, lines) in &session.breakpoints {
                            for (id, node) in session.world.nodes.iter() {
                                if let Some(ref node_file) = node.file {
                                    if node_file == path {
                                        if let Some(source) = session.sources.get(node_file) {
                                            let (line, _) =
                                                offset_to_line_col(source, node.span.start);
                                            if lines.contains(&(line as usize + 1)) {
                                                let signal_id =
                                                    continuum_foundation::SignalId::from(
                                                        id.to_string(),
                                                    );
                                                session.runtime.add_breakpoint(signal_id);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                ResponseBody::SetBreakpoints(dap::responses::SetBreakpointsResponse {
                    breakpoints: vec![],
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
                    vec![StackFrame {
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
                    }]
                } else {
                    vec![]
                };
                ResponseBody::StackTrace(dap::responses::StackTraceResponse {
                    stack_frames: frames,
                    total_frames: Some(1),
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
                        for (id, _signal) in &session.world.signals() {
                            if let Some(val) = session.runtime.get_signal(id) {
                                variables.push(Variable {
                                    name: id.to_string(),
                                    value: format!("{:?}", val),
                                    variables_reference: 0,
                                    ..Default::default()
                                });
                            }
                        }
                    } else if args.variables_reference == 2 {
                        for (name, (val, unit)) in &session.world.config {
                            variables.push(Variable {
                                name: name.clone(),
                                value: format!("{}{:?}", val, unit),
                                variables_reference: 0,
                                ..Default::default()
                            });
                        }
                    }
                }
                ResponseBody::Variables(dap::responses::VariablesResponse { variables })
            }
            Command::Continue(_) => {
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    session.status = SessionStatus::Running;
                    if !session.runtime.is_warmup_complete() {
                        let _ = session.runtime.execute_warmup();
                    }

                    match session.runtime.execute_until_breakpoint() {
                        Ok(result) => {
                            session.status = SessionStatus::Paused;
                            let body = match result {
                                continuum_runtime::types::StepResult::Breakpoint { signal } => {
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
                                _ => StoppedEventBody {
                                    reason: StoppedEventReason::Step,
                                    thread_id: Some(1),
                                    all_threads_stopped: Some(true),
                                    text: None,
                                    description: None,
                                    preserve_focus_hint: None,
                                    hit_breakpoint_ids: None,
                                },
                            };
                            self.send_event(Event::Stopped(body)).await;
                        }
                        Err(e) => {
                            error!("Execution failed: {}", e);
                        }
                    }
                }
                ResponseBody::Continue(dap::responses::ContinueResponse {
                    all_threads_continued: Some(true),
                })
            }
            Command::Next(_) => {
                let mut session_opt = self.session.lock().await;
                if let Some(ref mut session) = *session_opt {
                    if !session.runtime.is_warmup_complete() {
                        let _ = session.runtime.execute_warmup();
                    }
                    match session.runtime.execute_step() {
                        Ok(result) => {
                            let body = match result {
                                continuum_runtime::types::StepResult::Breakpoint { signal } => {
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
                                _ => StoppedEventBody {
                                    reason: StoppedEventReason::Step,
                                    thread_id: Some(1),
                                    all_threads_stopped: Some(true),
                                    text: None,
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
