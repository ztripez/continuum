# Tick Execution

This document specifies **how a single simulation tick executes**.

It describes the runtime flow, not the construction.

---

## 1. Tick Overview

A tick is the atomic unit of causal advancement.

```
Tick N
    ↓
1. Configure Phase
    ↓
2. Collect Phase
    ↓
3. Resolve Phase
    ↓
4. Fracture Phase
    ↓
5. Measure Phase
    ↓
Tick N+1
```

All phases complete before the tick ends.
No partial ticks are visible to observers.

---

## 2. Pre-Tick: Context Setup

Before phase execution begins:

### 2.1 Era Check

```
current_era = evaluate_era_transitions(prev_era, resolved_signals)
```

Era transitions are evaluated based on previous tick's resolved signals.
The new era is active for this tick.

### 2.2 Timestep Selection

```
dt = current_era.dt
```

The era determines the timestep for this tick.

### 2.3 Stratum Eligibility

```
for stratum in strata:
    eligible = era.stratum_policy(stratum)
    if eligible == gated:
        skip stratum this tick
    if eligible == active(stride: N):
        execute only if tick_number % N == 0
```

### 2.4 DAG Selection

```
active_dags = dag_set[current_era]
    .filter(|(phase, stratum)| stratum.is_eligible(tick_number))
```

---

## 3. Configure Phase

**Purpose:** Finalize tick execution context.

### 3.1 Operations

- Lock dt value
- Lock era identity
- Prepare stratum execution masks
- Initialize per-tick counters

### 3.2 Outputs

- `tick_context` frozen for remaining phases

### 3.3 DAG Execution

No user-defined DAG nodes execute in Configure.
This phase is engine-internal.

---

## 4. Collect Phase

**Purpose:** Gather inputs for signal resolution.

### 4.1 Input Channels

Each signal has an input channel:

```
InputChannel<T> {
    accumulated: T
    accumulator: fn(T, T) -> T  // must be commutative
}
```

Default accumulator is `sum`.

### 4.2 Operator Execution

For each `OperatorCollectNode` in the Collect DAG:

```
// Read resolved signals (from previous tick)
let inputs = node.reads.map(|s| prev_tick.resolved[s])

// Execute body
let outputs = execute(node.body, inputs)

// Write to input channels
for (channel, value) in outputs:
    input_channels[channel].accumulate(value)
```

### 4.3 Impulse Application

For each scheduled impulse this tick:

```
if impulse.scheduled_for(tick_number):
    let payload = impulse.payload
    execute(impulse.apply_body, payload)
    // writes to input channels
```

### 4.4 Execution Order

1. Execute Collect DAG levels sequentially
2. Apply scheduled impulses
3. All input channels are sealed

### 4.5 Phase Barrier

After Collect completes:
- Input channels become read-only
- No further accumulation allowed this tick

---

## 5. Resolve Phase

**Purpose:** Compute authoritative state.

### 5.1 Signal Resolution

For each `SignalResolveNode` in the Resolve DAG:

```
// Read previous value
let prev = prev_tick.resolved[signal.id]

// Read other resolved signals (previous tick for unresolved, current for resolved)
let reads = signal.reads.map(|s| {
    if already_resolved_this_tick(s):
        current_tick.resolved[s]
    else:
        prev_tick.resolved[s]
})

// Read accumulated inputs
let inputs = input_channels[signal.id].drain()

// Execute resolver
let value = execute(signal.resolve_body, prev, reads, inputs, dt)

// Validate assertions
for assertion in signal.assertions:
    if !assertion.check(value):
        emit_fault(assertion.fault_type)

// Store result
current_tick.resolved[signal.id] = value
```

### 5.2 Topological Execution

Signals resolve in topological level order:

```
for level in resolve_dag.levels:
    parallel_execute(level.nodes)
    barrier()
```

Within a level, nodes are independent and may parallelize.

### 5.3 Entity Resolution

Entity instances resolve as normal signals:

```
for instance in entity.instances:
    resolve(instance)  // follows dependency ordering
```

### 5.4 Phase Barrier

After Resolve completes:
- All signals have current-tick values
- Resolved state is immutable for remainder of tick

---

## 6. Fracture Phase

**Purpose:** Detect and respond to emergent tension.

### 6.1 Fracture Evaluation

For each `FractureNode`:

```
// Read resolved signals (current tick)
let reads = fracture.reads.map(|s| current_tick.resolved[s])

// Evaluate condition
if evaluate(fracture.when_condition, reads):
    // Execute emit block
    let outputs = execute(fracture.emit_body, reads)

    // Queue for next tick's Collect
    for (channel, value) in outputs:
        next_tick_inputs[channel].queue(value)
```

### 6.2 Fracture Timing

Fracture outputs affect **next tick**, not current tick.

The current tick's Resolve phase is complete.
Fracture outputs enter next tick's Collect phase.

### 6.3 Cascading Fractures

Fractures do not re-trigger within a tick.
Multiple fractures may fire, but each fires at most once per tick.

---

## 7. Measure Phase

**Purpose:** Produce observable outputs.

### 7.1 Field Emission

For each `FieldEmitNode`:

```
// Read resolved signals
let reads = field.reads.map(|s| current_tick.resolved[s])

// Execute measure body
let value = execute(field.measure_body, reads)

// Emit to observation buffer
observation_buffer.emit_field(field.id, value, field.topology)
```

### 7.2 Operator Measure Execution

For each `OperatorMeasureNode`:

```
// Similar to field emission but may emit multiple samples
execute(operator.body, reads)
// writes: field <- position, value
```

### 7.3 Chronicle Execution

For each `ChronicleNode`:

```
// Read resolved signals and emitted fields
let signal_reads = chronicle.reads.map(|s| current_tick.resolved[s])
let field_reads = chronicle.reads_fields.map(|f| observation_buffer[f])

// Execute observe body
execute(chronicle.observe_body, signal_reads, field_reads)
// May emit events to chronicle log
```

### 7.4 Observer Notification

After all emissions:

```
observer_system.notify(observation_buffer)
```

Observers receive completed tick data.

---

## 8. Post-Tick: State Transition

### 8.1 Tick Commit

```
prev_tick = current_tick
current_tick = new_tick()
tick_number += 1
```

### 8.2 Input Queue Transfer

```
// Fracture outputs become next tick's inputs
input_channels = next_tick_inputs
next_tick_inputs = empty()
```

### 8.3 History Recording

```
history.append(prev_tick.resolved)
```

If history recording is enabled.

---

## 9. Parallel Execution Model

### 9.1 Level Parallelism

Within a topological level:

```
level.nodes.par_iter().for_each(|node| {
    execute_node(node)
})
```

Nodes in the same level have no dependencies.

### 9.2 No Parallel Writes

All nodes write to distinct locations:
- Each signal writes to its own resolved slot
- Each input channel is lock-free with atomic accumulation

### 9.3 Deterministic Reduction

When multiple values accumulate:

```
// Sort by source ID for deterministic order
let sorted_contributions = contributions.sort_by_key(|c| c.source_id)
let result = sorted_contributions.fold(identity, accumulator)
```

Accumulation order is deterministic, not arrival-order.

---

## 10. Error Handling

### 10.1 Assertion Failures

```
if !assertion.check(value):
    match fault_policy:
        FaultPolicy::Fatal => abort_run()
        FaultPolicy::Halt => halt_at_boundary()
        FaultPolicy::Continue => log_fault_and_continue()
```

### 10.2 Numeric Errors

- NaN propagation: detected and faulted
- Infinity: detected and faulted (unless explicitly allowed)
- Overflow: saturates to bounds if constrained, else faults

### 10.3 Tick Boundary Guarantee

Errors halt at tick boundaries.
Partial ticks are never visible to observers.

---

## 11. Performance Considerations

### 11.1 Memory Layout

Signals are stored in contiguous arrays per stratum:

```
stratum_data[stratum_id].signals[signal_index] = value
```

Enables cache-friendly iteration.

### 11.2 Batch Kernels

When many nodes perform similar operations:

```
// Instead of per-node kernel dispatch:
for node in level.nodes:
    kernel.execute(node)

// Batch into single kernel call:
kernel.execute_batch(level.nodes)
```

Backend handles batching transparently.

### 11.3 Stratum Skipping

Gated strata have zero execution cost:

```
if !stratum.is_eligible():
    continue  // no DAG traversal
```

---

## 12. Tick Execution Pseudocode

```
fn execute_tick(state: &mut SimState, tick: u64, dag_set: &DagSet) {
    // Pre-tick
    let era = evaluate_era_transitions(&state, tick);
    let dt = era.dt;
    let eligible_strata = compute_eligible_strata(era, tick);

    // Configure (engine internal)
    let ctx = TickContext::new(tick, dt, era);

    // Collect
    for stratum in eligible_strata {
        execute_dag(&dag_set[era][Collect][stratum], &mut state);
    }
    apply_scheduled_impulses(&mut state, tick);
    seal_input_channels(&mut state);

    // Resolve
    for stratum in eligible_strata {
        execute_dag(&dag_set[era][Resolve][stratum], &mut state);
    }
    validate_assertions(&state);
    commit_resolved_signals(&mut state);

    // Fracture
    execute_fractures(&dag_set[era][Fracture], &mut state);
    // outputs queued for next tick

    // Measure
    for stratum in eligible_strata {
        execute_dag(&dag_set[era][Measure][stratum], &mut state);
    }
    execute_chronicles(&dag_set[era], &mut state);
    notify_observers(&state);

    // Post-tick
    advance_state(&mut state);
}
```

---

## Summary

- Ticks execute phases in strict order
- Configure prepares context, no user DAG
- Collect accumulates inputs from operators and impulses
- Resolve computes authoritative signals
- Fracture detects tension, queues for next tick
- Measure emits fields and chronicles
- Parallelism is safe within topological levels
- Errors halt at tick boundaries
- The flow is deterministic and reproducible
