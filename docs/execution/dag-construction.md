# DAG Construction

This document specifies **how execution DAGs are constructed** from the typed IR.

It bridges the DSL declarations and the executable graph.

---

## 1. Construction Overview

DAG construction proceeds in stages:

```
Typed IR
    ↓
1. Node Extraction
    ↓
2. Dependency Resolution
    ↓
3. Graph Partitioning
    ↓
4. Cycle Detection
    ↓
5. Topological Leveling
    ↓
Executable DAGs
```

Each stage is deterministic.
The same IR always produces the same DAGs.

---

## 2. Graph Partitioning Key

DAGs are partitioned by:

```
(phase × stratum × era)
```

This produces independent graphs for each combination.

Example for a world with:
- 5 phases
- 3 strata (thermal, tectonics, atmosphere)
- 2 eras (early, stable)

Results in up to 30 separate DAGs (some may be empty).

---

## 3. Node Types

### 3.1 SignalResolveNode

Created for each `signal.*` declaration.

```
SignalResolveNode {
    id: SignalId
    stratum: StratumId
    type: TypeRef
    reads: Set<SignalId>        // inferred from resolve block
    inputs: Set<InputChannelId> // accumulated inputs
    body: ResolveExpr
    assertions: Vec<Assertion>
}
```

Phase: **Resolve**

### 3.2 OperatorCollectNode

Created for `operator.*` with `phase(collect)`.

```
OperatorCollectNode {
    id: OperatorId
    stratum: StratumId
    reads: Set<SignalId>
    writes: Set<InputChannelId>  // signal <- value
    body: CollectExpr
}
```

Phase: **Collect**

### 3.3 OperatorMeasureNode

Created for `operator.*` with `phase(measure)`.

```
OperatorMeasureNode {
    id: OperatorId
    stratum: StratumId
    reads: Set<SignalId>
    writes: Set<FieldId>  // field <- position, value
    body: MeasureExpr
}
```

Phase: **Measure**

### 3.4 FieldEmitNode

Created for each `field.*` declaration with a `measure` block.

```
FieldEmitNode {
    id: FieldId
    stratum: StratumId
    type: TypeRef
    topology: Topology
    reads: Set<SignalId>
    body: MeasureExpr
}
```

Phase: **Measure**

### 3.5 ImpulseApplyNode

Created for each `impulse.*` declaration.

```
ImpulseApplyNode {
    id: ImpulseId
    payload_type: TypeRef
    writes: Set<InputChannelId>
    body: ApplyExpr
}
```

Phase: **Collect**

Impulse nodes are conditional - they execute only when an impulse is scheduled.

### 3.6 FractureNode

Created for each `fracture.*` declaration.

```
FractureNode {
    id: FractureId
    reads: Set<SignalId>
    condition: WhenExpr
    writes: Set<InputChannelId>  // signal <- value in emit
    body: EmitExpr
}
```

Phase: **Fracture**

### 3.7 ChronicleNode

Created for each `chronicle.*` declaration.

```
ChronicleNode {
    id: ChronicleId
    reads: Set<SignalId>
    reads_fields: Set<FieldId>
    body: ObserveExpr
}
```

Phase: **Measure** (post-field emission)

Chronicles are non-causal - they never write signals.

### 3.8 EntityInstanceNode

For `entity.*` declarations, nodes are **expanded per instance**.

```
EntityInstanceNode {
    entity_id: EntityId
    instance_index: u32
    inner: SignalResolveNode  // each instance resolves independently
}
```

Instance count is fixed at scenario application.

---

## 4. Dependency Edge Types

### 4.1 Signal Read Edge

```
A reads signal.x  →  edge from SignalResolveNode(x) to A
```

The reader must wait for the signal to resolve.

### 4.2 Input Channel Edge

When multiple sources write to a signal's input channel:

```
operator a <- signal.x
operator b <- signal.x
fracture c <- signal.x
```

All writers must complete before signal.x resolves.

```
OperatorCollectNode(a) → SignalResolveNode(x)
OperatorCollectNode(b) → SignalResolveNode(x)
FractureNode(c) → SignalResolveNode(x)  // fracture phase
```

### 4.3 Phase Boundary Edge

Implicit edges enforce phase ordering:

```
All Collect nodes → barrier → All Resolve nodes
All Resolve nodes → barrier → All Fracture nodes
All Fracture nodes → barrier → All Measure nodes
```

These are not explicit edges but structural barriers.

### 4.4 Stratum Coupling Edge

When stratum A reads a signal from stratum B:

```
signal a (stratum: thermal) {
    resolve {
        signal.b  // stratum: tectonics
    }
}
```

Creates a cross-stratum dependency.
Signal B must have resolved (in current or previous tick) before A resolves.

---

## 5. Dependency Inference

Dependencies are extracted from expression analysis.

### 5.1 Direct References

```
signal x
```

Creates read dependency on signal.x.

### 5.2 Prev References

```
prev
prev.field
```

References the signal's own previous value.
No inter-signal dependency, but creates a temporal dependency.

### 5.3 Entity Iteration

```
agg.sum(entity.stellar.moon, expr)
```

Creates dependencies on all entity instance signals referenced in expr.

### 5.4 Function Calls

```
fn physics.compute(signal.a, signal.b)
```

Dependencies propagate through function bodies.
Functions are inlined for dependency analysis.

### 5.5 Conditional Expressions

```
if signal.a > 0 { signal.b } else { signal.c }
```

Both branches contribute dependencies (signal.a, signal.b, signal.c).

---

## 6. Entity Expansion

Entities are expanded during DAG construction.

### 6.1 Instance Expansion

```
entity stellar.moon {
    : count(config.stellar.moon_count)
    ...
}
```

With `moon_count = 3`, creates:

```
SignalResolveNode(entity.stellar.moon[0])
SignalResolveNode(entity.stellar.moon[1])
SignalResolveNode(entity.stellar.moon[2])
```

### 6.2 Aggregate Dependencies

```
signal terra.tidal {
    resolve {
        agg.sum(entity.stellar.moon, fn.tidal_force(self.mass))
    }
}
```

Creates edges from all moon instance nodes to the tidal signal.

### 6.3 Pairwise Operations

```
agg.sum(pairs(entity.stellar.body), fn.gravity(self, other))
```

For N instances, creates N*(N-1)/2 pair computations.
Each pair depends on both instance nodes.

---

## 7. Cross-Strata Dependencies

### 7.1 Same-Tick Reads

Stratum A reading stratum B (same tick):

- If B executes this tick: A waits for B
- If B is gated: A reads B's last resolved value

### 7.2 Multi-Rate Resolution

When strata have different strides:

```
strata fast { stride: 1 }
strata slow { stride: 10 }
```

A signal in `fast` reading from `slow`:
- Reads the most recently resolved value
- Does not wait for slow to execute (it may not this tick)

This is explicit and deterministic.

### 7.3 Stratum Ordering

Within a phase, strata are ordered by:

1. Dependency (consumers after producers)
2. Lexicographic stratum ID (tie-breaker)

Ordering is fixed at compile time.

---

## 8. Cycle Detection

After edge construction, the graph is checked for cycles.

### 8.1 Intra-Stratum Cycles

```
signal a reads signal.b
signal b reads signal.a
```

**Error:** Circular dependency within stratum.

### 8.2 Cross-Stratum Cycles

```
signal a (stratum: x) reads signal.b (stratum: y)
signal b (stratum: y) reads signal.a (stratum: x)
```

**Error:** Cross-stratum circular dependency.

### 8.3 Temporal Dependencies Are Not Cycles

```
signal a reads prev  // own previous value
```

Not a cycle - reads previous tick, not current.

### 8.4 Cycle Error Reporting

Cycle detection reports:
- All signals in the cycle
- The dependency chain
- Stratum context

---

## 9. Topological Leveling

After cycle detection, nodes are assigned levels.

### 9.1 Level Assignment

```
level(node) = 0                           if no dependencies
level(node) = max(level(deps)) + 1        otherwise
```

### 9.2 Level Properties

- All nodes in a level are independent
- Nodes in the same level may execute in parallel
- Level N completes before level N+1 begins

### 9.3 Barrier Insertion

Between levels, a barrier ensures completion:

```
Level 0: [node_a, node_b, node_c]
         --- barrier ---
Level 1: [node_d, node_e]
         --- barrier ---
Level 2: [node_f]
```

---

## 10. Era-Specific DAGs

Each era produces its own DAG set.

### 10.1 Gated Strata

In era with:
```
strata {
    thermal: active
    tectonics: gated
}
```

The tectonics stratum DAG is **not executed**.
Signals in tectonics retain their last resolved values.

### 10.2 Stride Modifiers

```
strata {
    atmosphere: active(stride: 2)
}
```

The atmosphere DAG executes only on even ticks.

### 10.3 Era Transition

Era transitions do not modify DAGs.
They select which pre-built DAGs are active.

---

## 11. Construction Output

The final output is a set of executable DAGs:

```
DAGSet {
    era: EraId
    graphs: Map<(Phase, StratumId), ExecutableDAG>
}

ExecutableDAG {
    levels: Vec<Level>
    total_nodes: usize
}

Level {
    nodes: Vec<NodeRef>
    can_parallelize: bool
}
```

---

## 12. Invariants

DAG construction must preserve:

1. **Determinism**: Same IR → same DAGs
2. **Acyclicity**: No cycles within a tick
3. **Completeness**: All DSL entities represented
4. **Phase Correctness**: Nodes in correct phase graphs
5. **Stratum Isolation**: No cross-stratum execution within a level

---

## Summary

- DAGs are built from typed IR
- Nodes represent DSL constructs
- Edges represent data dependencies
- Entities expand to per-instance nodes
- Cycles are compile-time errors
- Levels enable safe parallelism
- Eras select pre-built DAG sets

The construction is mechanical and deterministic.
No runtime decisions affect DAG structure.
