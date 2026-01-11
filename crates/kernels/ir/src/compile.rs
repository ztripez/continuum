//! IR to Runtime DAG Compilation.
//!
//! This module transforms a [`CompiledWorld`] (the typed intermediate
//! representation) into executable runtime DAGs that can be scheduled
//! and executed by the runtime.
//!
//! # Compilation Process
//!
//! 1. **Index assignment**: Each signal, operator, field, and fracture is
//!    assigned a numeric index for efficient runtime lookup
//! 2. **Era compilation**: For each era, DAGs are built for every (phase, stratum) pair
//! 3. **DAG construction**: Nodes are added with their dependencies, then
//!    topologically sorted to create execution levels
//!
//! # Output Structure
//!
//! The compilation produces a [`DagSet`] containing:
//! - One [`EraDags`] per era
//! - Each era contains multiple [`ExecutableDag`] instances
//! - Each DAG represents a (phase, stratum) combination
//!
//! # Errors
//!
//! Compilation can fail with [`CompileError`] for issues like:
//! - Cyclic signal dependencies that prevent topological ordering
//! - References to undefined strata or eras

use indexmap::IndexMap;
use thiserror::Error;

use continuum_foundation::{EraId, EntityId as FoundationEntityId, MemberId, MemberSignalId, SignalId, StratumId};
use continuum_runtime::dag::{
    AggregateBarrier, BarrierDagBuilder, CycleError, DagBuilder, DagNode, DagSet, EraDags,
    ExecutableDag, NodeId, NodeKind,
};
use continuum_runtime::reductions::ReductionOp;
use continuum_runtime::types::{EntityId as RuntimeEntityId, Phase};

use crate::{AggregateOpIr, CompiledExpr, CompiledWorld, OperatorPhaseIr};

/// Checks if an expression contains any entity-related constructs.
///
/// Entity expressions (Aggregate, SelfField, EntityAccess, etc.) require the
/// EntityExecutor for evaluation and cannot be compiled to bytecode. Signals
/// with such expressions should not be added to the bytecode DAG.
fn contains_entity_expression(expr: &CompiledExpr) -> bool {
    match expr {
        // Entity-related expressions
        CompiledExpr::SelfField(_)
        | CompiledExpr::EntityAccess { .. }
        | CompiledExpr::Aggregate { .. }
        | CompiledExpr::Other { .. }
        | CompiledExpr::Pairs { .. }
        | CompiledExpr::Filter { .. }
        | CompiledExpr::First { .. }
        | CompiledExpr::Nearest { .. }
        | CompiledExpr::Within { .. } => true,

        // Impulse-related expressions (also not bytecode-compatible)
        CompiledExpr::Payload
        | CompiledExpr::PayloadField(_)
        | CompiledExpr::EmitSignal { .. } => true,

        // Recursive cases - check sub-expressions
        CompiledExpr::Binary { left, right, .. } => {
            contains_entity_expression(left) || contains_entity_expression(right)
        }
        CompiledExpr::Unary { operand, .. } => contains_entity_expression(operand),
        CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
            args.iter().any(contains_entity_expression)
        }
        CompiledExpr::DtRobustCall { args, .. } => args.iter().any(contains_entity_expression),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            contains_entity_expression(condition)
                || contains_entity_expression(then_branch)
                || contains_entity_expression(else_branch)
        }
        CompiledExpr::Let { value, body, .. } => {
            contains_entity_expression(value) || contains_entity_expression(body)
        }
        CompiledExpr::FieldAccess { object, .. } => contains_entity_expression(object),

        // Leaf expressions - no entity constructs
        CompiledExpr::Literal(_)
        | CompiledExpr::Prev
        | CompiledExpr::DtRaw
        | CompiledExpr::SimTime
        | CompiledExpr::Collected
        | CompiledExpr::Signal(_)
        | CompiledExpr::Const(_)
        | CompiledExpr::Config(_)
        | CompiledExpr::Local(_) => false,
    }
}

/// Errors that can occur during DAG compilation.
///
/// These errors represent problems converting the IR into executable DAGs.
/// They typically indicate structural issues that prevent deterministic
/// execution scheduling.
#[derive(Debug, Error)]
pub enum CompileError {
    /// A cycle was detected in signal dependencies during DAG construction.
    ///
    /// The execution model requires signals to form a directed acyclic graph
    /// (DAG) so that a valid execution order can be determined. Cycles make
    /// this impossible.
    ///
    /// The `nodes` field contains the node IDs involved in the cycle,
    /// which can help identify the problematic signals.
    #[error("cycle detected in signal dependencies: {nodes:?}")]
    CycleDetected {
        /// Node identifiers involved in the dependency cycle.
        nodes: Vec<String>,
    },

    /// A stratum referenced in a signal or operator was not defined.
    ///
    /// All strata must be declared before they can be referenced in
    /// signal or operator definitions.
    #[error("undefined stratum: {0}")]
    UndefinedStratum(String),

    /// An era referenced in a transition or configuration was not defined.
    ///
    /// All eras must be declared before they can be referenced in
    /// era transitions or world configuration.
    #[error("undefined era: {0}")]
    UndefinedEra(String),
}

impl From<CycleError> for CompileError {
    fn from(e: CycleError) -> Self {
        CompileError::CycleDetected {
            nodes: e.involved_nodes.into_iter().map(|n| n.0).collect(),
        }
    }
}

/// Result of compilation
pub struct CompilationResult {
    /// Executable DAG set for all eras
    pub dags: DagSet,
    /// Resolver function indices (signal_id -> index)
    pub resolver_indices: IndexMap<SignalId, usize>,
    /// Member signal resolver indices (member_id -> kernel index)
    pub member_indices: IndexMap<MemberId, usize>,
    /// Operator function indices
    pub operator_indices: IndexMap<String, usize>,
    /// Field emitter indices
    pub field_indices: IndexMap<String, usize>,
    /// Fracture detector indices
    pub fracture_indices: IndexMap<String, usize>,
    /// Aggregate function indices (aggregate_id -> index)
    pub aggregate_indices: IndexMap<String, usize>,
}

/// Compile a world to executable DAGs
pub fn compile(world: &CompiledWorld) -> Result<CompilationResult, CompileError> {
    let compiler = Compiler::new(world);
    compiler.compile()
}

struct Compiler<'a> {
    world: &'a CompiledWorld,
    resolver_indices: IndexMap<SignalId, usize>,
    member_indices: IndexMap<MemberId, usize>,
    operator_indices: IndexMap<String, usize>,
    field_indices: IndexMap<String, usize>,
    fracture_indices: IndexMap<String, usize>,
    aggregate_indices: IndexMap<String, usize>,
}

/// Information about an aggregate operation found in an expression.
#[derive(Debug, Clone)]
struct AggregateInfo {
    /// The entity being aggregated over.
    entity_id: FoundationEntityId,
    /// The reduction operation.
    op: AggregateOpIr,
    /// The body expression to evaluate per instance.
    body: CompiledExpr,
}

impl<'a> Compiler<'a> {
    fn new(world: &'a CompiledWorld) -> Self {
        Self {
            world,
            resolver_indices: IndexMap::new(),
            member_indices: IndexMap::new(),
            operator_indices: IndexMap::new(),
            field_indices: IndexMap::new(),
            fracture_indices: IndexMap::new(),
            aggregate_indices: IndexMap::new(),
        }
    }

    fn compile(mut self) -> Result<CompilationResult, CompileError> {
        // Assign indices to all entities
        self.assign_indices();

        // Build DAGs for each era
        let mut dag_set = DagSet::default();

        for (era_id, _era) in &self.world.eras {
            let era_dags = self.compile_era(era_id)?;
            dag_set.insert_era(era_id.clone(), era_dags);
        }

        Ok(CompilationResult {
            dags: dag_set,
            resolver_indices: self.resolver_indices,
            member_indices: self.member_indices,
            operator_indices: self.operator_indices,
            field_indices: self.field_indices,
            fracture_indices: self.fracture_indices,
            aggregate_indices: self.aggregate_indices,
        })
    }

    fn assign_indices(&mut self) {
        // Assign resolver indices
        for (idx, (signal_id, _)) in self.world.signals.iter().enumerate() {
            self.resolver_indices.insert(signal_id.clone(), idx);
        }

        // Assign member signal indices
        for (idx, (member_id, _)) in self.world.members.iter().enumerate() {
            self.member_indices.insert(member_id.clone(), idx);
        }

        // Assign operator indices
        for (idx, (op_id, _)) in self.world.operators.iter().enumerate() {
            self.operator_indices.insert(op_id.0.clone(), idx);
        }

        // Assign field indices (only for fields with measure expressions that are bytecode-compatible)
        // Fields with entity expressions are skipped - they require EntityExecutor at runtime
        let mut field_idx = 0;
        for (field_id, field) in &self.world.fields {
            if let Some(ref measure) = field.measure {
                if !contains_entity_expression(measure) {
                    self.field_indices.insert(field_id.0.clone(), field_idx);
                    field_idx += 1;
                }
            }
        }

        // Assign fracture indices
        for (idx, (fracture_id, _)) in self.world.fractures.iter().enumerate() {
            self.fracture_indices.insert(fracture_id.0.clone(), idx);
        }
    }

    /// Convert an IR aggregate operation to a runtime reduction operation.
    ///
    /// Note: Boolean aggregates (Any, All, None) are not directly supported by
    /// the runtime's ReductionOp. They would need special handling (e.g., mapping
    /// Any -> Max after converting to 0/1, All -> Min, None -> complement of Any).
    /// For now, these panic as they require runtime support.
    fn aggregate_op_to_reduction_op(op: &AggregateOpIr) -> ReductionOp {
        match op {
            AggregateOpIr::Sum => ReductionOp::Sum,
            AggregateOpIr::Product => ReductionOp::Product,
            AggregateOpIr::Min => ReductionOp::Min,
            AggregateOpIr::Max => ReductionOp::Max,
            AggregateOpIr::Mean => ReductionOp::Mean,
            AggregateOpIr::Count => ReductionOp::Count,
            // Boolean aggregates need special runtime support
            AggregateOpIr::Any | AggregateOpIr::All | AggregateOpIr::None => {
                panic!(
                    "Boolean aggregate {:?} not yet supported in DAG compilation",
                    op
                )
            }
        }
    }

    /// Extract all aggregate operations from an expression recursively.
    ///
    /// Returns a vector of `AggregateInfo` containing the entity, operation, and body
    /// for each aggregate found in the expression tree.
    fn extract_aggregates(expr: &CompiledExpr) -> Vec<AggregateInfo> {
        let mut aggregates = Vec::new();
        Self::extract_aggregates_recursive(expr, &mut aggregates);
        aggregates
    }

    fn extract_aggregates_recursive(expr: &CompiledExpr, aggregates: &mut Vec<AggregateInfo>) {
        match expr {
            CompiledExpr::Aggregate { op, entity, body } => {
                aggregates.push(AggregateInfo {
                    entity_id: FoundationEntityId(entity.0.clone()),
                    op: *op,
                    body: (**body).clone(),
                });
                // Also check inside the body for nested aggregates
                Self::extract_aggregates_recursive(body, aggregates);
            }
            CompiledExpr::Binary { left, right, .. } => {
                Self::extract_aggregates_recursive(left, aggregates);
                Self::extract_aggregates_recursive(right, aggregates);
            }
            CompiledExpr::Unary { operand, .. } => {
                Self::extract_aggregates_recursive(operand, aggregates);
            }
            CompiledExpr::Let { value, body, .. } => {
                Self::extract_aggregates_recursive(value, aggregates);
                Self::extract_aggregates_recursive(body, aggregates);
            }
            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::extract_aggregates_recursive(condition, aggregates);
                Self::extract_aggregates_recursive(then_branch, aggregates);
                Self::extract_aggregates_recursive(else_branch, aggregates);
            }
            CompiledExpr::Call { args, .. }
            | CompiledExpr::KernelCall { args, .. }
            | CompiledExpr::DtRobustCall { args, .. } => {
                for arg in args {
                    Self::extract_aggregates_recursive(arg, aggregates);
                }
            }
            CompiledExpr::FieldAccess { object, .. } => {
                Self::extract_aggregates_recursive(object, aggregates);
            }
            CompiledExpr::Other { body, .. } | CompiledExpr::Pairs { body, .. } => {
                Self::extract_aggregates_recursive(body, aggregates);
            }
            CompiledExpr::Filter {
                predicate, body, ..
            } => {
                Self::extract_aggregates_recursive(predicate, aggregates);
                Self::extract_aggregates_recursive(body, aggregates);
            }
            CompiledExpr::First { predicate, .. } => {
                Self::extract_aggregates_recursive(predicate, aggregates);
            }
            CompiledExpr::Nearest { position, .. } => {
                Self::extract_aggregates_recursive(position, aggregates);
            }
            CompiledExpr::Within {
                position,
                radius,
                body,
                ..
            } => {
                Self::extract_aggregates_recursive(position, aggregates);
                Self::extract_aggregates_recursive(radius, aggregates);
                Self::extract_aggregates_recursive(body, aggregates);
            }
            CompiledExpr::EmitSignal { value, .. } => {
                Self::extract_aggregates_recursive(value, aggregates);
            }
            // Leaf nodes - no recursion needed
            CompiledExpr::Literal(_)
            | CompiledExpr::Prev
            | CompiledExpr::DtRaw
            | CompiledExpr::SimTime
            | CompiledExpr::Collected
            | CompiledExpr::SelfField(_)
            | CompiledExpr::EntityAccess { .. }
            | CompiledExpr::Signal(_)
            | CompiledExpr::Config(_)
            | CompiledExpr::Const(_)
            | CompiledExpr::Local(_)
            | CompiledExpr::Payload
            | CompiledExpr::PayloadField(_) => {}
        }
    }

    fn compile_era(&mut self, _era_id: &EraId) -> Result<EraDags, CompileError> {
        let mut era_dags = EraDags::default();

        // Collect active strata for this era
        let active_strata: Vec<&StratumId> = self.world.strata.keys().collect();

        // Build DAGs per (phase, stratum)
        for stratum_id in active_strata {
            // Collect phase: operators
            if let Some(dag) = self.build_collect_dag(stratum_id)? {
                era_dags.insert(dag);
            }

            // Resolve phase: signals
            if let Some(dag) = self.build_resolve_dag(stratum_id)? {
                era_dags.insert(dag);
            }

            // Fracture phase: fracture detectors
            if let Some(dag) = self.build_fracture_dag(stratum_id)? {
                era_dags.insert(dag);
            }

            // Measure phase: fields
            if let Some(dag) = self.build_measure_dag(stratum_id)? {
                era_dags.insert(dag);
            }
        }

        Ok(era_dags)
    }

    fn build_collect_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Collect, (*stratum_id).clone());

        for (op_id, operator) in &self.world.operators {
            if operator.stratum != *stratum_id {
                continue;
            }

            if operator.phase != OperatorPhaseIr::Collect {
                continue;
            }

            let node = DagNode {
                id: NodeId(format!("op.{}", op_id.0)),
                reads: operator
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: None, // Operators don't write signals directly
                kind: NodeKind::OperatorCollect {
                    operator_idx: self.operator_indices[&op_id.0],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    fn build_resolve_dag(
        &mut self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        // First, check if any signals in this stratum use aggregates
        let mut has_aggregates = false;
        for (_, signal) in &self.world.signals {
            if signal.stratum != *stratum_id {
                continue;
            }
            if let Some(ref resolve) = signal.resolve {
                if !Self::extract_aggregates(resolve).is_empty() {
                    has_aggregates = true;
                    break;
                }
            }
        }

        // Use BarrierDagBuilder if aggregates are present, otherwise use DagBuilder
        if has_aggregates {
            self.build_resolve_dag_with_aggregates(stratum_id)
        } else {
            self.build_resolve_dag_simple(stratum_id)
        }
    }

    /// Build resolve DAG without aggregate handling (simpler path).
    fn build_resolve_dag_simple(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Resolve, (*stratum_id).clone());

        // Add regular signal resolve nodes
        for (signal_id, signal) in &self.world.signals {
            if signal.stratum != *stratum_id {
                continue;
            }

            // Skip signals without resolve expressions
            let Some(ref resolve_expr) = signal.resolve else {
                continue;
            };

            // Skip signals with entity expressions (require EntityExecutor, not bytecode)
            if contains_entity_expression(resolve_expr) {
                continue;
            }

            // Also check component expressions for vector signals
            if let Some(ref components) = signal.resolve_components {
                if components.iter().any(contains_entity_expression) {
                    continue;
                }
            }

            // Per docs/execution/phases.md: Resolve "reads resolved signals from the
            // previous tick" and "must be deterministic and order-independent".
            // Signal references always read previous tick values, so there are no
            // same-tick dependencies between signals. Use empty reads to allow
            // parallel resolution.
            let node = DagNode {
                id: NodeId(format!("sig.{}", signal_id.0)),
                reads: std::collections::HashSet::new(), // No same-tick dependencies
                writes: Some(continuum_runtime::SignalId(signal_id.0.clone())),
                kind: NodeKind::SignalResolve {
                    signal: continuum_runtime::SignalId(signal_id.0.clone()),
                    resolver_idx: self.resolver_indices[signal_id],
                },
            };
            builder.add_node(node);
        }

        // Add member signal resolve nodes
        for (member_id, member) in &self.world.members {
            if member.stratum != *stratum_id {
                continue;
            }

            // Skip members without resolve expressions
            let Some(ref resolve_expr) = member.resolve else {
                continue;
            };

            // Skip members with entity expressions (require EntityExecutor, not bytecode)
            if contains_entity_expression(resolve_expr) {
                continue;
            }

            let member_signal_id = MemberSignalId::new(
                continuum_runtime::types::EntityId(member.entity_id.0.clone()),
                member.signal_name.clone(),
            );

            // Per docs: signal references read previous tick values, no same-tick deps
            let node = DagNode {
                id: NodeId(format!("member.{}", member_id.0)),
                reads: std::collections::HashSet::new(), // No same-tick dependencies
                writes: None, // Member signals don't write to global signal namespace
                kind: NodeKind::MemberSignalResolve {
                    member_signal: member_signal_id,
                    kernel_idx: self.member_indices[member_id],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    /// Build resolve DAG with aggregate barrier handling.
    ///
    /// This uses `BarrierDagBuilder` to properly sequence:
    /// 1. Member signal resolution (all instances)
    /// 2. Aggregate barriers (population reduction)
    /// 3. Signal resolution (may depend on aggregate outputs)
    fn build_resolve_dag_with_aggregates(
        &mut self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = BarrierDagBuilder::new(Phase::Resolve, (*stratum_id).clone());

        // Track which member signals need to be added
        let mut member_signals_added: std::collections::HashSet<MemberSignalId> =
            std::collections::HashSet::new();

        // First pass: collect all aggregates and the member signals they depend on
        let mut signal_aggregates: Vec<(SignalId, Vec<AggregateInfo>)> = Vec::new();

        for (signal_id, signal) in &self.world.signals {
            if signal.stratum != *stratum_id {
                continue;
            }

            if let Some(ref resolve) = signal.resolve {
                let aggregates = Self::extract_aggregates(resolve);
                if !aggregates.is_empty() {
                    signal_aggregates.push((signal_id.clone(), aggregates));
                }
            }
        }

        // Add member signal resolve nodes for members in this stratum
        for (member_id, member) in &self.world.members {
            if member.stratum != *stratum_id {
                continue;
            }

            // Skip members without resolve expressions
            let Some(ref resolve_expr) = member.resolve else {
                continue;
            };

            // Skip members with entity expressions (require EntityExecutor, not bytecode)
            if contains_entity_expression(resolve_expr) {
                continue;
            }

            let member_signal_id = MemberSignalId::new(
                RuntimeEntityId(member.entity_id.0.clone()),
                member.signal_name.clone(),
            );

            builder.add_member_signal_resolve(
                member_signal_id.clone(),
                self.member_indices[member_id],
            );
            member_signals_added.insert(member_signal_id);
        }

        // Add aggregate barriers for each aggregate found
        let mut aggregate_counter = 0usize;

        for (signal_id, aggregates) in &signal_aggregates {
            for (agg_idx, agg_info) in aggregates.iter().enumerate() {
                // Extract the member signal name from the aggregate body
                // For count with literal body, find any member signal of the entity
                // For other aggregates, expect a self.X reference
                let member_name = match (&agg_info.op, &agg_info.body) {
                    (AggregateOpIr::Count, CompiledExpr::Literal(_)) => {
                        // Count with literal body - find any member of this entity
                        self.find_any_member_of_entity(&agg_info.entity_id)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Entity '{}' has no member signals for count aggregate",
                                    agg_info.entity_id.0
                                )
                            })
                    }
                    _ => Self::extract_member_name_from_body(&agg_info.body).unwrap_or_else(|| {
                        panic!(
                            "Aggregate body must be a simple self.X reference, got: {:?}",
                            agg_info.body
                        )
                    }),
                };

                let member_signal_id = MemberSignalId::new(
                    RuntimeEntityId(agg_info.entity_id.0.clone()),
                    member_name.clone(),
                );

                // Create the aggregate ID
                let agg_id = format!("agg.{}.{}", signal_id.0, agg_idx);

                // The output signal is the signal that contains this aggregate
                // For now, assume the entire resolve expression is just the aggregate
                let output_signal = continuum_runtime::SignalId(signal_id.0.clone());

                let barrier = AggregateBarrier {
                    id: NodeId(agg_id.clone()),
                    member_signal: member_signal_id,
                    reduction_op: Self::aggregate_op_to_reduction_op(&agg_info.op),
                    output_signal,
                    aggregate_idx: aggregate_counter,
                };

                builder.add_aggregate_barrier(barrier);

                // Track this aggregate's index
                self.aggregate_indices.insert(agg_id, aggregate_counter);
                aggregate_counter += 1;
            }
        }

        // Add regular signal resolve nodes (those without aggregates or entity expressions)
        for (signal_id, signal) in &self.world.signals {
            if signal.stratum != *stratum_id {
                continue;
            }

            // Skip signals without resolve expressions
            let Some(ref resolve) = signal.resolve else {
                continue;
            };

            // Skip signals that ARE aggregates (they're handled by aggregate barriers)
            if !Self::extract_aggregates(resolve).is_empty() {
                continue;
            }

            // Skip signals with entity expressions (require EntityExecutor, not bytecode)
            if contains_entity_expression(resolve) {
                continue;
            }

            // Also check component expressions for vector signals
            if let Some(ref components) = signal.resolve_components {
                if components.iter().any(contains_entity_expression) {
                    continue;
                }
            }

            // Signal references read from the PREVIOUS tick's values, so they
            // don't create same-tick dependencies. Pass empty reads to avoid
            // creating DAG edges between signals.
            // (signal.reads is informational only - for analysis, not scheduling)
            builder.add_signal_resolve(
                continuum_runtime::SignalId(signal_id.0.clone()),
                self.resolver_indices[signal_id],
                &[],
            );
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    /// Extract the member signal name from an aggregate body expression.
    ///
    /// Recursively searches for `self.X` references in the expression tree.
    /// Returns the first member signal name found.
    fn extract_member_name_from_body(body: &CompiledExpr) -> Option<String> {
        match body {
            CompiledExpr::SelfField(field_name) => Some(field_name.clone()),
            // Recurse into binary expressions
            CompiledExpr::Binary { left, right, .. } => {
                Self::extract_member_name_from_body(left)
                    .or_else(|| Self::extract_member_name_from_body(right))
            }
            // Recurse into unary expressions
            CompiledExpr::Unary { operand, .. } => Self::extract_member_name_from_body(operand),
            // Recurse into conditionals
            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => Self::extract_member_name_from_body(condition)
                .or_else(|| Self::extract_member_name_from_body(then_branch))
                .or_else(|| Self::extract_member_name_from_body(else_branch)),
            // Recurse into let expressions
            CompiledExpr::Let { value, body, .. } => Self::extract_member_name_from_body(value)
                .or_else(|| Self::extract_member_name_from_body(body)),
            // Recurse into function calls
            CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
                args.iter().find_map(Self::extract_member_name_from_body)
            }
            // Recurse into field access
            CompiledExpr::FieldAccess { object, .. } => Self::extract_member_name_from_body(object),
            // Leaf nodes that don't contain member references
            _ => None,
        }
    }

    /// Find any member signal belonging to the given entity.
    ///
    /// This is used for count aggregates where we need a member signal to iterate
    /// over entities but don't need a specific one.
    fn find_any_member_of_entity(&self, entity_id: &FoundationEntityId) -> Option<String> {
        for (_member_id, member) in &self.world.members {
            if member.entity_id.0 == entity_id.0 {
                return Some(member.signal_name.clone());
            }
        }
        None
    }

    fn build_fracture_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Fracture, (*stratum_id).clone());

        // Fractures aren't bound to a specific stratum in IR
        // Add all fractures to the first stratum only (to avoid duplicating execution)
        let first_stratum = self.world.strata.keys().next();
        if first_stratum != Some(stratum_id) {
            let dag = builder.build()?;
            return Ok(if dag.is_empty() { None } else { Some(dag) });
        }

        for (fracture_id, fracture) in &self.world.fractures {
            let node = DagNode {
                id: NodeId(format!("frac.{}", fracture_id.0)),
                reads: fracture
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: None,
                kind: NodeKind::Fracture {
                    fracture_idx: self.fracture_indices[&fracture_id.0],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }

    fn build_measure_dag(
        &self,
        stratum_id: &StratumId,
    ) -> Result<Option<ExecutableDag>, CompileError> {
        let mut builder = DagBuilder::new(Phase::Measure, (*stratum_id).clone());

        // Fields with measure expressions become OperatorMeasure nodes
        // Fields without measure expressions would be FieldEmit nodes (for dependency tracking only)
        // Fields with entity expressions are skipped - they require EntityExecutor
        for (field_id, field) in &self.world.fields {
            if field.stratum != *stratum_id {
                continue;
            }

            // Only create nodes for fields that have bytecode-compatible measure expressions
            if let Some(ref measure) = field.measure {
                if !contains_entity_expression(measure) {
                    let node = DagNode {
                        id: NodeId(format!("field.{}", field_id.0)),
                        reads: field
                            .reads
                            .iter()
                            .map(|s| continuum_runtime::SignalId(s.0.clone()))
                            .collect(),
                        writes: None,
                        kind: NodeKind::OperatorMeasure {
                            operator_idx: self.field_indices[&field_id.0],
                        },
                    };
                    builder.add_node(node);
                }
            }
        }

        // Also add measure-phase operators
        for (op_id, operator) in &self.world.operators {
            if operator.stratum != *stratum_id {
                continue;
            }

            if operator.phase != OperatorPhaseIr::Measure {
                continue;
            }

            let node = DagNode {
                id: NodeId(format!("op.{}", op_id.0)),
                reads: operator
                    .reads
                    .iter()
                    .map(|s| continuum_runtime::SignalId(s.0.clone()))
                    .collect(),
                writes: None,
                kind: NodeKind::OperatorMeasure {
                    operator_idx: self.operator_indices[&op_id.0],
                },
            };
            builder.add_node(node);
        }

        let dag = builder.build()?;
        if dag.is_empty() {
            Ok(None)
        } else {
            Ok(Some(dag))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower, CompiledWorld};
    use continuum_dsl::parse;

    fn parse_and_lower(src: &str) -> CompiledWorld {
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        lower(&unit.unwrap()).unwrap()
    }

    #[test]
    fn test_compile_empty() {
        let world = CompiledWorld {
            constants: IndexMap::new(),
            config: IndexMap::new(),
            functions: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
            entities: IndexMap::new(),
            members: IndexMap::new(),
            chronicles: IndexMap::new(),
            types: IndexMap::new(),
        };

        let result = compile(&world).unwrap();
        assert!(result.dags.is_empty());
    }

    #[test]
    fn test_compile_simple_signal() {
        let src = r#"
            strata.terra {
                : title("Terra")
            }

            era.hadean {
                : initial
            }

            signal.terra.temp {
                : strata(terra)
                resolve { prev + 1.0 }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Should have one era
        assert_eq!(result.dags.era_count(), 1);

        // Signal should have resolver index
        let sig_id = SignalId::from("terra.temp");
        assert!(result.resolver_indices.contains_key(&sig_id));
    }

    #[test]
    fn test_compile_signal_dependencies() {
        let src = r#"
            strata.terra {}

            era.main {
                : initial
            }

            signal.terra.a {
                : strata(terra)
                resolve { 1.0 }
            }

            signal.terra.b {
                : strata(terra)
                resolve { signal.terra.a * 2.0 }
            }

            signal.terra.c {
                : strata(terra)
                resolve { signal.terra.b + signal.terra.a }
            }
        "#;

        let world = parse_and_lower(src);
        let _result = compile(&world).unwrap();

        // Check that signal c depends on both a and b
        let sig_c = world.signals.get(&SignalId::from("terra.c")).unwrap();
        assert_eq!(sig_c.reads.len(), 2);
    }

    #[test]
    fn test_compile_member_signal() {
        let src = r#"
            strata.human {}

            era.main {
                : initial
            }

            entity.human.person {
                : count(1..100)
            }

            member.human.person.age {
                : Scalar
                : strata(human)
                resolve { prev + 1.0 }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Should have one era
        assert_eq!(result.dags.era_count(), 1);

        // Member should have index
        let member_id = continuum_foundation::MemberId::from("human.person.age");
        assert!(result.member_indices.contains_key(&member_id));
        assert_eq!(result.member_indices[&member_id], 0);
    }

    #[test]
    fn test_compile_multiple_member_signals_same_entity() {
        let src = r#"
            strata.stellar {}

            era.main {
                : initial
            }

            entity.stellar.moon {
                : count(1..10)
            }

            member.stellar.moon.mass {
                : Scalar<kg>
                : strata(stellar)
                resolve { prev }
            }

            member.stellar.moon.radius {
                : Scalar<m>
                : strata(stellar)
                resolve { prev * 1.01 }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Both members should have indices
        assert_eq!(result.member_indices.len(), 2);

        let mass_id = continuum_foundation::MemberId::from("stellar.moon.mass");
        let radius_id = continuum_foundation::MemberId::from("stellar.moon.radius");

        assert!(result.member_indices.contains_key(&mass_id));
        assert!(result.member_indices.contains_key(&radius_id));
    }

    #[test]
    fn test_compile_aggregate_sum() {
        let src = r#"
            strata.stellar {}

            era.main {
                : initial
            }

            entity.stellar.moon {
                : count(1..10)
            }

            member.stellar.moon.mass {
                : Scalar<kg>
                : strata(stellar)
                resolve { prev }
            }

            signal.stellar.total_mass {
                : strata(stellar)
                resolve { sum(entity.stellar.moon, self.mass) }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Should have one era
        assert_eq!(result.dags.era_count(), 1);

        // Member should have index
        let member_id = continuum_foundation::MemberId::from("stellar.moon.mass");
        assert!(result.member_indices.contains_key(&member_id));

        // Aggregate should have been assigned an index
        assert!(!result.aggregate_indices.is_empty());
        assert!(result.aggregate_indices.contains_key("agg.stellar.total_mass.0"));
    }

    #[test]
    fn test_compile_aggregate_count() {
        // Note: count(entity.X) is a special syntax that just counts instances
        // It doesn't take a body/predicate - the body is implicitly "1"
        // For count aggregates, we need at least one member signal to iterate over
        let src = r#"
            strata.human {}

            era.main {
                : initial
            }

            entity.human.person {
                : count(1..100)
            }

            member.human.person.age {
                : Scalar<s>
                : strata(human)
                resolve { prev }
            }

            signal.human.person_count {
                : strata(human)
                resolve { count(entity.human.person) }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Should have aggregates
        assert!(!result.aggregate_indices.is_empty());
        assert!(result.aggregate_indices.contains_key("agg.human.person_count.0"));
    }

    #[test]
    fn test_compile_multiple_aggregates_in_stratum() {
        let src = r#"
            strata.stellar {}

            era.main {
                : initial
            }

            entity.stellar.planet {
                : count(1..20)
            }

            member.stellar.planet.mass {
                : Scalar<kg>
                : strata(stellar)
                resolve { prev }
            }

            member.stellar.planet.radius {
                : Scalar<m>
                : strata(stellar)
                resolve { prev }
            }

            signal.stellar.total_mass {
                : strata(stellar)
                resolve { sum(entity.stellar.planet, self.mass) }
            }

            signal.stellar.max_radius {
                : strata(stellar)
                resolve { max(entity.stellar.planet, self.radius) }
            }
        "#;

        let world = parse_and_lower(src);
        let result = compile(&world).unwrap();

        // Should have two aggregates
        assert_eq!(result.aggregate_indices.len(), 2);
        assert!(result.aggregate_indices.contains_key("agg.stellar.total_mass.0"));
        assert!(result.aggregate_indices.contains_key("agg.stellar.max_radius.0"));

        // Should have two member signals
        assert_eq!(result.member_indices.len(), 2);
    }
}
