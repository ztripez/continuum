//! L3 lane kernel: Sub-DAG execution for complex inter-member dependencies.
//!
//! This module implements the L3 lowering strategy for member signals,
//! which builds a dependency DAG across member signals and executes them
//! per-entity with internal sub-DAG scheduling.
//!
//! # When to Use L3
//!
//! L3 is optimal when:
//! - Small population (< 2k) - parallelizing over instances has limited benefit
//! - Complex inter-member dependencies - many member signals that read each other
//! - Heavy per-member computation - individual resolvers are expensive
//!
//! # Architecture
//!
//! ```text
//! Entity Instance #42:
//!   Level 0: [visible_threats, visible_resources]  ← parallel within level
//!   Level 1: [threat_assessment]
//!   Level 2: [goal_priority]
//!   Level 3: [current_action]
//! ```
//!
//! # Hybrid L1+L3
//!
//! For medium populations (2k-10k), the hybrid strategy combines:
//! - Outer: parallel over entity instances (L1)
//! - Inner: sub-DAG per entity (L3)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;
use tracing::{debug, instrument, trace};

use crate::soa_storage::PopulationStorage;
use crate::storage::SignalStorage;
use crate::types::Dt;
use crate::vectorized::{EntityIndex, MemberSignalId};

use super::lane_kernel::{LaneKernel, LaneKernelError, LaneKernelResult, LoweringStrategy};

// ============================================================================
// Member DAG
// ============================================================================

/// A dependency edge between member signals.
///
/// Represents that `from` must be resolved before `to` because `to` reads `from`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberEdge {
    /// The member signal that is read (dependency)
    pub from: MemberSignalId,
    /// The member signal that reads `from`
    pub to: MemberSignalId,
}

impl MemberEdge {
    /// Create a new edge from dependency to dependent.
    pub fn new(from: MemberSignalId, to: MemberSignalId) -> Self {
        Self { from, to }
    }
}

/// Dependency DAG for member signals within an entity.
///
/// This structure represents the dependency graph across member signals,
/// enabling topological execution ordering for L3 strategy.
#[derive(Debug, Clone)]
pub struct MemberDag {
    /// All member signals in this DAG
    members: Vec<MemberSignalId>,
    /// Dependency edges (from → to means `to` depends on `from`)
    edges: Vec<MemberEdge>,
    /// Topological levels for parallel execution
    /// Members in the same level have no dependencies on each other
    levels: Vec<Vec<MemberSignalId>>,
}

impl MemberDag {
    /// Create a new empty MemberDag.
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            edges: Vec::new(),
            levels: Vec::new(),
        }
    }

    /// Add a member signal to the DAG.
    pub fn add_member(&mut self, member: MemberSignalId) {
        if !self.members.contains(&member) {
            self.members.push(member);
        }
    }

    /// Add a dependency edge: `to` reads `from`, so `from` must resolve first.
    pub fn add_dependency(&mut self, from: MemberSignalId, to: MemberSignalId) {
        // Ensure both members exist
        self.add_member(from.clone());
        self.add_member(to.clone());

        let edge = MemberEdge::new(from, to);
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Build the DAG with topological levels.
    ///
    /// Returns an error if there's a cycle in the dependencies.
    pub fn build(mut self) -> Result<Self, MemberDagError> {
        self.levels = self.compute_topological_levels()?;
        Ok(self)
    }

    /// Compute topological levels using Kahn's algorithm.
    ///
    /// Members with no dependencies go in level 0, members that only depend
    /// on level 0 go in level 1, etc.
    fn compute_topological_levels(&self) -> Result<Vec<Vec<MemberSignalId>>, MemberDagError> {
        if self.members.is_empty() {
            return Ok(Vec::new());
        }

        // Build adjacency and in-degree maps
        let mut in_degree: HashMap<&MemberSignalId, usize> = HashMap::new();
        let mut dependents: HashMap<&MemberSignalId, Vec<&MemberSignalId>> = HashMap::new();

        for member in &self.members {
            in_degree.insert(member, 0);
            dependents.insert(member, Vec::new());
        }

        for edge in &self.edges {
            *in_degree.get_mut(&edge.to).unwrap() += 1;
            dependents.get_mut(&edge.from).unwrap().push(&edge.to);
        }

        let mut levels: Vec<Vec<MemberSignalId>> = Vec::new();
        let mut remaining: HashSet<&MemberSignalId> = self.members.iter().collect();
        let mut processed_count = 0;

        // Process level by level
        while !remaining.is_empty() {
            // Find all nodes with in-degree 0
            let mut current_level: Vec<MemberSignalId> = remaining
                .iter()
                .filter(|m| in_degree[*m] == 0)
                .map(|m| (*m).clone())
                .collect();

            if current_level.is_empty() {
                // Cycle detected - no nodes with in-degree 0 but nodes remain
                let cycle_members: Vec<_> = remaining.iter().map(|m| m.to_string()).collect();
                return Err(MemberDagError::CycleDetected(cycle_members.join(", ")));
            }

            // Sort for determinism
            current_level.sort_by(|a, b| a.signal_name.cmp(&b.signal_name));

            // Remove processed nodes and update in-degrees
            for member in &current_level {
                remaining.remove(member);
                processed_count += 1;

                // Decrease in-degree of dependents
                if let Some(deps) = dependents.get(member) {
                    for dep in deps {
                        if let Some(degree) = in_degree.get_mut(dep) {
                            *degree = degree.saturating_sub(1);
                        }
                    }
                }
            }

            levels.push(current_level);
        }

        // Verify all nodes processed
        if processed_count != self.members.len() {
            return Err(MemberDagError::CycleDetected(
                "Not all members processed".to_string(),
            ));
        }

        Ok(levels)
    }

    /// Get the topological levels.
    pub fn levels(&self) -> &[Vec<MemberSignalId>] {
        &self.levels
    }

    /// Get all member signals.
    pub fn members(&self) -> &[MemberSignalId] {
        &self.members
    }

    /// Get all edges.
    pub fn edges(&self) -> &[MemberEdge] {
        &self.edges
    }

    /// Get number of members in the DAG.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get number of levels in the DAG.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

impl Default for MemberDag {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors during MemberDag construction.
#[derive(Debug, Clone)]
pub enum MemberDagError {
    /// Cycle detected in dependencies
    CycleDetected(String),
}

impl std::fmt::Display for MemberDagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemberDagError::CycleDetected(members) => {
                write!(f, "cycle detected in member dependencies: {}", members)
            }
        }
    }
}

impl std::error::Error for MemberDagError {}

// ============================================================================
// L3 Resolve Context
// ============================================================================

/// Context for L3 sub-DAG resolution, providing access to already-resolved
/// member values within the current entity.
pub struct L3ResolveContext<'a, T> {
    /// Previous tick's value for this member signal
    pub prev: T,
    /// Entity instance index
    pub index: EntityIndex,
    /// Read-only access to global signals
    pub signals: &'a SignalStorage,
    /// Already-resolved member values for this entity (current tick)
    /// Maps member signal name to its current value
    pub resolved_members: &'a HashMap<String, f64>,
    /// Already-resolved Vec3 members for this entity
    pub resolved_vec3_members: &'a HashMap<String, [f64; 3]>,
    /// Time step
    pub dt: Dt,
}

/// Typed L3 context for scalar members.
pub type ScalarL3ResolveContext<'a> = L3ResolveContext<'a, f64>;

/// Typed L3 context for Vec3 members.
pub type Vec3L3ResolveContext<'a> = L3ResolveContext<'a, [f64; 3]>;

// ============================================================================
// L3 Member Resolver
// ============================================================================

/// A resolver for a single member signal in the L3 DAG.
pub trait L3MemberResolver: Send + Sync {
    /// The value type this resolver produces.
    type Value: Clone + Send;

    /// Get the member signal ID this resolver handles.
    fn member_signal_id(&self) -> &MemberSignalId;

    /// Resolve a single entity instance.
    fn resolve_instance(
        &self,
        prev: Self::Value,
        index: EntityIndex,
        signals: &SignalStorage,
        resolved_scalars: &HashMap<String, f64>,
        resolved_vec3s: &HashMap<String, [f64; 3]>,
        dt: Dt,
    ) -> Self::Value;
}

/// Scalar L3 member resolver.
pub struct ScalarL3MemberResolver {
    member_signal_id: MemberSignalId,
    resolver: Arc<dyn Fn(&ScalarL3ResolveContext) -> f64 + Send + Sync>,
}

impl ScalarL3MemberResolver {
    /// Create a new scalar L3 resolver.
    pub fn new(
        member_signal_id: MemberSignalId,
        resolver: impl Fn(&ScalarL3ResolveContext) -> f64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            member_signal_id,
            resolver: Arc::new(resolver),
        }
    }
}

impl L3MemberResolver for ScalarL3MemberResolver {
    type Value = f64;

    fn member_signal_id(&self) -> &MemberSignalId {
        &self.member_signal_id
    }

    fn resolve_instance(
        &self,
        prev: f64,
        index: EntityIndex,
        signals: &SignalStorage,
        resolved_scalars: &HashMap<String, f64>,
        resolved_vec3s: &HashMap<String, [f64; 3]>,
        dt: Dt,
    ) -> f64 {
        let ctx = ScalarL3ResolveContext {
            prev,
            index,
            signals,
            resolved_members: resolved_scalars,
            resolved_vec3_members: resolved_vec3s,
            dt,
        };
        (self.resolver)(&ctx)
    }
}

/// Vec3 L3 member resolver.
pub struct Vec3L3MemberResolver {
    member_signal_id: MemberSignalId,
    resolver: Arc<dyn Fn(&Vec3L3ResolveContext) -> [f64; 3] + Send + Sync>,
}

impl Vec3L3MemberResolver {
    /// Create a new Vec3 L3 resolver.
    pub fn new(
        member_signal_id: MemberSignalId,
        resolver: impl Fn(&Vec3L3ResolveContext) -> [f64; 3] + Send + Sync + 'static,
    ) -> Self {
        Self {
            member_signal_id,
            resolver: Arc::new(resolver),
        }
    }
}

impl L3MemberResolver for Vec3L3MemberResolver {
    type Value = [f64; 3];

    fn member_signal_id(&self) -> &MemberSignalId {
        &self.member_signal_id
    }

    fn resolve_instance(
        &self,
        prev: [f64; 3],
        index: EntityIndex,
        signals: &SignalStorage,
        resolved_scalars: &HashMap<String, f64>,
        resolved_vec3s: &HashMap<String, [f64; 3]>,
        dt: Dt,
    ) -> [f64; 3] {
        let ctx = Vec3L3ResolveContext {
            prev,
            index,
            signals,
            resolved_members: resolved_scalars,
            resolved_vec3_members: resolved_vec3s,
            dt,
        };
        (self.resolver)(&ctx)
    }
}

// ============================================================================
// L3 Kernel
// ============================================================================

/// Type alias for scalar resolver in L3 kernel.
pub type ScalarL3ResolverFn = Arc<dyn Fn(&ScalarL3ResolveContext) -> f64 + Send + Sync>;

/// Type alias for Vec3 resolver in L3 kernel.
pub type Vec3L3ResolverFn = Arc<dyn Fn(&Vec3L3ResolveContext) -> [f64; 3] + Send + Sync>;

/// L3 lane kernel for executing a sub-DAG of member signals.
///
/// This kernel executes all member signals in the DAG for each entity instance,
/// respecting topological ordering within each instance.
pub struct L3Kernel {
    /// The "primary" member signal this kernel is associated with.
    /// In L3, multiple members are resolved together, but one is designated primary.
    primary_member: MemberSignalId,
    /// The member dependency DAG
    dag: MemberDag,
    /// Scalar resolvers keyed by signal name
    scalar_resolvers: HashMap<String, ScalarL3ResolverFn>,
    /// Vec3 resolvers keyed by signal name
    vec3_resolvers: HashMap<String, Vec3L3ResolverFn>,
    /// Expected population size
    population_hint: usize,
    /// Threshold for hybrid L1+L3 execution
    hybrid_threshold: usize,
}

impl L3Kernel {
    /// Create a new L3 kernel.
    pub fn new(primary_member: MemberSignalId, dag: MemberDag, population_hint: usize) -> Self {
        Self {
            primary_member,
            dag,
            scalar_resolvers: HashMap::new(),
            vec3_resolvers: HashMap::new(),
            population_hint,
            hybrid_threshold: 100, // Default: use hybrid above 100 instances
        }
    }

    /// Add a scalar resolver for a member signal.
    pub fn add_scalar_resolver(
        &mut self,
        signal_name: String,
        resolver: impl Fn(&ScalarL3ResolveContext) -> f64 + Send + Sync + 'static,
    ) {
        self.scalar_resolvers
            .insert(signal_name, Arc::new(resolver));
    }

    /// Add a Vec3 resolver for a member signal.
    pub fn add_vec3_resolver(
        &mut self,
        signal_name: String,
        resolver: impl Fn(&Vec3L3ResolveContext) -> [f64; 3] + Send + Sync + 'static,
    ) {
        self.vec3_resolvers.insert(signal_name, Arc::new(resolver));
    }

    /// Set the threshold for hybrid L1+L3 execution.
    pub fn with_hybrid_threshold(mut self, threshold: usize) -> Self {
        self.hybrid_threshold = threshold;
        self
    }

    /// Resolve all member signals for a single entity instance.
    fn resolve_entity(
        &self,
        entity_idx: usize,
        signals: &SignalStorage,
        population: &PopulationStorage,
        dt: Dt,
    ) -> Result<EntityResolveResult, LaneKernelError> {
        let mut resolved_scalars: HashMap<String, f64> = HashMap::new();
        let mut resolved_vec3s: HashMap<String, [f64; 3]> = HashMap::new();

        // Process levels sequentially, members within level can read from previous levels
        for level in self.dag.levels() {
            for member_id in level {
                let signal_name = &member_id.signal_name;

                // Try scalar resolver first
                if let Some(resolver) = self.scalar_resolvers.get(signal_name) {
                    let prev = population
                        .signals()
                        .prev_scalar_slice(signal_name)
                        .and_then(|slice| slice.get(entity_idx).copied())
                        .unwrap_or(0.0);

                    let ctx = ScalarL3ResolveContext {
                        prev,
                        index: EntityIndex(entity_idx),
                        signals,
                        resolved_members: &resolved_scalars,
                        resolved_vec3_members: &resolved_vec3s,
                        dt,
                    };
                    let result = resolver(&ctx);
                    resolved_scalars.insert(signal_name.clone(), result);
                }
                // Try Vec3 resolver
                else if let Some(resolver) = self.vec3_resolvers.get(signal_name) {
                    let prev = population
                        .signals()
                        .prev_vec3_slice(signal_name)
                        .and_then(|slice| slice.get(entity_idx).copied())
                        .unwrap_or([0.0, 0.0, 0.0]);

                    let ctx = Vec3L3ResolveContext {
                        prev,
                        index: EntityIndex(entity_idx),
                        signals,
                        resolved_members: &resolved_scalars,
                        resolved_vec3_members: &resolved_vec3s,
                        dt,
                    };
                    let result = resolver(&ctx);
                    resolved_vec3s.insert(signal_name.clone(), result);
                }
                // No resolver found - this is an error
                else {
                    return Err(LaneKernelError::ExecutionFailed(format!(
                        "no resolver for member signal: {}",
                        signal_name
                    )));
                }
            }
        }

        Ok(EntityResolveResult {
            scalars: resolved_scalars,
            vec3s: resolved_vec3s,
        })
    }

    /// Execute sequentially (pure L3).
    fn execute_sequential(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
        population_size: usize,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        trace!(population_size, "executing L3 kernel sequentially");

        // Resolve each entity
        let results: Vec<EntityResolveResult> = (0..population_size)
            .map(|idx| self.resolve_entity(idx, signals, population, dt))
            .collect::<Result<Vec<_>, _>>()?;

        // Write results back to population storage
        self.write_results(population, &results)?;

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: None,
        })
    }

    /// Execute with hybrid L1+L3 (parallel over entities, L3 per entity).
    fn execute_hybrid(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
        population_size: usize,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        trace!(population_size, "executing L3 kernel with hybrid strategy");

        // Parallel over entities, L3 sub-DAG per entity
        let results: Vec<EntityResolveResult> = (0..population_size)
            .into_par_iter()
            .map(|idx| self.resolve_entity(idx, signals, population, dt))
            .collect::<Result<Vec<_>, _>>()?;

        // Write results back (sequential, but fast)
        self.write_results(population, &results)?;

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: None,
        })
    }

    /// Write resolved results back to population storage.
    fn write_results(
        &self,
        population: &mut PopulationStorage,
        results: &[EntityResolveResult],
    ) -> Result<(), LaneKernelError> {
        let signals_mut = population.signals_mut();

        // Write scalar results
        for (signal_name, _) in &self.scalar_resolvers {
            if let Some(slice) = signals_mut.scalar_slice_mut(signal_name) {
                for (idx, result) in results.iter().enumerate() {
                    if let Some(&value) = result.scalars.get(signal_name) {
                        slice[idx] = value;
                    }
                }
            }
        }

        // Write Vec3 results
        for (signal_name, _) in &self.vec3_resolvers {
            if let Some(slice) = signals_mut.vec3_slice_mut(signal_name) {
                for (idx, result) in results.iter().enumerate() {
                    if let Some(&value) = result.vec3s.get(signal_name) {
                        slice[idx] = value;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Result of resolving all members for a single entity.
struct EntityResolveResult {
    scalars: HashMap<String, f64>,
    vec3s: HashMap<String, [f64; 3]>,
}

impl LaneKernel for L3Kernel {
    fn strategy(&self) -> LoweringStrategy {
        LoweringStrategy::SubDag
    }

    fn member_signal_id(&self) -> &MemberSignalId {
        &self.primary_member
    }

    fn population_hint(&self) -> usize {
        self.population_hint
    }

    #[instrument(skip_all, name = "l3_kernel", fields(
        member = %self.primary_member,
        population = self.population_hint,
        levels = self.dag.level_count(),
    ))]
    fn execute(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        let start = std::time::Instant::now();

        // Get population size from first scalar signal
        let population_size = self
            .scalar_resolvers
            .keys()
            .next()
            .and_then(|name| population.signals().prev_scalar_slice(name))
            .map(|s| s.len())
            .or_else(|| {
                self.vec3_resolvers
                    .keys()
                    .next()
                    .and_then(|name| population.signals().prev_vec3_slice(name))
                    .map(|s| s.len())
            })
            .unwrap_or(0);

        if population_size == 0 {
            return Ok(LaneKernelResult {
                instances_processed: 0,
                execution_ns: Some(0),
            });
        }

        // Choose execution strategy based on population size
        let mut result = if population_size <= self.hybrid_threshold {
            self.execute_sequential(signals, population, dt, population_size)?
        } else {
            self.execute_hybrid(signals, population, dt, population_size)?
        };

        result.execution_ns = Some(start.elapsed().as_nanos() as u64);
        debug!(
            population_size,
            elapsed_ns = result.execution_ns,
            "L3 kernel complete"
        );

        Ok(result)
    }
}

// ============================================================================
// Builder Pattern for L3 Kernel
// ============================================================================

/// Builder for constructing L3 kernels.
pub struct L3KernelBuilder {
    primary_member: Option<MemberSignalId>,
    dag: MemberDag,
    scalar_resolvers: HashMap<String, ScalarL3ResolverFn>,
    vec3_resolvers: HashMap<String, Vec3L3ResolverFn>,
    population_hint: usize,
    hybrid_threshold: usize,
}

impl L3KernelBuilder {
    /// Create a new L3 kernel builder.
    pub fn new() -> Self {
        Self {
            primary_member: None,
            dag: MemberDag::new(),
            scalar_resolvers: HashMap::new(),
            vec3_resolvers: HashMap::new(),
            population_hint: 100,
            hybrid_threshold: 100,
        }
    }

    /// Set the primary member signal.
    pub fn primary_member(mut self, member: MemberSignalId) -> Self {
        self.primary_member = Some(member);
        self
    }

    /// Add a scalar member signal with its resolver.
    pub fn add_scalar_member(
        mut self,
        member: MemberSignalId,
        resolver: impl Fn(&ScalarL3ResolveContext) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.dag.add_member(member.clone());
        self.scalar_resolvers
            .insert(member.signal_name.clone(), Arc::new(resolver));
        self
    }

    /// Add a Vec3 member signal with its resolver.
    pub fn add_vec3_member(
        mut self,
        member: MemberSignalId,
        resolver: impl Fn(&Vec3L3ResolveContext) -> [f64; 3] + Send + Sync + 'static,
    ) -> Self {
        self.dag.add_member(member.clone());
        self.vec3_resolvers
            .insert(member.signal_name.clone(), Arc::new(resolver));
        self
    }

    /// Add a dependency: `to` reads `from`.
    pub fn add_dependency(mut self, from: &MemberSignalId, to: &MemberSignalId) -> Self {
        self.dag.add_dependency(from.clone(), to.clone());
        self
    }

    /// Set the expected population size.
    pub fn population_hint(mut self, hint: usize) -> Self {
        self.population_hint = hint;
        self
    }

    /// Set the threshold for hybrid execution.
    pub fn hybrid_threshold(mut self, threshold: usize) -> Self {
        self.hybrid_threshold = threshold;
        self
    }

    /// Build the L3 kernel.
    pub fn build(self) -> Result<L3Kernel, MemberDagError> {
        let primary = self.primary_member.unwrap_or_else(|| {
            // Use first member as primary if not specified
            self.dag
                .members
                .first()
                .cloned()
                .expect("at least one member required")
        });

        let dag = self.dag.build()?;

        Ok(L3Kernel {
            primary_member: primary,
            dag,
            scalar_resolvers: self.scalar_resolvers,
            vec3_resolvers: self.vec3_resolvers,
            population_hint: self.population_hint,
            hybrid_threshold: self.hybrid_threshold,
        })
    }
}

impl Default for L3KernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soa_storage::ValueType;
    use crate::types::EntityId;

    fn make_member_signal_id(entity: &str, signal: &str) -> MemberSignalId {
        MemberSignalId::new(EntityId::from(entity), signal)
    }

    #[test]
    fn test_member_dag_empty() {
        let dag = MemberDag::new().build().unwrap();
        assert_eq!(dag.member_count(), 0);
        assert_eq!(dag.level_count(), 0);
    }

    #[test]
    fn test_member_dag_single_member() {
        let mut dag = MemberDag::new();
        let member_a = make_member_signal_id("entity", "a");
        dag.add_member(member_a.clone());

        let dag = dag.build().unwrap();

        assert_eq!(dag.member_count(), 1);
        assert_eq!(dag.level_count(), 1);
        assert_eq!(dag.levels()[0].len(), 1);
        assert_eq!(dag.levels()[0][0], member_a);
    }

    #[test]
    fn test_member_dag_independent_members() {
        let mut dag = MemberDag::new();
        let member_a = make_member_signal_id("entity", "a");
        let member_b = make_member_signal_id("entity", "b");

        dag.add_member(member_a.clone());
        dag.add_member(member_b.clone());

        let dag = dag.build().unwrap();

        // Both in same level (no dependencies)
        assert_eq!(dag.level_count(), 1);
        assert_eq!(dag.levels()[0].len(), 2);
    }

    #[test]
    fn test_member_dag_chain() {
        let mut dag = MemberDag::new();
        let member_a = make_member_signal_id("entity", "a");
        let member_b = make_member_signal_id("entity", "b");
        let member_c = make_member_signal_id("entity", "c");

        // Chain: A -> B -> C
        dag.add_dependency(member_a.clone(), member_b.clone());
        dag.add_dependency(member_b.clone(), member_c.clone());

        let dag = dag.build().unwrap();

        // Should have 3 levels
        assert_eq!(dag.level_count(), 3);
        assert_eq!(dag.levels()[0], vec![member_a]);
        assert_eq!(dag.levels()[1], vec![member_b]);
        assert_eq!(dag.levels()[2], vec![member_c]);
    }

    #[test]
    fn test_member_dag_diamond() {
        let mut dag = MemberDag::new();
        let a = make_member_signal_id("entity", "a");
        let b = make_member_signal_id("entity", "b");
        let c = make_member_signal_id("entity", "c");
        let d = make_member_signal_id("entity", "d");

        // Diamond: A -> B, A -> C, B -> D, C -> D
        dag.add_dependency(a.clone(), b.clone());
        dag.add_dependency(a.clone(), c.clone());
        dag.add_dependency(b.clone(), d.clone());
        dag.add_dependency(c.clone(), d.clone());

        let dag = dag.build().unwrap();

        // Level 0: A
        // Level 1: B, C (both only depend on A)
        // Level 2: D (depends on B and C)
        assert_eq!(dag.level_count(), 3);
        assert_eq!(dag.levels()[0].len(), 1); // A
        assert_eq!(dag.levels()[1].len(), 2); // B, C
        assert_eq!(dag.levels()[2].len(), 1); // D
    }

    #[test]
    fn test_member_dag_cycle_detection() {
        let mut dag = MemberDag::new();
        let a = make_member_signal_id("entity", "a");
        let b = make_member_signal_id("entity", "b");

        // Cycle: A -> B -> A
        dag.add_dependency(a.clone(), b.clone());
        dag.add_dependency(b.clone(), a.clone());

        let result = dag.build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MemberDagError::CycleDetected(_)));
    }

    #[test]
    fn test_l3_kernel_builder_basic() {
        let a = make_member_signal_id("entity", "a");
        let b = make_member_signal_id("entity", "b");

        let kernel = L3KernelBuilder::new()
            .add_scalar_member(a.clone(), |ctx| ctx.prev + 1.0)
            .add_scalar_member(b.clone(), |ctx| {
                let a_val = ctx.resolved_members.get("a").copied().unwrap_or(0.0);
                a_val * 2.0
            })
            .add_dependency(&a, &b)
            .primary_member(a.clone())
            .population_hint(100)
            .build()
            .unwrap();

        assert_eq!(kernel.strategy(), LoweringStrategy::SubDag);
        assert_eq!(kernel.member_signal_id(), &a);
        assert_eq!(kernel.population_hint(), 100);
        assert_eq!(kernel.dag.level_count(), 2);
    }

    #[test]
    fn test_l3_kernel_execution_chain() {
        let a = make_member_signal_id("test.entity", "a");
        let b = make_member_signal_id("test.entity", "b");

        let mut kernel = L3Kernel::new(a.clone(), {
            let mut dag = MemberDag::new();
            dag.add_dependency(a.clone(), b.clone());
            dag.build().unwrap()
        }, 5);

        // A: prev + 1
        kernel.add_scalar_resolver("a".to_string(), |ctx| ctx.prev + 1.0);
        // B: reads A, multiplies by 2
        kernel.add_scalar_resolver("b".to_string(), |ctx| {
            ctx.resolved_members.get("a").copied().unwrap_or(0.0) * 2.0
        });

        // Set up population
        let mut population = PopulationStorage::new("test.entity".into());
        population.register_signal("a".to_string(), ValueType::Scalar);
        population.register_signal("b".to_string(), ValueType::Scalar);

        for i in 0..5 {
            population.register_instance(format!("inst_{}", i));
        }
        population.finalize();

        // Initialize: A=10, B=0
        for i in 0..5 {
            population.set_current(
                &format!("inst_{}", i),
                "a",
                crate::types::Value::Scalar(10.0),
            );
            population.set_current(
                &format!("inst_{}", i),
                "b",
                crate::types::Value::Scalar(0.0),
            );
        }
        population.advance_tick();

        // Execute
        let signals = SignalStorage::default();
        let result = kernel.execute(&signals, &mut population, Dt(1.0)).unwrap();

        assert_eq!(result.instances_processed, 5);

        // Verify results: A = 10 + 1 = 11, B = 11 * 2 = 22
        for i in 0..5 {
            let a_val = population.get_current(&format!("inst_{}", i), "a");
            let b_val = population.get_current(&format!("inst_{}", i), "b");

            assert_eq!(a_val, Some(crate::types::Value::Scalar(11.0)));
            assert_eq!(b_val, Some(crate::types::Value::Scalar(22.0)));
        }
    }

    #[test]
    fn test_l3_kernel_execution_diamond() {
        let a = make_member_signal_id("test.entity", "a");
        let b = make_member_signal_id("test.entity", "b");
        let c = make_member_signal_id("test.entity", "c");
        let d = make_member_signal_id("test.entity", "d");

        let kernel = L3KernelBuilder::new()
            .add_scalar_member(a.clone(), |ctx| ctx.prev + 1.0) // A: prev + 1
            .add_scalar_member(b.clone(), |ctx| {
                ctx.resolved_members.get("a").copied().unwrap_or(0.0) + 10.0
            }) // B: A + 10
            .add_scalar_member(c.clone(), |ctx| {
                ctx.resolved_members.get("a").copied().unwrap_or(0.0) * 2.0
            }) // C: A * 2
            .add_scalar_member(d.clone(), |ctx| {
                let b_val = ctx.resolved_members.get("b").copied().unwrap_or(0.0);
                let c_val = ctx.resolved_members.get("c").copied().unwrap_or(0.0);
                b_val + c_val
            }) // D: B + C
            .add_dependency(&a, &b)
            .add_dependency(&a, &c)
            .add_dependency(&b, &d)
            .add_dependency(&c, &d)
            .primary_member(a.clone())
            .population_hint(3)
            .build()
            .unwrap();

        // Set up population
        let mut population = PopulationStorage::new("test.entity".into());
        for signal in ["a", "b", "c", "d"] {
            population.register_signal(signal.to_string(), ValueType::Scalar);
        }
        for i in 0..3 {
            population.register_instance(format!("inst_{}", i));
        }
        population.finalize();

        // Initialize: A=5, others=0
        for i in 0..3 {
            population.set_current(
                &format!("inst_{}", i),
                "a",
                crate::types::Value::Scalar(5.0),
            );
            for signal in ["b", "c", "d"] {
                population.set_current(
                    &format!("inst_{}", i),
                    signal,
                    crate::types::Value::Scalar(0.0),
                );
            }
        }
        population.advance_tick();

        // Execute
        let signals = SignalStorage::default();
        let result = kernel.execute(&signals, &mut population, Dt(1.0)).unwrap();

        assert_eq!(result.instances_processed, 3);

        // Verify: A=6, B=16, C=12, D=28
        for i in 0..3 {
            let a_val = population
                .get_current(&format!("inst_{}", i), "a")
                .unwrap()
                .as_scalar()
                .unwrap();
            let b_val = population
                .get_current(&format!("inst_{}", i), "b")
                .unwrap()
                .as_scalar()
                .unwrap();
            let c_val = population
                .get_current(&format!("inst_{}", i), "c")
                .unwrap()
                .as_scalar()
                .unwrap();
            let d_val = population
                .get_current(&format!("inst_{}", i), "d")
                .unwrap()
                .as_scalar()
                .unwrap();

            assert_eq!(a_val, 6.0, "A should be 5+1=6");
            assert_eq!(b_val, 16.0, "B should be 6+10=16");
            assert_eq!(c_val, 12.0, "C should be 6*2=12");
            assert_eq!(d_val, 28.0, "D should be 16+12=28");
        }
    }

    #[test]
    fn test_l3_kernel_deterministic() {
        let a = make_member_signal_id("test.entity", "a");
        let b = make_member_signal_id("test.entity", "b");

        let build_kernel = || {
            L3KernelBuilder::new()
                .add_scalar_member(a.clone(), |ctx| ctx.prev * 1.1 + ctx.index.0 as f64)
                .add_scalar_member(b.clone(), |ctx| {
                    ctx.resolved_members.get("a").copied().unwrap_or(0.0) + ctx.index.0 as f64
                })
                .add_dependency(&a, &b)
                .primary_member(a.clone())
                .population_hint(10)
                .build()
                .unwrap()
        };

        let kernel1 = build_kernel();
        let kernel2 = build_kernel();

        let setup_population = || {
            let mut population = PopulationStorage::new("test.entity".into());
            population.register_signal("a".to_string(), ValueType::Scalar);
            population.register_signal("b".to_string(), ValueType::Scalar);
            for i in 0..10 {
                population.register_instance(format!("inst_{}", i));
            }
            population.finalize();
            for i in 0..10 {
                population.set_current(
                    &format!("inst_{}", i),
                    "a",
                    crate::types::Value::Scalar(i as f64),
                );
                population.set_current(
                    &format!("inst_{}", i),
                    "b",
                    crate::types::Value::Scalar(0.0),
                );
            }
            population.advance_tick();
            population
        };

        let mut pop1 = setup_population();
        let mut pop2 = setup_population();
        let signals = SignalStorage::default();

        kernel1.execute(&signals, &mut pop1, Dt(1.0)).unwrap();
        kernel2.execute(&signals, &mut pop2, Dt(1.0)).unwrap();

        // Results must be identical
        for i in 0..10 {
            let a1 = pop1
                .get_current(&format!("inst_{}", i), "a")
                .unwrap()
                .as_scalar()
                .unwrap();
            let a2 = pop2
                .get_current(&format!("inst_{}", i), "a")
                .unwrap()
                .as_scalar()
                .unwrap();
            assert_eq!(a1, a2, "A values should be identical at index {}", i);

            let b1 = pop1
                .get_current(&format!("inst_{}", i), "b")
                .unwrap()
                .as_scalar()
                .unwrap();
            let b2 = pop2
                .get_current(&format!("inst_{}", i), "b")
                .unwrap()
                .as_scalar()
                .unwrap();
            assert_eq!(b1, b2, "B values should be identical at index {}", i);
        }
    }
}
