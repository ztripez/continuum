//! Operator Fusion Analysis.
//!
//! Compile-time analysis to identify operators that can be fused together,
//! reducing DAG node count, function call overhead, and enabling better
//! cache utilization.
//!
//! # Fusion Opportunities
//!
//! ## 1. Same-Signal Accumulation
//!
//! Multiple operators writing to the same signal input channel can be fused:
//!
//! ```cdsl
//! operator.heat_source {
//!     phase: collect
//!     signal.temperature <- 100.0
//! }
//!
//! operator.solar_input {
//!     phase: collect
//!     signal.temperature <- signal.solar_flux * 0.8
//! }
//! ```
//!
//! ## 2. Shared Input Dependencies
//!
//! Operators reading the same signals can share those reads:
//!
//! ```cdsl
//! operator.pressure_calc {
//!     phase: collect
//!     let density = signal.density
//!     signal.pressure <- density * GRAVITY * signal.depth
//! }
//!
//! operator.buoyancy_calc {
//!     phase: collect
//!     let density = signal.density
//!     signal.buoyancy <- (REF_DENSITY - density) * GRAVITY
//! }
//! ```
//!
//! ## 3. Kernel Call Batching
//!
//! Multiple operators calling the same kernel can be batched.
//!
//! # Safety
//!
//! Fusion is safe when:
//! - No write-write conflicts (both writing to same signal with ordering)
//! - No read-write conflicts (one reads what another writes within same level)
//! - Same phase and stratum

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use continuum_foundation::{OperatorId, SignalId, StratumId};

use crate::{CompiledExpr, CompiledOperator, CompiledWorld, OperatorPhaseIr};

// ============================================================================
// Operator Dependency Extraction
// ============================================================================

/// Dependencies extracted from an operator for fusion analysis.
#[derive(Debug, Clone)]
pub struct OperatorDeps {
    /// The operator's unique identifier.
    pub id: OperatorId,
    /// Stratum this operator belongs to.
    pub stratum: StratumId,
    /// Execution phase.
    pub phase: OperatorPhaseIr,
    /// Signals this operator reads.
    pub reads: HashSet<SignalId>,
    /// Signals this operator writes to (via accumulate/emit).
    pub writes: HashSet<SignalId>,
    /// Kernel functions called by this operator.
    pub kernel_calls: Vec<KernelCall>,
    /// Constants referenced.
    pub constants: HashSet<String>,
    /// Config values referenced.
    pub configs: HashSet<String>,
}

/// A kernel function call within an operator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelCall {
    /// Name of the kernel function.
    pub function: String,
    /// Number of arguments.
    pub arity: usize,
}

impl OperatorDeps {
    /// Extract dependencies from a compiled operator.
    pub fn from_operator(op: &CompiledOperator) -> Self {
        let mut deps = Self {
            id: op.id.clone(),
            stratum: op.stratum.clone(),
            phase: op.phase,
            reads: op.reads.iter().cloned().collect(),
            writes: HashSet::new(),
            kernel_calls: Vec::new(),
            constants: HashSet::new(),
            configs: HashSet::new(),
        };

        // Extract additional dependencies from body expression
        if let Some(ref body) = op.body {
            deps.extract_from_expr(body);
        }

        // Extract from assertions
        for assertion in &op.assertions {
            deps.extract_from_expr(&assertion.condition);
        }

        deps
    }

    /// Recursively extract dependencies from an expression.
    fn extract_from_expr(&mut self, expr: &CompiledExpr) {
        match expr {
            CompiledExpr::Signal(id) => {
                self.reads.insert(id.clone());
            }
            CompiledExpr::Const(name) => {
                self.constants.insert(name.clone());
            }
            CompiledExpr::Config(name) => {
                self.configs.insert(name.clone());
            }
            CompiledExpr::KernelCall { function, args } => {
                self.kernel_calls.push(KernelCall {
                    function: function.clone(),
                    arity: args.len(),
                });
                for arg in args {
                    self.extract_from_expr(arg);
                }
            }
            CompiledExpr::Call { args, .. } => {
                for arg in args {
                    self.extract_from_expr(arg);
                }
            }
            CompiledExpr::Binary { left, right, .. } => {
                self.extract_from_expr(left);
                self.extract_from_expr(right);
            }
            CompiledExpr::Unary { operand, .. } => {
                self.extract_from_expr(operand);
            }
            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.extract_from_expr(condition);
                self.extract_from_expr(then_branch);
                self.extract_from_expr(else_branch);
            }
            CompiledExpr::Let { value, body, .. } => {
                self.extract_from_expr(value);
                self.extract_from_expr(body);
            }
            CompiledExpr::DtRobustCall { args, .. } => {
                for arg in args {
                    self.extract_from_expr(arg);
                }
            }
            CompiledExpr::FieldAccess { object, .. } => {
                self.extract_from_expr(object);
            }
            CompiledExpr::Aggregate { body, .. } => {
                self.extract_from_expr(body);
            }
            CompiledExpr::Other { body, .. } => {
                self.extract_from_expr(body);
            }
            CompiledExpr::Pairs { body, .. } => {
                self.extract_from_expr(body);
            }
            CompiledExpr::Filter {
                predicate, body, ..
            } => {
                self.extract_from_expr(predicate);
                self.extract_from_expr(body);
            }
            CompiledExpr::First { predicate, .. } => {
                self.extract_from_expr(predicate);
            }
            CompiledExpr::Nearest { position, .. } => {
                self.extract_from_expr(position);
            }
            CompiledExpr::Within {
                position,
                radius,
                body,
                ..
            } => {
                self.extract_from_expr(position);
                self.extract_from_expr(radius);
                self.extract_from_expr(body);
            }
            CompiledExpr::EmitSignal { value, .. } => {
                self.extract_from_expr(value);
            }
            // These don't add dependencies to our tracking
            CompiledExpr::Literal(_)
            | CompiledExpr::Prev
            | CompiledExpr::DtRaw
            | CompiledExpr::Collected
            | CompiledExpr::Local(_)
            | CompiledExpr::SelfField(_)
            | CompiledExpr::EntityAccess { .. }
            | CompiledExpr::Payload
            | CompiledExpr::PayloadField(_) => {}
        }
    }
}

// ============================================================================
// Fusion Candidate Detection
// ============================================================================

/// A group of operators that can potentially be fused.
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Operators in this fusion group.
    pub operators: Vec<OperatorId>,
    /// The type of fusion opportunity.
    pub fusion_type: FusionType,
    /// Shared resources (signals for SharedReads, kernel names for SharedKernel).
    pub shared_resources: Vec<String>,
    /// Estimated benefit score (higher is better).
    pub benefit_score: f64,
}

/// The type of fusion opportunity identified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionType {
    /// Operators writing to the same signal input channel.
    SharedWrite,
    /// Operators reading many of the same signals.
    SharedReads,
    /// Operators calling the same kernel function.
    SharedKernel,
}

/// Detects fusion candidates from a set of operator dependencies.
pub struct FusionCandidateDetector {
    /// Minimum read overlap ratio to consider SharedReads fusion.
    pub min_read_overlap: f64,
    /// Minimum group size for fusion to be worthwhile.
    pub min_group_size: usize,
}

impl Default for FusionCandidateDetector {
    fn default() -> Self {
        Self {
            min_read_overlap: 0.5,
            min_group_size: 2,
        }
    }
}

impl FusionCandidateDetector {
    /// Create a new detector with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Find all fusion candidates from a collection of operator dependencies.
    pub fn find_candidates(&self, deps: &[OperatorDeps]) -> Vec<FusionGroup> {
        let mut groups = Vec::new();

        // Group by (phase, stratum) first - only operators in the same phase/stratum can fuse
        let mut by_phase_stratum: HashMap<(OperatorPhaseIr, StratumId), Vec<&OperatorDeps>> =
            HashMap::new();
        for dep in deps {
            by_phase_stratum
                .entry((dep.phase, dep.stratum.clone()))
                .or_default()
                .push(dep);
        }

        for (_, phase_stratum_deps) in by_phase_stratum {
            // Find shared write groups
            groups.extend(self.find_shared_write_groups(&phase_stratum_deps));

            // Find shared read groups
            groups.extend(self.find_shared_read_groups(&phase_stratum_deps));

            // Find shared kernel groups
            groups.extend(self.find_shared_kernel_groups(&phase_stratum_deps));
        }

        // Sort by benefit score (highest first)
        groups.sort_by(|a, b| b.benefit_score.partial_cmp(&a.benefit_score).unwrap());

        groups
    }

    /// Find operators that write to the same signal.
    fn find_shared_write_groups(&self, deps: &[&OperatorDeps]) -> Vec<FusionGroup> {
        let mut by_write: HashMap<&SignalId, Vec<&OperatorDeps>> = HashMap::new();

        for dep in deps {
            for write in &dep.writes {
                by_write.entry(write).or_default().push(dep);
            }
        }

        by_write
            .into_iter()
            .filter(|(_, ops)| ops.len() >= self.min_group_size)
            .map(|(signal, ops)| {
                let operator_ids: Vec<_> = ops.iter().map(|d| d.id.clone()).collect();
                let benefit = (operator_ids.len() - 1) as f64; // Eliminated calls

                FusionGroup {
                    operators: operator_ids,
                    fusion_type: FusionType::SharedWrite,
                    shared_resources: vec![signal.0.clone()],
                    benefit_score: benefit,
                }
            })
            .collect()
    }

    /// Find operators with high read overlap.
    fn find_shared_read_groups(&self, deps: &[&OperatorDeps]) -> Vec<FusionGroup> {
        let mut groups = Vec::new();

        // Compare all pairs for read overlap
        for i in 0..deps.len() {
            for j in (i + 1)..deps.len() {
                let overlap = self.compute_read_overlap(deps[i], deps[j]);
                if overlap >= self.min_read_overlap {
                    // Find shared signals
                    let shared: Vec<_> = deps[i]
                        .reads
                        .intersection(&deps[j].reads)
                        .map(|s| s.0.clone())
                        .collect();

                    groups.push(FusionGroup {
                        operators: vec![deps[i].id.clone(), deps[j].id.clone()],
                        fusion_type: FusionType::SharedReads,
                        shared_resources: shared.clone(),
                        benefit_score: shared.len() as f64 * overlap,
                    });
                }
            }
        }

        // Merge overlapping groups
        self.merge_overlapping_groups(groups, FusionType::SharedReads)
    }

    /// Find operators calling the same kernel.
    fn find_shared_kernel_groups(&self, deps: &[&OperatorDeps]) -> Vec<FusionGroup> {
        let mut by_kernel: HashMap<&KernelCall, Vec<&OperatorDeps>> = HashMap::new();

        for dep in deps {
            for kernel in &dep.kernel_calls {
                by_kernel.entry(kernel).or_default().push(dep);
            }
        }

        by_kernel
            .into_iter()
            .filter(|(_, ops)| ops.len() >= self.min_group_size)
            .map(|(kernel, ops)| {
                let operator_ids: Vec<_> = ops.iter().map(|d| d.id.clone()).collect();
                // Benefit: batched kernel dispatch saves setup overhead
                let benefit = (operator_ids.len() - 1) as f64 * 2.0;

                FusionGroup {
                    operators: operator_ids,
                    fusion_type: FusionType::SharedKernel,
                    shared_resources: vec![kernel.function.clone()],
                    benefit_score: benefit,
                }
            })
            .collect()
    }

    /// Compute the Jaccard similarity of reads between two operators.
    fn compute_read_overlap(&self, a: &OperatorDeps, b: &OperatorDeps) -> f64 {
        if a.reads.is_empty() || b.reads.is_empty() {
            return 0.0;
        }

        let intersection = a.reads.intersection(&b.reads).count();
        let union = a.reads.union(&b.reads).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Merge groups that share operators.
    fn merge_overlapping_groups(
        &self,
        groups: Vec<FusionGroup>,
        _fusion_type: FusionType,
    ) -> Vec<FusionGroup> {
        // Simple greedy merge: if two groups share an operator, merge them
        let mut merged = Vec::new();
        let mut used: HashSet<usize> = HashSet::new();

        for i in 0..groups.len() {
            if used.contains(&i) {
                continue;
            }

            let mut current = groups[i].clone();
            used.insert(i);

            // Find all overlapping groups
            for j in (i + 1)..groups.len() {
                if used.contains(&j) {
                    continue;
                }

                let has_overlap = groups[j]
                    .operators
                    .iter()
                    .any(|op| current.operators.contains(op));

                if has_overlap {
                    // Merge
                    for op in &groups[j].operators {
                        if !current.operators.contains(op) {
                            current.operators.push(op.clone());
                        }
                    }
                    for res in &groups[j].shared_resources {
                        if !current.shared_resources.contains(res) {
                            current.shared_resources.push(res.clone());
                        }
                    }
                    current.benefit_score += groups[j].benefit_score;
                    used.insert(j);
                }
            }

            if current.operators.len() >= self.min_group_size {
                merged.push(current);
            }
        }

        merged
    }
}

// ============================================================================
// Fusion Safety Validation
// ============================================================================

/// Validates whether a fusion group is safe to fuse.
#[derive(Debug, Clone)]
pub struct FusionValidator {
    /// All operator dependencies.
    deps_map: HashMap<OperatorId, OperatorDeps>,
}

/// Error when fusion is unsafe.
#[derive(Debug, Clone)]
pub enum FusionUnsafe {
    /// Write-write conflict: both operators write to the same signal.
    WriteWriteConflict {
        op_a: OperatorId,
        op_b: OperatorId,
        signal: SignalId,
    },
    /// Read-write conflict: one operator reads what another writes.
    ReadWriteConflict {
        reader: OperatorId,
        writer: OperatorId,
        signal: SignalId,
    },
    /// Phase mismatch: operators are in different phases.
    PhaseMismatch {
        op_a: OperatorId,
        op_b: OperatorId,
    },
    /// Stratum mismatch: operators are in different strata.
    StratumMismatch {
        op_a: OperatorId,
        op_b: OperatorId,
    },
}

impl FusionValidator {
    /// Create a validator from operator dependencies.
    pub fn new(deps: &[OperatorDeps]) -> Self {
        let deps_map = deps.iter().map(|d| (d.id.clone(), d.clone())).collect();
        Self { deps_map }
    }

    /// Validate that a fusion group is safe to fuse.
    pub fn validate(&self, group: &FusionGroup) -> Result<(), Vec<FusionUnsafe>> {
        let mut errors = Vec::new();

        // Get all deps for operators in the group
        let group_deps: Vec<_> = group
            .operators
            .iter()
            .filter_map(|id| self.deps_map.get(id))
            .collect();

        if group_deps.len() < 2 {
            return Ok(()); // Nothing to validate
        }

        // Check phase/stratum consistency
        let first = &group_deps[0];
        for dep in &group_deps[1..] {
            if dep.phase != first.phase {
                errors.push(FusionUnsafe::PhaseMismatch {
                    op_a: first.id.clone(),
                    op_b: dep.id.clone(),
                });
            }
            if dep.stratum != first.stratum {
                errors.push(FusionUnsafe::StratumMismatch {
                    op_a: first.id.clone(),
                    op_b: dep.id.clone(),
                });
            }
        }

        // Check for read-write and write-write conflicts
        for i in 0..group_deps.len() {
            for j in (i + 1)..group_deps.len() {
                let a = group_deps[i];
                let b = group_deps[j];

                // Write-write conflicts
                for signal in a.writes.intersection(&b.writes) {
                    errors.push(FusionUnsafe::WriteWriteConflict {
                        op_a: a.id.clone(),
                        op_b: b.id.clone(),
                        signal: signal.clone(),
                    });
                }

                // Read-write conflicts (a reads what b writes)
                for signal in a.reads.intersection(&b.writes) {
                    errors.push(FusionUnsafe::ReadWriteConflict {
                        reader: a.id.clone(),
                        writer: b.id.clone(),
                        signal: signal.clone(),
                    });
                }

                // Read-write conflicts (b reads what a writes)
                for signal in b.reads.intersection(&a.writes) {
                    errors.push(FusionUnsafe::ReadWriteConflict {
                        reader: b.id.clone(),
                        writer: a.id.clone(),
                        signal: signal.clone(),
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ============================================================================
// Fused Operator Generation
// ============================================================================

/// A fused operator combining multiple original operators.
#[derive(Debug, Clone)]
pub struct FusedOperator {
    /// Generated ID for the fused operator.
    pub id: OperatorId,
    /// Original operator IDs that were fused.
    pub original_ids: Vec<OperatorId>,
    /// Combined stratum (should be same for all).
    pub stratum: StratumId,
    /// Combined phase (should be same for all).
    pub phase: OperatorPhaseIr,
    /// Combined reads (union of all).
    pub combined_reads: HashSet<SignalId>,
    /// Combined writes (union of all).
    pub combined_writes: HashSet<SignalId>,
    /// Fused body expressions in execution order.
    pub bodies: Vec<CompiledExpr>,
}

impl FusedOperator {
    /// Create a fused operator from a fusion group and the original operators.
    pub fn from_group(
        group: &FusionGroup,
        operators: &IndexMap<OperatorId, CompiledOperator>,
    ) -> Option<Self> {
        if group.operators.len() < 2 {
            return None;
        }

        let mut original_ids = Vec::new();
        let mut combined_reads = HashSet::new();
        let combined_writes = HashSet::new();
        let mut bodies = Vec::new();
        let mut stratum = None;
        let mut phase = None;

        for op_id in &group.operators {
            if let Some(op) = operators.get(op_id) {
                original_ids.push(op_id.clone());

                // Set stratum/phase from first operator
                if stratum.is_none() {
                    stratum = Some(op.stratum.clone());
                    phase = Some(op.phase);
                }

                // Collect reads
                for read in &op.reads {
                    combined_reads.insert(read.clone());
                }

                // Collect body
                if let Some(ref body) = op.body {
                    bodies.push(body.clone());
                }
            }
        }

        // Generate fused ID
        let fused_id = OperatorId(format!(
            "fused.{}",
            original_ids
                .iter()
                .map(|id| id.0.replace('.', "_"))
                .collect::<Vec<_>>()
                .join("_")
        ));

        Some(FusedOperator {
            id: fused_id,
            original_ids,
            stratum: stratum?,
            phase: phase?,
            combined_reads,
            combined_writes,
            bodies,
        })
    }
}

// ============================================================================
// Fusion Cost Model
// ============================================================================

/// Cost model for deciding whether fusion is beneficial.
#[derive(Debug, Clone)]
pub struct FusionCostModel {
    /// Minimum benefit score to proceed with fusion.
    pub min_benefit: f64,
    /// Weight for shared read savings.
    pub shared_read_weight: f64,
    /// Weight for eliminated operator calls.
    pub call_elimination_weight: f64,
    /// Weight for kernel batching.
    pub kernel_batch_weight: f64,
    /// Maximum fused body size multiplier (relative to sum of individual).
    pub max_size_multiplier: f64,
}

impl Default for FusionCostModel {
    fn default() -> Self {
        Self {
            min_benefit: 1.0,
            shared_read_weight: 1.0,
            call_elimination_weight: 2.0,
            kernel_batch_weight: 3.0,
            max_size_multiplier: 2.0,
        }
    }
}

impl FusionCostModel {
    /// Decide whether a fusion group should be fused.
    pub fn should_fuse(&self, group: &FusionGroup, deps: &[OperatorDeps]) -> bool {
        let benefit = self.compute_benefit(group, deps);
        benefit >= self.min_benefit
    }

    /// Compute the benefit score for a fusion group.
    pub fn compute_benefit(&self, group: &FusionGroup, _deps: &[OperatorDeps]) -> f64 {
        let mut benefit = 0.0;

        // Call elimination: (n-1) fewer operator invocations
        let call_savings = (group.operators.len() as f64 - 1.0) * self.call_elimination_weight;
        benefit += call_savings;

        // Type-specific benefits
        match group.fusion_type {
            FusionType::SharedReads => {
                // Shared read savings
                let shared_count = group.shared_resources.len() as f64;
                benefit += shared_count * self.shared_read_weight;
            }
            FusionType::SharedKernel => {
                // Kernel batching benefit
                benefit += self.kernel_batch_weight;
            }
            FusionType::SharedWrite => {
                // Already counted in call savings
            }
        }

        benefit
    }
}

// ============================================================================
// Fusion Analysis Pipeline
// ============================================================================

/// Complete fusion analysis result.
#[derive(Debug, Clone)]
pub struct FusionAnalysis {
    /// Fusion groups that passed validation and cost check.
    pub fused_groups: Vec<FusionGroup>,
    /// Fused operators ready for use.
    pub fused_operators: Vec<FusedOperator>,
    /// Statistics about the analysis.
    pub stats: FusionStats,
}

/// Statistics from fusion analysis.
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Total operators analyzed.
    pub total_operators: usize,
    /// Candidate groups found.
    pub candidate_groups: usize,
    /// Groups that passed safety validation.
    pub safe_groups: usize,
    /// Groups that passed cost analysis.
    pub beneficial_groups: usize,
    /// Operators fused.
    pub operators_fused: usize,
    /// Fused operators created.
    pub fused_operators_created: usize,
    /// Estimated call savings.
    pub estimated_call_savings: usize,
    /// Estimated shared read savings.
    pub estimated_read_savings: usize,
}

/// Run the complete fusion analysis pipeline.
pub fn analyze_fusion(world: &CompiledWorld) -> FusionAnalysis {
    let mut stats = FusionStats::default();

    // Step 1: Extract dependencies from all operators
    let deps: Vec<_> = world
        .operators
        .values()
        .map(OperatorDeps::from_operator)
        .collect();
    stats.total_operators = deps.len();

    // Step 2: Find fusion candidates
    let detector = FusionCandidateDetector::default();
    let candidates = detector.find_candidates(&deps);
    stats.candidate_groups = candidates.len();

    // Step 3: Validate candidates
    let validator = FusionValidator::new(&deps);
    let safe_candidates: Vec<_> = candidates
        .into_iter()
        .filter(|g| validator.validate(g).is_ok())
        .collect();
    stats.safe_groups = safe_candidates.len();

    // Step 4: Apply cost model
    let cost_model = FusionCostModel::default();
    let beneficial_groups: Vec<_> = safe_candidates
        .into_iter()
        .filter(|g| cost_model.should_fuse(g, &deps))
        .collect();
    stats.beneficial_groups = beneficial_groups.len();

    // Step 5: Generate fused operators
    let mut fused_operators = Vec::new();
    let mut fused_op_ids: HashSet<OperatorId> = HashSet::new();

    for group in &beneficial_groups {
        // Skip if any operator is already part of another fusion
        if group.operators.iter().any(|id| fused_op_ids.contains(id)) {
            continue;
        }

        if let Some(fused) = FusedOperator::from_group(group, &world.operators) {
            for id in &fused.original_ids {
                fused_op_ids.insert(id.clone());
            }
            stats.operators_fused += fused.original_ids.len();
            fused_operators.push(fused);
        }
    }
    stats.fused_operators_created = fused_operators.len();

    // Calculate estimated savings
    stats.estimated_call_savings = stats.operators_fused.saturating_sub(stats.fused_operators_created);
    stats.estimated_read_savings = beneficial_groups
        .iter()
        .filter(|g| g.fusion_type == FusionType::SharedReads)
        .map(|g| g.shared_resources.len())
        .sum();

    FusionAnalysis {
        fused_groups: beneficial_groups,
        fused_operators,
        stats,
    }
}

// ============================================================================
// Display Implementation
// ============================================================================

impl std::fmt::Display for FusionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Operator Fusion Report:")?;
        writeln!(
            f,
            "  Analyzed {} operators, found {} candidate groups",
            self.total_operators, self.candidate_groups
        )?;
        writeln!(
            f,
            "  {} groups passed safety validation",
            self.safe_groups
        )?;
        writeln!(
            f,
            "  {} groups passed cost analysis",
            self.beneficial_groups
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  Fused {} operators into {} groups",
            self.operators_fused, self.fused_operators_created
        )?;
        writeln!(f)?;
        writeln!(f, "  Estimated savings:")?;
        writeln!(
            f,
            "    - {} fewer operator calls per tick",
            self.estimated_call_savings
        )?;
        writeln!(
            f,
            "    - {} shared signal reads",
            self.estimated_read_savings
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op_deps(
        id: &str,
        stratum: &str,
        phase: OperatorPhaseIr,
        reads: &[&str],
        writes: &[&str],
    ) -> OperatorDeps {
        OperatorDeps {
            id: OperatorId(id.to_string()),
            stratum: StratumId(stratum.to_string()),
            phase,
            reads: reads.iter().map(|s| SignalId(s.to_string())).collect(),
            writes: writes.iter().map(|s| SignalId(s.to_string())).collect(),
            kernel_calls: Vec::new(),
            constants: HashSet::new(),
            configs: HashSet::new(),
        }
    }

    #[test]
    fn test_find_shared_write_groups() {
        let deps = vec![
            make_op_deps("heat_source", "terra", OperatorPhaseIr::Collect, &[], &["temperature"]),
            make_op_deps("solar_input", "terra", OperatorPhaseIr::Collect, &["solar_flux"], &["temperature"]),
            make_op_deps("geothermal", "terra", OperatorPhaseIr::Collect, &[], &["temperature"]),
            make_op_deps("unrelated", "terra", OperatorPhaseIr::Collect, &[], &["pressure"]),
        ];

        let detector = FusionCandidateDetector::default();
        let candidates = detector.find_candidates(&deps);

        // Should find the temperature writers group
        let temp_group = candidates
            .iter()
            .find(|g| g.fusion_type == FusionType::SharedWrite && g.shared_resources.contains(&"temperature".to_string()));
        assert!(temp_group.is_some());
        assert_eq!(temp_group.unwrap().operators.len(), 3);
    }

    #[test]
    fn test_find_shared_read_groups() {
        let deps = vec![
            make_op_deps("pressure_calc", "terra", OperatorPhaseIr::Collect, &["density", "depth"], &["pressure"]),
            make_op_deps("buoyancy_calc", "terra", OperatorPhaseIr::Collect, &["density", "gravity"], &["buoyancy"]),
        ];

        let mut detector = FusionCandidateDetector::default();
        detector.min_read_overlap = 0.3; // Lower threshold for this test

        let candidates = detector.find_candidates(&deps);

        // Should find shared reads group (density is shared)
        let read_group = candidates
            .iter()
            .find(|g| g.fusion_type == FusionType::SharedReads);
        assert!(read_group.is_some());
    }

    #[test]
    fn test_fusion_validator_safe() {
        let deps = vec![
            make_op_deps("op_a", "terra", OperatorPhaseIr::Collect, &["x"], &["y"]),
            make_op_deps("op_b", "terra", OperatorPhaseIr::Collect, &["x"], &["z"]),
        ];

        let group = FusionGroup {
            operators: vec![OperatorId("op_a".to_string()), OperatorId("op_b".to_string())],
            fusion_type: FusionType::SharedReads,
            shared_resources: vec!["x".to_string()],
            benefit_score: 1.0,
        };

        let validator = FusionValidator::new(&deps);
        assert!(validator.validate(&group).is_ok());
    }

    #[test]
    fn test_fusion_validator_read_write_conflict() {
        let deps = vec![
            make_op_deps("op_a", "terra", OperatorPhaseIr::Collect, &["x"], &["y"]),
            make_op_deps("op_b", "terra", OperatorPhaseIr::Collect, &["y"], &["z"]),
        ];

        let group = FusionGroup {
            operators: vec![OperatorId("op_a".to_string()), OperatorId("op_b".to_string())],
            fusion_type: FusionType::SharedReads,
            shared_resources: vec![],
            benefit_score: 1.0,
        };

        let validator = FusionValidator::new(&deps);
        let result = validator.validate(&group);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, FusionUnsafe::ReadWriteConflict { .. })));
    }

    #[test]
    fn test_fusion_validator_phase_mismatch() {
        let deps = vec![
            make_op_deps("op_a", "terra", OperatorPhaseIr::Collect, &["x"], &[]),
            make_op_deps("op_b", "terra", OperatorPhaseIr::Measure, &["x"], &[]),
        ];

        let group = FusionGroup {
            operators: vec![OperatorId("op_a".to_string()), OperatorId("op_b".to_string())],
            fusion_type: FusionType::SharedReads,
            shared_resources: vec!["x".to_string()],
            benefit_score: 1.0,
        };

        let validator = FusionValidator::new(&deps);
        let result = validator.validate(&group);
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_model_should_fuse() {
        let deps = vec![
            make_op_deps("op_a", "terra", OperatorPhaseIr::Collect, &["x", "y", "z"], &[]),
            make_op_deps("op_b", "terra", OperatorPhaseIr::Collect, &["x", "y", "z"], &[]),
            make_op_deps("op_c", "terra", OperatorPhaseIr::Collect, &["x", "y", "z"], &[]),
        ];

        let group = FusionGroup {
            operators: vec![
                OperatorId("op_a".to_string()),
                OperatorId("op_b".to_string()),
                OperatorId("op_c".to_string()),
            ],
            fusion_type: FusionType::SharedReads,
            shared_resources: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            benefit_score: 5.0,
        };

        let cost_model = FusionCostModel::default();
        assert!(cost_model.should_fuse(&group, &deps));
    }

    #[test]
    fn test_cost_model_not_beneficial() {
        let deps = vec![
            make_op_deps("op_a", "terra", OperatorPhaseIr::Collect, &["x"], &[]),
            make_op_deps("op_b", "terra", OperatorPhaseIr::Collect, &["y"], &[]),
        ];

        let group = FusionGroup {
            operators: vec![OperatorId("op_a".to_string()), OperatorId("op_b".to_string())],
            fusion_type: FusionType::SharedReads,
            shared_resources: vec![], // No shared resources
            benefit_score: 0.5,
        };

        // First compute the actual benefit to understand what threshold to set
        let cost_model = FusionCostModel::default();
        let actual_benefit = cost_model.compute_benefit(&group, &deps);

        // Verify our test setup produces some benefit (sanity check)
        assert!(
            actual_benefit > 0.0,
            "Test expects some positive benefit from call elimination, got {}",
            actual_benefit
        );

        // Now set threshold higher than the computed benefit
        let mut high_threshold_model = cost_model.clone();
        high_threshold_model.min_benefit = actual_benefit + 1.0;

        // Verify should_fuse correctly rejects when benefit < threshold
        assert!(
            !high_threshold_model.should_fuse(&group, &deps),
            "should_fuse should return false when min_benefit ({}) > actual_benefit ({})",
            high_threshold_model.min_benefit,
            actual_benefit
        );
    }

    #[test]
    fn test_fusion_stats_display() {
        let stats = FusionStats {
            total_operators: 10,
            candidate_groups: 5,
            safe_groups: 4,
            beneficial_groups: 3,
            operators_fused: 8,
            fused_operators_created: 3,
            estimated_call_savings: 5,
            estimated_read_savings: 12,
        };

        let display = format!("{}", stats);
        assert!(display.contains("10 operators"));
        assert!(display.contains("5 candidate groups"));
        assert!(display.contains("8 operators into 3 groups"));
    }
}
