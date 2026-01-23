//!
//! The Executor is the core of the Continuum runtime. It is responsible for:
//! 1. **DAG Scheduling** - Determining the execution order based on signal dependencies.
//! 2. **Phase Management** - Orchestrating the execution flow (Collect, Resolve, Fracture, Measure).
//! 3. **Memory Management** - Handling signal storage (SoA), entity instances, and event buffers.
//! 4. **Bytecode Execution** - Running compiled DSL blocks within the appropriate context.
//!
//! # Execution Architecture
//!
//! Simulation progress is driven by `execute_tick()`, which iterates through all phases
//! for all active strata.
//!
//! ```text
//! Tick
//!  ├─ Configure (engine internal)
//!  ├─ Collect (impulses -> input channels)
//!  ├─ Resolve (inputs -> signals)
//!  ├─ Fracture (tension detection -> spawn/destroy)
//!  └─ Measure (signals -> fields/observers)
//! ```
//!
//! Each phase uses a specialized `Context` to provide the required capabilities
//! while enforcing the observer boundary. For example, `ResolveContext` allows
//! reading `prev` values but forbids `emit` calls to fields.
//!
//! # Determinism
//!
//! The executor guarantees bit-for-bit determinism by:
//! - Using stable topological sorting for DAG nodes.
//! - Enforcing explicit iteration order for entities and members.
//! - Preventing any non-causal data (fields) from influencing causal phases.
//!
//! # Failure Handling (Fail Loudly)
//!
//! Any runtime violation—such as missing dependencies, type mismatches, or
//! assertion failures—is immediately surfaced as a `RunError`. The runtime
//! does not attempt to "fix" or "clamp" invalid states silently.

mod assertions;
pub mod bytecode;
mod context;
pub mod cost_model;
pub mod kernel_registry;
pub mod l1_kernels;
pub mod l3_kernel;
pub mod lane_kernel;
pub mod lowering_strategy;
pub mod member_executor;
mod phases;
mod run;
mod runtime;
mod warmup;

// Re-export public types
pub use crate::types::AssertionSeverity;
pub use assertions::{AssertionChecker, AssertionFailure, AssertionFn, SignalAssertion};
pub use context::{
    AssertContext, ChronicleContext, CollectContext, FractureContext, ImpulseContext,
    MeasureContext, ResolveContext, WarmupContext,
};
pub use kernel_registry::LaneKernelRegistry;
pub use l1_kernels::{ScalarKernelFn, ScalarL1Kernel, Vec3KernelFn, Vec3L1Kernel};
pub use l3_kernel::{
    L3Kernel, L3KernelBuilder, MemberDag, MemberDagError, ScalarL3MemberResolver,
    ScalarL3ResolverFn, Vec3L3MemberResolver, Vec3L3ResolverFn,
};
pub use lane_kernel::{LaneKernel, LaneKernelError, LaneKernelResult};
pub use lowering_strategy::{LoweringHeuristics, LoweringStrategy};
pub use member_executor::{
    ChunkConfig, MemberResolveContext, MemberSignalResolver, ScalarL1Resolver,
    ScalarResolveContext, ScalarResolverFn, Vec3L1Resolver, Vec3ResolveContext, Vec3ResolverFn,
};
pub use phases::{
    ChronicleFn, CollectFn, EmittedEvent, FractureFn, ImpulseFn, MeasureFn, PhaseExecutor,
    ResolverFn,
};
pub use run::{run_simulation, CheckpointOptions, RunError, RunOptions, RunReport};
pub use runtime::{AggregateResolverFn, EraConfig, Runtime, TransitionFn};
pub use warmup::{WarmupExecutor, WarmupFn};
