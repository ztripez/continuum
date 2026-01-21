//! World container for a compiled simulation.
//!
//! A [`World`] represents the complete, resolved, and verified state of a
//! simulation definition. It is produced by the compiler pipeline after
//! all resolution and validation passes have completed.

use crate::ast::declaration::{Declaration, WorldDecl};
use crate::ast::node::Node;
use crate::ast::structural::{Entity, Era, Stratum};
use crate::foundation::{EntityId, EraId, Path};
use crate::resolve::graph::DagSet;
use indexmap::IndexMap;

/// A compiled and resolved Continuum world.
///
/// Contains all primitives (signals, fields, operators) and structural
/// definitions (entities, strata, eras) after semantic analysis, ready for
/// DAG construction.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::ast::{World, WorldDecl};
/// use continuum_cdsl::foundation::{Path, Span};
///
/// let decl = WorldDecl {
///     path: Path::from_path_str("demo"),
///     title: None,
///     version: None,
///     warmup: None,
///     attributes: Vec::new(),
///     span: Span::new(0, 0, 0, 0),
///     doc: None,
///     debug: false,
/// };
/// let world = World::new(decl);
/// assert!(world.eras.is_empty());
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct World {
    /// Declaration metadata from the `world` block, including path/title/version.
    pub metadata: WorldDecl,

    /// Era ID to start execution in, if an era was marked `:initial`.
    ///
    /// Compiler validation requires exactly one `:initial` era.
    #[serde(default)]
    pub initial_era: Option<EraId>,

    /// Global nodes keyed by fully-qualified path.
    pub globals: IndexMap<Path, Node<()>>,

    /// Per-entity members keyed by fully-qualified path.
    pub members: IndexMap<Path, Node<EntityId>>,

    /// Entity definitions keyed by fully-qualified path.
    pub entities: IndexMap<Path, Entity>,

    /// Stratum definitions keyed by fully-qualified path.
    pub strata: IndexMap<Path, Stratum>,

    /// Era definitions keyed by fully-qualified path.
    pub eras: IndexMap<Path, Era>,

    /// Desugared declarations in source order, preserved for diagnostics and tooling.
    pub declarations: Vec<Declaration>,
}

impl World {
    /// Create a new empty world with metadata.
    ///
    /// # Parameters
    /// - `metadata`: World declaration metadata from the parser.
    ///
    /// # Returns
    /// An empty [`World`] with no nodes or structural declarations.
    ///
    /// # Examples
    /// ```rust
    /// use continuum_cdsl::ast::{World, WorldDecl};
    /// use continuum_cdsl::foundation::{Path, Span};
    ///
    /// let decl = WorldDecl {
    ///     path: Path::from_path_str("demo"),
    ///     title: None,
    ///     version: None,
    ///     warmup: None,
    ///     attributes: Vec::new(),
    ///     span: Span::new(0, 0, 0, 0),
    ///     doc: None,
    ///     debug: false,
    /// };

    /// let world = World::new(decl);
    /// assert!(world.globals.is_empty());
    /// ```
    pub fn new(metadata: WorldDecl) -> Self {
        Self {
            metadata,
            initial_era: None,
            globals: IndexMap::new(),
            members: IndexMap::new(),
            entities: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            declarations: Vec::new(),
        }
    }
}

/// A fully compiled Continuum world, including execution graphs.
///
/// This is the final output of the compiler pipeline, ready to be
/// consumed by the runtime.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::ast::{CompiledWorld, World, WorldDecl};
/// use continuum_cdsl::foundation::{Path, Span};
/// use continuum_cdsl::resolve::graph::DagSet;
///
/// let decl = WorldDecl {
///     path: Path::from_path_str("demo"),
///     title: None,
///     version: None,
///     warmup: None,
///     attributes: Vec::new(),
///     span: Span::new(0, 0, 0, 0),
///     doc: None,
///     debug: false,
/// };
/// let world = World::new(decl);
/// let dag_set = DagSet::default();
/// let compiled = CompiledWorld::new(world, dag_set);
/// let _ = compiled;
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledWorld {
    /// The resolved world structure
    // Note: We might want to optimize what we include here for serialization
    pub world: World,

    /// The compiled execution DAGs
    pub dag_set: DagSet,
}

impl CompiledWorld {
    /// Create a new compiled world from a resolved world and its DAGs.
    ///
    /// # Parameters
    /// - `world`: Resolved world containing declarations and structural metadata.
    /// - `dag_set`: Execution DAGs derived from the resolved world.
    ///
    /// # Returns
    /// A [`CompiledWorld`] bundling the world and its DAGs.
    ///
    /// # Examples
    /// ```rust
    /// use continuum_cdsl::ast::{CompiledWorld, World, WorldDecl};
    /// use continuum_cdsl::foundation::{Path, Span};
    /// use continuum_cdsl::resolve::graph::DagSet;
    ///
    /// let decl = WorldDecl {
    ///     path: Path::from_path_str("demo"),
    ///     title: None,
    ///     version: None,
    ///     warmup: None,
    ///     attributes: Vec::new(),
    ///     span: Span::new(0, 0, 0, 0),
    ///     doc: None,
    ///     debug: false,
    /// };

    /// let world = World::new(decl);
    /// let dag_set = DagSet::default();
    /// let compiled = CompiledWorld::new(world, dag_set);
    /// let _ = compiled;
    /// ```
    pub fn new(world: World, dag_set: DagSet) -> Self {
        Self { world, dag_set }
    }
}

/// Binary bundle for serialized compiled worlds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BinaryBundle {
    /// The compiled world.
    pub world: CompiledWorld,
}
