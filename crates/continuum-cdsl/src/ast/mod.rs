// AST module - unified Node<I> structure and role system
//
// This module defines the core AST structure used throughout compilation.
// Everything is Node<I> with RoleData - no separate AST→IR copying.

mod node;
mod role;

pub use node::*;
pub use role::*;
