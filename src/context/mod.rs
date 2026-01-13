//! Context manager for dynamic prompt injection
//!
//! This module provides the ContextManager which handles:
//! - System prompt management
//! - Hidden prompts (injected context)
//! - Dynamic context generation (e.g., file structure, git status)

mod manager;
mod providers;

pub use manager::ContextManager;
pub use providers::{ContextProvider, FileStructureProvider, GitStatusProvider};
