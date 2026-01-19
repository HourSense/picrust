//! Permission system for tool execution
//!
//! This module provides a three-tier permission system:
//! - **Global**: Shared across all agents via `Arc<GlobalPermissions>`
//! - **Local**: Agent-type specific rules
//! - **Session**: Rules added during the current session
//!
//! ## Rule Types
//!
//! - `AllowTool`: Allow an entire tool (e.g., Read is always allowed)
//! - `AllowPrefix`: Allow commands starting with a prefix (e.g., `cd` for Bash)
//!
//! ## Example
//!
//! ```rust,ignore
//! use shadow_agent_sdk::permissions::{GlobalPermissions, PermissionManager, PermissionRule};
//! use std::sync::Arc;
//!
//! // Create shared global permissions
//! let global = Arc::new(GlobalPermissions::new());
//! global.add_rule(PermissionRule::allow_tool("Read"));
//!
//! // Create per-agent manager
//! let mut manager = PermissionManager::new(global.clone(), "my-agent");
//!
//! // Check permission
//! match manager.check("Read", "file.txt") {
//!     CheckResult::Allowed => { /* execute */ }
//!     CheckResult::AskUser => { /* prompt user */ }
//!     CheckResult::Denied => { /* reject */ }
//! }
//! ```

mod manager;

pub use manager::{
    CheckResult, GlobalPermissions, PermissionDecision, PermissionManager, PermissionRequest,
    PermissionRule, PermissionScope, RuleType,
};
