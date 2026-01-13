//! Permission system for tool execution
//!
//! This module provides permission management for tool execution.
//! Currently supports simple approval/denial, but is designed to be
//! extensible for future fine-grained rules.

mod manager;

pub use manager::{Permission, PermissionDecision, PermissionManager, PermissionRequest};
