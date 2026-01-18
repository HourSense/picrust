//! Tool system for the agent framework
//!
//! This module provides:
//! - `Tool` trait - Interface for implementing tools
//! - `ToolResult` - Result type for tool execution
//! - `ToolRegistry` - Registry for managing available tools
//! - `common` - Built-in tools (Bash, Read, Write, Edit, Glob, Grep, Todo)

mod registry;
mod tool;

/// Common/built-in tools
pub mod common;

// Core exports
pub use registry::ToolRegistry;
pub use tool::{Tool, ToolInfo, ToolResult};

// Re-export common tools for convenience
pub use common::{
    BashTool, EditTool, GlobTool, GrepTool, ReadTool, TodoWriteTool, WriteTool,
};
