//! Tool system for the agent framework
//!
//! This module provides:
//! - `Tool` trait - Interface for implementing tools
//! - `ToolResult` - Result type for tool execution
//! - `ToolRegistry` - Registry for managing available tools
//! - `ToolProvider` trait - Interface for dynamic tool sources (MCP, OpenAPI, etc.)
//! - `common` - Built-in tools (Bash, Read, Write, Edit, Glob, Grep, Todo)

mod provider;
mod registry;
mod tool;

/// Common/built-in tools
pub mod common;

// Core exports
pub use provider::ToolProvider;
pub use registry::ToolRegistry;
pub use tool::{Tool, ToolInfo, ToolResult, ToolResultData};

// Re-export common tools for convenience
pub use common::{
    AskUserQuestionTool, BashTool, EditTool, GlobTool, GrepTool, PresentFileTool, ReadTool,
    TodoWriteTool, WriteTool,
};
