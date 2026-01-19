//! Tool setup for the test agent
//!
//! Registers Read, Write, Bash, and TodoWrite tools.

use anyhow::Result;
use shadow_agent_sdk::tools::{BashTool, ReadTool, TodoWriteTool, ToolRegistry, WriteTool};

/// Create a tool registry with Read, Write, Bash, and TodoWrite tools
pub fn create_registry() -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();

    // Register tools
    registry.register(ReadTool::new()?);
    registry.register(WriteTool::new()?);
    registry.register(BashTool::new()?);
    registry.register(TodoWriteTool::new());

    Ok(registry)
}

