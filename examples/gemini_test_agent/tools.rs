//! Tool setup for the Gemini test agent

use anyhow::Result;
use shadow_agent_sdk::tools::{BashTool, ReadTool, TodoWriteTool, ToolRegistry, WriteTool, GrepTool, GlobTool, EditTool, AskUserQuestionTool};

/// Create a tool registry with standard tools
pub fn create_registry() -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();

    registry.register(ReadTool::new()?);
    registry.register(WriteTool::new()?);
    registry.register(BashTool::new()?);
    registry.register(TodoWriteTool::new());
    registry.register(GrepTool::new()?);
    registry.register(GlobTool::new()?);
    registry.register(EditTool::new()?);
    registry.register(AskUserQuestionTool::new());

    Ok(registry)
}
