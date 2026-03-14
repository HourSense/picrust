//! Tool setup for the OpenAI test agent

use anyhow::Result;
use picrust::tools::{BashTool, ReadTool, GrepTool, GlobTool, ToolRegistry};

pub fn create_registry() -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    registry.register(ReadTool::new()?);
    registry.register(BashTool::new()?);
    registry.register(GrepTool::new()?);
    registry.register(GlobTool::new()?);
    Ok(registry)
}
