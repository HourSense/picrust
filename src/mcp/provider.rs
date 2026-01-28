//! MCP Tool Provider
//!
//! Implements ToolProvider trait for MCP servers

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use crate::tools::{Tool, ToolProvider};

use super::manager::MCPServerManager;
use super::tool_adapter::MCPToolAdapter;

/// Tool provider that fetches tools from MCP servers
pub struct MCPToolProvider {
    /// Manager for MCP servers
    manager: Arc<MCPServerManager>,
}

impl MCPToolProvider {
    /// Create a new MCP tool provider
    pub fn new(manager: Arc<MCPServerManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl ToolProvider for MCPToolProvider {
    async fn get_tools(&self) -> Result<Vec<Arc<dyn Tool>>> {
        tracing::info!("[MCPToolProvider] Fetching tools from all MCP servers");

        let mcp_tools = self.manager.get_all_tools().await?;

        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

        for mcp_tool_info in mcp_tools {
            let adapter = MCPToolAdapter::new(
                mcp_tool_info.server_id,
                mcp_tool_info.server,
                mcp_tool_info.tool_def,
            );

            tools.push(Arc::new(adapter));
        }

        tracing::info!(
            "[MCPToolProvider] Created {} tool adapters from MCP servers",
            tools.len()
        );

        Ok(tools)
    }

    async fn refresh(&self) -> Result<()> {
        tracing::info!("[MCPToolProvider] Refreshing MCP tools");
        // Re-fetching is handled by get_tools()
        Ok(())
    }

    fn name(&self) -> &str {
        "MCP"
    }

    fn is_dynamic(&self) -> bool {
        true
    }
}
