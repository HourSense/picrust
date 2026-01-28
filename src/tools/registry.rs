//! Tool registry for managing available tools
//!
//! The registry holds all tools that are available to the agent.
//! It supports both static tools (registered directly) and dynamic tools
//! from providers (like MCP servers).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;

use super::provider::ToolProvider;
use super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::ToolDefinition;
use crate::runtime::AgentInternals;

/// Registry that holds all available tools
pub struct ToolRegistry {
    /// Static tools registered directly
    tools: HashMap<String, Arc<dyn Tool>>,

    /// Dynamic tool providers (MCP, etc.)
    providers: Vec<Arc<dyn ToolProvider>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            providers: Vec::new(),
        }
    }

    /// Register a static tool in the registry
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        tracing::info!("Registering tool: {}", name);
        self.tools.insert(name, Arc::new(tool));
    }

    /// Add a tool provider (MCP, etc.)
    ///
    /// This will immediately fetch all tools from the provider and add them to the registry.
    /// Returns an error if any tool name conflicts with existing tools.
    pub async fn add_provider(&mut self, provider: Arc<dyn ToolProvider>) -> Result<()> {
        tracing::info!(
            "[ToolRegistry] Adding provider '{}' (dynamic: {})",
            provider.name(),
            provider.is_dynamic()
        );

        let tools = provider.get_tools().await?;

        for tool in tools {
            let name = tool.name().to_string();

            // Check for conflicts
            if self.tools.contains_key(&name) {
                return Err(anyhow::anyhow!(
                    "Tool name conflict: '{}' already exists (from provider '{}')",
                    name,
                    provider.name()
                ));
            }

            tracing::info!(
                "[ToolRegistry] Registering tool '{}' from provider '{}'",
                name,
                provider.name()
            );
            self.tools.insert(name, tool);
        }

        self.providers.push(provider);

        Ok(())
    }

    /// Refresh all dynamic providers
    ///
    /// This will re-fetch tools from all dynamic providers and update the registry.
    /// Useful for MCP servers where tools can change at runtime.
    pub async fn refresh_providers(&mut self) -> Result<()> {
        tracing::info!("[ToolRegistry] Refreshing all dynamic providers");

        // Remove all tools from providers
        let provider_names: Vec<_> = self.providers.iter().map(|p| p.name()).collect();

        self.tools.retain(|name, _| {
            // Keep static tools, remove provider tools
            !provider_names.iter().any(|p| name.starts_with(&format!("{}:", p)))
        });

        // Re-add tools from all providers
        for provider in &self.providers {
            if provider.is_dynamic() {
                provider.refresh().await?;
            }

            let tools = provider.get_tools().await?;

            for tool in tools {
                let name = tool.name().to_string();
                self.tools.insert(name, tool);
            }
        }

        tracing::info!("[ToolRegistry] Provider refresh complete");

        Ok(())
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Get all tool definitions for the Anthropic API
    pub fn get_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Get information about a tool invocation
    pub fn get_tool_info(&self, name: &str, input: &Value) -> Option<ToolInfo> {
        self.tools.get(name).map(|t| t.get_info(input))
    }

    /// Execute a tool by name
    pub async fn execute(
        &self,
        name: &str,
        input: &Value,
        internals: &mut AgentInternals,
    ) -> Result<ToolResult> {
        let tool = self
            .tools
            .get(name)
            .with_context(|| format!("Tool not found: {}", name))?;

        tracing::info!("Executing tool: {}", name);
        tracing::debug!("Input: {:?}", input);

        let result = tool.execute(input, internals).await?;

        tracing::debug!(
            "Tool {} completed. Is error: {}",
            name,
            result.is_error
        );

        Ok(result)
    }

    /// Check if a tool requires permission
    pub fn requires_permission(&self, name: &str) -> bool {
        self.tools
            .get(name)
            .map(|t| t.requires_permission())
            .unwrap_or(true)
    }

    /// Get the list of tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.get("nonexistent").is_none());
    }
}
