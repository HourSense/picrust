//! Tool Provider trait
//!
//! Abstraction for dynamic tool sources (MCP servers, OpenAPI specs, etc.)

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use super::tool::Tool;

/// Trait for dynamic tool providers
///
/// Providers can fetch tools from external sources (MCP servers, OpenAPI specs, etc.)
/// and expose them as Tool implementations.
#[async_trait]
pub trait ToolProvider: Send + Sync {
    /// Get all tools from this provider
    ///
    /// This method is called during initialization and when refreshing tools.
    async fn get_tools(&self) -> Result<Vec<Arc<dyn Tool>>>;

    /// Refresh the tool list
    ///
    /// For dynamic providers (like MCP), this re-fetches the tool list from the source.
    /// For static providers, this is a no-op.
    async fn refresh(&self) -> Result<()> {
        Ok(())
    }

    /// Provider name for logging and debugging
    fn name(&self) -> &str;

    /// Whether this provider is dynamic (tools can change at runtime)
    fn is_dynamic(&self) -> bool {
        false
    }
}
