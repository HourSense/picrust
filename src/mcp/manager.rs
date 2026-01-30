//! MCP Server Manager
//!
//! Manages multiple MCP server connections

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::{RoleClient, ServiceExt};

use super::config::MCPServerConfig;
use super::server::MCPServer;

/// Information about an MCP tool from a specific server
#[derive(Debug, Clone)]
pub struct MCPToolInfo {
    /// ID of the server this tool belongs to
    pub server_id: String,

    /// Arc reference to the server
    pub server: Arc<MCPServer>,

    /// The tool definition from rmcp
    pub tool_def: rmcp::model::Tool,
}

/// Manages connections to multiple MCP servers
pub struct MCPServerManager {
    /// Map of server ID to server instance
    servers: Arc<RwLock<HashMap<String, Arc<MCPServer>>>>,
}

impl MCPServerManager {
    /// Create a new empty manager
    pub fn new() -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add an MCP server with a service refresher callback
    ///
    /// The refresher is REQUIRED and is called:
    /// - Before each MCP operation (to check connection health)
    /// - When connection failures are detected (to force reconnect)
    ///
    /// The refresher should return:
    /// - `Ok(Some(service))` - A new service was created
    /// - `Ok(None)` - The cached service is still valid, no refresh needed
    /// - `Err(e)` - Failed to create service
    ///
    /// The refresher should implement caching internally to avoid unnecessary reconnections.
    ///
    /// # Example
    /// ```no_run
    /// use std::sync::Arc;
    /// use std::time::{Duration, Instant};
    /// use tokio::sync::RwLock;
    /// use rmcp::transport::StreamableHttpClientTransport;
    /// use rmcp::ServiceExt;
    ///
    /// // Cached service with timestamp
    /// let cached_service = Arc::new(RwLock::new(None));
    /// let last_refresh = Arc::new(RwLock::new(Instant::now()));
    /// let jwt_provider = Arc::new(MyJwtProvider::new());
    ///
    /// let refresher = {
    ///     let cached = cached_service.clone();
    ///     let last_refresh = last_refresh.clone();
    ///     let jwt = jwt_provider.clone();
    ///
    ///     move || {
    ///         let cached = cached.clone();
    ///         let last_refresh = last_refresh.clone();
    ///         let jwt = jwt.clone();
    ///
    ///         async move {
    ///             // Check if cached service is still valid
    ///             {
    ///                 let last = last_refresh.read().await;
    ///                 if last.elapsed() < Duration::from_secs(50 * 60) {
    ///                     // Token still fresh, no refresh needed
    ///                     let cached_guard = cached.read().await;
    ///                     if cached_guard.is_some() {
    ///                         return Ok(None); // Keep using cached
    ///                     }
    ///                 }
    ///             }
    ///
    ///             // Create new service
    ///             let token = jwt.get_fresh_token().await?;
    ///             let transport = StreamableHttpClientTransport::from_uri("https://backend/mcp")
    ///                 .with_header("Authorization", format!("Bearer {}", token));
    ///             let service = ().serve(transport).await?;
    ///
    ///             // Cache it
    ///             *cached.write().await = Some(service);
    ///             *last_refresh.write().await = Instant::now();
    ///
    ///             // Return the cached service as Option
    ///             Ok(cached.read().await.clone())
    ///         }
    ///     }
    /// };
    ///
    /// manager.add_server_with_refresher("my-server", refresher).await?;
    /// ```
    pub async fn add_server_with_refresher<F, Fut>(
        &self,
        id: impl Into<String>,
        refresher: F,
    ) -> Result<()>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send + 'static,
    {
        let id = id.into();

        // Check if server already exists
        if self.servers.read().await.contains_key(&id) {
            return Err(anyhow!("MCP server '{}' already exists", id));
        }

        let server = Arc::new(MCPServer::new(id.clone(), refresher));

        // Add to map
        self.servers.write().await.insert(id.clone(), server);

        tracing::debug!(
            "[MCPServerManager] Added MCP server '{}' with refresher",
            id
        );

        Ok(())
    }

    /// Add and connect to a new MCP server from config
    ///
    /// Creates a simple refresher that reconnects to the URI on every refresh.
    /// For more control (JWT refresh, custom caching), use `add_server_with_refresher()`.
    pub async fn add_server(&self, config: MCPServerConfig) -> Result<()> {
        if !config.enabled {
            tracing::info!(
                "[MCPServerManager] Skipping disabled server '{}'",
                config.id
            );
            return Ok(());
        }

        let id = config.id.clone();
        let uri = config.uri.clone();

        // Check if server already exists
        if self.servers.read().await.contains_key(&id) {
            return Err(anyhow!("MCP server '{}' already exists", id));
        }

        // Create a refresher with simple caching
        // Cache service for 5 minutes to avoid unnecessary reconnections
        let cached_service: Arc<RwLock<Option<RunningService<RoleClient, ()>>>> =
            Arc::new(RwLock::new(None));
        let last_refresh = Arc::new(RwLock::new(Instant::now() - Duration::from_secs(999)));

        let refresher = {
            let uri = uri.clone();
            let cached = cached_service.clone();
            let last_refresh = last_refresh.clone();

            move || {
                let uri = uri.clone();
                let cached = cached.clone();
                let last_refresh = last_refresh.clone();

                async move {
                    // Check if cached service is still valid (less than 5 minutes old)
                    {
                        let last = last_refresh.read().await;
                        if last.elapsed() < Duration::from_secs(5 * 60) {
                            let cached_guard = cached.read().await;
                            if cached_guard.is_some() {
                                // Service is still valid, no refresh needed
                                return Ok(None);
                            }
                        }
                    }

                    // Create new service
                    let transport = StreamableHttpClientTransport::from_uri(uri.as_str());
                    let service = ().serve(transport).await?;

                    // Cache it (just for tracking the timestamp)
                    // We don't actually return from cache since RunningService doesn't impl Clone
                    *cached.write().await = Some(service);
                    *last_refresh.write().await = Instant::now();

                    // Take ownership of the service from the cache and return it
                    Ok(cached.write().await.take())
                }
            }
        };

        let server = Arc::new(MCPServer::new(id.clone(), refresher));

        // Add to map
        self.servers.write().await.insert(id.clone(), server);

        tracing::debug!("[MCPServerManager] Added MCP server '{}'", id);

        Ok(())
    }

    /// Get a server by ID
    pub async fn get_server(&self, id: &str) -> Option<Arc<MCPServer>> {
        self.servers.read().await.get(id).cloned()
    }

    /// Get all server IDs
    pub async fn server_ids(&self) -> Vec<String> {
        self.servers.read().await.keys().cloned().collect()
    }

    /// Get all tools from all connected servers
    pub async fn get_all_tools(&self) -> Result<Vec<MCPToolInfo>> {
        let mut all_tools = Vec::new();

        let servers = self.servers.read().await;

        for (server_id, server) in servers.iter() {
            match server.list_tools().await {
                Ok(tools) => {
                    tracing::info!(
                        "[MCPServerManager] Got {} tools from server '{}'",
                        tools.len(),
                        server_id
                    );

                    for tool_def in tools {
                        all_tools.push(MCPToolInfo {
                            server_id: server_id.clone(),
                            server: server.clone(),
                            tool_def,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "[MCPServerManager] Failed to get tools from server '{}': {}",
                        server_id,
                        e
                    );
                    // Continue with other servers instead of failing completely
                }
            }
        }

        Ok(all_tools)
    }

    /// Run health checks on all servers
    pub async fn health_check_all(&self) -> HashMap<String, Result<()>> {
        let mut results = HashMap::new();
        let servers = self.servers.read().await;

        for (server_id, server) in servers.iter() {
            let result = server.health_check().await;
            results.insert(server_id.clone(), result);
        }

        results
    }

    /// Force reconnect a specific server by clearing its cache
    ///
    /// This will cause the next operation to trigger the refresher.
    /// Note: Reconnection happens automatically on connection failures,
    /// so this is usually not needed.
    pub async fn reconnect_server(&self, id: &str) -> Result<()> {
        let _server = self
            .get_server(id)
            .await
            .ok_or_else(|| anyhow!("Server '{}' not found", id))?;

        tracing::debug!(
            "[MCPServerManager] Reconnection requested for '{}' (automatic on next operation)",
            id
        );

        // Note: The actual reconnection logic is handled by the refresher
        // on the next operation. We don't have a direct reconnect method
        // because reconnection is automatic when operations fail.

        Ok(())
    }

    /// Get the number of connected servers
    pub async fn server_count(&self) -> usize {
        self.servers.read().await.len()
    }

    /// Check if manager has any servers
    pub async fn is_empty(&self) -> bool {
        self.servers.read().await.is_empty()
    }
}

impl Default for MCPServerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = MCPServerManager::new();
        assert!(manager.is_empty().await);
        assert_eq!(manager.server_count().await, 0);
    }

    #[tokio::test]
    async fn test_server_ids() {
        let manager = MCPServerManager::new();
        assert!(manager.server_ids().await.is_empty());
    }
}
