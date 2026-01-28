//! MCP Server Manager
//!
//! Manages multiple MCP server connections

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

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

    /// Add an MCP server from an existing RunningService
    ///
    /// For simple cases without auth refresh. For JWT scenarios,
    /// use `add_service_with_refresher()` instead.
    ///
    /// # Example
    /// ```no_run
    /// use rmcp::transport::StreamableHttpClientTransport;
    /// use rmcp::ServiceExt;
    ///
    /// let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp")
    ///     .with_header("Authorization", "Bearer token");
    /// let service = ().serve(transport).await?;
    /// manager.add_service("my-server", service).await?;
    /// ```
    pub async fn add_service(
        &self,
        id: impl Into<String>,
        service: rmcp::service::RunningService<rmcp::RoleClient, ()>,
    ) -> Result<()> {
        let id = id.into();

        // Check if server already exists
        if self.servers.read().await.contains_key(&id) {
            return Err(anyhow!("MCP server '{}' already exists", id));
        }

        let server = Arc::new(MCPServer::from_service(id.clone(), service));

        // Add to map
        self.servers.write().await.insert(id.clone(), server);

        tracing::info!("[MCPServerManager] Added MCP server '{}'", id);

        Ok(())
    }

    /// Add an MCP server with a service refresher callback
    ///
    /// This is the recommended way for JWT/auth scenarios where tokens expire.
    /// The refresher is called before every MCP operation.
    ///
    /// # Example
    /// ```no_run
    /// use std::sync::Arc;
    /// use std::time::{Duration, Instant};
    /// use tokio::sync::RwLock;
    /// use rmcp::transport::StreamableHttpClientTransport;
    /// use rmcp::ServiceExt;
    ///
    /// let last_refresh = Arc::new(RwLock::new(Instant::now()));
    /// let jwt_provider = Arc::new(MyJwtProvider::new());
    ///
    /// let refresher = {
    ///     let last_refresh = last_refresh.clone();
    ///     let jwt = jwt_provider.clone();
    ///
    ///     move || {
    ///         let last_refresh = last_refresh.clone();
    ///         let jwt = jwt.clone();
    ///         async move {
    ///             let mut last = last_refresh.write().await;
    ///             if last.elapsed() < Duration::from_secs(50 * 60) {
    ///                 return Ok(None); // Still valid
    ///             }
    ///
    ///             let token = jwt.get_fresh_token().await?;
    ///             let transport = StreamableHttpClientTransport::from_uri("https://backend/mcp")
    ///                 .with_header("Authorization", format!("Bearer {}", token));
    ///             let service = ().serve(transport).await?;
    ///             *last = Instant::now();
    ///             Ok(Some(service))
    ///         }
    ///     }
    /// };
    ///
    /// let initial_service = create_initial_service().await?;
    /// manager.add_service_with_refresher("my-server", initial_service, refresher).await?;
    /// ```
    pub async fn add_service_with_refresher<F, Fut>(
        &self,
        id: impl Into<String>,
        initial_service: rmcp::service::RunningService<rmcp::RoleClient, ()>,
        refresher: F,
    ) -> Result<()>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<
                Output = Result<Option<rmcp::service::RunningService<rmcp::RoleClient, ()>>>,
            > + Send
            + 'static,
    {
        let id = id.into();

        // Check if server already exists
        if self.servers.read().await.contains_key(&id) {
            return Err(anyhow!("MCP server '{}' already exists", id));
        }

        let server = Arc::new(MCPServer::with_service_refresher(
            id.clone(),
            initial_service,
            refresher,
        ));

        // Add to map
        self.servers.write().await.insert(id.clone(), server);

        tracing::info!(
            "[MCPServerManager] Added MCP server '{}' with service refresher",
            id
        );

        Ok(())
    }

    /// Add and connect to a new MCP server using a simple URI
    ///
    /// This is a convenience method for simple cases. For more control (auth, custom headers),
    /// use `add_service()` instead.
    pub async fn add_server(&self, config: MCPServerConfig) -> Result<()> {
        if !config.enabled {
            tracing::info!(
                "[MCPServerManager] Skipping disabled server '{}'",
                config.id
            );
            return Ok(());
        }

        let id = config.id.clone();

        // Check if server already exists
        if self.servers.read().await.contains_key(&id) {
            return Err(anyhow!("MCP server '{}' already exists", id));
        }

        // Connect to server
        let server = Arc::new(MCPServer::new(config).await?);

        // Add to map
        self.servers.write().await.insert(id.clone(), server);

        tracing::info!("[MCPServerManager] Added MCP server '{}'", id);

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

    /// Reconnect a specific server
    pub async fn reconnect_server(&self, id: &str) -> Result<()> {
        let server = self
            .get_server(id)
            .await
            .ok_or_else(|| anyhow!("Server '{}' not found", id))?;

        server.reconnect().await
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
