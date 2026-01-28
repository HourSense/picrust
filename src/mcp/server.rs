//! MCP Server wrapper
//!
//! Wraps rmcp service to manage connections to individual MCP servers

use anyhow::{anyhow, Result};
use rmcp::model::{CallToolRequestParams, CallToolResult, ListToolsResult, Tool};
use rmcp::service::RunningService;
use rmcp::transport::{
    streamable_http_client::StreamableHttpClientTransportConfig,
    StreamableHttpClientTransport,
};
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Map, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::MCPServerConfig;

/// The concrete transport type we use for HTTP MCP connections
pub type HttpClientTransport = StreamableHttpClientTransport<reqwest::Client>;

/// Type alias for service refresher callback future
pub type ServiceRefreshFuture =
    Pin<Box<dyn Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send>>;

/// Trait for providing MCP service with optional refresh logic
///
/// This is called before each MCP operation (list_tools, call_tool).
/// Implement this to handle JWT expiration, token refresh, etc.
///
/// Return:
/// - `Ok(None)` - current service is still valid, no refresh needed
/// - `Ok(Some(service))` - replace current service with this new one
/// - `Err(...)` - refresh failed
pub trait ServiceRefresher: Send + Sync {
    /// Check if service needs refresh and optionally return a new service
    fn refresh(&self) -> ServiceRefreshFuture;
}

/// Wrapper to implement ServiceRefresher for async closures
pub struct FnServiceRefresher<F> {
    func: F,
}

impl<F, Fut> ServiceRefresher for FnServiceRefresher<F>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send + 'static,
{
    fn refresh(&self) -> ServiceRefreshFuture {
        Box::pin((self.func)())
    }
}

/// Create a service refresher from an async closure
pub fn service_refresher<F, Fut>(func: F) -> FnServiceRefresher<F>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send + 'static,
{
    FnServiceRefresher { func }
}

/// Wrapper around an rmcp service connection
pub struct MCPServer {
    /// Unique identifier for this server
    id: String,

    /// URI of the server (optional - only used for reconnection)
    uri: Option<String>,

    /// The underlying rmcp service (None if not connected)
    service: Arc<RwLock<Option<RunningService<RoleClient, ()>>>>,

    /// Optional service refresher callback (for JWT expiration, etc.)
    refresher: Option<Arc<dyn ServiceRefresher>>,
}

impl std::fmt::Debug for MCPServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MCPServer")
            .field("id", &self.id)
            .field("uri", &self.uri)
            .field("has_refresher", &self.refresher.is_some())
            .finish()
    }
}

impl MCPServer {
    /// Create a new MCP server from an existing RunningService
    ///
    /// For simple cases without token refresh. For JWT/auth scenarios,
    /// use `with_service_refresher()` instead.
    ///
    /// # Example
    /// ```no_run
    /// use rmcp::transport::StreamableHttpClientTransport;
    /// use rmcp::ServiceExt;
    ///
    /// let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp")
    ///     .with_header("Authorization", "Bearer token");
    /// let service = ().serve(transport).await?;
    /// let server = MCPServer::from_service("my-server", service);
    /// ```
    pub fn from_service(id: impl Into<String>, service: RunningService<RoleClient, ()>) -> Self {
        let id = id.into();
        tracing::info!("[MCPServer] Created MCP server '{}'", id);

        Self {
            id,
            uri: None,
            service: Arc::new(RwLock::new(Some(service))),
            refresher: None,
        }
    }

    /// Create an MCP server with a service refresher callback
    ///
    /// The refresher is called before EVERY operation (list_tools, call_tool).
    /// This is useful for JWT tokens that expire and need periodic refresh.
    ///
    /// # Example
    /// ```no_run
    /// use std::sync::Arc;
    /// use std::time::{Duration, Instant};
    /// use tokio::sync::RwLock;
    /// use rmcp::transport::StreamableHttpClientTransport;
    /// use rmcp::ServiceExt;
    ///
    /// // Track when we last refreshed
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
    ///
    ///         async move {
    ///             // Check if we need to refresh (e.g., every 50 minutes for 1hr JWT)
    ///             let mut last = last_refresh.write().await;
    ///             if last.elapsed() < Duration::from_secs(50 * 60) {
    ///                 return Ok(None); // Still valid
    ///             }
    ///
    ///             // Get fresh JWT and create new service
    ///             let token = jwt.get_fresh_token().await?;
    ///             let transport = StreamableHttpClientTransport::from_uri("https://my-backend/mcp")
    ///                 .with_header("Authorization", format!("Bearer {}", token));
    ///             let service = ().serve(transport).await?;
    ///
    ///             *last = Instant::now();
    ///             Ok(Some(service)) // Use this new service
    ///         }
    ///     }
    /// };
    ///
    /// // Create initial service
    /// let initial_token = jwt_provider.get_fresh_token().await?;
    /// let transport = StreamableHttpClientTransport::from_uri("https://my-backend/mcp")
    ///     .with_header("Authorization", format!("Bearer {}", initial_token));
    /// let service = ().serve(transport).await?;
    ///
    /// let server = MCPServer::with_service_refresher("my-server", service, refresher);
    /// ```
    pub fn with_service_refresher<F, Fut>(
        id: impl Into<String>,
        initial_service: RunningService<RoleClient, ()>,
        refresher: F,
    ) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send + 'static,
    {
        let id = id.into();
        tracing::info!("[MCPServer] Created MCP server '{}' with service refresher", id);

        Self {
            id,
            uri: None,
            service: Arc::new(RwLock::new(Some(initial_service))),
            refresher: Some(Arc::new(service_refresher(refresher))),
        }
    }

    /// Create a new MCP server and connect to it using a simple URI
    ///
    /// This is a convenience method for simple cases. For more control (auth, custom headers),
    /// use `from_service()` or `with_service_refresher()` instead.
    pub async fn new(config: MCPServerConfig) -> Result<Self> {
        let id = config.id.clone();
        let uri = config.uri.clone();

        tracing::info!("[MCPServer] Connecting to '{}' at {}", id, uri);

        let service = Self::create_service(&uri).await?;

        Ok(Self {
            id,
            uri: Some(uri),
            service: Arc::new(RwLock::new(Some(service))),
            refresher: None,
        })
    }

    /// Create an rmcp service connection
    async fn create_service(uri: &str) -> Result<RunningService<RoleClient, ()>> {
        let transport_config = StreamableHttpClientTransportConfig::with_uri(uri);
        let transport: HttpClientTransport = HttpClientTransport::from_config(transport_config);

        // Serve the transport to create a service
        let service = ().serve(transport).await?;

        Ok(service)
    }

    /// Get the server ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the server URI (if available)
    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }

    /// Check if the server is connected
    pub async fn is_connected(&self) -> bool {
        self.service.read().await.is_some()
    }

    /// Ensure service is valid, calling refresher if needed
    ///
    /// This is called before every operation (list_tools, call_tool).
    async fn ensure_service_valid(&self) -> Result<()> {
        // If we have a refresher, call it to check if we need a new service
        if let Some(ref refresher) = self.refresher {
            tracing::debug!("[MCPServer] Checking if service needs refresh for '{}'", self.id);

            match refresher.refresh().await {
                Ok(Some(new_service)) => {
                    // Refresher returned a new service, replace the old one
                    tracing::info!("[MCPServer] Refreshing service for '{}'", self.id);
                    let mut service_guard = self.service.write().await;
                    *service_guard = Some(new_service);
                    tracing::info!("[MCPServer] Service refreshed successfully for '{}'", self.id);
                }
                Ok(None) => {
                    // No refresh needed, service is still valid
                    tracing::trace!("[MCPServer] Service still valid for '{}'", self.id);
                }
                Err(e) => {
                    tracing::warn!(
                        "[MCPServer] Service refresh failed for '{}': {}",
                        self.id,
                        e
                    );
                    // Don't fail the operation, try to use existing service
                    // The operation itself will fail if the service is truly invalid
                }
            }
        }

        Ok(())
    }

    /// Reconnect to the server
    ///
    /// Only works for servers created with `new()` that have a URI.
    /// Servers created with `from_service()` cannot be reconnected automatically.
    pub async fn reconnect(&self) -> Result<()> {
        let uri = self.uri.as_ref()
            .ok_or_else(|| anyhow!("Cannot reconnect server '{}': no URI available (created from external service)", self.id))?;

        tracing::info!("[MCPServer] Reconnecting to '{}'", self.id);

        let mut service_guard = self.service.write().await;

        // Drop old connection
        *service_guard = None;

        // Create new connection
        let service = Self::create_service(uri).await?;
        *service_guard = Some(service);

        tracing::info!("[MCPServer] Successfully reconnected to '{}'", self.id);
        Ok(())
    }

    /// List all tools available on this server
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        // Ensure service is valid (refreshes if needed)
        self.ensure_service_valid().await?;

        let service_guard = self.service.read().await;
        let service = service_guard
            .as_ref()
            .ok_or_else(|| anyhow!("MCP server '{}' is not connected", self.id))?;

        tracing::debug!("[MCPServer] Listing tools from '{}'", self.id);

        let result: ListToolsResult = service.list_tools(Default::default()).await?;

        tracing::info!(
            "[MCPServer] Got {} tools from '{}'",
            result.tools.len(),
            self.id
        );

        Ok(result.tools)
    }

    /// Call a tool on this server
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Map<String, Value>>,
    ) -> Result<CallToolResult> {
        // Ensure service is valid (refreshes if needed)
        self.ensure_service_valid().await?;

        let service_guard = self.service.read().await;
        let service = service_guard
            .as_ref()
            .ok_or_else(|| anyhow!("MCP server '{}' is not connected", self.id))?;

        tracing::info!(
            "[MCPServer] Calling tool '{}' on server '{}'",
            name,
            self.id
        );
        tracing::debug!("[MCPServer] Arguments: {:?}", arguments);

        let result = service
            .call_tool(CallToolRequestParams {
                meta: None,
                name: name.to_string().into(),
                arguments,
                task: None,
            })
            .await?;

        tracing::debug!("[MCPServer] Tool call completed for '{}'", name);

        Ok(result)
    }

    /// Health check - try to list tools to verify connection
    pub async fn health_check(&self) -> Result<()> {
        self.list_tools().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires a running MCP server
    async fn test_mcp_server_connection() {
        let config = MCPServerConfig::new("test-server", "http://localhost:8005/mcp");

        let server = MCPServer::new(config).await.unwrap();
        assert!(server.is_connected().await);

        let tools = server.list_tools().await.unwrap();
        assert!(!tools.is_empty());
    }
}
