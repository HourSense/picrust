//! MCP Server wrapper
//!
//! Wraps rmcp service to manage connections to individual MCP servers

use anyhow::{anyhow, Result};
use rmcp::model::{CallToolRequestParams, CallToolResult, Tool};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::RoleClient;
use serde_json::{Map, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

/// The concrete transport type we use for HTTP MCP connections
pub type HttpClientTransport = StreamableHttpClientTransport<reqwest::Client>;

/// Type alias for service refresher callback future
pub type ServiceRefreshFuture =
    Pin<Box<dyn Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send>>;

/// Trait for providing MCP service with refresh/reconnection logic
///
/// This is called:
/// 1. Before each MCP operation (to check JWT expiry, etc.)
/// 2. When connection failures are detected (to force reconnect)
///
/// The refresher MUST return a service. It's responsible for:
/// - Checking if the current service is still valid (token not expired)
/// - Creating a new service if needed (token expired or connection dead)
/// - Caching to avoid unnecessary reconnections
///
/// The framework will call this frequently, so implement caching!
pub trait ServiceRefresher: Send + Sync {
    /// Get or create a valid service
    ///
    /// Returns a service that is ready to use. This can be:
    /// - The same cached service (if still valid)
    /// - A newly created service (if token expired or forced reconnect)
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

    /// The underlying rmcp service (None if not connected)
    service: Arc<RwLock<Option<RunningService<RoleClient, ()>>>>,

    /// Service refresher callback (REQUIRED - handles both JWT refresh and reconnection)
    refresher: Arc<dyn ServiceRefresher>,
}

impl std::fmt::Debug for MCPServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MCPServer")
            .field("id", &self.id)
            .finish()
    }
}

impl MCPServer {
    /// Create an MCP server with a service refresher callback
    ///
    /// The refresher is REQUIRED and is called:
    /// - Before each MCP operation (to check JWT expiry, etc.)
    /// - When connection failures are detected (to force reconnect)
    ///
    /// The refresher should implement caching to avoid unnecessary reconnections.
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
    ///                     // Token still fresh, return cached service
    ///                     let cached_guard = cached.read().await;
    ///                     if let Some(service) = cached_guard.as_ref() {
    ///                         return Ok(service.clone()); // Reuse cached
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
    ///             *cached.write().await = Some(service.clone());
    ///             *last_refresh.write().await = Instant::now();
    ///
    ///             Ok(service)
    ///         }
    ///     }
    /// };
    ///
    /// let server = MCPServer::new("my-server", refresher);
    /// ```
    pub fn new<F, Fut>(id: impl Into<String>, refresher: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<RunningService<RoleClient, ()>>>> + Send + 'static,
    {
        let id = id.into();
        tracing::debug!("[MCPServer] Created MCP server '{}'", id);

        Self {
            id,
            service: Arc::new(RwLock::new(None)),
            refresher: Arc::new(service_refresher(refresher)),
        }
    }

    /// Get the server ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if the server is connected
    pub async fn is_connected(&self) -> bool {
        self.service.read().await.is_some()
    }

    /// Ensure service is valid, calling refresher if needed
    ///
    /// This is called before every operation (list_tools, call_tool).
    async fn ensure_service_valid(&self) -> Result<()> {
        tracing::debug!("[MCPServer] Checking if service needs refresh for '{}'", self.id);

        match self.refresher.refresh().await {
            Ok(Some(new_service)) => {
                // Refresher returned a new service, replace the current one
                tracing::debug!("[MCPServer] Got new service from refresher for '{}'", self.id);
                let mut service_guard = self.service.write().await;
                *service_guard = Some(new_service);
            }
            Ok(None) => {
                // Refresher said no refresh needed, keep current service
                tracing::debug!("[MCPServer] Refresher said no refresh needed for '{}'", self.id);
            }
            Err(e) => {
                tracing::warn!(
                    "[MCPServer] Service refresh failed for '{}': {}",
                    self.id,
                    e
                );
                return Err(e);
            }
        }

        Ok(())
    }

    /// List all tools available on this server
    ///
    /// This method includes automatic retry logic with reconnection.
    /// If the connection is dead, it will attempt to refresh the service and retry.
    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        self.list_tools_with_retry(2).await
    }

    /// Internal: List tools with retry logic
    async fn list_tools_with_retry(&self, max_retries: u32) -> Result<Vec<Tool>> {
        let mut attempts = 0;
        let mut last_error = None;

        tracing::info!(
            "[MCPServer] Starting list_tools for '{}' (will attempt {} times)",
            self.id,
            max_retries + 1
        );

        while attempts <= max_retries {
            tracing::info!(
                "[MCPServer] list_tools attempt {}/{} for '{}'",
                attempts + 1,
                max_retries + 1,
                self.id
            );

            // Ensure service is valid (refreshes if needed)
            if let Err(e) = self.ensure_service_valid().await {
                tracing::warn!(
                    "[MCPServer] ensure_service_valid failed for '{}': {}",
                    self.id,
                    e
                );
                last_error = Some(e);
                attempts += 1;
                continue;
            }

            let service_guard = self.service.read().await;
            let service = match service_guard.as_ref() {
                Some(s) => {
                    tracing::debug!("[MCPServer] Service is available for '{}'", self.id);
                    s
                }
                None => {
                    tracing::warn!("[MCPServer] Service is None for '{}'", self.id);
                    last_error = Some(anyhow!("MCP server '{}' is not connected", self.id));
                    attempts += 1;
                    drop(service_guard);
                    continue;
                }
            };

            // Very short timeout - this is a quick health check
            let timeout_duration = std::time::Duration::from_secs(5);
            tracing::info!(
                "[MCPServer] Calling list_tools on '{}' with {}s timeout...",
                self.id,
                timeout_duration.as_secs()
            );

            let list_future = service.list_tools(Default::default());

            match tokio::time::timeout(timeout_duration, list_future).await {
                Ok(Ok(result)) => {
                    tracing::debug!(
                        "[MCPServer] list_tools SUCCESS for '{}' - got {} tools",
                        self.id,
                        result.tools.len()
                    );
                    return Ok(result.tools);
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "[MCPServer] list_tools FAILED for '{}': {}",
                        self.id,
                        e
                    );
                    last_error = Some(e.into());

                    // Drop the read lock before attempting to reconnect
                    drop(service_guard);

                    // Connection might be dead, try to force refresh on next attempt
                    if attempts < max_retries {
                        tracing::info!(
                            "[MCPServer] Will attempt FORCED RECONNECTION for '{}' before retry",
                            self.id
                        );

                        // Drop old service to force refresher to create a new one
                        {
                            let mut write_guard = self.service.write().await;
                            *write_guard = None;
                            tracing::debug!("[MCPServer] Dropped old service for '{}'", self.id);
                        }

                        // Force service refresh by calling refresher
                        tracing::info!(
                            "[MCPServer] Calling refresher callback for '{}' to FORCE reconnect...",
                            self.id
                        );
                        match self.refresher.refresh().await {
                            Ok(Some(new_service)) => {
                                tracing::info!(
                                    "[MCPServer] Refresher provided new service for '{}'",
                                    self.id
                                );
                                let mut write_guard = self.service.write().await;
                                *write_guard = Some(new_service);
                                tracing::info!(
                                    "[MCPServer] New service installed for '{}'",
                                    self.id
                                );
                            }
                            Ok(None) => {
                                tracing::warn!(
                                    "[MCPServer] Refresher returned None after forced reconnect for '{}'",
                                    self.id
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "[MCPServer] Refresher FAILED for '{}': {}",
                                    self.id,
                                    e
                                );
                            }
                        }
                    }

                    attempts += 1;
                }
                Err(_) => {
                    tracing::error!(
                        "[MCPServer] TIMEOUT listing tools from '{}' after {}s",
                        self.id,
                        timeout_duration.as_secs()
                    );
                    last_error = Some(anyhow!(
                        "Timeout listing tools from '{}' after {}s",
                        self.id,
                        timeout_duration.as_secs()
                    ));

                    // Drop lock and try to reconnect
                    drop(service_guard);

                    if attempts < max_retries {
                        tracing::info!(
                            "[MCPServer] Will attempt FORCED RECONNECTION for '{}' after timeout",
                            self.id
                        );

                        // Drop old service to force refresher to create a new one
                        {
                            let mut write_guard = self.service.write().await;
                            *write_guard = None;
                            tracing::debug!("[MCPServer] Dropped old service for '{}'", self.id);
                        }

                        tracing::info!(
                            "[MCPServer] Calling refresher callback for '{}' to FORCE reconnect...",
                            self.id
                        );
                        match self.refresher.refresh().await {
                            Ok(Some(new_service)) => {
                                tracing::info!(
                                    "[MCPServer] Refresher provided new service for '{}'",
                                    self.id
                                );
                                let mut write_guard = self.service.write().await;
                                *write_guard = Some(new_service);
                                tracing::info!(
                                    "[MCPServer] New service installed for '{}'",
                                    self.id
                                );
                            }
                            Ok(None) => {
                                tracing::warn!(
                                    "[MCPServer] Refresher returned None after timeout/forced reconnect for '{}'",
                                    self.id
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "[MCPServer] Refresher FAILED for '{}': {}",
                                    self.id,
                                    e
                                );
                            }
                        }
                    }

                    attempts += 1;
                }
            }
        }

        tracing::error!(
            "[MCPServer] EXHAUSTED all {} attempts for list_tools on '{}'",
            max_retries + 1,
            self.id
        );
        Err(last_error.unwrap_or_else(|| {
            anyhow!(
                "Failed to list tools after {} attempts",
                max_retries + 1
            )
        }))
    }

    /// Call a tool on this server
    ///
    /// This method first checks server connectivity by calling list_tools()
    /// to ensure the server is alive before executing the actual tool call.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Map<String, Value>>,
    ) -> Result<CallToolResult> {
        // Health check: call list_tools() to verify server is up
        // This will automatically retry and reconnect if the server crashed
        tracing::info!(
            "[MCPServer] HEALTH CHECK before calling tool '{}' on '{}'",
            name,
            self.id
        );

        match self.list_tools().await {
            Ok(_) => {
                tracing::info!(
                    "[MCPServer] HEALTH CHECK PASSED for '{}' - proceeding with tool call",
                    self.id
                );
            }
            Err(e) => {
                tracing::error!(
                    "[MCPServer] HEALTH CHECK FAILED for '{}': {}",
                    self.id,
                    e
                );
                tracing::error!(
                    "[MCPServer] Tool '{}' will NOT execute - server is unreachable",
                    name
                );
                return Err(anyhow!(
                    "Health check failed for MCP server '{}': {}",
                    self.id,
                    e
                ));
            }
        }

        let service_guard = self.service.read().await;
        let service = service_guard
            .as_ref()
            .ok_or_else(|| anyhow!("MCP server '{}' is not connected", self.id))?;

        tracing::info!(
            "[MCPServer] Calling tool '{}' on server '{}'",
            name,
            self.id
        );
        tracing::debug!("[MCPServer] Tool arguments: {:?}", arguments);

        // Add timeout to tool call as well
        let timeout_duration = std::time::Duration::from_secs(120); // 2 minutes for tool execution
        tracing::info!(
            "[MCPServer] Executing '{}' with {}s timeout...",
            name,
            timeout_duration.as_secs()
        );

        let call_future = service.call_tool(CallToolRequestParams {
            meta: None,
            name: name.to_string().into(),
            arguments,
            task: None,
        });

        let result = match tokio::time::timeout(timeout_duration, call_future).await {
            Ok(Ok(result)) => {
                tracing::info!(
                    "[MCPServer] Tool '{}' executed successfully on '{}'",
                    name,
                    self.id
                );
                result
            }
            Ok(Err(e)) => {
                tracing::error!(
                    "[MCPServer] Tool '{}' FAILED on '{}': {}",
                    name,
                    self.id,
                    e
                );
                return Err(e.into());
            }
            Err(_) => {
                tracing::error!(
                    "[MCPServer] TIMEOUT executing tool '{}' on '{}' after {}s",
                    name,
                    self.id,
                    timeout_duration.as_secs()
                );
                return Err(anyhow!(
                    "Timeout calling tool '{}' on '{}' after {}s",
                    name,
                    self.id,
                    timeout_duration.as_secs()
                ));
            }
        };

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
        use rmcp::transport::StreamableHttpClientTransport;
        use rmcp::ServiceExt;

        let uri = "http://localhost:8005/mcp";

        // Create a simple refresher that always creates a new service
        let refresher = {
            let uri = uri.to_string();
            move || {
                let uri = uri.clone();
                async move {
                    let transport = StreamableHttpClientTransport::from_uri(uri.as_str());
                    let service = ().serve(transport).await?;
                    Ok(Some(service))
                }
            }
        };

        let server = MCPServer::new("test-server", refresher);
        assert!(server.is_connected().await);

        let tools = server.list_tools().await.unwrap();
        assert!(!tools.is_empty());
    }
}
