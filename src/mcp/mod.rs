//! MCP (Model Context Protocol) Support
//!
//! This module provides integration with MCP servers, allowing agents to use
//! tools from external MCP servers as if they were native tools.
//!
//! # Architecture
//!
//! - `MCPServer`: Wraps rmcp service, manages connection to a single MCP server
//! - `MCPServerManager`: Manages multiple MCP servers
//! - `MCPToolAdapter`: Adapts MCP tools to implement the Tool trait
//! - `MCPToolProvider`: Implements ToolProvider to expose MCP tools to the registry
//!
//! # Usage
//!
//! ## Recommended: With JWT Refresh (For Proxied/Auth Servers)
//!
//! For servers requiring JWT tokens that expire:
//!
//! ```ignore
//! use shadow_agent_sdk::mcp::{MCPServerManager, MCPToolProvider};
//! use rmcp::transport::StreamableHttpClientTransport;
//! use rmcp::ServiceExt;
//! use std::sync::Arc;
//! use std::time::{Duration, Instant};
//! use tokio::sync::RwLock;
//!
//! // Track when we last refreshed
//! let last_refresh = Arc::new(RwLock::new(Instant::now()));
//! let jwt_provider = Arc::new(MyJwtProvider::new());
//!
//! // Create refresher callback (called before EVERY MCP operation)
//! let refresher = {
//!     let last_refresh = last_refresh.clone();
//!     let jwt = jwt_provider.clone();
//!
//!     move || {
//!         let last_refresh = last_refresh.clone();
//!         let jwt = jwt.clone();
//!
//!         async move {
//!             // Check if we need to refresh (e.g., every 50 minutes for 1hr JWT)
//!             let mut last = last_refresh.write().await;
//!             if last.elapsed() < Duration::from_secs(50 * 60) {
//!                 return Ok(None); // Still valid, no refresh needed
//!             }
//!
//!             // Get fresh JWT and create new service
//!             let token = jwt.get_fresh_token().await?;
//!             let transport = StreamableHttpClientTransport::from_uri("https://backend/mcp-proxy")
//!                 .with_header("Authorization", format!("Bearer {}", token));
//!             let service = ().serve(transport).await?;
//!
//!             *last = Instant::now();
//!             Ok(Some(service)) // Replace with new service
//!         }
//!     }
//! };
//!
//! // Create initial service
//! let initial_token = jwt_provider.get_fresh_token().await?;
//! let transport = StreamableHttpClientTransport::from_uri("https://backend/mcp-proxy")
//!     .with_header("Authorization", format!("Bearer {}", initial_token));
//! let service = ().serve(transport).await?;
//!
//! // Add to manager with refresher
//! let mcp_manager = Arc::new(MCPServerManager::new());
//! mcp_manager.add_service_with_refresher("remote-mcp", service, refresher).await?;
//!
//! // Add to tool registry
//! let mcp_provider = Arc::new(MCPToolProvider::new(mcp_manager));
//! tool_registry.add_provider(mcp_provider).await?;
//! ```
//!
//! ## Simple: Static Auth Headers
//!
//! For servers with static auth (no token refresh):
//!
//! ```ignore
//! use shadow_agent_sdk::mcp::{MCPServerManager, MCPToolProvider};
//! use rmcp::transport::StreamableHttpClientTransport;
//! use rmcp::ServiceExt;
//! use std::sync::Arc;
//!
//! // Create transport with static auth
//! let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp")
//!     .with_header("Authorization", "Bearer static-token");
//!
//! let service = ().serve(transport).await?;
//!
//! // Add to manager (no refresher)
//! let mcp_manager = Arc::new(MCPServerManager::new());
//! mcp_manager.add_service("filesystem", service).await?;
//!
//! // Add to tool registry
//! let mcp_provider = Arc::new(MCPToolProvider::new(mcp_manager));
//! tool_registry.add_provider(mcp_provider).await?;
//! ```
//!
//! ## Simple: URI-based Configuration
//!
//! For simple cases without auth:
//!
//! ```ignore
//! use shadow_agent_sdk::mcp::{MCPServerManager, MCPServerConfig, MCPToolProvider};
//!
//! let mcp_manager = Arc::new(MCPServerManager::new());
//! mcp_manager.add_server(MCPServerConfig::new(
//!     "filesystem",
//!     "http://localhost:8005/mcp"
//! )).await?;
//!
//! let mcp_provider = Arc::new(MCPToolProvider::new(mcp_manager));
//! tool_registry.add_provider(mcp_provider).await?;
//! ```
//!
//! # Tool Namespacing
//!
//! MCP tools are automatically namespaced with their server ID to avoid conflicts:
//! - Server ID: `filesystem`
//! - Original tool name: `read_file`
//! - Exposed name: `filesystem__read_file`

mod config;
mod manager;
mod provider;
mod server;
mod tool_adapter;

// Public exports
pub use config::{MCPConfig, MCPServerConfig};
pub use manager::{MCPServerManager, MCPToolInfo};
pub use provider::MCPToolProvider;
pub use server::{service_refresher, MCPServer, ServiceRefreshFuture, ServiceRefresher};
pub use tool_adapter::MCPToolAdapter;
