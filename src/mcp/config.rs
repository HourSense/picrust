//! MCP Server Configuration
//!
//! Configuration types for MCP servers

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for a single MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    /// Unique identifier for this server (used for namespacing tools)
    pub id: String,

    /// URI of the MCP server (e.g., "http://localhost:8005/mcp")
    pub uri: String,

    /// Whether this server is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Number of reconnection attempts on failure
    #[serde(default = "default_reconnect_attempts")]
    pub reconnect_attempts: u32,

    /// Optional health check interval in seconds
    pub health_check_interval_secs: Option<u64>,
}

fn default_enabled() -> bool {
    true
}

fn default_reconnect_attempts() -> u32 {
    3
}

impl MCPServerConfig {
    /// Create a new MCP server configuration
    pub fn new(id: impl Into<String>, uri: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            uri: uri.into(),
            enabled: true,
            reconnect_attempts: 3,
            health_check_interval_secs: None,
        }
    }

    /// Set whether this server is enabled
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set reconnection attempts
    pub fn with_reconnect_attempts(mut self, attempts: u32) -> Self {
        self.reconnect_attempts = attempts;
        self
    }

    /// Set health check interval
    pub fn with_health_check_interval(mut self, interval_secs: u64) -> Self {
        self.health_check_interval_secs = Some(interval_secs);
        self
    }

    /// Get health check interval as Duration
    pub fn health_check_interval(&self) -> Option<Duration> {
        self.health_check_interval_secs.map(Duration::from_secs)
    }
}

/// Global MCP configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MCPConfig {
    /// List of MCP servers to connect to
    pub servers: Vec<MCPServerConfig>,

    /// Global timeout for MCP tool calls in milliseconds
    pub global_timeout_ms: Option<u64>,
}

impl MCPConfig {
    /// Create a new empty MCP configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a server configuration
    pub fn add_server(mut self, server: MCPServerConfig) -> Self {
        self.servers.push(server);
        self
    }

    /// Set global timeout
    pub fn with_global_timeout(mut self, timeout_ms: u64) -> Self {
        self.global_timeout_ms = Some(timeout_ms);
        self
    }

    /// Get global timeout as Duration
    pub fn global_timeout(&self) -> Option<Duration> {
        self.global_timeout_ms.map(Duration::from_millis)
    }
}
