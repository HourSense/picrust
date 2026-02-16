//! Authentication providers for LLM APIs
//!
//! Supports both static and dynamic authentication:
//! - Static: API key set once at creation
//! - Dynamic: Callback that provides fresh credentials before each request
//!
//! # Example: Dynamic auth with JWT refresh
//!
//! ```ignore
//! use picrust::llm::{AnthropicProvider, AuthConfig};
//!
//! let llm = AnthropicProvider::with_auth_provider(|| async {
//!     // Fetch fresh JWT from your auth service
//!     let jwt = refresh_jwt().await?;
//!     Ok(AuthConfig {
//!         api_key: jwt,
//!         base_url: Some("https://proxy.example.com/v1/messages".into()),
//!     })
//! });
//! ```

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Authentication configuration for API requests
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// API key or token for authentication
    pub api_key: String,
    /// Optional custom base URL (overrides default API endpoint)
    pub base_url: Option<String>,
}

impl AuthConfig {
    /// Create a new auth config with just an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
        }
    }

    /// Create a new auth config with API key and custom base URL
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: Some(base_url.into()),
        }
    }
}

/// Type alias for the boxed future returned by auth providers
pub type AuthFuture<'a> = Pin<Box<dyn Future<Output = Result<AuthConfig>> + Send + 'a>>;

/// Trait for providing authentication credentials dynamically
///
/// Implement this trait to provide fresh credentials before each API request.
/// This is useful for:
/// - JWT tokens that expire frequently
/// - Proxy servers that require per-request auth
/// - Rotating API keys
pub trait AuthProvider: Send + Sync {
    /// Get authentication configuration
    ///
    /// Called before each API request. Implementations should handle caching
    /// and refresh logic internally.
    fn get_auth(&self) -> AuthFuture<'_>;
}

/// Wrapper to implement AuthProvider for async closures
pub struct FnAuthProvider<F> {
    func: F,
}

impl<F, Fut> AuthProvider for FnAuthProvider<F>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<AuthConfig>> + Send + 'static,
{
    fn get_auth(&self) -> AuthFuture<'_> {
        Box::pin((self.func)())
    }
}

/// Create an auth provider from an async closure
///
/// # Example
///
/// ```ignore
/// let provider = auth_provider(|| async {
///     let token = fetch_fresh_token().await?;
///     Ok(AuthConfig::new(token))
/// });
/// ```
pub fn auth_provider<F, Fut>(func: F) -> FnAuthProvider<F>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<AuthConfig>> + Send + 'static,
{
    FnAuthProvider { func }
}

/// Internal auth source - either static or dynamic
pub(crate) enum AuthSource {
    /// Static credentials set at creation time
    Static(AuthConfig),
    /// Dynamic credentials from a provider
    Dynamic(Arc<dyn AuthProvider>),
}

impl Clone for AuthSource {
    fn clone(&self) -> Self {
        match self {
            AuthSource::Static(config) => AuthSource::Static(config.clone()),
            AuthSource::Dynamic(provider) => AuthSource::Dynamic(Arc::clone(provider)),
        }
    }
}

impl AuthSource {
    /// Get auth config (either returns static or calls provider)
    pub(crate) async fn get_auth(&self) -> Result<AuthConfig> {
        match self {
            AuthSource::Static(config) => Ok(config.clone()),
            AuthSource::Dynamic(provider) => provider.get_auth().await,
        }
    }
}
