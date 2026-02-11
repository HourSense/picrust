//! Swappable LLM provider for runtime model switching
//!
//! Wraps any `LlmProvider` and allows swapping the underlying provider at runtime.
//! This enables features like switching between "pro" and "fast" models without
//! restarting the agent.
//!
//! # Example
//!
//! ```ignore
//! use shadow_agent_sdk::llm::{GeminiProvider, LlmProvider, SwappableLlmProvider};
//!
//! // Create initial provider
//! let fast = Arc::new(GeminiProvider::new("key")?.with_model("gemini-3-flash-preview"));
//! let swappable = SwappableLlmProvider::new(fast);
//!
//! // Get a handle for external switching
//! let handle = swappable.handle();
//!
//! // Use with agent (agent sees Arc<dyn LlmProvider>)
//! let llm: Arc<dyn LlmProvider> = Arc::new(swappable);
//! let agent = StandardAgent::new(config, llm);
//!
//! // Later, switch to pro model (from UI handler, etc.)
//! let pro = Arc::new(GeminiProvider::new("key")?.with_model("gemini-3-pro-preview"));
//! handle.set_provider(pro).await;
//! ```

use anyhow::Result;
use futures::stream::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::provider::LlmProvider;
use super::types::{
    Message, MessageResponse, StreamEvent, SystemPrompt, ThinkingConfig, ToolChoice,
    ToolDefinition,
};

/// A swappable LLM provider that delegates to an inner provider which can be
/// changed at runtime.
///
/// The agent sees this as a normal `Arc<dyn LlmProvider>`. External code uses
/// the [`LlmProviderHandle`] (obtained via [`handle()`](SwappableLlmProvider::handle))
/// to swap the underlying provider between turns.
pub struct SwappableLlmProvider {
    inner: Arc<RwLock<Arc<dyn LlmProvider>>>,
}

impl SwappableLlmProvider {
    /// Create a new swappable provider wrapping the given initial provider.
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(provider)),
        }
    }

    /// Get a handle that can be used to swap the provider from outside the agent.
    ///
    /// The handle is cheap to clone and can be shared across threads.
    pub fn handle(&self) -> LlmProviderHandle {
        LlmProviderHandle {
            inner: self.inner.clone(),
        }
    }
}

/// Handle for swapping the LLM provider from outside the agent loop.
///
/// Obtained via [`SwappableLlmProvider::handle()`]. Multiple handles can coexist;
/// they all point to the same inner provider.
#[derive(Clone)]
pub struct LlmProviderHandle {
    inner: Arc<RwLock<Arc<dyn LlmProvider>>>,
}

impl LlmProviderHandle {
    /// Swap the underlying LLM provider.
    ///
    /// The new provider will be used for the next LLM call. Any in-flight
    /// request that already obtained a reference to the old provider will
    /// complete using the old provider.
    pub async fn set_provider(&self, provider: Arc<dyn LlmProvider>) {
        let mut guard = self.inner.write().await;
        *guard = provider;
    }

    /// Get the current model name.
    pub async fn current_model(&self) -> String {
        let guard = self.inner.read().await;
        guard.model()
    }
}

#[async_trait::async_trait]
impl LlmProvider for SwappableLlmProvider {
    async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<String> {
        let provider = self.inner.read().await.clone();
        provider
            .send_message(user_message, conversation_history, system_prompt, session_id)
            .await
    }

    async fn send_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<MessageResponse> {
        let provider = self.inner.read().await.clone();
        provider
            .send_with_tools_and_system(messages, system, tools, tool_choice, thinking, session_id)
            .await
    }

    async fn stream_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let provider = self.inner.read().await.clone();
        provider
            .stream_with_tools_and_system(
                messages, system, tools, tool_choice, thinking, session_id,
            )
            .await
    }

    fn model(&self) -> String {
        match self.inner.try_read() {
            Ok(guard) => guard.model(),
            Err(_) => String::new(),
        }
    }

    fn provider_name(&self) -> &str {
        // Provider names are static strings, so we can match the inner
        // provider's name to a 'static &str to avoid lifetime issues.
        let name = match self.inner.try_read() {
            Ok(guard) => guard.provider_name().to_string(),
            Err(_) => return "swappable",
        };
        match name.as_str() {
            "anthropic" => "anthropic",
            "gemini" => "gemini",
            _ => "unknown",
        }
    }

    fn create_variant(&self, model: &str, max_tokens: u32) -> Arc<dyn LlmProvider> {
        // For variants (e.g., conversation naming), we create from the current
        // inner provider. The variant is NOT swappable - it's a lightweight
        // fixed-model clone used for specific purposes like naming.
        if let Ok(guard) = self.inner.try_read() {
            guard.create_variant(model, max_tokens)
        } else {
            Arc::new(Self {
                inner: self.inner.clone(),
            })
        }
    }
}
