//! LLM Provider trait
//!
//! Abstracts the LLM interface so that different providers (Anthropic, Gemini, etc.)
//! can be used interchangeably with the StandardAgent.

use anyhow::Result;
use futures::stream::Stream;
use std::pin::Pin;
use std::sync::Arc;

use super::types::{
    Message, MessageResponse, StreamEvent, SystemPrompt, ThinkingConfig, ToolChoice,
    ToolDefinition,
};

/// Trait for LLM providers that can be used with StandardAgent.
///
/// This trait abstracts the interface needed by the agent loop, allowing
/// different LLM backends (Anthropic, Gemini, etc.) to be used interchangeably.
///
/// All providers work with the same internal message types (which follow Anthropic's
/// format). Providers that use a different wire format (e.g., Gemini) handle
/// translation internally.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a simple message and get a text response (no tool calling).
    ///
    /// Used by ConversationNamer and other simple use cases.
    async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<String>;

    /// Send a request with tools and system prompt, returning the full response.
    ///
    /// This is the primary method used by the agent loop for non-streaming requests.
    async fn send_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<MessageResponse>;

    /// Stream a request with tools and system prompt.
    ///
    /// Returns an async stream of StreamEvent that yields events as they arrive.
    /// This is the primary method used by the agent loop for streaming requests.
    async fn stream_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>>;

    /// Get the current model name.
    fn model(&self) -> String;

    /// Get the provider name (e.g., "anthropic", "gemini").
    fn provider_name(&self) -> &str;

    /// Create a lightweight variant of this provider with a different model and max tokens.
    ///
    /// Used by ConversationNamer to create a Haiku-based namer that shares
    /// the same authentication configuration.
    fn create_variant(&self, model: &str, max_tokens: u32) -> Arc<dyn LlmProvider>;
}
