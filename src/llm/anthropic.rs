//! Anthropic API client using HTTP requests
//!
//! This module provides a direct HTTP client for the Anthropic Messages API,
//! without relying on any community SDK.
//!
//! # Authentication
//!
//! Supports both static and dynamic authentication:
//!
//! ```ignore
//! // Static auth from environment
//! let llm = AnthropicProvider::from_env()?;
//!
//! // Static auth with explicit key
//! let llm = AnthropicProvider::new("sk-...")?;
//!
//! // Dynamic auth with callback (for JWT/proxy scenarios)
//! let llm = AnthropicProvider::with_auth_provider(|| async {
//!     let jwt = refresh_token().await?;
//!     Ok(AuthConfig::with_base_url(jwt, "https://proxy.example.com/v1/messages"))
//! });
//! ```

use anyhow::{Context, Result};
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use std::env;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;

use super::auth::{auth_provider, AuthConfig, AuthProvider, AuthSource};
use super::provider::LlmProvider;
use super::types::{
    Message, MessageRequest, MessageResponse, RawStreamEvent, StreamEvent, SystemPrompt,
    ThinkingConfig, ToolChoice, ToolDefinition,
};

const DEFAULT_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic LLM provider using direct HTTP calls
///
/// Supports both static and dynamic authentication for scenarios like:
/// - Standard API key authentication
/// - JWT tokens with expiration (proxy servers)
/// - Per-request credential refresh
pub struct AnthropicProvider {
    client: Client,
    auth: AuthSource,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from environment variables
    ///
    /// Reads from:
    /// - `ANTHROPIC_API_KEY` (required)
    /// - `ANTHROPIC_MODEL` (required)
    /// - `ANTHROPIC_BASE_URL` (optional, defaults to Anthropic API)
    /// - `ANTHROPIC_MAX_TOKENS` (optional, defaults to 32000)
    pub fn from_env() -> Result<Self> {
        tracing::info!("Creating Anthropic provider from environment");

        let api_key = env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY environment variable not set")?;

        let base_url = env::var("ANTHROPIC_BASE_URL").ok();

        let model = env::var("ANTHROPIC_MODEL")
            .context("ANTHROPIC_MODEL environment variable not set")?;

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32000); // Must be > thinking.budget_tokens (16000)

        tracing::info!("Using model: {}", model);
        tracing::info!("Max tokens: {}", max_tokens);
        if let Some(ref url) = base_url {
            tracing::info!("Using custom base URL: {}", url);
        }

        let client = Client::new();

        Ok(Self {
            client,
            auth: AuthSource::Static(AuthConfig {
                api_key,
                base_url,
            }),
            model,
            max_tokens,
        })
    }

    /// Create a new Anthropic provider with a specific API key
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = Client::new();

        Ok(Self {
            client,
            auth: AuthSource::Static(AuthConfig::new(api_key)),
            model: "".to_string(),
            max_tokens: 32000,
        })
    }

    /// Create a new Anthropic provider with an auth provider callback
    ///
    /// The callback is called before each API request to get fresh credentials.
    /// This is useful for:
    /// - JWT tokens that expire frequently
    /// - Proxy servers that require per-request auth
    /// - Rotating API keys
    ///
    /// # Example
    ///
    /// ```ignore
    /// let llm = AnthropicProvider::with_auth_provider(|| async {
    ///     let jwt = my_auth_service.get_fresh_token().await?;
    ///     Ok(AuthConfig::with_base_url(jwt, "https://my-proxy.com/v1/messages"))
    /// });
    /// ```
    pub fn with_auth_provider<F, Fut>(provider: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AuthConfig>> + Send + 'static,
    {
        Self {
            client: Client::new(),
            auth: AuthSource::Dynamic(Arc::new(auth_provider(provider))),
            model: "".to_string(),
            max_tokens: 32000,
        }
    }

    /// Create a new Anthropic provider with a trait object auth provider
    ///
    /// Use this when you have a custom `AuthProvider` implementation.
    pub fn with_auth_provider_boxed(provider: Arc<dyn AuthProvider>) -> Self {
        Self {
            client: Client::new(),
            auth: AuthSource::Dynamic(provider),
            model: "".to_string(),
            max_tokens: 32000,
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the max tokens for responses
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the current max tokens
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Create a new provider with a different model, sharing the same auth
    ///
    /// This is useful for creating lightweight models (like Haiku) for simple tasks
    /// while reusing the same authentication configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let main_llm = AnthropicProvider::from_env()?;
    /// let haiku_llm = main_llm.with_model_override("claude-3-5-haiku-20241022");
    /// ```
    pub fn with_model_override(&self, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            auth: self.auth.clone(),
            model: model.into(),
            max_tokens: self.max_tokens,
        }
    }

    /// Create a new provider with a different model and max tokens, sharing the same auth
    pub fn with_model_and_tokens_override(
        &self,
        model: impl Into<String>,
        max_tokens: u32,
    ) -> Self {
        Self {
            client: Client::new(),
            auth: self.auth.clone(),
            model: model.into(),
            max_tokens,
        }
    }

    /// Send a message and get a complete response (no tool calling)
    ///
    /// This is a simple method for basic conversations without tools.
    pub async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<String> {
        tracing::info!("Sending message to Anthropic API");
        tracing::debug!("User message: {}", user_message);
        tracing::debug!(
            "Conversation history length: {} messages",
            conversation_history.len()
        );
        tracing::debug!("System prompt: {:?}", system_prompt);

        // Build messages array
        let mut messages: Vec<Message> = conversation_history.to_vec();
        messages.push(Message::user(user_message));

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system: system_prompt.map(|s| SystemPrompt::Text(s.to_string())),
            tools: None,
            tool_choice: None,
            thinking: None,
            temperature: None,
            stream: None,
        };

        let response = self.send_request(&request, session_id).await?;
        Ok(response.text())
    }

    /// Send a request with tools and get the full response
    ///
    /// This method supports tool calling and extended thinking,
    /// returning the complete response including any tool use and thinking blocks.
    pub async fn send_with_tools(
        &self,
        messages: Vec<Message>,
        system_prompt: Option<&str>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
    ) -> Result<MessageResponse> {
        tracing::info!("Sending message with tools to Anthropic API");
        tracing::debug!("Messages count: {}", messages.len());
        tracing::debug!("Tools count: {}", tools.len());
        tracing::debug!("Thinking enabled: {}", thinking.is_some());

        // When thinking is enabled, temperature must be 1 (required by Anthropic API)
        let temperature = if thinking.is_some() { Some(1.0) } else { None };

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system: system_prompt.map(|s| SystemPrompt::Text(s.to_string())),
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            thinking,
            temperature,
            stream: None,
        };

        self.send_request(&request, None).await
    }

    /// Send a request with tools and system prompt (with caching support)
    ///
    /// This variant accepts `Option<SystemPrompt>` instead of `Option<&str>`,
    /// allowing for prompt caching via SystemPrompt::Blocks.
    pub async fn send_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<MessageResponse> {
        tracing::info!("Sending message with tools to Anthropic API");
        tracing::debug!("Messages count: {}", messages.len());
        tracing::debug!("Tools count: {}", tools.len());
        tracing::debug!("Thinking enabled: {}", thinking.is_some());

        // When thinking is enabled, temperature must be 1 (required by Anthropic API)
        let temperature = if thinking.is_some() { Some(1.0) } else { None };

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            thinking,
            temperature,
            stream: None,
        };

        self.send_request(&request, session_id).await
    }

    /// Send a raw request to the Anthropic API
    async fn send_request(&self, request: &MessageRequest, session_id: Option<&str>) -> Result<MessageResponse> {
        tracing::debug!("Model: {}", request.model);
        tracing::debug!("Max tokens: {}", request.max_tokens);

        // Get auth credentials (static or from provider)
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_url = auth_config.base_url.as_deref().unwrap_or(DEFAULT_API_URL);

        let request_json = serde_json::to_string(request)
            .context("Failed to serialize request")?;
        tracing::debug!("Request JSON: {}", request_json);

        let mut request_builder = self
            .client
            .post(api_url)
            .header("Content-Type", "application/json")
            .header("x-api-key", &auth_config.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", "interleaved-thinking-2025-05-14");

        // Add agent-session-id header if session_id is provided
        if let Some(sid) = session_id {
            request_builder = request_builder.header("agent-session-id", sid);
        }

        let response = request_builder
            .body(request_json)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .context("Failed to read response body")?;

        tracing::debug!("Response status: {}", status);
        tracing::debug!("Response body: {}", response_text);

        if !status.is_success() {
            tracing::error!("API error: {} - {}", status, response_text);
            anyhow::bail!("Anthropic API error ({}): {}", status, response_text);
        }

        let response: MessageResponse = serde_json::from_str(&response_text)
            .context("Failed to parse API response")?;

        tracing::info!("Received response from Anthropic API");
        tracing::debug!("Response ID: {}", response.id);
        tracing::debug!("Stop reason: {:?}", response.stop_reason);
        tracing::debug!("Content blocks: {}", response.content.len());
        tracing::debug!(
            "Usage: {} input, {} output tokens",
            response.usage.input_tokens,
            response.usage.output_tokens
        );

        // Log cache metrics if present
        if let Some(cache_creation_tokens) = response.usage.cache_creation_input_tokens {
            tracing::debug!("Cache creation tokens: {}", cache_creation_tokens);
        }
        if let Some(cache_read_tokens) = response.usage.cache_read_input_tokens {
            tracing::debug!("Cache read tokens: {}", cache_read_tokens);
        }

        Ok(response)
    }

    /// Stream a message and get incremental responses via SSE
    ///
    /// Returns an async stream of `StreamEvent` that yields events as they arrive.
    /// The stream follows this flow:
    /// 1. `MessageStart` - initial message metadata
    /// 2. `ContentBlockStart` - start of each content block
    /// 3. `ContentBlockDelta` - incremental updates (text, tool input, thinking)
    /// 4. `ContentBlockStop` - end of each content block
    /// 5. `MessageDelta` - final stop reason and usage
    /// 6. `MessageStop` - stream complete
    pub async fn stream_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        tracing::info!("Streaming message from Anthropic API");
        tracing::debug!("User message: {}", user_message);

        // Build messages array
        let mut messages: Vec<Message> = conversation_history.to_vec();
        messages.push(Message::user(user_message));

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system: system_prompt.map(|s| SystemPrompt::Text(s.to_string())),
            tools: None,
            tool_choice: None,
            thinking: None,
            temperature: None,
            stream: Some(true),
        };

        self.send_streaming_request(&request, None).await
    }

    /// Stream a message with tools and get incremental responses
    ///
    /// Similar to `stream_message` but supports tools and extended thinking.
    pub async fn stream_with_tools(
        &self,
        messages: Vec<Message>,
        system_prompt: Option<&str>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        tracing::info!("Streaming message with tools from Anthropic API");
        tracing::debug!("Messages count: {}", messages.len());
        tracing::debug!("Tools count: {}", tools.len());
        tracing::debug!("Thinking enabled: {}", thinking.is_some());

        // When thinking is enabled, temperature must be 1 (required by Anthropic API)
        let temperature = if thinking.is_some() { Some(1.0) } else { None };

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system: system_prompt.map(|s| SystemPrompt::Text(s.to_string())),
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            thinking,
            temperature,
            stream: Some(true),
        };

        self.send_streaming_request(&request, None).await
    }

    /// Stream a message with tools and system prompt (with caching support)
    ///
    /// This variant accepts `Option<SystemPrompt>` instead of `Option<&str>`,
    /// allowing for prompt caching via SystemPrompt::Blocks.
    pub async fn stream_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        tracing::info!("Streaming message with tools from Anthropic API");
        tracing::debug!("Messages count: {}", messages.len());
        tracing::debug!("Tools count: {}", tools.len());
        tracing::debug!("Thinking enabled: {}", thinking.is_some());

        // When thinking is enabled, temperature must be 1 (required by Anthropic API)
        let temperature = if thinking.is_some() { Some(1.0) } else { None };

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            thinking,
            temperature,
            stream: Some(true),
        };

        self.send_streaming_request(&request, session_id).await
    }

    /// Send a streaming request to the Anthropic API
    async fn send_streaming_request(
        &self,
        request: &MessageRequest,
        session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        tracing::debug!("Model: {}", request.model);
        tracing::debug!("Max tokens: {}", request.max_tokens);

        // Get auth credentials (static or from provider)
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_url = auth_config.base_url.as_deref().unwrap_or(DEFAULT_API_URL);

        let request_json =
            serde_json::to_string(request).context("Failed to serialize request")?;
        tracing::debug!("Request JSON: {}", request_json);

        let mut request_builder = self
            .client
            .post(api_url)
            .header("Content-Type", "application/json")
            .header("x-api-key", &auth_config.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("anthropic-beta", "interleaved-thinking-2025-05-14");

        // Add agent-session-id header if session_id is provided
        if let Some(sid) = session_id {
            request_builder = request_builder.header("X-Agent-Session-Id", sid);
        }

        let response = request_builder
            .body(request_json)
            .send()
            .await
            .context("Failed to send streaming request to Anthropic API")?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            tracing::error!("API error: {} - {}", status, error_text);
            anyhow::bail!("Anthropic API error ({}): {}", status, error_text);
        }

        tracing::info!("Streaming response started from Anthropic API");

        // Convert the response body stream to an async reader
        let byte_stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            byte_stream.map(|result| result.map_err(|e| std::io::Error::other(e.to_string()))),
        );
        let buf_reader = tokio::io::BufReader::new(stream_reader);

        // Create async stream that parses SSE events
        let stream = async_stream::try_stream! {
            let mut lines = buf_reader.lines();
            let mut current_event: Option<String> = None;
            let mut current_data = String::new();

            while let Some(line) = lines.next_line().await? {
                if line.starts_with("event: ") {
                    current_event = Some(line[7..].to_string());
                    current_data.clear();
                } else if line.starts_with("data: ") {
                    current_data.push_str(&line[6..]);
                } else if line.is_empty() && current_event.is_some() {
                    // Empty line signals end of event
                    if let Some(ref event_type) = current_event {
                        tracing::trace!("SSE event: {} data: {}", event_type, current_data);

                        // Parse the data based on event type
                        match parse_sse_event(event_type, &current_data) {
                            Ok(Some(event)) => yield event,
                            Ok(None) => {
                                // Ping or other non-data event
                            }
                            Err(e) => {
                                tracing::warn!("Failed to parse SSE event: {} - {}", event_type, e);
                            }
                        }
                    }
                    current_event = None;
                    current_data.clear();
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Parse an SSE event from its type and data
fn parse_sse_event(event_type: &str, data: &str) -> Result<Option<StreamEvent>> {
    match event_type {
        "message_start" | "content_block_start" | "content_block_delta" | "content_block_stop"
        | "message_delta" | "message_stop" | "ping" | "error" => {
            let raw_event: RawStreamEvent =
                serde_json::from_str(data).context("Failed to parse SSE event data")?;
            Ok(Some(raw_event.into_stream_event()))
        }
        _ => {
            tracing::debug!("Unknown SSE event type: {}", event_type);
            Ok(None)
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<String> {
        self.send_message(user_message, conversation_history, system_prompt, session_id)
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
        self.send_with_tools_and_system(messages, system, tools, tool_choice, thinking, session_id)
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
        self.stream_with_tools_and_system(messages, system, tools, tool_choice, thinking, session_id)
            .await
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn create_variant(&self, model: &str, max_tokens: u32) -> Arc<dyn LlmProvider> {
        Arc::new(self.with_model_and_tokens_override(model, max_tokens))
    }
}

/// Helper function to build a simple tool definition
pub fn define_tool(
    name: impl Into<String>,
    description: impl Into<String>,
    properties: serde_json::Value,
    required: Vec<String>,
) -> ToolDefinition {
    use super::types::{CustomTool, ToolInputSchema};

    ToolDefinition::Custom(CustomTool {
        name: name.into(),
        description: Some(description.into()),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: Some(properties),
            required: Some(required),
        },
        tool_type: None,
        cache_control: None,
    })
}
