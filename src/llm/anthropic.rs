//! Anthropic API client using HTTP requests
//!
//! This module provides a direct HTTP client for the Anthropic Messages API,
//! without relying on any community SDK.

use anyhow::{Context, Result};
use reqwest::Client;
use std::env;

use super::types::{
    Message, MessageRequest, MessageResponse, ThinkingConfig, ToolChoice, ToolDefinition,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic LLM provider using direct HTTP calls
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from environment variables
    pub fn from_env() -> Result<Self> {
        tracing::info!("Creating Anthropic provider from environment");

        let api_key = env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY environment variable not set")?;

        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-5-20250929".to_string());

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16000); // Higher default for extended thinking

        tracing::info!("Using model: {}", model);
        tracing::info!("Max tokens: {}", max_tokens);

        let client = Client::new();

        Ok(Self {
            client,
            api_key,
            model,
            max_tokens,
        })
    }

    /// Create a new Anthropic provider with a specific API key
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = Client::new();

        Ok(Self {
            client,
            api_key: api_key.into(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 16000,
        })
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

    /// Send a message and get a complete response (no tool calling)
    ///
    /// This is a simple method for basic conversations without tools.
    pub async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
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
            system: system_prompt.map(String::from),
            tools: None,
            tool_choice: None,
            thinking: None,
            temperature: None,
            stream: None,
        };

        let response = self.send_request(&request).await?;
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

        let request = MessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system: system_prompt.map(String::from),
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            thinking,
            temperature: None,
            stream: None,
        };

        self.send_request(&request).await
    }

    /// Send a raw request to the Anthropic API
    async fn send_request(&self, request: &MessageRequest) -> Result<MessageResponse> {
        tracing::debug!("Model: {}", request.model);
        tracing::debug!("Max tokens: {}", request.max_tokens);

        let request_json = serde_json::to_string(request)
            .context("Failed to serialize request")?;
        tracing::debug!("Request JSON: {}", request_json);

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
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

        Ok(response)
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
    })
}
