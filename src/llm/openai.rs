//! OpenAI API client
//!
//! This module provides a direct HTTP client for the OpenAI Chat Completions API,
//! translating between the framework's internal message types (Anthropic format)
//! and the OpenAI API format.
//!
//! # Authentication
//!
//! Uses an OpenAI API key (set via `OPENAI_API_KEY` environment variable or passed directly).
//!
//! ```ignore
//! // From environment variable
//! let llm = OpenAiProvider::from_env()?;
//!
//! // With explicit API key
//! let llm = OpenAiProvider::new("sk-...")?;
//! ```

use anyhow::{Context, Result};
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio_util::io::StreamReader;

use super::auth::{auth_provider, AuthConfig, AuthProvider, AuthSource};
use super::provider::LlmProvider;
use super::types::{
    ContentBlock, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
    ContentBlockStopEvent, ContentDelta, DeltaUsage, Message, MessageContent,
    MessageDeltaData, MessageDeltaEvent, MessageResponse, MessageStartData, MessageStartEvent,
    StopReason, StreamEvent, SystemPrompt, ThinkingConfig, ToolChoice, ToolDefinition, Usage,
};

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

// ============================================================================
// OpenAI-specific request/response types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAiStreamOptions>,
}

#[derive(Debug, Serialize)]
struct OpenAiStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum OpenAiContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum OpenAiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiImageUrl {
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String, // JSON string
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunctionDefinition,
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiToolChoice {
    Mode(String), // "auto", "none", "required"
    Function { r#type: String, function: OpenAiFunctionChoice },
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionChoice {
    name: String,
}

// Response types

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    index: u32,
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// Streaming types

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    index: u32,
    delta: OpenAiDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    tool_type: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ============================================================================
// OpenAiProvider
// ============================================================================

/// OpenAI LLM provider
///
/// Translates between the framework's internal message types and the OpenAI API format.
/// All internal types follow Anthropic's format; translation happens at the boundary.
///
/// # Authentication
///
/// Supports both static and dynamic authentication:
///
/// ```ignore
/// // Static auth with API key
/// let llm = OpenAiProvider::new("sk-...")?;
///
/// // Dynamic auth with callback (for JWT/proxy scenarios)
/// let llm = OpenAiProvider::with_auth_provider(|| async {
///     let token = refresh_token().await?;
///     Ok(AuthConfig::new(token))
/// });
/// ```
pub struct OpenAiProvider {
    client: Client,
    auth: AuthSource,
    model: String,
    max_tokens: u32,
    api_base: String,
}

impl OpenAiProvider {
    /// Create a new OpenAI provider from environment variables
    ///
    /// Reads from:
    /// - `OPENAI_API_KEY` (required)
    /// - `OPENAI_MODEL` (required)
    /// - `OPENAI_BASE_URL` (optional, defaults to OpenAI API)
    /// - `OPENAI_MAX_TOKENS` (optional, defaults to 4096)
    pub fn from_env() -> Result<Self> {
        tracing::info!("Creating OpenAI provider from environment");

        let api_key = env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;

        let model = env::var("OPENAI_MODEL")
            .context("OPENAI_MODEL environment variable not set")?;

        let base_url = env::var("OPENAI_BASE_URL").ok();

        let max_tokens = env::var("OPENAI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);

        tracing::info!("Using model: {}", model);
        tracing::info!("Max tokens: {}", max_tokens);
        if let Some(ref url) = base_url {
            tracing::info!("Using custom base URL: {}", url);
        }

        Ok(Self {
            client: Client::new(),
            auth: AuthSource::Static(AuthConfig {
                api_key,
                base_url,
            }),
            model,
            max_tokens,
            api_base: DEFAULT_API_BASE.to_string(),
        })
    }

    /// Create a new OpenAI provider with a specific API key
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            auth: AuthSource::Static(AuthConfig::new(api_key)),
            model: "".to_string(),
            max_tokens: 4096,
            api_base: DEFAULT_API_BASE.to_string(),
        })
    }

    /// Create a new OpenAI provider with a dynamic auth provider callback
    ///
    /// The callback is called before each API request to get fresh credentials.
    /// This is useful for:
    /// - JWT tokens that expire frequently
    /// - Proxy servers that require per-request auth
    /// - Rotating API keys
    ///
    /// The `AuthConfig.api_key` is used as the OpenAI API key (Bearer token).
    /// The `AuthConfig.base_url` (if set) overrides the default OpenAI API base URL.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let llm = OpenAiProvider::with_auth_provider(|| async {
    ///     let token = my_backend.get_openai_key().await?;
    ///     Ok(AuthConfig::new(token))
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
            max_tokens: 4096,
            api_base: DEFAULT_API_BASE.to_string(),
        }
    }

    /// Create a new OpenAI provider with a trait object auth provider
    ///
    /// Use this when you have a custom `AuthProvider` implementation.
    pub fn with_auth_provider_boxed(provider: Arc<dyn AuthProvider>) -> Self {
        Self {
            client: Client::new(),
            auth: AuthSource::Dynamic(provider),
            model: "".to_string(),
            max_tokens: 4096,
            api_base: DEFAULT_API_BASE.to_string(),
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

    /// Create a variant with different model/tokens, sharing the same auth config
    fn create_variant_impl(&self, model: &str, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            auth: self.auth.clone(),
            model: model.to_string(),
            max_tokens,
            api_base: self.api_base.clone(),
        }
    }

    // ========================================================================
    // Format conversion: Internal (Anthropic) -> OpenAI
    // ========================================================================

    /// Convert internal messages to OpenAI format
    fn convert_messages(&self, messages: &[Message], system_prompt: &Option<SystemPrompt>) -> Vec<OpenAiMessage> {
        let mut openai_messages: Vec<OpenAiMessage> = Vec::new();

        // Add system message first if present
        if let Some(system) = system_prompt {
            let system_text = match system {
                SystemPrompt::Text(text) => text.clone(),
                SystemPrompt::Blocks(blocks) => {
                    blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("\n")
                }
            };

            openai_messages.push(OpenAiMessage {
                role: "system".to_string(),
                content: Some(OpenAiContent::Text(system_text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert conversation messages
        for msg in messages {
            match &msg.content {
                MessageContent::Text(text) => {
                    openai_messages.push(OpenAiMessage {
                        role: msg.role.clone(),
                        content: Some(OpenAiContent::Text(text.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                MessageContent::Blocks(blocks) => {
                    self.convert_blocks_to_messages(blocks, &msg.role, &mut openai_messages);
                }
            }
        }

        openai_messages
    }

    /// Convert content blocks to OpenAI messages
    fn convert_blocks_to_messages(
        &self,
        blocks: &[ContentBlock],
        role: &str,
        openai_messages: &mut Vec<OpenAiMessage>,
    ) {
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<OpenAiToolCall> = Vec::new();
        let mut tool_results: Vec<(String, String)> = Vec::new(); // (tool_call_id, content)

        for block in blocks {
            match block {
                ContentBlock::Text { text, .. } => {
                    if !text.is_empty() {
                        text_parts.push(text.clone());
                    }
                }
                ContentBlock::ToolUse { id, name, input, .. } => {
                    // Convert tool use to OpenAI tool call
                    let arguments = serde_json::to_string(input)
                        .unwrap_or_else(|_| "{}".to_string());
                    
                    tool_calls.push(OpenAiToolCall {
                        id: id.clone(),
                        tool_type: "function".to_string(),
                        function: OpenAiFunctionCall {
                            name: name.clone(),
                            arguments,
                        },
                    });
                }
                ContentBlock::ToolResult { tool_use_id, content, is_error, .. } => {
                    // Tool results need to be separate messages with role "tool"
                    let result_content = content
                        .clone()
                        .unwrap_or_else(|| "No output".to_string());
                    
                    let formatted_content = if is_error.unwrap_or(false) {
                        format!("Error: {}", result_content)
                    } else {
                        result_content
                    };
                    
                    tool_results.push((tool_use_id.clone(), formatted_content));
                }
                ContentBlock::Thinking { thinking, .. } => {
                    // OpenAI doesn't have native thinking blocks, so we include them as text
                    // with a marker to distinguish them
                    text_parts.push(format!("[Internal reasoning: {}]", thinking));
                }
                ContentBlock::RedactedThinking { .. } => {
                    // Skip redacted thinking
                }
                ContentBlock::Image { source, .. } => {
                    // OpenAI uses base64 data URLs for images
                    let data_url = format!("data:{};base64,{}", source.media_type, source.data);
                    // Images need to be handled in content parts
                    // We'll need to restructure this if we have mixed content
                }
                ContentBlock::Document { .. } => {
                    // OpenAI doesn't support document blocks directly
                    // Skip for now or convert to text
                }
            }
        }

        // Build the message based on what we collected
        if role == "assistant" {
            // Assistant messages can have both text and tool calls
            let content = if text_parts.is_empty() {
                None
            } else {
                Some(OpenAiContent::Text(text_parts.join("\n")))
            };

            openai_messages.push(OpenAiMessage {
                role: role.to_string(),
                content,
                tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                tool_call_id: None,
            });
        } else if role == "user" {
            // User messages - handle tool results separately
            if !text_parts.is_empty() {
                openai_messages.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(OpenAiContent::Text(text_parts.join("\n"))),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        // Add tool result messages (they have role "tool" in OpenAI)
        for (tool_call_id, content) in tool_results {
            openai_messages.push(OpenAiMessage {
                role: "tool".to_string(),
                content: Some(OpenAiContent::Text(content)),
                tool_calls: None,
                tool_call_id: Some(tool_call_id),
            });
        }
    }

    /// Convert internal tool definitions to OpenAI format
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Option<Vec<OpenAiTool>> {
        if tools.is_empty() {
            return None;
        }

        let openai_tools: Vec<OpenAiTool> = tools
            .iter()
            .filter_map(|tool| {
                match tool {
                    ToolDefinition::Custom(custom) => {
                        // Build parameters object from input_schema
                        let parameters = if custom.input_schema.properties.is_some()
                            || custom.input_schema.required.is_some()
                        {
                            let mut params = serde_json::json!({
                                "type": custom.input_schema.schema_type,
                            });
                            if let Some(ref props) = custom.input_schema.properties {
                                params["properties"] = props.clone();
                            }
                            if let Some(ref req) = custom.input_schema.required {
                                params["required"] = serde_json::json!(req);
                            }
                            Some(params)
                        } else {
                            None
                        };

                        Some(OpenAiTool {
                            tool_type: "function".to_string(),
                            function: OpenAiFunctionDefinition {
                                name: custom.name.clone(),
                                description: custom.description.clone(),
                                parameters,
                            },
                        })
                    }
                    // Built-in Anthropic tools don't map to OpenAI - skip them
                    ToolDefinition::Bash(_) | ToolDefinition::TextEditor(_) => None,
                }
            })
            .collect();

        if openai_tools.is_empty() {
            None
        } else {
            Some(openai_tools)
        }
    }

    /// Convert tool choice to OpenAI format
    fn convert_tool_choice(&self, tool_choice: &Option<ToolChoice>) -> Option<OpenAiToolChoice> {
        match tool_choice {
            Some(ToolChoice::Auto { .. }) | None => {
                Some(OpenAiToolChoice::Mode("auto".to_string()))
            }
            Some(ToolChoice::Any { .. }) => {
                Some(OpenAiToolChoice::Mode("required".to_string()))
            }
            Some(ToolChoice::None) => {
                Some(OpenAiToolChoice::Mode("none".to_string()))
            }
            Some(ToolChoice::Tool { name, .. }) => {
                Some(OpenAiToolChoice::Function {
                    r#type: "function".to_string(),
                    function: OpenAiFunctionChoice {
                        name: name.clone(),
                    },
                })
            }
        }
    }

    // ========================================================================
    // Format conversion: OpenAI -> Internal (Anthropic)
    // ========================================================================

    /// Convert OpenAI response to internal MessageResponse format
    fn convert_response(&self, openai_resp: OpenAiResponse) -> Result<MessageResponse> {
        let choice = openai_resp
            .choices
            .first()
            .context("No choices in OpenAI response")?;

        let content_blocks = self.convert_openai_message_to_blocks(&choice.message);

        let stop_reason = choice.finish_reason.as_deref().map(|r| match r {
            "stop" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" => StopReason::ToolUse,
            "content_filter" => StopReason::Refusal,
            _ => StopReason::EndTurn,
        });

        let usage = openai_resp.usage.as_ref().map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            thoughts_token_count: None,
        }).unwrap_or(Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            thoughts_token_count: None,
        });

        Ok(MessageResponse {
            id: openai_resp.id,
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: content_blocks,
            model: openai_resp.model,
            stop_reason,
            stop_sequence: None,
            usage,
        })
    }

    /// Convert OpenAI message to internal ContentBlocks
    fn convert_openai_message_to_blocks(&self, message: &OpenAiMessage) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        // Add text content if present
        if let Some(ref content) = message.content {
            match content {
                OpenAiContent::Text(text) => {
                    if !text.is_empty() {
                        blocks.push(ContentBlock::Text {
                            text: text.clone(),
                            cache_control: None,
                        });
                    }
                }
                OpenAiContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            OpenAiContentPart::Text { text } => {
                                if !text.is_empty() {
                                    blocks.push(ContentBlock::Text {
                                        text: text.clone(),
                                        cache_control: None,
                                    });
                                }
                            }
                            OpenAiContentPart::ImageUrl { .. } => {
                                // Skip image URLs in responses
                            }
                        }
                    }
                }
            }
        }

        // Add tool calls if present
        if let Some(ref tool_calls) = message.tool_calls {
            for tool_call in tool_calls {
                let input: Value = serde_json::from_str(&tool_call.function.arguments)
                    .unwrap_or_else(|_| serde_json::json!({}));

                blocks.push(ContentBlock::tool_use(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    input,
                ));
            }
        }

        blocks
    }

    // ========================================================================
    // API methods
    // ========================================================================

    /// Send a non-streaming request to the OpenAI API
    async fn send_openai_request(&self, request: &OpenAiRequest, _session_id: Option<&str>) -> Result<OpenAiResponse> {
        // Get auth credentials (static or from provider)
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_base = auth_config.base_url.as_deref().unwrap_or(&self.api_base);
        let url = format!("{}/chat/completions", api_base);

        let request_json = serde_json::to_string(request)
            .context("Failed to serialize OpenAI request")?;
        tracing::debug!("[OpenAI] Request JSON: {}", request_json);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", auth_config.api_key))
            .body(request_json)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .context("Failed to read OpenAI response body")?;

        tracing::debug!("[OpenAI] Response status: {}", status);
        tracing::debug!("[OpenAI] Response body: {}", response_text);

        if !status.is_success() {
            tracing::error!("[OpenAI] API error: {} - {}", status, response_text);
            anyhow::bail!("OpenAI API error ({}): {}", status, response_text);
        }

        let openai_response: OpenAiResponse = serde_json::from_str(&response_text)
            .context("Failed to parse OpenAI API response")?;

        Ok(openai_response)
    }

    /// Send a streaming request to the OpenAI API
    async fn send_openai_streaming_request(
        &self,
        request: &OpenAiRequest,
        _session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        // Get auth credentials (static or from provider)
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_base = auth_config.base_url.as_deref().unwrap_or(&self.api_base);
        let url = format!("{}/chat/completions", api_base);

        let request_json = serde_json::to_string(request)
            .context("Failed to serialize OpenAI streaming request")?;
        tracing::debug!("[OpenAI] Streaming request JSON: {}", request_json);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", auth_config.api_key))
            .body(request_json)
            .send()
            .await
            .context("Failed to send streaming request to OpenAI API")?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            tracing::error!("[OpenAI] Streaming API error: {} - {}", status, error_text);
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }

        tracing::info!("[OpenAI] Streaming response started");

        // Parse SSE stream and convert to internal StreamEvent format
        let byte_stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            byte_stream.map(|result| result.map_err(|e| std::io::Error::other(e.to_string()))),
        );
        let buf_reader = tokio::io::BufReader::new(stream_reader);
        let model = self.model.clone();

        let stream = async_stream::try_stream! {
            let mut lines = buf_reader.lines();
            let mut chunk_index: usize = 0;
            let mut content_block_started = false;
            let mut current_tool_calls: std::collections::HashMap<u32, (String, String, String)> = std::collections::HashMap::new();
            let mut finished = false;

            tracing::info!("[OpenAI] Stream: starting to read lines");

            while let Some(line) = lines.next_line().await? {
                if finished {
                    tracing::debug!("[OpenAI] Stream: ignoring line after finish");
                    continue;
                }
                
                tracing::trace!("[OpenAI] Stream: got line: {}", line);

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    tracing::info!("[OpenAI] Stream: received [DONE] marker");
                    finished = true;
                    break;
                }

                if data.is_empty() {
                    continue;
                }

                tracing::debug!("[OpenAI] Stream: parsing chunk #{}", chunk_index);

                let chunk: OpenAiStreamChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("[OpenAI] Failed to parse streaming chunk: {}", e);
                        continue;
                    }
                };

                // First chunk - emit MessageStart
                if chunk_index == 0 {
                    let usage = chunk.usage.as_ref().map(|u| Usage {
                        input_tokens: u.prompt_tokens,
                        output_tokens: u.completion_tokens,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                        thoughts_token_count: None,
                    }).unwrap_or(Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                        thoughts_token_count: None,
                    });

                    yield StreamEvent::MessageStart(MessageStartEvent {
                        message: MessageStartData {
                            id: chunk.id.clone(),
                            message_type: "message".to_string(),
                            role: "assistant".to_string(),
                            content: vec![],
                            model: model.clone(),
                            stop_reason: None,
                            stop_sequence: None,
                            usage,
                        },
                    });
                }

                // Process choice delta
                if let Some(choice) = chunk.choices.first() {
                    let delta = &choice.delta;

                    // Handle text content
                    if let Some(ref content) = delta.content {
                        if !content.is_empty() {
                            if !content_block_started {
                                yield StreamEvent::ContentBlockStart(
                                    ContentBlockStartEvent {
                                        index: 0,
                                        content_block: ContentBlockStart::Text {
                                            text: String::new(),
                                        },
                                    }
                                );
                                content_block_started = true;
                            }
                            yield StreamEvent::ContentBlockDelta(
                                ContentBlockDeltaEvent {
                                    index: 0,
                                    delta: ContentDelta::TextDelta {
                                        text: content.clone(),
                                    },
                                }
                            );
                        }
                    }

                    // Handle tool call deltas
                    if let Some(ref tool_call_deltas) = delta.tool_calls {
                        for tool_delta in tool_call_deltas {
                            let index = tool_delta.index;
                            
                            // Update our tracking of tool calls
                            let entry = current_tool_calls.entry(index).or_insert((
                                String::new(),
                                String::new(),
                                String::new(),
                            ));

                            if let Some(ref id) = tool_delta.id {
                                entry.0 = id.clone();
                            }

                            if let Some(ref function) = tool_delta.function {
                                if let Some(ref name) = function.name {
                                    entry.1 = name.clone();
                                    
                                    // Start new content block for tool use
                                    if content_block_started {
                                        yield StreamEvent::ContentBlockStop(
                                            ContentBlockStopEvent { index: 0 }
                                        );
                                    }
                                    
                                    yield StreamEvent::ContentBlockStart(
                                        ContentBlockStartEvent {
                                            index: 0,
                                            content_block: ContentBlockStart::ToolUse {
                                                id: entry.0.clone(),
                                                name: name.clone(),
                                                input: Value::Object(Default::default()),
                                                signature: None,
                                            },
                                        }
                                    );
                                    content_block_started = true;
                                }

                                if let Some(ref args) = function.arguments {
                                    entry.2.push_str(args);
                                    yield StreamEvent::ContentBlockDelta(
                                        ContentBlockDeltaEvent {
                                            index: 0,
                                            delta: ContentDelta::InputJsonDelta {
                                                partial_json: args.clone(),
                                            },
                                        }
                                    );
                                }
                            }
                        }
                    }

                    // Check finish reason
                    if let Some(ref reason) = choice.finish_reason {
                        tracing::info!("[OpenAI] Stream: finish_reason={}", reason);

                        if content_block_started {
                            yield StreamEvent::ContentBlockStop(
                                ContentBlockStopEvent { index: 0 }
                            );
                            content_block_started = false;
                        }

                        let stop_reason = match reason.as_str() {
                            "stop" => StopReason::EndTurn,
                            "length" => StopReason::MaxTokens,
                            "tool_calls" => StopReason::ToolUse,
                            "content_filter" => StopReason::Refusal,
                            _ => StopReason::EndTurn,
                        };

                        let output_tokens = chunk.usage.as_ref()
                            .map(|u| u.completion_tokens)
                            .unwrap_or(0);

                        yield StreamEvent::MessageDelta(MessageDeltaEvent {
                            delta: MessageDeltaData {
                                stop_reason: Some(stop_reason),
                                stop_sequence: None,
                            },
                            usage: DeltaUsage { output_tokens },
                        });

                        finished = true;
                    }
                }

                chunk_index += 1;

                if finished {
                    break;
                }
            }

            tracing::info!("[OpenAI] Stream: loop ended after {} chunks, finished={}", chunk_index, finished);

            // Ensure we close any open content block
            if content_block_started {
                yield StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 });
            }

            tracing::info!("[OpenAI] Stream: yielding MessageStop");
            yield StreamEvent::MessageStop;
        };

        Ok(Box::pin(stream))
    }

    /// Build an OpenAiRequest from internal types
    fn build_request(
        &self,
        messages: &[Message],
        system: &Option<SystemPrompt>,
        tools: &[ToolDefinition],
        tool_choice: &Option<ToolChoice>,
        _thinking: &Option<ThinkingConfig>,
    ) -> OpenAiRequest {
        let openai_messages = self.convert_messages(messages, system);
        let openai_tools = self.convert_tools(tools);
        let openai_tool_choice = if openai_tools.is_some() {
            self.convert_tool_choice(tool_choice)
        } else {
            None
        };

        // Reasoning models (o-series, gpt-5.x) use max_completion_tokens instead of max_tokens
        let is_reasoning_model = self.model.starts_with("o1-") 
            || self.model.starts_with("o3-")
            || self.model.starts_with("gpt-5");

        let (max_tokens, max_completion_tokens) = if is_reasoning_model {
            (None, Some(self.max_tokens))
        } else {
            (Some(self.max_tokens), None)
        };

        // Reasoning models don't support temperature parameter
        let temperature = if is_reasoning_model {
            None
        } else {
            Some(1.0)
        };

        OpenAiRequest {
            model: self.model.clone(),
            messages: openai_messages,
            max_tokens,
            max_completion_tokens,
            temperature,
            tools: openai_tools,
            tool_choice: openai_tool_choice,
            stream: None,
            stream_options: None,
        }
    }
}

// ============================================================================
// LlmProvider implementation
// ============================================================================

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<String> {
        tracing::info!("[OpenAI] Sending message");

        let mut messages: Vec<Message> = conversation_history.to_vec();
        messages.push(Message::user(user_message));

        let system = system_prompt.map(|s| SystemPrompt::Text(s.to_string()));
        let request = self.build_request(&messages, &system, &[], &None, &None);

        let openai_response = self.send_openai_request(&request, session_id).await?;
        let response = self.convert_response(openai_response)?;

        Ok(response.text())
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
        tracing::info!("[OpenAI] Sending message with tools");
        tracing::debug!("[OpenAI] Messages count: {}", messages.len());
        tracing::debug!("[OpenAI] Tools count: {}", tools.len());

        let request = self.build_request(&messages, &system, &tools, &tool_choice, &thinking);
        let openai_response = self.send_openai_request(&request, session_id).await?;
        self.convert_response(openai_response)
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
        tracing::info!("[OpenAI] Streaming message with tools");
        tracing::debug!("[OpenAI] Messages count: {}", messages.len());
        tracing::debug!("[OpenAI] Tools count: {}", tools.len());

        let mut request = self.build_request(&messages, &system, &tools, &tool_choice, &thinking);
        request.stream = Some(true);
        request.stream_options = Some(OpenAiStreamOptions {
            include_usage: true,
        });

        self.send_openai_streaming_request(&request, session_id).await
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn create_variant(&self, model: &str, max_tokens: u32) -> Arc<dyn LlmProvider> {
        Arc::new(self.create_variant_impl(model, max_tokens))
    }
}
