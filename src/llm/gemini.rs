//! Gemini API client
//!
//! This module provides a direct HTTP client for the Google Gemini API,
//! translating between the framework's internal message types (Anthropic format)
//! and the Gemini API format.
//!
//! # Authentication
//!
//! Uses a Gemini API key (set via `GEMINI_API_KEY` environment variable or passed directly).
//!
//! ```ignore
//! // From environment variable
//! let llm = GeminiProvider::from_env()?;
//!
//! // With explicit API key
//! let llm = GeminiProvider::new("AIza...")?;
//! ```

use anyhow::{Context, Result};
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::AsyncBufReadExt;
use tokio::sync::Mutex;
use tokio_util::io::StreamReader;

use super::auth::{auth_provider, AuthConfig, AuthProvider, AuthSource};
use super::provider::LlmProvider;
use super::types::{
    ContentBlock, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
    ContentBlockStopEvent, ContentDelta, DeltaUsage, Message, MessageContent,
    MessageDeltaData, MessageDeltaEvent, MessageResponse, MessageStartData, MessageStartEvent,
    StopReason, StreamEvent, SystemPrompt, ThinkingConfig, ToolChoice, ToolDefinition, Usage,
};

const DEFAULT_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

// ============================================================================
// Gemini-specific request/response types
// ============================================================================

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
}

impl Default for GeminiPart {
    fn default() -> Self {
        Self {
            text: None,
            function_call: None,
            function_response: None,
            inline_data: None,
            thought_signature: None,
            thought: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolConfig {
    function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionCallingConfig {
    mode: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

// Response types

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    usage_metadata: Option<GeminiUsageMetadata>,
    #[allow(dead_code)]
    model_version: Option<String>,
    #[allow(dead_code)]
    response_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
    #[allow(dead_code)]
    index: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
    #[serde(default)]
    thoughts_token_count: Option<u32>,
}

// ============================================================================
// GeminiProvider
// ============================================================================

/// Google Gemini LLM provider
///
/// Translates between the framework's internal message types and the Gemini API format.
/// All internal types follow Anthropic's format; translation happens at the boundary.
///
/// # Authentication
///
/// Supports both static and dynamic authentication:
///
/// ```ignore
/// // Static auth with API key
/// let llm = GeminiProvider::new("AIza...")?;
///
/// // Dynamic auth with callback (for JWT/proxy scenarios)
/// let llm = GeminiProvider::with_auth_provider(|| async {
///     let token = refresh_token().await?;
///     Ok(AuthConfig::new(token))
/// });
/// ```
pub struct GeminiProvider {
    client: Client,
    auth: AuthSource,
    model: String,
    max_tokens: u32,
    api_base: String,
    /// Cache of thought signatures for Gemini 3 function calling.
    /// Maps tool_use_id -> thought_signature.
    /// Gemini 3 requires thought signatures to be sent back with function calls.
    thought_signatures: Arc<Mutex<HashMap<String, String>>>,
}

impl GeminiProvider {
    /// Create a new Gemini provider from environment variables
    ///
    /// Reads from:
    /// - `GEMINI_API_KEY` (required)
    /// - `GEMINI_MODEL` (required)
    /// - `GEMINI_MAX_TOKENS` (optional, defaults to 8192)
    pub fn from_env() -> Result<Self> {
        tracing::info!("Creating Gemini provider from environment");

        let api_key = env::var("GEMINI_API_KEY")
            .context("GEMINI_API_KEY environment variable not set")?;

        let model = env::var("GEMINI_MODEL")
            .context("GEMINI_MODEL environment variable not set")?;

        let max_tokens = env::var("GEMINI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);

        tracing::info!("Using model: {}", model);
        tracing::info!("Max tokens: {}", max_tokens);

        Ok(Self {
            client: Client::new(),
            auth: AuthSource::Static(AuthConfig::new(api_key)),
            model,
            max_tokens,
            api_base: DEFAULT_API_BASE.to_string(),
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create a new Gemini provider with a specific API key
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            auth: AuthSource::Static(AuthConfig::new(api_key)),
            model: "".to_string(),
            max_tokens: 8192,
            api_base: DEFAULT_API_BASE.to_string(),
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create a new Gemini provider with a dynamic auth provider callback
    ///
    /// The callback is called before each API request to get fresh credentials.
    /// This is useful for:
    /// - JWT tokens that expire frequently
    /// - Proxy servers that require per-request auth
    /// - Rotating API keys
    ///
    /// The `AuthConfig.api_key` is used as the Gemini API key.
    /// The `AuthConfig.base_url` (if set) overrides the default Gemini API base URL.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let llm = GeminiProvider::with_auth_provider(|| async {
    ///     let token = my_backend.get_gemini_key().await?;
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
            max_tokens: 8192,
            api_base: DEFAULT_API_BASE.to_string(),
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new Gemini provider with a trait object auth provider
    ///
    /// Use this when you have a custom `AuthProvider` implementation.
    pub fn with_auth_provider_boxed(provider: Arc<dyn AuthProvider>) -> Self {
        Self {
            client: Client::new(),
            auth: AuthSource::Dynamic(provider),
            model: "".to_string(),
            max_tokens: 8192,
            api_base: DEFAULT_API_BASE.to_string(),
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
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
            thought_signatures: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    // ========================================================================
    // Format conversion: Internal (Anthropic) -> Gemini
    // ========================================================================

    /// Convert internal messages to Gemini format
    async fn convert_messages(&self, messages: &[Message]) -> Vec<GeminiContent> {
        let mut gemini_contents: Vec<GeminiContent> = Vec::new();

        for msg in messages {
            let gemini_role = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "model",
                _ => "user",
            };

            let parts = self.convert_content_to_parts(&msg.content, &msg.role).await;

            // Check if parts contain function responses - those should be role "user"
            let has_function_response = parts.iter().any(|p| p.function_response.is_some());
            let role = if has_function_response {
                "user".to_string()
            } else {
                gemini_role.to_string()
            };

            // Split: function responses go as separate "user" content,
            // function calls stay with "model" content
            if has_function_response && parts.iter().any(|p| p.function_response.is_none()) {
                // Mixed content - separate function responses from other parts
                let fn_parts: Vec<GeminiPart> = parts.iter()
                    .filter(|p| p.function_response.is_some())
                    .cloned()
                    .collect();
                let other_parts: Vec<GeminiPart> = parts.iter()
                    .filter(|p| p.function_response.is_none())
                    .cloned()
                    .collect();

                if !other_parts.is_empty() {
                    gemini_contents.push(GeminiContent {
                        role: gemini_role.to_string(),
                        parts: other_parts,
                    });
                }
                if !fn_parts.is_empty() {
                    gemini_contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: fn_parts,
                    });
                }
            } else {
                if !parts.is_empty() {
                    gemini_contents.push(GeminiContent { role, parts });
                }
            }
        }

        // Gemini requires alternating user/model turns - merge consecutive same-role messages
        self.merge_consecutive_roles(gemini_contents)
    }

    /// Merge consecutive messages with the same role (Gemini requires alternation)
    fn merge_consecutive_roles(&self, contents: Vec<GeminiContent>) -> Vec<GeminiContent> {
        let mut merged: Vec<GeminiContent> = Vec::new();

        for content in contents {
            if let Some(last) = merged.last_mut() {
                if last.role == content.role {
                    // Merge parts into the last message
                    last.parts.extend(content.parts);
                    continue;
                }
            }
            merged.push(content);
        }

        merged
    }

    /// Convert internal content to Gemini parts
    async fn convert_content_to_parts(&self, content: &MessageContent, _role: &str) -> Vec<GeminiPart> {
        match content {
            MessageContent::Text(text) => {
                vec![GeminiPart {
                    text: Some(text.clone()),
                    ..Default::default()
                }]
            }
            MessageContent::Blocks(blocks) => {
                let mut parts = Vec::new();
                // Track tool names for tool_result -> functionResponse mapping
                // We need to find the corresponding tool_use name for each tool_result
                let tool_use_names: std::collections::HashMap<String, String> = blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolUse { id, name, .. } = b {
                            Some((id.clone(), name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => {
                            // Skip empty text blocks
                            if !text.is_empty() {
                                parts.push(GeminiPart {
                                    text: Some(text.clone()),
                                    ..Default::default()
                                });
                            }
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            // Look up cached thought signature for this tool call
                            let sig = {
                                let sigs = self.thought_signatures.lock().await;
                                sigs.get(id).cloned()
                                    .or_else(|| sigs.get(&format!("name:{}", name)).cloned())
                            };
                            parts.push(GeminiPart {
                                function_call: Some(GeminiFunctionCall {
                                    name: name.clone(),
                                    args: input.clone(),
                                }),
                                thought_signature: sig,
                                ..Default::default()
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                            ..
                        } => {
                            // Find the tool name from our context or from the history
                            let tool_name = tool_use_names
                                .get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| {
                                    // Try to find from previous messages stored in session
                                    // Fallback: use the tool_use_id as a stand-in
                                    tool_use_id.clone()
                                });

                            let result_content = content
                                .clone()
                                .unwrap_or_else(|| "No output".to_string());

                            let response = if is_error.unwrap_or(false) {
                                serde_json::json!({
                                    "error": result_content
                                })
                            } else {
                                serde_json::json!({
                                    "result": result_content
                                })
                            };

                            parts.push(GeminiPart {
                                function_response: Some(GeminiFunctionResponse {
                                    name: tool_name,
                                    response,
                                }),
                                ..Default::default()
                            });
                        }
                        ContentBlock::Thinking { thinking, .. } => {
                            // Send thinking as a thought-flagged text part
                            parts.push(GeminiPart {
                                text: Some(thinking.clone()),
                                thought: Some(true),
                                ..Default::default()
                            });
                        }
                        ContentBlock::RedactedThinking { .. } => {
                            // Skip redacted thinking
                        }
                        ContentBlock::Image { source, .. } => {
                            parts.push(GeminiPart {
                                inline_data: Some(GeminiInlineData {
                                    mime_type: source.media_type.clone(),
                                    data: source.data.clone(),
                                }),
                                ..Default::default()
                            });
                        }
                        ContentBlock::Document { source, .. } => {
                            parts.push(GeminiPart {
                                inline_data: Some(GeminiInlineData {
                                    mime_type: source.media_type.clone(),
                                    data: source.data.clone(),
                                }),
                                ..Default::default()
                            });
                        }
                    }
                }

                // For user messages with tool results, we need to map tool_use_ids to names.
                // The tool_use_names map above handles this for blocks within the same message.
                // For cross-message lookups, we rely on the session history being passed correctly.

                parts
            }
        }
    }

    /// Convert internal tool definitions to Gemini function declarations
    fn convert_tools(&self, tools: &[ToolDefinition]) -> Option<Vec<GeminiTool>> {
        if tools.is_empty() {
            return None;
        }

        let declarations: Vec<GeminiFunctionDeclaration> = tools
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
                                // Clean properties for Gemini compatibility
                                params["properties"] = Self::clean_schema_for_gemini(props);
                            }
                            if let Some(ref req) = custom.input_schema.required {
                                params["required"] = serde_json::json!(req);
                            }
                            Some(params)
                        } else {
                            None
                        };

                        Some(GeminiFunctionDeclaration {
                            name: custom.name.clone(),
                            description: custom.description.clone().unwrap_or_default(),
                            parameters,
                        })
                    }
                    // Built-in Anthropic tools don't map to Gemini - skip them
                    ToolDefinition::Bash(_) | ToolDefinition::TextEditor(_) => None,
                }
            })
            .collect();

        if declarations.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: declarations,
            }])
        }
    }

    /// Clean JSON schema for Gemini compatibility
    ///
    /// Gemini's function declarations don't support some JSON Schema fields
    /// that Anthropic does. This recursively strips unsupported fields.
    fn clean_schema_for_gemini(value: &Value) -> Value {
        // Fields not supported by Gemini's function declaration schema
        const UNSUPPORTED_FIELDS: &[&str] = &[
            "additionalProperties",
            "$schema",
            "definitions",
            "$ref",
            "patternProperties",
            "if", "then", "else",
            "allOf", "anyOf", "oneOf", "not",
            "default",
        ];

        match value {
            Value::Object(map) => {
                let mut cleaned = serde_json::Map::new();
                for (key, val) in map {
                    if UNSUPPORTED_FIELDS.contains(&key.as_str()) {
                        continue;
                    }
                    cleaned.insert(key.clone(), Self::clean_schema_for_gemini(val));
                }
                Value::Object(cleaned)
            }
            Value::Array(arr) => {
                Value::Array(arr.iter().map(|v| Self::clean_schema_for_gemini(v)).collect())
            }
            other => other.clone(),
        }
    }

    /// Convert system prompt to Gemini format
    fn convert_system_prompt(&self, system: &Option<SystemPrompt>) -> Option<GeminiSystemInstruction> {
        match system {
            Some(SystemPrompt::Text(text)) => Some(GeminiSystemInstruction {
                parts: vec![GeminiPart {
                    text: Some(text.clone()),
                    ..Default::default()
                }],
            }),
            Some(SystemPrompt::Blocks(blocks)) => {
                let parts: Vec<GeminiPart> = blocks
                    .iter()
                    .map(|b| GeminiPart {
                        text: Some(b.text.clone()),
                        ..Default::default()
                    })
                    .collect();
                Some(GeminiSystemInstruction { parts })
            }
            None => None,
        }
    }

    /// Convert tool choice to Gemini format
    fn convert_tool_config(&self, tool_choice: &Option<ToolChoice>) -> Option<GeminiToolConfig> {
        match tool_choice {
            Some(ToolChoice::Auto { .. }) | None => {
                // AUTO is the default, only set if tools are present
                Some(GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: "AUTO".to_string(),
                    },
                })
            }
            Some(ToolChoice::Any { .. }) => Some(GeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: "ANY".to_string(),
                },
            }),
            Some(ToolChoice::None) => Some(GeminiToolConfig {
                function_calling_config: GeminiFunctionCallingConfig {
                    mode: "NONE".to_string(),
                },
            }),
            Some(ToolChoice::Tool { .. }) => {
                // Gemini doesn't have a direct equivalent of "must use specific tool"
                // Fall back to ANY mode
                Some(GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: "ANY".to_string(),
                    },
                })
            }
        }
    }

    // ========================================================================
    // Format conversion: Gemini -> Internal (Anthropic)
    // ========================================================================

    /// Convert Gemini response to internal MessageResponse format
    async fn convert_response(&self, gemini_resp: GeminiResponse) -> Result<MessageResponse> {
        let candidate = gemini_resp
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .context("No candidates in Gemini response")?;

        let content_blocks = self.convert_gemini_parts_to_blocks(
            candidate.content.as_ref().map(|c| &c.parts[..]).unwrap_or(&[]),
        ).await;

        let stop_reason = candidate.finish_reason.as_deref().map(|r| match r {
            "STOP" => StopReason::EndTurn,
            "MAX_TOKENS" => StopReason::MaxTokens,
            "SAFETY" => StopReason::Refusal,
            "RECITATION" => StopReason::Refusal,
            _ => StopReason::EndTurn,
        });

        // Check if response contains function calls - if so, stop reason should be ToolUse
        let has_tool_use = content_blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
        let stop_reason = if has_tool_use {
            Some(StopReason::ToolUse)
        } else {
            stop_reason
        };

        let usage = gemini_resp.usage_metadata.as_ref().map(|u| Usage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        }).unwrap_or(Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        });

        Ok(MessageResponse {
            id: gemini_resp.response_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: content_blocks,
            model: gemini_resp.model_version.unwrap_or_else(|| self.model.clone()),
            stop_reason,
            stop_sequence: None,
            usage,
        })
    }

    /// Convert Gemini parts to internal ContentBlocks, caching thought signatures
    async fn convert_gemini_parts_to_blocks(&self, parts: &[GeminiPart]) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        let mut tool_call_counter: u32 = 0;

        for part in parts {
            if let Some(ref text) = part.text {
                if part.thought == Some(true) {
                    // This is a thinking block
                    blocks.push(ContentBlock::Thinking {
                        thinking: text.clone(),
                        signature: part.thought_signature.clone().unwrap_or_default(),
                    });
                } else if !text.is_empty() {
                    blocks.push(ContentBlock::Text {
                        text: text.clone(),
                        cache_control: None,
                    });
                }
            }

            if let Some(ref fc) = part.function_call {
                tool_call_counter += 1;
                let tool_id = format!("gemini_tool_{}", tool_call_counter);

                // Cache thought signature for this tool call
                if let Some(ref sig) = part.thought_signature {
                    let mut sigs = self.thought_signatures.lock().await;
                    sigs.insert(tool_id.clone(), sig.clone());
                    // Also store by name for cross-reference
                    sigs.insert(format!("name:{}", fc.name), sig.clone());
                }

                blocks.push(ContentBlock::ToolUse {
                    id: tool_id,
                    name: fc.name.clone(),
                    input: fc.args.clone(),
                });
            }
        }

        blocks
    }

    // ========================================================================
    // API methods
    // ========================================================================

    /// Build the API URL for a given operation, using the provided base URL
    fn api_url_with_base(&self, base: &str, operation: &str) -> String {
        format!("{}/models/{}:{}", base, self.model, operation)
    }

    /// Send a non-streaming request to the Gemini API
    async fn send_gemini_request(&self, request: &GeminiRequest) -> Result<GeminiResponse> {
        // Get auth credentials (static or from provider)
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_base = auth_config.base_url.as_deref().unwrap_or(&self.api_base);
        let url = self.api_url_with_base(api_base, "generateContent");

        let request_json = serde_json::to_string(request)
            .context("Failed to serialize Gemini request")?;
        tracing::debug!("[Gemini] Request JSON: {}", request_json);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &auth_config.api_key)
            .body(request_json)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .context("Failed to read Gemini response body")?;

        tracing::debug!("[Gemini] Response status: {}", status);
        tracing::debug!("[Gemini] Response body: {}", response_text);

        if !status.is_success() {
            tracing::error!("[Gemini] API error: {} - {}", status, response_text);
            anyhow::bail!("Gemini API error ({}): {}", status, response_text);
        }

        let gemini_response: GeminiResponse = serde_json::from_str(&response_text)
            .context("Failed to parse Gemini API response")?;

        Ok(gemini_response)
    }

    /// Send a streaming request to the Gemini API
    async fn send_gemini_streaming_request(
        &self,
        request: &GeminiRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        // Get auth credentials (static or from provider)
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_base = auth_config.base_url.as_deref().unwrap_or(&self.api_base);
        let url = format!("{}?alt=sse", self.api_url_with_base(api_base, "streamGenerateContent"));

        let request_json = serde_json::to_string(request)
            .context("Failed to serialize Gemini streaming request")?;
        tracing::debug!("[Gemini] Streaming request JSON: {}", request_json);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &auth_config.api_key)
            .body(request_json)
            .send()
            .await
            .context("Failed to send streaming request to Gemini API")?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            tracing::error!("[Gemini] Streaming API error: {} - {}", status, error_text);
            anyhow::bail!("Gemini API error ({}): {}", status, error_text);
        }

        tracing::info!("[Gemini] Streaming response started");

        // Parse SSE stream and convert to internal StreamEvent format
        let byte_stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            byte_stream.map(|result| result.map_err(|e| std::io::Error::other(e.to_string()))),
        );
        let buf_reader = tokio::io::BufReader::new(stream_reader);
        let model = self.model.clone();
        let thought_sigs = self.thought_signatures.clone();

        let stream = async_stream::try_stream! {
            let mut lines = buf_reader.lines();
            let mut chunk_index: usize = 0;
            let mut content_block_started = false;
            let mut prev_had_text = false;
            let mut prev_had_function = false;
            let mut finished = false;

            tracing::info!("[Gemini] Stream: starting to read lines");

            while let Some(line) = lines.next_line().await? {
                if finished {
                    tracing::debug!("[Gemini] Stream: ignoring line after finish");
                    continue;
                }
                tracing::debug!("[Gemini] Stream: got line ({} chars): {}", line.len(), &line[..line.len().min(100)]);

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data.is_empty() {
                    continue;
                }

                tracing::info!("[Gemini] Stream: parsing chunk #{}", chunk_index);

                let gemini_resp: GeminiResponse = match serde_json::from_str(data) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("[Gemini] Failed to parse streaming chunk: {}", e);
                        continue;
                    }
                };

                // First chunk - emit MessageStart
                if chunk_index == 0 {
                    let usage = gemini_resp.usage_metadata.as_ref().map(|u| Usage {
                        input_tokens: u.prompt_token_count,
                        output_tokens: u.candidates_token_count,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    }).unwrap_or(Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    });

                    yield StreamEvent::MessageStart(MessageStartEvent {
                        message: MessageStartData {
                            id: gemini_resp.response_id.clone().unwrap_or_default(),
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

                // Process candidate content
                if let Some(candidates) = &gemini_resp.candidates {
                    for candidate in candidates {
                        if let Some(content) = &candidate.content {
                            for part in &content.parts {
                                // Handle text parts
                                if let Some(ref text) = part.text {
                                    let is_thinking = part.thought == Some(true);

                                    if is_thinking {
                                        // Thinking block
                                        if !content_block_started || prev_had_text || prev_had_function {
                                            if content_block_started {
                                                yield StreamEvent::ContentBlockStop(
                                                    ContentBlockStopEvent { index: 0 }
                                                );
                                            }
                                            yield StreamEvent::ContentBlockStart(
                                                ContentBlockStartEvent {
                                                    index: 0,
                                                    content_block: ContentBlockStart::Thinking {
                                                        thinking: String::new(),
                                                    },
                                                }
                                            );
                                            content_block_started = true;
                                            prev_had_text = false;
                                            prev_had_function = false;
                                        }
                                        yield StreamEvent::ContentBlockDelta(
                                            ContentBlockDeltaEvent {
                                                index: 0,
                                                delta: ContentDelta::ThinkingDelta {
                                                    thinking: text.clone(),
                                                },
                                            }
                                        );
                                        // If there's a thought signature, send it
                                        if let Some(ref sig) = part.thought_signature {
                                            yield StreamEvent::ContentBlockDelta(
                                                ContentBlockDeltaEvent {
                                                    index: 0,
                                                    delta: ContentDelta::SignatureDelta {
                                                        signature: sig.clone(),
                                                    },
                                                }
                                            );
                                        }
                                    } else if !text.is_empty() {
                                        // Regular text block
                                        if !content_block_started || prev_had_function {
                                            if content_block_started {
                                                yield StreamEvent::ContentBlockStop(
                                                    ContentBlockStopEvent { index: 0 }
                                                );
                                            }
                                            yield StreamEvent::ContentBlockStart(
                                                ContentBlockStartEvent {
                                                    index: 0,
                                                    content_block: ContentBlockStart::Text {
                                                        text: String::new(),
                                                    },
                                                }
                                            );
                                            content_block_started = true;
                                            prev_had_text = true;
                                            prev_had_function = false;
                                        }
                                        yield StreamEvent::ContentBlockDelta(
                                            ContentBlockDeltaEvent {
                                                index: 0,
                                                delta: ContentDelta::TextDelta {
                                                    text: text.clone(),
                                                },
                                            }
                                        );
                                    } else if text.is_empty() && part.thought_signature.is_some() {
                                        // Empty text with thought signature - just metadata, ignore
                                    }
                                }

                                // Handle function calls
                                if let Some(ref fc) = part.function_call {
                                    if content_block_started {
                                        yield StreamEvent::ContentBlockStop(
                                            ContentBlockStopEvent { index: 0 }
                                        );
                                    }

                                    let tool_id = format!("gemini_tool_{}", chunk_index);

                                    // Cache thought signature for this tool call
                                    if let Some(ref sig) = part.thought_signature {
                                        let mut sigs = thought_sigs.lock().await;
                                        sigs.insert(tool_id.clone(), sig.clone());
                                        sigs.insert(format!("name:{}", fc.name), sig.clone());
                                    }

                                    let input_json = serde_json::to_string(&fc.args)
                                        .unwrap_or_else(|_| "{}".to_string());

                                    yield StreamEvent::ContentBlockStart(
                                        ContentBlockStartEvent {
                                            index: 0,
                                            content_block: ContentBlockStart::ToolUse {
                                                id: tool_id,
                                                name: fc.name.clone(),
                                                input: Value::Object(Default::default()),
                                            },
                                        }
                                    );
                                    content_block_started = true;
                                    prev_had_function = true;
                                    prev_had_text = false;

                                    yield StreamEvent::ContentBlockDelta(
                                        ContentBlockDeltaEvent {
                                            index: 0,
                                            delta: ContentDelta::InputJsonDelta {
                                                partial_json: input_json,
                                            },
                                        }
                                    );
                                }
                            }
                        }

                        // Check finish reason
                        if let Some(ref reason) = candidate.finish_reason {
                            tracing::info!("[Gemini] Stream: finish_reason={}", reason);

                            if content_block_started {
                                yield StreamEvent::ContentBlockStop(
                                    ContentBlockStopEvent { index: 0 }
                                );
                                content_block_started = false;
                            }

                            let stop_reason = match reason.as_str() {
                                "STOP" => StopReason::EndTurn,
                                "MAX_TOKENS" => StopReason::MaxTokens,
                                "SAFETY" => StopReason::Refusal,
                                _ => StopReason::EndTurn,
                            };

                            // Override with ToolUse if we had function calls
                            let stop_reason = if prev_had_function {
                                StopReason::ToolUse
                            } else {
                                stop_reason
                            };

                            let output_tokens = gemini_resp.usage_metadata.as_ref()
                                .map(|u| u.candidates_token_count)
                                .unwrap_or(0);

                            yield StreamEvent::MessageDelta(MessageDeltaEvent {
                                delta: MessageDeltaData {
                                    stop_reason: Some(stop_reason),
                                    stop_sequence: None,
                                },
                                usage: DeltaUsage { output_tokens },
                            });

                            // Mark as finished - don't wait for more data
                            finished = true;
                        }
                    }
                }

                chunk_index += 1;

                // Break out of the loop once finished
                if finished {
                    break;
                }
            }

            tracing::info!("[Gemini] Stream: loop ended after {} chunks, finished={}", chunk_index, finished);

            // Ensure we close any open content block
            if content_block_started {
                yield StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 });
            }

            tracing::info!("[Gemini] Stream: yielding MessageStop");
            yield StreamEvent::MessageStop;
        };

        Ok(Box::pin(stream))
    }

    /// Build a GeminiRequest from internal types
    async fn build_request(
        &self,
        messages: &[Message],
        system: &Option<SystemPrompt>,
        tools: &[ToolDefinition],
        tool_choice: &Option<ToolChoice>,
        _thinking: &Option<ThinkingConfig>,
    ) -> GeminiRequest {
        let contents = self.convert_messages(messages).await;
        let system_instruction = self.convert_system_prompt(system);
        let gemini_tools = self.convert_tools(tools);
        let tool_config = if gemini_tools.is_some() {
            self.convert_tool_config(tool_choice)
        } else {
            None
        };

        GeminiRequest {
            contents,
            system_instruction,
            tools: gemini_tools,
            tool_config,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: Some(self.max_tokens),
                temperature: Some(1.0),
            }),
        }
    }
}

// ============================================================================
// LlmProvider implementation
// ============================================================================

#[async_trait::async_trait]
impl LlmProvider for GeminiProvider {
    async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        _session_id: Option<&str>,
    ) -> Result<String> {
        tracing::info!("[Gemini] Sending message");

        let mut messages: Vec<Message> = conversation_history.to_vec();
        messages.push(Message::user(user_message));

        let system = system_prompt.map(|s| SystemPrompt::Text(s.to_string()));
        let request = self.build_request(&messages, &system, &[], &None, &None).await;

        let gemini_response = self.send_gemini_request(&request).await?;
        let response = self.convert_response(gemini_response).await?;

        Ok(response.text())
    }

    async fn send_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        _session_id: Option<&str>,
    ) -> Result<MessageResponse> {
        tracing::info!("[Gemini] Sending message with tools");
        tracing::debug!("[Gemini] Messages count: {}", messages.len());
        tracing::debug!("[Gemini] Tools count: {}", tools.len());

        let request = self.build_request(&messages, &system, &tools, &tool_choice, &thinking).await;
        let gemini_response = self.send_gemini_request(&request).await?;
        self.convert_response(gemini_response).await
    }

    async fn stream_with_tools_and_system(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        _session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        tracing::info!("[Gemini] Streaming message with tools");
        tracing::debug!("[Gemini] Messages count: {}", messages.len());
        tracing::debug!("[Gemini] Tools count: {}", tools.len());

        let request = self.build_request(&messages, &system, &tools, &tool_choice, &thinking).await;
        self.send_gemini_streaming_request(&request).await
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn provider_name(&self) -> &str {
        "gemini"
    }

    fn create_variant(&self, model: &str, max_tokens: u32) -> Arc<dyn LlmProvider> {
        Arc::new(self.create_variant_impl(model, max_tokens))
    }
}
