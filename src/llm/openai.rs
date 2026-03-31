//! OpenAI Responses API client
//!
//! Translates between the framework's internal message types (Anthropic format)
//! and the OpenAI Responses API format.
//!
//! # Authentication
//!
//! ```ignore
//! // From environment variables (OPENAI_API_KEY, OPENAI_MODEL)
//! let llm = OpenAIProvider::from_env()?;
//!
//! // With explicit API key
//! let llm = OpenAIProvider::new("sk-...")?.with_model("gpt-4o");
//!
//! // With custom base URL (e.g. Azure, local proxy)
//! let llm = OpenAIProvider::new("sk-...")?
//!     .with_model("gpt-4o")
//!     .with_base_url("https://my-proxy.example.com/v1/responses");
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

const DEFAULT_API_URL: &str = "https://api.openai.com/v1/responses";

// ============================================================================
// OpenAI Responses API request types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    max_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAIReasoning>,
    stream: bool,
}

/// Reasoning configuration for o-series and reasoning-capable models
#[derive(Debug, Serialize)]
struct OpenAIReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

/// Top-level item in the `input` array
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum InputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: InputContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

/// Content for a message input item — either a plain string or an array of parts
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum InputContent {
    Text(String),
    Parts(Vec<InputContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum InputContentPart {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "input_image")]
    Image { image_url: String },
}

/// OpenAI function tool definition
#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String, // always "function"
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    strict: bool,
}

// ============================================================================
// OpenAI Responses API response types
// ============================================================================

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    #[serde(default)]
    output: Vec<OutputItem>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
    #[serde(default)]
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum OutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        #[serde(default)]
        content: Vec<OutputContentPart>,
        #[serde(default)]
        status: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default)]
        status: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<ReasoningSummaryPart>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ReasoningSummaryPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum OutputContentPart {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct OutputTokensDetails {
    #[serde(default)]
    reasoning_tokens: u32,
}

// ============================================================================
// Streaming event types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum OpenAIStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated { response: OpenAIStreamResponse },

    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: OutputItemPartial,
    },

    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        output_index: usize,
        content_index: usize,
        part: ContentPartPartial,
    },

    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        output_index: usize,
        content_index: usize,
        delta: String,
    },

    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        output_index: usize,
        delta: String,
    },

    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: OutputItem,
    },

    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        output_index: usize,
        summary_index: usize,
    },

    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        output_index: usize,
        #[allow(dead_code)]
        summary_index: usize,
        delta: String,
    },

    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        #[allow(dead_code)]
        output_index: usize,
        #[allow(dead_code)]
        summary_index: usize,
        #[allow(dead_code)]
        text: String,
    },

    #[serde(rename = "response.completed")]
    ResponseCompleted { response: OpenAIStreamResponse },

    #[serde(rename = "response.failed")]
    ResponseFailed { response: OpenAIStreamResponse },

    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIStreamResponse {
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
    #[serde(default)]
    output: Vec<OutputItem>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum OutputItemPartial {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum ContentPartPartial {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(other)]
    Unknown,
}

// ============================================================================
// OpenAI Provider
// ============================================================================

/// OpenAI Responses API provider
///
/// Translates between the internal Anthropic-format message types and
/// the OpenAI Responses API wire format.
pub struct OpenAIProvider {
    client: Client,
    auth: AuthSource,
    model: String,
    max_tokens: u32,
}

impl OpenAIProvider {
    /// Create a provider from environment variables.
    ///
    /// Reads:
    /// - `OPENAI_API_KEY` (required)
    /// - `OPENAI_MODEL` (required)
    /// - `OPENAI_BASE_URL` (optional, defaults to `https://api.openai.com/v1/responses`)
    /// - `OPENAI_MAX_TOKENS` (optional, defaults to 32000)
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;
        let model = env::var("OPENAI_MODEL")
            .context("OPENAI_MODEL environment variable not set")?;
        let base_url = env::var("OPENAI_BASE_URL").ok();
        let max_tokens = env::var("OPENAI_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32000);

        tracing::info!("Creating OpenAI provider from environment");
        tracing::info!("Using model: {}", model);
        tracing::info!("Max tokens: {}", max_tokens);
        if let Some(ref url) = base_url {
            tracing::info!("Using custom base URL: {}", url);
        }

        Ok(Self {
            client: Client::new(),
            auth: AuthSource::Static(AuthConfig { api_key, base_url }),
            model,
            max_tokens,
        })
    }

    /// Create a provider with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            auth: AuthSource::Static(AuthConfig::new(api_key)),
            model: String::new(),
            max_tokens: 32000,
        })
    }

    /// Create a provider with a dynamic auth callback.
    pub fn with_auth_provider<F, Fut>(provider: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AuthConfig>> + Send + 'static,
    {
        Self {
            client: Client::new(),
            auth: AuthSource::Dynamic(Arc::new(auth_provider(provider))),
            model: String::new(),
            max_tokens: 32000,
        }
    }

    /// Create a provider with a boxed `AuthProvider`.
    pub fn with_auth_provider_boxed(provider: Arc<dyn AuthProvider>) -> Self {
        Self {
            client: Client::new(),
            auth: AuthSource::Dynamic(provider),
            model: String::new(),
            max_tokens: 32000,
        }
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the maximum output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override the base URL (e.g. for Azure OpenAI or a local proxy).
    ///
    /// The URL should point directly to the responses endpoint, e.g.:
    /// `https://my-proxy.example.com/v1/responses`
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        match &mut self.auth {
            AuthSource::Static(config) => {
                config.base_url = Some(base_url.into());
            }
            AuthSource::Dynamic(_) => {
                // For dynamic auth, the provider controls the base URL.
                // Log a warning that this is a no-op.
                tracing::warn!("with_base_url() has no effect when using a dynamic auth provider; set base_url inside the AuthConfig returned by your provider");
            }
        }
        self
    }

    /// Clone with a different model and max_tokens (shares auth).
    pub fn with_model_and_tokens_override(&self, model: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            auth: self.auth.clone(),
            model: model.into(),
            max_tokens,
        }
    }

    // ------------------------------------------------------------------ //
    // Internal helpers
    // ------------------------------------------------------------------ //

    async fn send_request_internal(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<MessageResponse> {
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_url = auth_config.base_url.as_deref().unwrap_or(DEFAULT_API_URL);

        let openai_req = build_request(
            &self.model,
            self.max_tokens,
            messages,
            system,
            tools,
            tool_choice,
            thinking,
            false,
        );

        let req_json = serde_json::to_string(&openai_req)
            .context("Failed to serialize OpenAI request")?;
        tracing::debug!("OpenAI request JSON: {}", req_json);

        let mut builder = self.client
            .post(api_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", auth_config.api_key));

        if let Some(sid) = session_id {
            builder = builder.header("agent-session-id", sid);
        }

        let response = builder
            .body(req_json)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        let status = response.status();
        let body = response.text().await.context("Failed to read OpenAI response body")?;

        tracing::debug!("OpenAI response status: {}", status);
        tracing::debug!("OpenAI response body: {}", body);

        if !status.is_success() {
            anyhow::bail!("OpenAI API error ({}): {}", status, body);
        }

        let openai_resp: OpenAIResponse = serde_json::from_str(&body)
            .context("Failed to parse OpenAI response")?;

        Ok(openai_response_to_anthropic(openai_resp))
    }

    async fn stream_request_internal(
        &self,
        messages: Vec<Message>,
        system: Option<SystemPrompt>,
        tools: Vec<ToolDefinition>,
        tool_choice: Option<ToolChoice>,
        thinking: Option<ThinkingConfig>,
        session_id: Option<&str>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let auth_config = self.auth.get_auth().await
            .context("Failed to get authentication credentials")?;
        let api_url = auth_config.base_url.as_deref().unwrap_or(DEFAULT_API_URL);

        let openai_req = build_request(
            &self.model,
            self.max_tokens,
            messages,
            system,
            tools,
            tool_choice,
            thinking,
            true,
        );

        let req_json = serde_json::to_string(&openai_req)
            .context("Failed to serialize OpenAI request")?;
        tracing::debug!("OpenAI streaming request JSON: {}", req_json);

        let mut builder = self.client
            .post(api_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", auth_config.api_key));

        if let Some(sid) = session_id {
            builder = builder.header("agent-session-id", sid);
        }

        let response = builder
            .body(req_json)
            .send()
            .await
            .context("Failed to send streaming request to OpenAI API")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());
            anyhow::bail!("OpenAI API error ({}): {}", status, error_text);
        }

        let model = self.model.clone();
        let byte_stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            byte_stream.map(|r| r.map_err(|e| std::io::Error::other(e.to_string()))),
        );
        let buf_reader = tokio::io::BufReader::new(stream_reader);

        let stream = async_stream::try_stream! {
            let mut lines = buf_reader.lines();
            let mut current_data;

            // State needed to synthesise Anthropic-style block indices
            // Each output_item gets its own block index.
            let mut block_index: usize = 0;

            while let Some(line) = lines.next_line().await? {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" {
                        break;
                    }
                    current_data = data.to_string();

                    match serde_json::from_str::<OpenAIStreamEvent>(&current_data) {
                        Ok(event) => {
                            for stream_event in translate_stream_event(event, &model, &mut block_index) {
                                yield stream_event;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse OpenAI SSE event: {} - data: {}", e, current_data);
                        }
                    }
                }
                // Ignore lines that are not "data: ..." (e.g. "event: ...", empty lines)
            }
        };

        Ok(Box::pin(stream))
    }
}

// ============================================================================
// Translation: Anthropic → OpenAI
// ============================================================================

fn build_request(
    model: &str,
    max_tokens: u32,
    messages: Vec<Message>,
    system: Option<SystemPrompt>,
    tools: Vec<ToolDefinition>,
    tool_choice: Option<ToolChoice>,
    thinking: Option<ThinkingConfig>,
    stream: bool,
) -> OpenAIRequest {
    let instructions = system.map(system_prompt_to_string);
    let input = messages_to_input_items(messages);
    let openai_tools = if tools.is_empty() {
        None
    } else {
        Some(tools.into_iter().filter_map(tool_def_to_openai).collect())
    };
    let openai_tool_choice = tool_choice.map(tool_choice_to_openai);
    let reasoning = thinking_to_reasoning(thinking);

    OpenAIRequest {
        model: model.to_string(),
        input,
        instructions,
        tools: openai_tools,
        tool_choice: openai_tool_choice,
        max_output_tokens: max_tokens,
        temperature: None,
        reasoning,
        stream,
    }
}

/// Convert internal ThinkingConfig to OpenAI reasoning format
fn thinking_to_reasoning(thinking: Option<ThinkingConfig>) -> Option<OpenAIReasoning> {
    thinking.map(|config| {
        let effort = match config.budget_tokens {
            0 => "low",
            1..=2048 => "low",
            2049..=8192 => "medium",
            _ => "high",
        };
        tracing::info!("[OpenAI] Reasoning enabled: effort={}, budget_tokens={}", effort, config.budget_tokens);
        OpenAIReasoning {
            effort: Some(effort.to_string()),
            summary: Some("auto".to_string()),
        }
    })
}

fn system_prompt_to_string(system: SystemPrompt) -> String {
    match system {
        SystemPrompt::Text(s) => s,
        SystemPrompt::Blocks(blocks) => blocks
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn messages_to_input_items(messages: Vec<Message>) -> Vec<InputItem> {
    let mut items = Vec::new();

    for msg in messages {
        let is_assistant = msg.role == "assistant";

        match msg.content {
            MessageContent::Text(text) => {
                items.push(InputItem::Message {
                    role: msg.role,
                    content: InputContent::Text(text),
                });
            }
            MessageContent::Blocks(blocks) => {
                // For assistant messages, text content must be sent as a plain string
                // (OpenAI only accepts "output_text"/"refusal" for assistant content parts,
                // but "input_text" for user parts — so it's simpler to use a plain string).
                // For user messages, text blocks become "input_text" parts.
                let mut text_parts: Vec<InputContentPart> = Vec::new();
                let mut text_strings: Vec<String> = Vec::new(); // for assistant messages

                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => {
                            if is_assistant {
                                text_strings.push(text);
                            } else {
                                text_parts.push(InputContentPart::Text { text });
                            }
                        }
                        ContentBlock::Image { source, .. } => {
                            // Images only make sense in user messages
                            if !is_assistant {
                                let image_url = format!(
                                    "data:{};base64,{}",
                                    source.media_type, source.data
                                );
                                text_parts.push(InputContentPart::Image { image_url });
                            }
                        }
                        ContentBlock::ToolUse { id, name, input, signature } => {
                            // Flush any accumulated text before this tool call
                            if is_assistant {
                                if !text_strings.is_empty() {
                                    let joined = std::mem::take(&mut text_strings).join("");
                                    items.push(InputItem::Message {
                                        role: msg.role.clone(),
                                        content: InputContent::Text(joined),
                                    });
                                }
                            } else if !text_parts.is_empty() {
                                let parts = std::mem::take(&mut text_parts);
                                items.push(InputItem::Message {
                                    role: msg.role.clone(),
                                    content: InputContent::Parts(parts),
                                });
                            }
                            let arguments = serde_json::to_string(&input)
                                .unwrap_or_else(|_| "{}".to_string());
                            // signature stores the original OpenAI fc_... id (required by API).
                            // id is the call_id used for round-tripping tool results.
                            let fc_id = signature.unwrap_or_else(|| format!("fc_{}", id));
                            items.push(InputItem::FunctionCall {
                                id: fc_id,
                                call_id: id,
                                name,
                                arguments,
                            });
                        }
                        ContentBlock::ToolResult { tool_use_id, content, .. } => {
                            let output = content.unwrap_or_default();
                            items.push(InputItem::FunctionCallOutput {
                                call_id: tool_use_id,
                                output,
                            });
                        }
                        // Thinking blocks have no OpenAI equivalent — drop them
                        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
                        ContentBlock::Document { .. } => {
                            // Documents not supported in OpenAI Responses API input
                        }
                    }
                }

                // Emit any remaining text
                if is_assistant {
                    if !text_strings.is_empty() {
                        items.push(InputItem::Message {
                            role: msg.role,
                            content: InputContent::Text(text_strings.join("")),
                        });
                    }
                } else if !text_parts.is_empty() {
                    items.push(InputItem::Message {
                        role: msg.role,
                        content: InputContent::Parts(text_parts),
                    });
                }
            }
        }
    }

    items
}

fn tool_def_to_openai(tool: ToolDefinition) -> Option<OpenAITool> {
    match tool {
        ToolDefinition::Custom(custom) => {
            let parameters = {
                let props = custom.input_schema.properties.unwrap_or(serde_json::json!({}));
                let required = custom.input_schema.required.unwrap_or_default();
                serde_json::json!({
                    "type": "object",
                    "properties": props,
                    "required": required,
                })
            };
            Some(OpenAITool {
                tool_type: "function".to_string(),
                name: custom.name,
                description: custom.description,
                parameters: Some(parameters),
                strict: false,
            })
        }
        // Built-in bash/text-editor tools don't map to OpenAI functions
        ToolDefinition::Bash(_) | ToolDefinition::TextEditor(_) => None,
    }
}

fn tool_choice_to_openai(tc: ToolChoice) -> Value {
    match tc {
        ToolChoice::Auto { .. } => Value::String("auto".to_string()),
        ToolChoice::Any { .. } => Value::String("required".to_string()),
        ToolChoice::None => Value::String("none".to_string()),
        ToolChoice::Tool { name, .. } => {
            serde_json::json!({ "type": "function", "name": name })
        }
    }
}

// ============================================================================
// Translation: OpenAI → Anthropic
// ============================================================================

fn openai_response_to_anthropic(resp: OpenAIResponse) -> MessageResponse {
    let mut content_blocks: Vec<ContentBlock> = Vec::new();
    let mut has_tool_use = false;

    for item in resp.output {
        match item {
            OutputItem::Message { content, .. } => {
                for part in content {
                    if let OutputContentPart::Text { text } = part {
                        content_blocks.push(ContentBlock::Text {
                            text,
                            cache_control: None,
                        });
                    }
                }
            }
            OutputItem::FunctionCall { id: fc_id, call_id, name, arguments, .. } => {
                let input: Value = serde_json::from_str(&arguments)
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                content_blocks.push(ContentBlock::ToolUse {
                    // call_id is used as our internal id (for round-tripping tool results)
                    id: call_id,
                    name,
                    input,
                    // Store the fc_... id so we can reconstruct history correctly
                    signature: if fc_id.is_empty() { None } else { Some(fc_id) },
                });
                has_tool_use = true;
            }
            OutputItem::Reasoning { summary, .. } => {
                let text = summary.into_iter()
                    .filter_map(|part| match part {
                        ReasoningSummaryPart::Text { text } => Some(text),
                        ReasoningSummaryPart::Unknown => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                if !text.is_empty() {
                    content_blocks.push(ContentBlock::Thinking {
                        thinking: text,
                        signature: String::new(),
                    });
                }
            }
            OutputItem::Unknown => {}
        }
    }

    let stop_reason = if has_tool_use {
        Some(StopReason::ToolUse)
    } else if resp.status == "incomplete" {
        Some(StopReason::MaxTokens)
    } else {
        Some(StopReason::EndTurn)
    };

    let usage = resp.usage.unwrap_or_default();

    let reasoning_tokens = usage.output_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .filter(|&t| t > 0);

    MessageResponse {
        id: resp.id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: content_blocks,
        model: String::new(), // model not echoed in all OpenAI responses
        stop_reason,
        stop_sequence: None,
        usage: Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            thoughts_token_count: reasoning_tokens,
        },
    }
}

// ============================================================================
// Streaming translation
// ============================================================================

fn translate_stream_event(
    event: OpenAIStreamEvent,
    model: &str,
    block_index: &mut usize,
) -> Vec<StreamEvent> {
    match event {
        OpenAIStreamEvent::ResponseCreated { response } => {
            let usage = response.usage.unwrap_or_default();
            vec![StreamEvent::MessageStart(MessageStartEvent {
                message: MessageStartData {
                    id: response.id,
                    message_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content: vec![],
                    model: model.to_string(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                        thoughts_token_count: None,
                    },
                },
            })]
        }

        OpenAIStreamEvent::OutputItemAdded { output_index, item } => {
            // Assign a stable block index for this output_index
            *block_index = output_index;
            let cb_start = match item {
                OutputItemPartial::Message { .. } => {
                    ContentBlockStart::Text { text: String::new() }
                }
                OutputItemPartial::FunctionCall { call_id, name, .. } => {
                    ContentBlockStart::ToolUse {
                        id: call_id.clone(),
                        name,
                        input: Value::Null,
                        signature: None,
                    }
                }
                OutputItemPartial::Reasoning { .. } => {
                    ContentBlockStart::Thinking {
                        thinking: String::new(),
                    }
                }
                OutputItemPartial::Unknown => return vec![],
            };
            vec![StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index: output_index,
                content_block: cb_start,
            })]
        }

        OpenAIStreamEvent::OutputTextDelta { output_index, delta, .. } => {
            vec![StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: output_index,
                delta: ContentDelta::TextDelta { text: delta },
            })]
        }

        OpenAIStreamEvent::FunctionCallArgumentsDelta { output_index, delta } => {
            vec![StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: output_index,
                delta: ContentDelta::InputJsonDelta { partial_json: delta },
            })]
        }

        OpenAIStreamEvent::OutputItemDone { output_index, item: _ } => {
            // For function_call items we want to emit a final ToolUse block start
            // (with complete data) followed by a stop. But since we already emitted
            // a start event from OutputItemAdded, here we just emit the stop.
            // The agent loop reconstructs tool input from the accumulated JSON deltas.
            vec![StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: output_index,
            })]
        }

        OpenAIStreamEvent::ReasoningSummaryPartAdded { .. } => {
            // Already handled by OutputItemAdded(Reasoning) → ContentBlockStart(Thinking)
            vec![]
        }

        OpenAIStreamEvent::ReasoningSummaryTextDelta { output_index, delta, .. } => {
            vec![StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: output_index,
                delta: ContentDelta::ThinkingDelta { thinking: delta },
            })]
        }

        OpenAIStreamEvent::ReasoningSummaryTextDone { .. } => {
            // Accumulated text is handled by the agent loop via ThinkingDelta events
            vec![]
        }

        OpenAIStreamEvent::ResponseCompleted { response } => {
            let usage = response.usage.unwrap_or_default();
            let has_tool_use = response.output.iter().any(|item| {
                matches!(item, OutputItem::FunctionCall { .. })
            });
            let stop_reason = if has_tool_use {
                Some(StopReason::ToolUse)
            } else if response.status == "incomplete" {
                Some(StopReason::MaxTokens)
            } else {
                Some(StopReason::EndTurn)
            };
            vec![
                StreamEvent::MessageDelta(MessageDeltaEvent {
                    delta: MessageDeltaData {
                        stop_reason,
                        stop_sequence: None,
                    },
                    usage: DeltaUsage {
                        output_tokens: usage.output_tokens,
                    },
                }),
                StreamEvent::MessageStop,
            ]
        }

        OpenAIStreamEvent::ResponseFailed { .. } => {
            // Let the stream end; the caller will surface the error from the HTTP status.
            vec![StreamEvent::MessageStop]
        }

        // Ignore events we don't need to translate
        OpenAIStreamEvent::ContentPartAdded { .. } | OpenAIStreamEvent::Unknown => vec![],
    }
}

// ============================================================================
// LlmProvider impl
// ============================================================================

#[async_trait::async_trait]
impl LlmProvider for OpenAIProvider {
    async fn send_message(
        &self,
        user_message: &str,
        conversation_history: &[Message],
        system_prompt: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<String> {
        let mut messages = conversation_history.to_vec();
        messages.push(Message::user(user_message));

        let system = system_prompt.map(|s| SystemPrompt::Text(s.to_string()));
        let resp = self
            .send_request_internal(messages, system, vec![], None, None, session_id)
            .await?;
        Ok(resp.text())
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
        self.send_request_internal(messages, system, tools, tool_choice, thinking, session_id)
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
        self.stream_request_internal(messages, system, tools, tool_choice, thinking, session_id)
            .await
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn create_variant(&self, model: &str, max_tokens: u32) -> Arc<dyn LlmProvider> {
        Arc::new(self.with_model_and_tokens_override(model, max_tokens))
    }
}
