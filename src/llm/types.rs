//! Anthropic API types matching the official REST API specification
//!
//! These types are designed to serialize/deserialize correctly with the Anthropic Messages API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Cache Control
// ============================================================================

/// Cache control configuration for prompt caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache type (always "ephemeral")
    #[serde(rename = "type")]
    pub cache_type: String,

    /// Time to live ("5m" or "1h")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    /// Create ephemeral cache control with 5-minute TTL
    pub fn ephemeral_5m() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: Some("5m".to_string()),
        }
    }

    /// Create ephemeral cache control with 1-hour TTL
    pub fn ephemeral_1h() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: Some("1h".to_string()),
        }
    }

    /// Create ephemeral cache control with default TTL (5 minutes)
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: None,
        }
    }
}

// ============================================================================
// Request Types
// ============================================================================

/// System prompt - either a simple string or array of text blocks (for caching)
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple text system prompt
    Text(String),
    /// Array of text blocks (for prompt caching)
    Blocks(Vec<SystemBlock>),
}

/// System block with cache control
#[derive(Debug, Clone, Serialize)]
pub struct SystemBlock {
    /// Type (always "text")
    #[serde(rename = "type")]
    pub block_type: String,
    /// Text content
    pub text: String,
    /// Cache control (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    /// Create a new system block
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            block_type: "text".to_string(),
            text: text.into(),
            cache_control: None,
        }
    }

    /// Add cache control to this block
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Request body for the Anthropic Messages API
#[derive(Debug, Clone, Serialize)]
pub struct MessageRequest {
    /// The model to use
    pub model: String,

    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// Input messages
    pub messages: Vec<Message>,

    /// System prompt (optional) - can be string or array of blocks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Tools available to the model (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// How the model should use tools (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Extended thinking configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Temperature for sampling (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Whether to stream the response (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Extended thinking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Type of thinking ("enabled")
    #[serde(rename = "type")]
    pub thinking_type: String,

    /// Budget tokens for thinking
    pub budget_tokens: u32,
}

impl ThinkingConfig {
    /// Create a new thinking config with enabled thinking
    pub fn enabled(budget_tokens: u32) -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens,
        }
    }
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender ("user" or "assistant")
    pub role: String,

    /// Content of the message - can be a string or array of content blocks
    pub content: MessageContent,
}

/// Message content - either a simple string or array of content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Array of content blocks (for tool use, tool results, thinking, etc.)
    Blocks(Vec<ContentBlock>),
}

impl Message {
    /// Create a simple user message with text content
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Text(text.into()),
        }
    }

    /// Create a simple assistant message with text content
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Text(text.into()),
        }
    }

    /// Create a user message with content blocks (for tool results)
    pub fn user_with_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }

    /// Create an assistant message with content blocks (for tool use)
    pub fn assistant_with_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }

    /// Get text content if this is a simple text message
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Blocks(_) => None,
        }
    }

    /// Get content blocks if this message has blocks
    pub fn blocks(&self) -> Option<&[ContentBlock]> {
        match &self.content {
            MessageContent::Text(_) => None,
            MessageContent::Blocks(blocks) => Some(blocks),
        }
    }

    /// Append text to this message
    ///
    /// - For Text messages: appends to the string
    /// - For Block messages: appends a new text block
    pub fn append_text(&mut self, text: &str) {
        match &mut self.content {
            MessageContent::Text(s) => {
                s.push_str(text);
            }
            MessageContent::Blocks(blocks) => {
                blocks.push(ContentBlock::Text {
                    text: text.to_string(),
                    cache_control: None,
                });
            }
        }
    }

    /// Prepend text to this message
    ///
    /// - For Text messages: prepends to the string
    /// - For Block messages: inserts a new text block at the beginning
    pub fn prepend_text(&mut self, text: &str) {
        match &mut self.content {
            MessageContent::Text(s) => {
                *s = format!("{}{}", text, s);
            }
            MessageContent::Blocks(blocks) => {
                blocks.insert(
                    0,
                    ContentBlock::Text {
                        text: text.to_string(),
                        cache_control: None,
                    },
                );
            }
        }
    }
}

// ============================================================================
// Content Blocks
// ============================================================================

/// Image source for image content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    /// Source type (always "base64")
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type (e.g., "image/png", "image/jpeg", "image/gif", "image/webp")
    pub media_type: String,
    /// Base64-encoded image data
    pub data: String,
}

impl ImageSource {
    /// Create a new image source from base64 data
    pub fn base64(data: String, media_type: String) -> Self {
        Self {
            source_type: "base64".to_string(),
            media_type,
            data,
        }
    }
}

/// Document source for document content blocks (PDFs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    /// Source type (always "base64")
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type (e.g., "application/pdf")
    pub media_type: String,
    /// Base64-encoded document data
    pub data: String,
}

impl DocumentSource {
    /// Create a new document source from base64 data
    pub fn base64(data: String, media_type: String) -> Self {
        Self {
            source_type: "base64".to_string(),
            media_type,
            data,
        }
    }
}

/// Content block in a message - supports text, tool_use, tool_result, and thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Tool use request from the model
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    /// Tool result from the user
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Thinking block (for extended thinking)
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
    },

    /// Redacted thinking block
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },

    /// Image block
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
    },

    /// Document block (for PDFs)
    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
    },
}

impl ContentBlock {
    /// Create a text content block
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Create a text content block with cache control
    pub fn text_with_cache(text: impl Into<String>, cache_control: CacheControl) -> Self {
        ContentBlock::Text {
            text: text.into(),
            cache_control: Some(cache_control),
        }
    }

    /// Create a tool use content block
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a tool result content block
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: Some(content.into()),
            is_error: if is_error { Some(true) } else { None },
            cache_control: None,
        }
    }

    /// Create a tool result content block with cache control
    pub fn tool_result_with_cache(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
        cache_control: CacheControl,
    ) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: Some(content.into()),
            is_error: if is_error { Some(true) } else { None },
            cache_control: Some(cache_control),
        }
    }

    /// Create an image content block from base64 data
    pub fn image(data: String, media_type: String) -> Self {
        ContentBlock::Image {
            source: ImageSource::base64(data, media_type),
        }
    }

    /// Create a document content block from base64 data
    pub fn document(data: String, media_type: String) -> Self {
        ContentBlock::Document {
            source: DocumentSource::base64(data, media_type),
        }
    }

    /// Add cache control to this content block (if applicable)
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        match &mut self {
            ContentBlock::Text { cache_control: cc, .. } => {
                *cc = Some(cache_control);
            }
            ContentBlock::ToolResult { cache_control: cc, .. } => {
                *cc = Some(cache_control);
            }
            _ => {
                // Other block types don't support cache control
            }
        }
        self
    }

    /// Get the text content if this is a text block
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }

    /// Get the tool use info if this is a tool use block
    pub fn as_tool_use(&self) -> Option<(&str, &str, &Value)> {
        match self {
            ContentBlock::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        }
    }
}

// ============================================================================
// Tool Definitions
// ============================================================================

/// Tool definition for the API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolDefinition {
    /// Custom tool with JSON schema
    Custom(CustomTool),
    /// Built-in bash tool
    Bash(BashTool),
    /// Built-in text editor tool
    TextEditor(TextEditorTool),
}

impl ToolDefinition {
    /// Add cache control to this tool definition
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        match &mut self {
            ToolDefinition::Custom(tool) => {
                tool.cache_control = Some(cache_control);
            }
            ToolDefinition::Bash(tool) => {
                tool.cache_control = Some(cache_control);
            }
            ToolDefinition::TextEditor(tool) => {
                tool.cache_control = Some(cache_control);
            }
        }
        self
    }
}

/// Custom tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTool {
    /// Tool name
    pub name: String,

    /// Tool description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON schema for the tool input
    pub input_schema: ToolInputSchema,

    /// Optional type field (always "custom" for custom tools)
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,

    /// Cache control (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// JSON schema for tool input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// Type (always "object")
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Properties of the input object
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<Value>,

    /// Required properties
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl ToolInputSchema {
    /// Create a new tool input schema
    pub fn new() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: None,
            required: None,
        }
    }

    /// Set the properties
    pub fn with_properties(mut self, properties: Value) -> Self {
        self.properties = Some(properties);
        self
    }

    /// Set the required fields
    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = Some(required);
        self
    }
}

impl Default for ToolInputSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in bash tool (bash_20250124)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashTool {
    /// Tool name (always "bash")
    pub name: String,

    /// Tool type (always "bash_20250124")
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Cache control (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Default for BashTool {
    fn default() -> Self {
        Self {
            name: "bash".to_string(),
            tool_type: "bash_20250124".to_string(),
            cache_control: None,
        }
    }
}

/// Built-in text editor tool (text_editor_20250124)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEditorTool {
    /// Tool name (always "str_replace_editor")
    pub name: String,

    /// Tool type
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Cache control (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Default for TextEditorTool {
    fn default() -> Self {
        Self {
            name: "str_replace_editor".to_string(),
            tool_type: "text_editor_20250124".to_string(),
            cache_control: None,
        }
    }
}

// ============================================================================
// Tool Choice
// ============================================================================

/// How the model should use tools
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolChoice {
    /// Model decides whether to use tools
    #[serde(rename = "auto")]
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model must use at least one tool
    #[serde(rename = "any")]
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model must use the specified tool
    #[serde(rename = "tool")]
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },

    /// Model cannot use tools
    #[serde(rename = "none")]
    None,
}

impl ToolChoice {
    /// Create auto tool choice
    pub fn auto() -> Self {
        ToolChoice::Auto {
            disable_parallel_tool_use: None,
        }
    }

    /// Create any tool choice
    pub fn any() -> Self {
        ToolChoice::Any {
            disable_parallel_tool_use: None,
        }
    }

    /// Create tool choice for a specific tool
    pub fn tool(name: impl Into<String>) -> Self {
        ToolChoice::Tool {
            name: name.into(),
            disable_parallel_tool_use: None,
        }
    }

    /// Create none tool choice
    pub fn none() -> Self {
        ToolChoice::None
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Response from the Anthropic Messages API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    /// Unique message ID
    pub id: String,

    /// Type (always "message")
    #[serde(rename = "type")]
    pub response_type: String,

    /// Role (always "assistant")
    pub role: String,

    /// Content blocks in the response
    pub content: Vec<ContentBlock>,

    /// Model used
    pub model: String,

    /// Reason for stopping
    pub stop_reason: Option<StopReason>,

    /// Stop sequence that was matched (if any)
    pub stop_sequence: Option<String>,

    /// Token usage
    pub usage: Usage,
}

impl MessageResponse {
    /// Get all text content from the response
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all tool use blocks from the response
    pub fn tool_uses(&self) -> Vec<(&str, &str, &Value)> {
        self.content
            .iter()
            .filter_map(|block| block.as_tool_use())
            .collect()
    }

    /// Check if the response contains tool use
    pub fn has_tool_use(&self) -> bool {
        self.content.iter().any(|block| matches!(block, ContentBlock::ToolUse { .. }))
    }

    /// Check if the model wants to stop
    pub fn is_end_turn(&self) -> bool {
        matches!(self.stop_reason, Some(StopReason::EndTurn))
    }
}

/// Reason why the model stopped generating
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model reached a natural stopping point
    EndTurn,
    /// Max tokens reached
    MaxTokens,
    /// Stop sequence matched
    StopSequence,
    /// Model invoked tools
    ToolUse,
    /// Long-running turn was paused
    PauseTurn,
    /// Policy violation refusal
    Refusal,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens used
    pub input_tokens: u32,

    /// Output tokens generated
    pub output_tokens: u32,

    /// Cache creation tokens (if caching enabled)
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,

    /// Cache read tokens (if caching enabled)
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

// ============================================================================
// Error Types
// ============================================================================

/// Error response from the Anthropic API
#[derive(Debug, Clone, Deserialize)]
pub struct ApiError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error details
    pub error: ApiErrorDetails,
}

/// Details of an API error
#[derive(Debug, Clone, Deserialize)]
pub struct ApiErrorDetails {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error message
    pub message: String,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Server-sent event from the streaming API
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Initial message with metadata
    MessageStart(MessageStartEvent),
    /// Start of a content block
    ContentBlockStart(ContentBlockStartEvent),
    /// Delta update to a content block
    ContentBlockDelta(ContentBlockDeltaEvent),
    /// End of a content block
    ContentBlockStop(ContentBlockStopEvent),
    /// Final message delta with stop reason and usage
    MessageDelta(MessageDeltaEvent),
    /// Stream complete
    MessageStop,
    /// Keep-alive ping
    Ping,
    /// Error event
    Error(StreamError),
}

/// Event data for message_start
#[derive(Debug, Clone, Deserialize)]
pub struct MessageStartEvent {
    /// The message object (with empty content)
    pub message: MessageStartData,
}

/// Message data in message_start event
#[derive(Debug, Clone, Deserialize)]
pub struct MessageStartData {
    /// Unique message ID
    pub id: String,
    /// Type (always "message")
    #[serde(rename = "type")]
    pub message_type: String,
    /// Role (always "assistant")
    pub role: String,
    /// Content (empty array at start)
    pub content: Vec<ContentBlock>,
    /// Model used
    pub model: String,
    /// Stop reason (null at start)
    pub stop_reason: Option<StopReason>,
    /// Stop sequence (null at start)
    pub stop_sequence: Option<String>,
    /// Initial usage
    pub usage: Usage,
}

/// Event data for content_block_start
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockStartEvent {
    /// Index of this content block
    pub index: usize,
    /// The content block (type only, content is empty)
    pub content_block: ContentBlockStart,
}

/// Content block start data
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlockStart {
    /// Text block start
    #[serde(rename = "text")]
    Text { text: String },
    /// Tool use block start
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: Value },
    /// Thinking block start
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

/// Event data for content_block_delta
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockDeltaEvent {
    /// Index of the content block being updated
    pub index: usize,
    /// The delta update
    pub delta: ContentDelta,
}

/// Delta types for content block updates
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentDelta {
    /// Text delta
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    /// JSON delta for tool input
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    /// Thinking delta
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    /// Signature delta (at end of thinking block)
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

/// Event data for content_block_stop
#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockStopEvent {
    /// Index of the content block that stopped
    pub index: usize,
}

/// Event data for message_delta
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaEvent {
    /// Delta changes to the message
    pub delta: MessageDeltaData,
    /// Cumulative usage
    pub usage: DeltaUsage,
}

/// Delta data in message_delta event
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaData {
    /// Stop reason
    pub stop_reason: Option<StopReason>,
    /// Stop sequence
    pub stop_sequence: Option<String>,
}

/// Usage in delta events (may only have output_tokens)
#[derive(Debug, Clone, Deserialize)]
pub struct DeltaUsage {
    /// Output tokens (cumulative)
    pub output_tokens: u32,
}

/// Error in stream
#[derive(Debug, Clone, Deserialize)]
pub struct StreamError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error details
    pub error: StreamErrorDetails,
}

/// Stream error details
#[derive(Debug, Clone, Deserialize)]
pub struct StreamErrorDetails {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message
    pub message: String,
}

/// Raw SSE event data structure for deserialization
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum RawStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: usize, content_block: ContentBlockStart },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: ContentDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaData, usage: DeltaUsage },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: StreamErrorDetails },
}

impl RawStreamEvent {
    /// Convert to StreamEvent
    pub fn into_stream_event(self) -> StreamEvent {
        match self {
            RawStreamEvent::MessageStart { message } => {
                StreamEvent::MessageStart(MessageStartEvent { message })
            }
            RawStreamEvent::ContentBlockStart { index, content_block } => {
                StreamEvent::ContentBlockStart(ContentBlockStartEvent { index, content_block })
            }
            RawStreamEvent::ContentBlockDelta { index, delta } => {
                StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent { index, delta })
            }
            RawStreamEvent::ContentBlockStop { index } => {
                StreamEvent::ContentBlockStop(ContentBlockStopEvent { index })
            }
            RawStreamEvent::MessageDelta { delta, usage } => {
                StreamEvent::MessageDelta(MessageDeltaEvent { delta, usage })
            }
            RawStreamEvent::MessageStop => StreamEvent::MessageStop,
            RawStreamEvent::Ping => StreamEvent::Ping,
            RawStreamEvent::Error { error } => {
                StreamEvent::Error(StreamError {
                    error_type: "error".to_string(),
                    error,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
    }

    #[test]
    fn test_content_block_serialization() {
        let block = ContentBlock::text("Hello");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_tool_result_serialization() {
        let block = ContentBlock::tool_result("toolu_123", "output", false);
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_result\""));
        assert!(json.contains("\"tool_use_id\":\"toolu_123\""));
    }

    #[test]
    fn test_stream_event_deserialization() {
        let json = r#"{"type": "ping"}"#;
        let event: RawStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RawStreamEvent::Ping));
    }

    #[test]
    fn test_text_delta_deserialization() {
        let json = r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#;
        let event: RawStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            RawStreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::TextDelta { text } => assert_eq!(text, "Hello"),
                    _ => panic!("Expected TextDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta"),
        }
    }
}
