//! Tool trait definition
//!
//! All tools implement this trait to provide a consistent interface.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::ToolDefinition;
use crate::runtime::AgentInternals;

/// Content type for tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResultData {
    /// Text content
    Text(String),
    /// Image content (raw bytes and media type)
    Image { data: Vec<u8>, media_type: String },
    /// Document content (raw bytes, media type, and description)
    Document {
        data: Vec<u8>,
        media_type: String,
        description: String,
    },
}

/// Result of executing a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The content of the tool result
    pub content: ToolResultData,
    /// Whether the tool execution resulted in an error
    pub is_error: bool,
}

impl ToolResult {
    /// Create a successful tool result with text content
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            content: ToolResultData::Text(output.into()),
            is_error: false,
        }
    }

    /// Create an error tool result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: ToolResultData::Text(message.into()),
            is_error: true,
        }
    }

    /// Create a successful image result
    pub fn image(data: Vec<u8>, media_type: impl Into<String>) -> Self {
        Self {
            content: ToolResultData::Image {
                data,
                media_type: media_type.into(),
            },
            is_error: false,
        }
    }

    /// Create a successful document result
    pub fn document(
        data: Vec<u8>,
        media_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            content: ToolResultData::Document {
                data,
                media_type: media_type.into(),
                description: description.into(),
            },
            is_error: false,
        }
    }
}

/// Information about a tool for permission prompts
#[derive(Debug, Clone)]
pub struct ToolInfo {
    /// Name of the tool
    pub name: String,
    /// Human-readable description of what this invocation will do
    pub action_description: String,
    /// Additional details about the action (e.g., command to run, file to edit)
    pub details: Option<String>,
}

/// Trait for tools that the agent can use
///
/// All tools must implement this trait to be usable by the agent.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the name of this tool
    fn name(&self) -> &str;

    /// Get a description of this tool
    fn description(&self) -> &str;

    /// Get the tool definition for the Anthropic API
    fn definition(&self) -> ToolDefinition;

    /// Get information about what this tool invocation will do
    ///
    /// This is used to display permission prompts to the user.
    fn get_info(&self, input: &Value) -> ToolInfo;

    /// Execute the tool with the given input
    ///
    /// The input is a JSON value that matches the tool's input schema.
    /// The internals provide access to agent context, output channel, etc.
    async fn execute(&self, input: &Value, internals: &mut AgentInternals) -> Result<ToolResult>;

    /// Check if this tool requires permission before execution
    ///
    /// Default is true - tools should generally require permission.
    fn requires_permission(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("output");
        match result.content {
            ToolResultData::Text(text) => assert_eq!(text, "output"),
            _ => panic!("Expected text content"),
        }
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("error message");
        match result.content {
            ToolResultData::Text(text) => assert_eq!(text, "error message"),
            _ => panic!("Expected text content"),
        }
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_result_image() {
        let data = vec![1, 2, 3, 4];
        let result = ToolResult::image(data.clone(), "image/png");
        match result.content {
            ToolResultData::Image {
                data: img_data,
                media_type,
            } => {
                assert_eq!(img_data, data);
                assert_eq!(media_type, "image/png");
            }
            _ => panic!("Expected image content"),
        }
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_document() {
        let data = vec![1, 2, 3, 4];
        let result = ToolResult::document(data.clone(), "application/pdf", "Test PDF");
        match result.content {
            ToolResultData::Document {
                data: doc_data,
                media_type,
                description,
            } => {
                assert_eq!(doc_data, data);
                assert_eq!(media_type, "application/pdf");
                assert_eq!(description, "Test PDF");
            }
            _ => panic!("Expected document content"),
        }
        assert!(!result.is_error);
    }
}
