//! Tool trait definition
//!
//! All tools implement this trait to provide a consistent interface.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::llm::ToolDefinition;

/// Result of executing a tool
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The output of the tool
    pub output: String,
    /// Whether the tool execution resulted in an error
    pub is_error: bool,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
        }
    }

    /// Create an error tool result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            output: message.into(),
            is_error: true,
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
    async fn execute(&self, input: &Value) -> Result<ToolResult>;

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
        assert_eq!(result.output, "output");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("error message");
        assert_eq!(result.output, "error message");
        assert!(result.is_error);
    }
}
