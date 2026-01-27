//! Present File tool for displaying files to the user
//!
//! This tool presents a file to the user with an option to open it.
//! Use this after creating or modifying a file that the user should access.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};
use crate::runtime::AgentInternals;

/// Present File tool for displaying files to the user
pub struct PresentFileTool;

/// Input for the present file tool
#[derive(Debug, Deserialize)]
struct PresentFileInput {
    /// Absolute path to the file (required)
    file_path: String,
    /// Display name for the file (required)
    file_name: String,
    /// Brief description of what the file contains (optional)
    description: Option<String>,
}

impl PresentFileTool {
    /// Create a new Present File tool
    pub fn new() -> Self {
        Self
    }
}

impl Default for PresentFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for PresentFileTool {
    fn name(&self) -> &str {
        "PresentFile"
    }

    fn description(&self) -> &str {
        "Present a file to the user with an option to open it."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "PresentFile".to_string(),
            description: Some(
                "Present a file to the user with an option to open it. \
                Use this after creating or modifying a file that the user should access."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Display name for the file"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what the file contains"
                    }
                })),
                required: Some(vec!["file_path".to_string(), "file_name".to_string()]),
            },
            tool_type: None,
            cache_control: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let file_name = input
            .get("file_name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");

        ToolInfo {
            name: "PresentFile".to_string(),
            action_description: format!("Present file: {}", file_name),
            details: None,
        }
    }

    async fn execute(&self, input: &Value, _internals: &mut AgentInternals) -> Result<ToolResult> {
        let present_input: PresentFileInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid present file input: {}", e))?;

        let message = format!(
            "File ready: {} at {}",
            present_input.file_name, present_input.file_path
        );

        tracing::info!("{}", message);
        if let Some(desc) = &present_input.description {
            tracing::info!("Description: {}", desc);
        }

        Ok(ToolResult::success(message))
    }

    fn requires_permission(&self) -> bool {
        false // Read-only presentation doesn't need permission
    }
}
