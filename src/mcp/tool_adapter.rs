//! MCP Tool Adapter
//!
//! Adapts MCP tools to implement the framework's Tool trait

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::llm::{ToolDefinition, ToolInputSchema};
use crate::runtime::AgentInternals;
use crate::tools::{Tool, ToolInfo, ToolResult};

use super::server::MCPServer;

/// Adapter that wraps an MCP tool to implement the Tool trait
pub struct MCPToolAdapter {
    /// ID of the server this tool belongs to
    server_id: String,

    /// Reference to the MCP server
    server: Arc<MCPServer>,

    /// Original tool name (used when calling MCP server)
    tool_name: String,

    /// Exposed name with namespace (e.g., "filesystem:read_file")
    exposed_name: String,

    /// Tool definition converted to framework format
    tool_definition: ToolDefinition,
}

impl MCPToolAdapter {
    /// Create a new MCP tool adapter with namespacing
    pub fn new(
        server_id: String,
        server: Arc<MCPServer>,
        rmcp_tool: rmcp::model::Tool,
    ) -> Self {
        // Create namespaced name: "server_id__tool_name" (double underscore for clarity)
        let exposed_name = format!("{}__{}", server_id, rmcp_tool.name);

        // Convert rmcp tool definition to framework ToolDefinition
        let tool_definition = Self::convert_tool_definition(&exposed_name, &rmcp_tool);

        Self {
            server_id,
            server,
            tool_name: rmcp_tool.name.to_string(),
            exposed_name,
            tool_definition,
        }
    }

    /// Convert rmcp Tool definition to framework ToolDefinition
    fn convert_tool_definition(name: &str, rmcp_tool: &rmcp::model::Tool) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        // rmcp tool's input_schema is an Arc<JsonObject> (Map<String, Value>)
        // We need to convert it to our ToolInputSchema
        let schema_obj = rmcp_tool.input_schema.as_ref();

        let input_schema = ToolInputSchema {
            schema_type: schema_obj
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("object")
                .to_string(),
            properties: schema_obj.get("properties").cloned(),
            required: schema_obj
                .get("required")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                }),
        };

        ToolDefinition::Custom(CustomTool {
            tool_type: Some("custom".to_string()),
            name: name.to_string(),
            description: rmcp_tool.description.as_ref().map(|d| d.to_string()),
            input_schema,
            cache_control: None,
        })
    }

    /// Convert rmcp CallToolResult to framework ToolResult
    fn convert_mcp_result(&self, rmcp_result: rmcp::model::CallToolResult) -> Result<ToolResult> {
        use rmcp::model::RawContent;

        let is_error = rmcp_result.is_error.unwrap_or(false);

        // Aggregate all content
        let mut text_parts = Vec::new();

        for content in rmcp_result.content {
            // Extract the raw content from the annotated wrapper
            match &content.raw {
                RawContent::Text(text_content) => {
                    text_parts.push(text_content.text.clone());
                }
                RawContent::Image(image_content) => {
                    // Return image directly
                    use base64::Engine;
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(&image_content.data)
                        .map_err(|e| anyhow::anyhow!("Failed to decode base64 image: {}", e))?;
                    return Ok(ToolResult::image(decoded, image_content.mime_type.clone()));
                }
                RawContent::Resource(resource_content) => {
                    // Serialize resource as JSON
                    text_parts.push(serde_json::to_string_pretty(&resource_content.resource)?);
                }
                _ => {
                    // Handle other content types (Audio, ResourceLink) as JSON
                    text_parts.push(serde_json::to_string_pretty(&content)?);
                }
            }
        }

        let output = text_parts.join("\n\n");

        if is_error {
            Ok(ToolResult::error(output))
        } else {
            Ok(ToolResult::success(output))
        }
    }
}

#[async_trait]
impl Tool for MCPToolAdapter {
    fn name(&self) -> &str {
        &self.exposed_name
    }

    fn description(&self) -> &str {
        // Extract description from CustomTool
        match &self.tool_definition {
            ToolDefinition::Custom(custom) => {
                custom.description.as_deref().unwrap_or("MCP tool (no description)")
            }
            _ => "MCP tool (no description)",
        }
    }

    fn definition(&self) -> ToolDefinition {
        self.tool_definition.clone()
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        ToolInfo {
            name: self.exposed_name.clone(),
            action_description: format!(
                "Call MCP tool '{}' on server '{}'",
                self.tool_name, self.server_id
            ),
            details: Some(format!("Input: {}", input)),
        }
    }

    async fn execute(&self, input: &Value, _internals: &mut AgentInternals) -> Result<ToolResult> {
        tracing::info!(
            "[MCPToolAdapter] Executing '{}' on server '{}'",
            self.tool_name,
            self.server_id
        );
        tracing::debug!("[MCPToolAdapter] Input: {}", input);

        // Convert JSON Value to Map<String, Value> for rmcp
        let arguments = input.as_object().cloned();

        // Call the MCP server with the ORIGINAL tool name (not namespaced)
        let rmcp_result = self.server.call_tool(&self.tool_name, arguments).await?;

        // Convert rmcp result to framework ToolResult
        let result = self.convert_mcp_result(rmcp_result)?;

        tracing::debug!(
            "[MCPToolAdapter] Tool '{}' completed. Is error: {}",
            self.tool_name,
            result.is_error
        );

        Ok(result)
    }

    fn requires_permission(&self) -> bool {
        // All MCP tools require permission by default
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn test_tool_definition_conversion() {
        use rmcp::model::Tool as RmcpTool;

        let input_schema = Arc::new(serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Test input"
                }
            },
            "required": ["input"]
        })).unwrap());

        let rmcp_tool = RmcpTool {
            name: "test_tool".into(),
            title: None,
            description: Some("A test tool".into()),
            input_schema,
            output_schema: None,
            annotations: None,
            icons: None,
            meta: None,
        };

        let def = MCPToolAdapter::convert_tool_definition("server__test_tool", &rmcp_tool);

        // Check the name through pattern matching
        match &def {
            ToolDefinition::Custom(custom) => {
                assert_eq!(custom.name, "server__test_tool");
                assert_eq!(custom.description, Some("A test tool".to_string()));
            }
            _ => panic!("Expected CustomTool"),
        }
    }
}
