//! SummarizeFileTool - A tool that spawns FileSummaryAgent to summarize files
//!
//! This demonstrates how to create a tool that spawns a subagent.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use shadow_agent_sdk::{
    core::OutputChunk,
    llm::{AnthropicProvider, ToolDefinition},
    llm::types::CustomTool,
    runtime::{AgentInternals, AgentRuntime},
    tools::{Tool, ToolInfo, ToolResult},
};

use crate::file_summary_agent::{spawn_file_summary_agent, FileSummaryAgentConfig};

/// Tool that spawns a FileSummaryAgent to summarize a file
pub struct SummarizeFileTool {
    /// Shared LLM provider
    llm: Arc<AnthropicProvider>,
    /// Shared runtime for spawning subagents
    runtime: AgentRuntime,
}

impl SummarizeFileTool {
    pub fn new(llm: Arc<AnthropicProvider>, runtime: AgentRuntime) -> Self {
        Self { llm, runtime }
    }
}

#[async_trait]
impl Tool for SummarizeFileTool {
    fn name(&self) -> &str {
        "SummarizeFile"
    }

    fn description(&self) -> &str {
        "Spawns a subagent to read and summarize a file. Use this when you need a concise summary of a file's contents. The subagent will read the file and return a ~100 word summary."
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::Custom(CustomTool {
            name: self.name().to_string(),
            description: Some(self.description().to_string()),
            input_schema: shadow_agent_sdk::llm::types::ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to summarize"
                    }
                })),
                required: Some(vec!["file_path".to_string()]),
            },
            tool_type: None,
        })
    }

    fn get_info(&self, input: &serde_json::Value) -> ToolInfo {
        let file_path = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown>");

        ToolInfo {
            name: self.name().to_string(),
            action_description: format!("Summarize file: {}", file_path),
            details: Some(format!("Spawns FileSummaryAgent to read and summarize {}", file_path)),
        }
    }

    fn requires_permission(&self) -> bool {
        // Require permission since this spawns a subagent
        true
    }

    async fn execute(&self, input: &serde_json::Value, internals: &mut AgentInternals) -> Result<ToolResult> {
        // Extract file_path from input
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolResult::error("Missing required parameter: file_path")),
        };

        // Get parent session info for subagent linking
        let parent_session_id = internals.session.session_id().to_string();
        let tool_use_id = internals
            .context
            .current_tool_use_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        // Notify that we're spawning a subagent
        internals.send(OutputChunk::SubAgentSpawned {
            session_id: format!("file-summary-{}", &tool_use_id),
            agent_type: "file-summary-agent".to_string(),
        });

        // Create config for spawning
        let config = FileSummaryAgentConfig::new(self.llm.clone(), self.runtime.clone());

        // Spawn the subagent
        let handle = match spawn_file_summary_agent(
            &config,
            &parent_session_id,
            &tool_use_id,
            file_path,
        )
        .await
        {
            Ok(h) => h,
            Err(e) => return Ok(ToolResult::error(format!("Failed to spawn subagent: {}", e))),
        };

        // Subscribe to subagent output
        let mut rx = handle.subscribe();

        // Collect the summary from subagent output
        let mut summary = String::new();

        // Wait for subagent to complete
        loop {
            match rx.recv().await {
                Ok(chunk) => {
                    match &chunk {
                        OutputChunk::TextDelta(text) => {
                            summary.push_str(text);
                        }
                        OutputChunk::TextComplete(text) => {
                            summary = text.clone();
                        }
                        OutputChunk::Done => {
                            break;
                        }
                        OutputChunk::Error(message) => {
                            return Ok(ToolResult::error(format!("Subagent error: {}", message)));
                        }
                        _ => {}
                    }

                    // Forward subagent output to parent
                    internals.send(OutputChunk::SubAgentOutput {
                        session_id: handle.session_id().to_string(),
                        chunk: Box::new(chunk),
                    });
                }
                Err(e) => {
                    // Channel closed or lagged
                    if summary.is_empty() {
                        return Ok(ToolResult::error(format!("Subagent channel error: {}", e)));
                    }
                    break;
                }
            }
        }

        // Notify completion
        internals.send(OutputChunk::SubAgentComplete {
            session_id: handle.session_id().to_string(),
            result: Some(summary.clone()),
        });

        // Return the summary as the tool result
        if summary.is_empty() {
            Ok(ToolResult::error("Subagent returned empty summary"))
        } else {
            Ok(ToolResult::success(summary))
        }
    }
}
