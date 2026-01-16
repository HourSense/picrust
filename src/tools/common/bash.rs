//! Bash tool for executing shell commands
//!
//! This tool executes bash commands with optional timeout and description.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Default timeout in milliseconds (2 minutes)
const DEFAULT_TIMEOUT_MS: u64 = 120000;
/// Maximum timeout in milliseconds (10 minutes)
const MAX_TIMEOUT_MS: u64 = 600000;
/// Maximum output length in characters
const MAX_OUTPUT_LENGTH: usize = 30000;

/// Bash tool for executing shell commands
pub struct BashTool {
    /// Working directory for command execution
    working_dir: String,
}

/// Input for the bash tool
#[derive(Debug, Deserialize)]
struct BashInput {
    /// The command to execute (required)
    command: String,
    /// Optional timeout in milliseconds (max 600000)
    timeout: Option<u64>,
    /// Optional description of what this command does
    description: Option<String>,
}

impl BashTool {
    /// Create a new Bash tool with the current directory as working directory
    pub fn new() -> Result<Self> {
        let working_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self { working_dir })
    }

    /// Create a new Bash tool with a specific working directory
    pub fn with_working_dir(working_dir: impl Into<String>) -> Self {
        Self {
            working_dir: working_dir.into(),
        }
    }

    /// Execute a bash command with optional timeout
    async fn run_command(&self, command: &str, timeout_ms: u64) -> Result<(String, i32)> {
        tracing::info!("Executing bash command: {}", command);
        tracing::debug!("Working directory: {}", self.working_dir);
        tracing::debug!("Timeout: {}ms", timeout_ms);

        let duration = Duration::from_millis(timeout_ms.min(MAX_TIMEOUT_MS));

        let output_future = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&self.working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        let output = match timeout(duration, output_future).await {
            Ok(result) => result?,
            Err(_) => {
                return Ok((
                    format!("Command timed out after {}ms", timeout_ms),
                    -1,
                ));
            }
        };

        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Combine stdout and stderr
        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("STDERR:\n");
            result.push_str(&stderr);
        }

        // Truncate if too long
        if result.len() > MAX_OUTPUT_LENGTH {
            result.truncate(MAX_OUTPUT_LENGTH);
            result.push_str("\n... (output truncated)");
        }

        tracing::debug!("Command exit code: {}", exit_code);
        tracing::debug!("Output length: {} chars", result.len());

        Ok((result, exit_code))
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::with_working_dir(".")
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the shell. Use for terminal operations like git, npm, docker, etc."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Bash".to_string(),
            description: Some(
                "Executes a given bash command in a persistent shell session with optional timeout. \
                Use this for terminal operations like git, npm, docker, etc. \
                DO NOT use it for file operations (reading, writing, editing) - use the specialized tools instead."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in milliseconds (max 600000). Default is 120000ms (2 minutes)."
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear, concise description of what this command does in 5-10 words, in active voice."
                    }
                })),
                required: Some(vec!["command".to_string()]),
            },
            tool_type: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown command>");

        let description = input
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from);

        let action = description.unwrap_or_else(|| format!("Execute: {}", command));

        ToolInfo {
            name: "Bash".to_string(),
            action_description: action,
            details: Some(format!("Command: {}", command)),
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let bash_input: BashInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid bash input: {}", e))?;

        let timeout_ms = bash_input.timeout.unwrap_or(DEFAULT_TIMEOUT_MS);

        if let Some(ref desc) = bash_input.description {
            tracing::info!("Command description: {}", desc);
        }

        match self.run_command(&bash_input.command, timeout_ms).await {
            Ok((output, exit_code)) => {
                if exit_code == 0 {
                    if output.is_empty() {
                        Ok(ToolResult::success("Command completed successfully (no output)"))
                    } else {
                        Ok(ToolResult::success(output))
                    }
                } else {
                    Ok(ToolResult::error(format!(
                        "Command failed with exit code {}\n{}",
                        exit_code, output
                    )))
                }
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to execute command: {}", e))),
        }
    }

    fn requires_permission(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_command() {
        let tool = BashTool::with_working_dir(".");
        let input = json!({
            "command": "echo 'hello world'",
            "description": "Print hello world"
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("hello world"));
    }

    #[tokio::test]
    async fn test_failing_command() {
        let tool = BashTool::with_working_dir(".");
        let input = json!({ "command": "exit 1" });
        let result = tool.execute(&input).await.unwrap();
        assert!(result.is_error);
        assert!(result.output.contains("exit code 1"));
    }

    #[tokio::test]
    async fn test_timeout() {
        let tool = BashTool::with_working_dir(".");
        let input = json!({
            "command": "sleep 5",
            "timeout": 100
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(result.is_error);
        assert!(result.output.contains("timed out"));
    }
}
