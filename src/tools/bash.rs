//! Bash tool for executing shell commands
//!
//! This tool executes bash commands without session persistence.
//! Each command runs in a fresh shell starting from the working directory.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::process::Command;

use super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Bash tool for executing shell commands
pub struct BashTool {
    /// Working directory for command execution
    working_dir: String,
    /// Maximum output length in characters
    max_output_length: usize,
}

/// Input for the bash tool
#[derive(Debug, Deserialize)]
struct BashInput {
    /// The command to execute
    command: String,
    /// Optional working directory override
    working_dir: Option<String>,
}

impl BashTool {
    /// Create a new Bash tool with the current directory as working directory
    pub fn new() -> Result<Self> {
        let working_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self {
            working_dir,
            max_output_length: 50000,
        })
    }

    /// Create a new Bash tool with a specific working directory
    pub fn with_working_dir(working_dir: impl Into<String>) -> Self {
        Self {
            working_dir: working_dir.into(),
            max_output_length: 50000,
        }
    }

    /// Set the maximum output length
    pub fn with_max_output_length(mut self, max_length: usize) -> Self {
        self.max_output_length = max_length;
        self
    }

    /// Execute a bash command and return the output
    async fn run_command(&self, command: &str, working_dir: &str) -> Result<(String, i32)> {
        tracing::info!("Executing bash command: {}", command);
        tracing::debug!("Working directory: {}", working_dir);

        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

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
                result.push_str("\n");
            }
            result.push_str("STDERR:\n");
            result.push_str(&stderr);
        }

        // Truncate if too long
        if result.len() > self.max_output_length {
            result.truncate(self.max_output_length);
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
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the shell. Each command runs in a fresh shell without session persistence."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "bash".to_string(),
            description: Some(self.description().to_string()),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory for the command. Defaults to the project root."
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

        let working_dir = input
            .get("working_dir")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| self.working_dir.clone());

        ToolInfo {
            name: "bash".to_string(),
            action_description: format!("Execute command: {}", command),
            details: Some(format!("Working directory: {}", working_dir)),
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let bash_input: BashInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid bash input: {}", e))?;

        let working_dir = bash_input
            .working_dir
            .as_deref()
            .unwrap_or(&self.working_dir);

        match self.run_command(&bash_input.command, working_dir).await {
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
        let input = json!({ "command": "echo 'hello world'" });
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
}
