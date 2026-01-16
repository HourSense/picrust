//! Read tool for reading files
//!
//! Reads files from the local filesystem with line numbers.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Maximum lines to read by default
const DEFAULT_LINE_LIMIT: usize = 2000;
/// Maximum characters per line before truncation
const MAX_LINE_LENGTH: usize = 2000;

/// Read tool for reading files
pub struct ReadTool {
    /// Base directory for file operations
    base_dir: String,
}

/// Input for the read tool
#[derive(Debug, Deserialize)]
struct ReadInput {
    /// The absolute path to the file to read (required)
    file_path: String,
    /// The line number to start reading from (1-indexed)
    offset: Option<usize>,
    /// The number of lines to read
    limit: Option<usize>,
}

impl ReadTool {
    /// Create a new Read tool with the current directory as base
    pub fn new() -> Result<Self> {
        let base_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self { base_dir })
    }

    /// Create a new Read tool with a specific base directory
    pub fn with_base_dir(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Resolve a path (handle both absolute and relative)
    fn resolve_path(&self, path: &str) -> String {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_string_lossy().to_string()
        } else {
            Path::new(&self.base_dir)
                .join(path)
                .to_string_lossy()
                .to_string()
        }
    }

    /// Read file contents with optional offset and limit
    fn read_file(&self, file_path: &str, offset: Option<usize>, limit: Option<usize>) -> Result<String> {
        let resolved_path = self.resolve_path(file_path);
        tracing::info!("Reading file: {}", resolved_path);

        let content = fs::read_to_string(&resolved_path)
            .with_context(|| format!("Failed to read file: {}", resolved_path))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = offset.unwrap_or(1).saturating_sub(1);
        let count = limit.unwrap_or(DEFAULT_LINE_LIMIT);
        let end = (start + count).min(total_lines);

        if start >= total_lines {
            return Ok(format!(
                "File has {} lines. Requested offset {} is out of range.",
                total_lines,
                start + 1
            ));
        }

        let mut result = String::new();

        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            let display_line = if line.len() > MAX_LINE_LENGTH {
                format!("{}...", &line[..MAX_LINE_LENGTH])
            } else {
                line.to_string()
            };
            // Use cat -n format: right-aligned line number + tab + content
            result.push_str(&format!("{:>6}\t{}\n", line_num, display_line));
        }

        if end < total_lines {
            result.push_str(&format!(
                "\n... ({} more lines, use offset and limit to read more)\n",
                total_lines - end
            ));
        }

        Ok(result)
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::with_base_dir(".")
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "Read"
    }

    fn description(&self) -> &str {
        "Read a file from the local filesystem. Returns content with line numbers."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Read".to_string(),
            description: Some(
                "Reads a file from the local filesystem. \
                The file_path parameter must be an absolute path. \
                By default, reads up to 2000 lines. \
                You can optionally specify offset and limit for long files. \
                Results are returned with line numbers starting at 1."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    },
                    "offset": {
                        "type": "number",
                        "description": "The line number to start reading from (1-indexed). Only provide if the file is too large."
                    },
                    "limit": {
                        "type": "number",
                        "description": "The number of lines to read. Only provide if the file is too large."
                    }
                })),
                required: Some(vec!["file_path".to_string()]),
            },
            tool_type: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let file_path = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("?");

        ToolInfo {
            name: "Read".to_string(),
            action_description: format!("Read file: {}", file_path),
            details: None,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let read_input: ReadInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid read input: {}", e))?;

        match self.read_file(&read_input.file_path, read_input.offset, read_input.limit) {
            Ok(output) => Ok(ToolResult::success(output)),
            Err(e) => Ok(ToolResult::error(format!("{}", e))),
        }
    }

    fn requires_permission(&self) -> bool {
        false // Read-only operation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_file() {
        let tool = ReadTool::with_base_dir(".");
        let input = json!({ "file_path": "Cargo.toml" });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("[package]") || result.output.contains("name"));
    }
}
