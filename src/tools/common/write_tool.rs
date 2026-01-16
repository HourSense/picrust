//! Write tool for creating/writing files
//!
//! Writes content to files on the local filesystem.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Write tool for creating files
pub struct WriteTool {
    /// Base directory for file operations
    base_dir: String,
}

/// Input for the write tool
#[derive(Debug, Deserialize)]
struct WriteInput {
    /// The absolute path to the file to write (required)
    file_path: String,
    /// The content to write to the file (required)
    content: String,
}

impl WriteTool {
    /// Create a new Write tool with the current directory as base
    pub fn new() -> Result<Self> {
        let base_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self { base_dir })
    }

    /// Create a new Write tool with a specific base directory
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

    /// Write content to a file
    fn write_file(&self, file_path: &str, content: &str) -> Result<String> {
        let resolved_path = self.resolve_path(file_path);
        tracing::info!("Writing file: {}", resolved_path);

        // Create parent directories if needed
        if let Some(parent) = Path::new(&resolved_path).parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        let existed = Path::new(&resolved_path).exists();

        fs::write(&resolved_path, content)
            .with_context(|| format!("Failed to write file: {}", resolved_path))?;

        if existed {
            Ok(format!("File updated successfully: {}", file_path))
        } else {
            Ok(format!("File created successfully: {}", file_path))
        }
    }
}

impl Default for WriteTool {
    fn default() -> Self {
        Self::with_base_dir(".")
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "Write"
    }

    fn description(&self) -> &str {
        "Write content to a file on the local filesystem."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Write".to_string(),
            description: Some(
                "Writes a file to the local filesystem. \
                This will overwrite the existing file if there is one. \
                ALWAYS prefer editing existing files. NEVER write new files unless explicitly required."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write (must be absolute, not relative)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                })),
                required: Some(vec!["file_path".to_string(), "content".to_string()]),
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
            name: "Write".to_string(),
            action_description: format!("Write file: {}", file_path),
            details: None,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let write_input: WriteInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid write input: {}", e))?;

        match self.write_file(&write_input.file_path, &write_input.content) {
            Ok(output) => Ok(ToolResult::success(output)),
            Err(e) => Ok(ToolResult::error(format!("{}", e))),
        }
    }

    fn requires_permission(&self) -> bool {
        true // Modifies files
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_write_file() {
        let dir = tempdir().unwrap();
        let tool = WriteTool::with_base_dir(dir.path().to_string_lossy().to_string());

        let input = json!({
            "file_path": "test.txt",
            "content": "Hello World"
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("created"));

        let content = fs::read_to_string(dir.path().join("test.txt")).unwrap();
        assert_eq!(content, "Hello World");
    }
}
