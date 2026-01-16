//! Edit tool for modifying files
//!
//! Performs exact string replacements in files.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Edit tool for string replacement in files
pub struct EditTool {
    /// Base directory for file operations
    base_dir: String,
}

/// Input for the edit tool
#[derive(Debug, Deserialize)]
struct EditInput {
    /// The absolute path to the file to modify (required)
    file_path: String,
    /// The text to replace (required)
    old_string: String,
    /// The text to replace it with (required)
    new_string: String,
    /// Replace all occurrences (default false)
    #[serde(default)]
    replace_all: bool,
}

impl EditTool {
    /// Create a new Edit tool with the current directory as base
    pub fn new() -> Result<Self> {
        let base_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self { base_dir })
    }

    /// Create a new Edit tool with a specific base directory
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

    /// Perform string replacement in a file
    fn str_replace(
        &self,
        file_path: &str,
        old_str: &str,
        new_str: &str,
        replace_all: bool,
    ) -> Result<String> {
        let resolved_path = self.resolve_path(file_path);
        tracing::info!("Editing file: {}", resolved_path);

        if old_str == new_str {
            anyhow::bail!("old_string and new_string must be different");
        }

        let content = fs::read_to_string(&resolved_path)
            .with_context(|| format!("Failed to read file: {}", resolved_path))?;

        let occurrences = content.matches(old_str).count();

        if occurrences == 0 {
            anyhow::bail!(
                "String not found in file. Make sure to include exact text including whitespace."
            );
        }

        if !replace_all && occurrences > 1 {
            anyhow::bail!(
                "Found {} occurrences of the string. Either provide a more specific string \
                to ensure only one match, or use replace_all: true to change every instance.",
                occurrences
            );
        }

        let new_content = if replace_all {
            content.replace(old_str, new_str)
        } else {
            content.replacen(old_str, new_str, 1)
        };

        fs::write(&resolved_path, &new_content)
            .with_context(|| format!("Failed to write file: {}", resolved_path))?;

        if replace_all {
            Ok(format!(
                "Successfully replaced {} occurrences in {}",
                occurrences, file_path
            ))
        } else {
            Ok(format!("Successfully replaced text in {}", file_path))
        }
    }
}

impl Default for EditTool {
    fn default() -> Self {
        Self::with_base_dir(".")
    }
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "Edit"
    }

    fn description(&self) -> &str {
        "Perform exact string replacements in files."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Edit".to_string(),
            description: Some(
                "Performs exact string replacements in files. \
                The edit will FAIL if old_string is not unique in the file unless replace_all is true. \
                Use replace_all for replacing and renaming strings across the file."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to modify"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with (must be different from old_string)"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": false,
                        "description": "Replace all occurrences of old_string (default false)"
                    }
                })),
                required: Some(vec![
                    "file_path".to_string(),
                    "old_string".to_string(),
                    "new_string".to_string(),
                ]),
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
            name: "Edit".to_string(),
            action_description: format!("Edit file: {}", file_path),
            details: None,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let edit_input: EditInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid edit input: {}", e))?;

        match self.str_replace(
            &edit_input.file_path,
            &edit_input.old_string,
            &edit_input.new_string,
            edit_input.replace_all,
        ) {
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
    async fn test_str_replace() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "Hello World").unwrap();

        let tool = EditTool::with_base_dir(dir.path().to_string_lossy().to_string());

        let input = json!({
            "file_path": "test.txt",
            "old_string": "World",
            "new_string": "Rust"
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello Rust");
    }
}
