//! Glob tool for file pattern matching
//!
//! Fast file pattern matching tool that works with any codebase size.

use anyhow::Result;
use async_trait::async_trait;
use glob::glob;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Glob tool for file pattern matching
pub struct GlobTool {
    /// Base directory for searches
    base_dir: String,
}

/// Input for the glob tool
#[derive(Debug, Deserialize)]
struct GlobInput {
    /// The glob pattern to match files against (required)
    pattern: String,
    /// The directory to search in (optional)
    path: Option<String>,
}

impl GlobTool {
    /// Create a new Glob tool with the current directory as base
    pub fn new() -> Result<Self> {
        let base_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self { base_dir })
    }

    /// Create a new Glob tool with a specific base directory
    pub fn with_base_dir(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Search for files matching the glob pattern
    fn search(&self, pattern: &str, search_dir: Option<&str>) -> Result<Vec<String>> {
        let base = search_dir.unwrap_or(&self.base_dir);

        let full_pattern = if Path::new(pattern).is_absolute() {
            pattern.to_string()
        } else {
            format!("{}/{}", base, pattern)
        };

        tracing::info!("Searching with glob pattern: {}", full_pattern);

        let mut entries: Vec<(String, std::time::SystemTime)> = glob(&full_pattern)?
            .filter_map(|entry| entry.ok())
            .filter_map(|path| {
                let mtime = path.metadata().ok()?.modified().ok()?;
                let display_path = path
                    .strip_prefix(&self.base_dir)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| path.to_string_lossy().to_string());
                Some((display_path, mtime))
            })
            .collect();

        // Sort by modification time (most recent first)
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(entries.into_iter().map(|(path, _)| path).collect())
    }
}

impl Default for GlobTool {
    fn default() -> Self {
        Self::with_base_dir(".")
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Fast file pattern matching tool. Supports glob patterns like **/*.js or src/**/*.ts."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Glob".to_string(),
            description: Some(
                "Fast file pattern matching tool that works with any codebase size. \
                Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\". \
                Returns matching file paths sorted by modification time."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files against"
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in. If not specified, the current working directory will be used."
                    }
                })),
                required: Some(vec!["pattern".to_string()]),
            },
            tool_type: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let pattern = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("*");

        ToolInfo {
            name: "Glob".to_string(),
            action_description: format!("Search files: {}", pattern),
            details: None,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let glob_input: GlobInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid glob input: {}", e))?;

        match self.search(&glob_input.pattern, glob_input.path.as_deref()) {
            Ok(entries) => {
                if entries.is_empty() {
                    Ok(ToolResult::success(format!(
                        "No files found matching pattern: {}",
                        glob_input.pattern
                    )))
                } else {
                    let mut result = format!(
                        "Found {} files matching '{}':\n",
                        entries.len(),
                        glob_input.pattern
                    );
                    for entry in entries.iter().take(100) {
                        result.push_str(&format!("{}\n", entry));
                    }
                    if entries.len() > 100 {
                        result.push_str(&format!("... and {} more\n", entries.len() - 100));
                    }
                    Ok(ToolResult::success(result))
                }
            }
            Err(e) => Ok(ToolResult::error(format!("Glob search failed: {}", e))),
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
    async fn test_glob_search() {
        let tool = GlobTool::with_base_dir(".");
        let input = json!({ "pattern": "*.toml" });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
    }
}
