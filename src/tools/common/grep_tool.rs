//! Grep tool for content search using ripgrep
//!
//! A powerful search tool built on ripgrep for fast regex searching.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::process::Command;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Grep tool for content search
pub struct GrepTool {
    /// Base directory for searches
    base_dir: String,
}

/// Output mode for grep results
#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OutputMode {
    Content,
    FilesWithMatches,
    Count,
}

impl Default for OutputMode {
    fn default() -> Self {
        OutputMode::FilesWithMatches
    }
}

/// Input for the grep tool
#[derive(Debug, Deserialize)]
struct GrepInput {
    /// The regex pattern to search for (required)
    pattern: String,
    /// File or directory to search in
    path: Option<String>,
    /// Glob pattern to filter files
    glob: Option<String>,
    /// Output mode
    output_mode: Option<OutputMode>,
    /// Lines before match
    #[serde(rename = "-B")]
    before_context: Option<u32>,
    /// Lines after match
    #[serde(rename = "-A")]
    after_context: Option<u32>,
    /// Lines around match
    #[serde(rename = "-C")]
    context: Option<u32>,
    /// Show line numbers
    #[serde(rename = "-n")]
    line_numbers: Option<bool>,
    /// Case insensitive
    #[serde(rename = "-i")]
    case_insensitive: Option<bool>,
    /// File type filter
    #[serde(rename = "type")]
    file_type: Option<String>,
    /// Limit output lines
    head_limit: Option<usize>,
    /// Skip first N entries
    offset: Option<usize>,
    /// Multiline mode
    multiline: Option<bool>,
}

impl GrepTool {
    /// Create a new Grep tool with the current directory as base
    pub fn new() -> Result<Self> {
        let base_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self { base_dir })
    }

    /// Create a new Grep tool with a specific base directory
    pub fn with_base_dir(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Execute ripgrep with the given options
    async fn search(&self, input: &GrepInput) -> Result<String> {
        let search_path = input.path.as_deref().unwrap_or(&self.base_dir);
        let output_mode = input.output_mode.unwrap_or_default();

        let mut cmd = Command::new("rg");

        // Add pattern
        cmd.arg(&input.pattern);

        // Add path
        cmd.arg(search_path);

        // Output mode
        match output_mode {
            OutputMode::FilesWithMatches => {
                cmd.arg("-l");
            }
            OutputMode::Count => {
                cmd.arg("-c");
            }
            OutputMode::Content => {
                // Default rg behavior, show matching lines
                if input.line_numbers.unwrap_or(true) {
                    cmd.arg("-n");
                }
            }
        }

        // Context options (only for content mode)
        if output_mode == OutputMode::Content {
            if let Some(b) = input.before_context {
                cmd.arg("-B").arg(b.to_string());
            }
            if let Some(a) = input.after_context {
                cmd.arg("-A").arg(a.to_string());
            }
            if let Some(c) = input.context {
                cmd.arg("-C").arg(c.to_string());
            }
        }

        // Case insensitive
        if input.case_insensitive.unwrap_or(false) {
            cmd.arg("-i");
        }

        // File type filter
        if let Some(ref ft) = input.file_type {
            cmd.arg("--type").arg(ft);
        }

        // Glob filter
        if let Some(ref g) = input.glob {
            cmd.arg("--glob").arg(g);
        }

        // Multiline mode
        if input.multiline.unwrap_or(false) {
            cmd.arg("-U").arg("--multiline-dotall");
        }

        // Don't use colors in output
        cmd.arg("--color=never");

        tracing::info!("Running ripgrep: {:?}", cmd);

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !output.status.success() && stdout.is_empty() {
            if stderr.contains("No such file or directory") {
                return Ok(format!("Path not found: {}", search_path));
            }
            if output.status.code() == Some(1) {
                // No matches found
                return Ok(format!("No matches found for pattern: {}", input.pattern));
            }
            return Ok(format!("Search error: {}", stderr));
        }

        let mut lines: Vec<&str> = stdout.lines().collect();

        // Apply offset
        if let Some(offset) = input.offset {
            if offset < lines.len() {
                lines = lines[offset..].to_vec();
            } else {
                lines.clear();
            }
        }

        // Apply head limit
        if let Some(limit) = input.head_limit {
            lines.truncate(limit);
        }

        Ok(lines.join("\n"))
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::with_base_dir(".")
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex patterns. Uses ripgrep for fast searching."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Grep".to_string(),
            description: Some(
                "A powerful search tool built on ripgrep. \
                Supports full regex syntax. \
                Output modes: 'content' shows matching lines, 'files_with_matches' shows only file paths (default), 'count' shows match counts."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "pattern": {
                        "type": "string",
                        "description": "The regular expression pattern to search for in file contents"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in. Defaults to current working directory."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\")"
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode: 'content', 'files_with_matches' (default), or 'count'"
                    },
                    "-B": {
                        "type": "number",
                        "description": "Number of lines to show before each match"
                    },
                    "-A": {
                        "type": "number",
                        "description": "Number of lines to show after each match"
                    },
                    "-C": {
                        "type": "number",
                        "description": "Number of lines to show before and after each match"
                    },
                    "-n": {
                        "type": "boolean",
                        "description": "Show line numbers in output. Defaults to true."
                    },
                    "-i": {
                        "type": "boolean",
                        "description": "Case insensitive search"
                    },
                    "type": {
                        "type": "string",
                        "description": "File type to search (e.g. 'js', 'py', 'rust')"
                    },
                    "head_limit": {
                        "type": "number",
                        "description": "Limit output to first N lines/entries"
                    },
                    "offset": {
                        "type": "number",
                        "description": "Skip first N lines/entries"
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Enable multiline mode where . matches newlines"
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
            .unwrap_or("?");

        ToolInfo {
            name: "Grep".to_string(),
            action_description: format!("Search for: {}", pattern),
            details: None,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let grep_input: GrepInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid grep input: {}", e))?;

        match self.search(&grep_input).await {
            Ok(output) => {
                if output.is_empty() {
                    Ok(ToolResult::success(format!(
                        "No matches found for pattern: {}",
                        grep_input.pattern
                    )))
                } else {
                    Ok(ToolResult::success(output))
                }
            }
            Err(e) => Ok(ToolResult::error(format!("Search failed: {}", e))),
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
    async fn test_grep_search() {
        let tool = GrepTool::with_base_dir(".");
        let input = json!({
            "pattern": "fn main",
            "glob": "*.rs"
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
    }
}
