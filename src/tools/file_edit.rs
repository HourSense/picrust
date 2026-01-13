//! File editing tool with glob patterns and str_replace functionality
//!
//! This tool provides file operations including:
//! - View files (with optional line range)
//! - Create new files
//! - Replace text in files (str_replace)
//! - Search files using glob patterns

use anyhow::{Context, Result};
use async_trait::async_trait;
use glob::glob;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

use super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// File edit tool for viewing and modifying files
pub struct FileEditTool {
    /// Base directory for file operations
    base_dir: String,
    /// Maximum file size to read (in bytes)
    max_file_size: usize,
}

/// Input commands for the file edit tool
#[derive(Debug, Deserialize)]
#[serde(tag = "command")]
enum FileCommand {
    /// View a file's contents
    #[serde(rename = "view")]
    View {
        path: String,
        #[serde(default)]
        start_line: Option<usize>,
        #[serde(default)]
        end_line: Option<usize>,
    },

    /// Create a new file
    #[serde(rename = "create")]
    Create {
        path: String,
        content: String,
    },

    /// Replace text in a file
    #[serde(rename = "str_replace")]
    StrReplace {
        path: String,
        old_str: String,
        new_str: String,
    },

    /// Search for files using glob pattern
    #[serde(rename = "glob")]
    Glob {
        pattern: String,
    },

    /// Insert text at a specific line
    #[serde(rename = "insert")]
    Insert {
        path: String,
        line: usize,
        content: String,
    },
}

impl FileEditTool {
    /// Create a new FileEditTool with the current directory as base
    pub fn new() -> Result<Self> {
        let base_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();

        Ok(Self {
            base_dir,
            max_file_size: 1_000_000, // 1MB default
        })
    }

    /// Create a new FileEditTool with a specific base directory
    pub fn with_base_dir(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            max_file_size: 1_000_000,
        }
    }

    /// Set the maximum file size
    pub fn with_max_file_size(mut self, max_size: usize) -> Self {
        self.max_file_size = max_size;
        self
    }

    /// Resolve a path relative to the base directory
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

    /// View file contents
    fn view_file(&self, path: &str, start_line: Option<usize>, end_line: Option<usize>) -> Result<String> {
        let resolved_path = self.resolve_path(path);
        tracing::info!("Viewing file: {}", resolved_path);

        let metadata = fs::metadata(&resolved_path)
            .with_context(|| format!("Failed to access file: {}", resolved_path))?;

        if metadata.len() as usize > self.max_file_size {
            anyhow::bail!(
                "File too large ({} bytes). Maximum allowed: {} bytes",
                metadata.len(),
                self.max_file_size
            );
        }

        let content = fs::read_to_string(&resolved_path)
            .with_context(|| format!("Failed to read file: {}", resolved_path))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = start_line.unwrap_or(1).saturating_sub(1);
        let end = end_line.unwrap_or(total_lines).min(total_lines);

        if start >= total_lines {
            return Ok(format!("File has {} lines. Requested start line {} is out of range.", total_lines, start + 1));
        }

        let mut result = String::new();
        result.push_str(&format!("File: {} ({} lines total)\n", path, total_lines));
        result.push_str(&format!("Showing lines {}-{}:\n\n", start + 1, end));

        for (i, line) in lines[start..end].iter().enumerate() {
            result.push_str(&format!("{:>4} | {}\n", start + i + 1, line));
        }

        Ok(result)
    }

    /// Create a new file
    fn create_file(&self, path: &str, content: &str) -> Result<String> {
        let resolved_path = self.resolve_path(path);
        tracing::info!("Creating file: {}", resolved_path);

        // Create parent directories if needed
        if let Some(parent) = Path::new(&resolved_path).parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        fs::write(&resolved_path, content)
            .with_context(|| format!("Failed to write file: {}", resolved_path))?;

        Ok(format!("File created successfully: {}", path))
    }

    /// Replace text in a file
    fn str_replace(&self, path: &str, old_str: &str, new_str: &str) -> Result<String> {
        let resolved_path = self.resolve_path(path);
        tracing::info!("Replacing text in file: {}", resolved_path);

        let content = fs::read_to_string(&resolved_path)
            .with_context(|| format!("Failed to read file: {}", resolved_path))?;

        let occurrences = content.matches(old_str).count();

        if occurrences == 0 {
            anyhow::bail!("String not found in file. Make sure to include exact text including whitespace.");
        }

        if occurrences > 1 {
            anyhow::bail!(
                "Found {} occurrences of the string. Please provide a more specific string to ensure only one match.",
                occurrences
            );
        }

        let new_content = content.replace(old_str, new_str);
        fs::write(&resolved_path, &new_content)
            .with_context(|| format!("Failed to write file: {}", resolved_path))?;

        Ok(format!(
            "Successfully replaced text in {}. The string was replaced 1 time.",
            path
        ))
    }

    /// Search for files using glob pattern
    fn glob_search(&self, pattern: &str) -> Result<String> {
        let full_pattern = if Path::new(pattern).is_absolute() {
            pattern.to_string()
        } else {
            format!("{}/{}", self.base_dir, pattern)
        };

        tracing::info!("Searching with glob pattern: {}", full_pattern);

        let entries: Vec<String> = glob(&full_pattern)
            .with_context(|| format!("Invalid glob pattern: {}", pattern))?
            .filter_map(|entry| entry.ok())
            .map(|path| {
                // Try to make the path relative to base_dir for cleaner output
                path.strip_prefix(&self.base_dir)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| path.to_string_lossy().to_string())
            })
            .collect();

        if entries.is_empty() {
            Ok(format!("No files found matching pattern: {}", pattern))
        } else {
            let mut result = format!("Found {} files matching '{}':\n", entries.len(), pattern);
            for entry in entries.iter().take(100) {
                result.push_str(&format!("  {}\n", entry));
            }
            if entries.len() > 100 {
                result.push_str(&format!("  ... and {} more\n", entries.len() - 100));
            }
            Ok(result)
        }
    }

    /// Insert text at a specific line
    fn insert_at_line(&self, path: &str, line: usize, content: &str) -> Result<String> {
        let resolved_path = self.resolve_path(path);
        tracing::info!("Inserting text at line {} in file: {}", line, resolved_path);

        let file_content = fs::read_to_string(&resolved_path)
            .with_context(|| format!("Failed to read file: {}", resolved_path))?;

        let mut lines: Vec<&str> = file_content.lines().collect();
        let insert_index = line.saturating_sub(1).min(lines.len());

        lines.insert(insert_index, content);
        let new_content = lines.join("\n");

        fs::write(&resolved_path, &new_content)
            .with_context(|| format!("Failed to write file: {}", resolved_path))?;

        Ok(format!(
            "Successfully inserted text at line {} in {}",
            line, path
        ))
    }
}

impl Default for FileEditTool {
    fn default() -> Self {
        Self::with_base_dir(".")
    }
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "file_edit"
    }

    fn description(&self) -> &str {
        "View, create, and edit files. Supports viewing files with line numbers, creating new files, replacing text (str_replace), and searching with glob patterns."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "file_edit".to_string(),
            description: Some(self.description().to_string()),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "command": {
                        "type": "string",
                        "description": "The command to execute: 'view', 'create', 'str_replace', 'glob', or 'insert'",
                        "enum": ["view", "create", "str_replace", "glob", "insert"]
                    },
                    "path": {
                        "type": "string",
                        "description": "File path (relative to project root or absolute)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for 'create' or 'insert' commands"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to find for 'str_replace' command"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "String to replace with for 'str_replace' command"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for 'glob' command (e.g., '**/*.rs')"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number for 'view' command (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number for 'view' command (inclusive)"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number for 'insert' command (1-indexed)"
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
            .unwrap_or("unknown");

        let (action, details) = match command {
            "view" => {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                ("View file".to_string(), Some(format!("Path: {}", path)))
            }
            "create" => {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                ("Create file".to_string(), Some(format!("Path: {}", path)))
            }
            "str_replace" => {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                ("Replace text in file".to_string(), Some(format!("Path: {}", path)))
            }
            "glob" => {
                let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
                ("Search files".to_string(), Some(format!("Pattern: {}", pattern)))
            }
            "insert" => {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                let line = input.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                ("Insert text in file".to_string(), Some(format!("Path: {}, Line: {}", path, line)))
            }
            _ => ("Unknown file operation".to_string(), None),
        };

        ToolInfo {
            name: "file_edit".to_string(),
            action_description: action,
            details,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let command: FileCommand = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid file_edit input: {}", e))?;

        let result = match command {
            FileCommand::View { path, start_line, end_line } => {
                self.view_file(&path, start_line, end_line)
            }
            FileCommand::Create { path, content } => {
                self.create_file(&path, &content)
            }
            FileCommand::StrReplace { path, old_str, new_str } => {
                self.str_replace(&path, &old_str, &new_str)
            }
            FileCommand::Glob { pattern } => {
                self.glob_search(&pattern)
            }
            FileCommand::Insert { path, line, content } => {
                self.insert_at_line(&path, line, &content)
            }
        };

        match result {
            Ok(output) => Ok(ToolResult::success(output)),
            Err(e) => Ok(ToolResult::error(format!("{}", e))),
        }
    }

    fn requires_permission(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_glob_search() {
        let tool = FileEditTool::with_base_dir(".");
        let input = json!({
            "command": "glob",
            "pattern": "*.toml"
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("Cargo.toml") || result.output.contains("No files found"));
    }

    #[tokio::test]
    async fn test_create_and_view_file() {
        let dir = tempdir().unwrap();
        let tool = FileEditTool::with_base_dir(dir.path().to_string_lossy().to_string());

        // Create a file
        let create_input = json!({
            "command": "create",
            "path": "test.txt",
            "content": "Hello\nWorld\nTest"
        });
        let result = tool.execute(&create_input).await.unwrap();
        assert!(!result.is_error);

        // View the file
        let view_input = json!({
            "command": "view",
            "path": "test.txt"
        });
        let result = tool.execute(&view_input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("Hello"));
        assert!(result.output.contains("World"));
    }

    #[tokio::test]
    async fn test_str_replace() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "Hello World").unwrap();

        let tool = FileEditTool::with_base_dir(dir.path().to_string_lossy().to_string());

        let input = json!({
            "command": "str_replace",
            "path": "test.txt",
            "old_str": "World",
            "new_str": "Rust"
        });
        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello Rust");
    }
}
