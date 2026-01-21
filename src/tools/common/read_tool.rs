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
use crate::runtime::AgentInternals;

/// Maximum lines to read by default
const DEFAULT_LINE_LIMIT: usize = 2000;
/// Maximum characters per line before truncation
const MAX_LINE_LENGTH: usize = 2000;
/// Maximum file size for images (5MB per Claude docs)
const MAX_IMAGE_SIZE: u64 = 5 * 1024 * 1024;
/// Maximum file size for PDFs (32MB per user requirement)
const MAX_PDF_SIZE: u64 = 32 * 1024 * 1024;

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

    /// Read file contents - dispatches to appropriate handler based on file type
    fn read_file(&self, file_path: &str, offset: Option<usize>, limit: Option<usize>) -> Result<ToolResult> {
        let resolved_path = self.resolve_path(file_path);
        tracing::info!("Reading file: {}", resolved_path);

        // Get file extension to determine type
        let extension = Path::new(&resolved_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase());

        match extension.as_deref() {
            Some("png") | Some("jpg") | Some("jpeg") | Some("gif") | Some("webp") => {
                self.read_image(&resolved_path)
            }
            Some("pdf") => {
                self.read_pdf(&resolved_path)
            }
            _ => {
                // Default to text reading for all other files
                self.read_text_file(&resolved_path, offset, limit)
            }
        }
    }

    /// Read a text file with optional offset and limit
    fn read_text_file(&self, resolved_path: &str, offset: Option<usize>, limit: Option<usize>) -> Result<ToolResult> {
        let content = fs::read_to_string(resolved_path)
            .with_context(|| format!("Failed to read file: {}", resolved_path))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = offset.unwrap_or(1).saturating_sub(1);
        let count = limit.unwrap_or(DEFAULT_LINE_LIMIT);
        let end = (start + count).min(total_lines);

        if start >= total_lines {
            return Ok(ToolResult::success(format!(
                "File has {} lines. Requested offset {} is out of range.",
                total_lines,
                start + 1
            )));
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

        Ok(ToolResult::success(result))
    }

    /// Read an image file
    fn read_image(&self, resolved_path: &str) -> Result<ToolResult> {
        // Check file size first
        let metadata = fs::metadata(resolved_path)
            .with_context(|| format!("Failed to get file metadata: {}", resolved_path))?;

        if metadata.len() > MAX_IMAGE_SIZE {
            return Ok(ToolResult::error(format!(
                "Image file too large: {} bytes (max: {} bytes)",
                metadata.len(),
                MAX_IMAGE_SIZE
            )));
        }

        // Read the file as bytes
        let data = fs::read(resolved_path)
            .with_context(|| format!("Failed to read image file: {}", resolved_path))?;

        // Determine media type from extension
        let extension = Path::new(resolved_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let media_type = match extension.as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            _ => "application/octet-stream",
        };

        tracing::info!(
            "Read image: {} ({} bytes, type: {})",
            resolved_path,
            data.len(),
            media_type
        );

        Ok(ToolResult::image(data, media_type))
    }

    /// Read a PDF file
    fn read_pdf(&self, resolved_path: &str) -> Result<ToolResult> {
        // Check file size first
        let metadata = fs::metadata(resolved_path)
            .with_context(|| format!("Failed to get file metadata: {}", resolved_path))?;

        if metadata.len() > MAX_PDF_SIZE {
            return Ok(ToolResult::error(format!(
                "PDF file too large: {} bytes (max: {} bytes)",
                metadata.len(),
                MAX_PDF_SIZE
            )));
        }

        // Read the file as bytes
        let data = fs::read(resolved_path)
            .with_context(|| format!("Failed to read PDF file: {}", resolved_path))?;

        // Format file size for description
        let size_kb = data.len() as f64 / 1024.0;
        let size_str = if size_kb < 1024.0 {
            format!("{:.1}KB", size_kb)
        } else {
            format!("{:.1}MB", size_kb / 1024.0)
        };

        let description = format!("PDF file read: {} ({})", resolved_path, size_str);

        tracing::info!("{}", description);

        Ok(ToolResult::document(data, "application/pdf", description))
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
        "Read a file from the local filesystem. Supports text files, images (PNG, JPEG, GIF, WebP), and PDFs."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "Read".to_string(),
            description: Some(
                "Reads a file from the local filesystem. Supports text files, images, and PDFs. \
                The file_path parameter must be an absolute path. \
                \n\n\
                For TEXT FILES: \
                - By default, reads up to 2000 lines with line numbers. \
                - You can optionally specify offset and limit for long files. \
                \n\n\
                For IMAGES (PNG, JPEG, GIF, WebP): \
                - Reads and returns the image for vision analysis. \
                - Maximum file size: 5MB. \
                - The image will be sent to Claude for visual understanding. \
                \n\n\
                For PDFs: \
                - Reads and returns the PDF document for analysis. \
                - Maximum file size: 32MB. \
                - The PDF will be sent to Claude for document understanding. \
                \n\n\
                The tool automatically detects the file type based on the extension."
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
            cache_control: None,
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

    async fn execute(&self, input: &Value, _internals: &mut AgentInternals) -> Result<ToolResult> {
        let read_input: ReadInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid read input: {}", e))?;

        match self.read_file(&read_input.file_path, read_input.offset, read_input.limit) {
            Ok(result) => Ok(result),
            Err(e) => Ok(ToolResult::error(format!("{}", e))),
        }
    }

    fn requires_permission(&self) -> bool {
        false // Read-only operation
    }
}

// Tests temporarily disabled - require AgentInternals test helper
// TODO: Create test infrastructure for tools that need AgentInternals
