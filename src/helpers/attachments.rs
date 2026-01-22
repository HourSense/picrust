//! Attachment processing for user messages
//!
//! This module handles parsing and processing of attachment tags in user input.
//! Attachments are specified using `<vibe-work-attachment>path</vibe-work-attachment>` tags.

use anyhow::Result;
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::llm::ContentBlock;

/// Maximum file size for images (5MB per Claude docs)
const MAX_IMAGE_SIZE: u64 = 5 * 1024 * 1024;
/// Maximum file size for PDFs (32MB)
const MAX_PDF_SIZE: u64 = 32 * 1024 * 1024;
/// Maximum lines to read for text files
const DEFAULT_LINE_LIMIT: usize = 2000;
/// Maximum characters per line before truncation
const MAX_LINE_LENGTH: usize = 2000;

/// Process attachments from user input
///
/// Scans the input for `<vibe-work-attachment>path</vibe-work-attachment>` tags,
/// reads each file, and returns a Vec of ContentBlocks representing the attachments.
///
/// Features:
/// - Deduplicates files (same file referenced multiple times is only processed once)
/// - Handles directories (lists contents instead of trying to read)
/// - Preserves order of first occurrence
///
/// # Arguments
/// * `input` - The user input text containing attachment tags
/// * `base_dir` - Base directory for resolving relative paths
///
/// # Returns
/// A vector of ContentBlocks, one for each attachment found (in order)
pub fn process_attachments(input: &str, base_dir: &str) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    let mut processed_paths: HashSet<String> = HashSet::new();

    // Parse attachment tags using regex
    let re = match Regex::new(r"<vibe-work-attachment>(.*?)</vibe-work-attachment>") {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("[Attachments] Failed to compile regex: {}", e);
            return blocks;
        }
    };

    // Extract all attachment paths in order
    for cap in re.captures_iter(input) {
        if let Some(path_match) = cap.get(1) {
            let file_path = path_match.as_str().trim();
            let resolved_path = resolve_path(file_path, base_dir);

            // Check if we've already processed this file
            if processed_paths.contains(&resolved_path) {
                tracing::debug!("[Attachments] Skipping duplicate: {}", file_path);
                blocks.push(ContentBlock::Text {
                    text: format!("Note: File {} was already attached above", file_path),
                    cache_control: None,
                });
                continue;
            }

            tracing::info!("[Attachments] Processing attachment: {}", file_path);

            // Read the file and convert to content blocks
            match read_attachment(file_path, base_dir) {
                Ok(mut content_blocks) => {
                    processed_paths.insert(resolved_path);
                    blocks.append(&mut content_blocks);
                }
                Err(e) => {
                    // On error, add a text block describing the error
                    let error_text = format!(
                        "Error: Cannot read file {} - {}",
                        file_path, e
                    );
                    tracing::warn!("[Attachments] {}", error_text);
                    blocks.push(ContentBlock::Text {
                        text: error_text,
                        cache_control: None,
                    });
                }
            }
        }
    }

    blocks
}

/// Read a single attachment file and convert to ContentBlocks
fn read_attachment(file_path: &str, base_dir: &str) -> Result<Vec<ContentBlock>> {
    let resolved_path = resolve_path(file_path, base_dir);
    let path_obj = Path::new(&resolved_path);

    // Check if path is a directory
    if path_obj.is_dir() {
        return read_directory(&resolved_path, file_path);
    }

    // Get file extension to determine type
    let extension = path_obj
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());

    match extension.as_deref() {
        Some("png") | Some("jpg") | Some("jpeg") | Some("gif") | Some("webp") => {
            read_image(&resolved_path, file_path)
        }
        Some("pdf") => read_pdf(&resolved_path, file_path),
        _ => {
            // Default to text reading for all other files
            read_text_file(&resolved_path, file_path)
        }
    }
}

/// Resolve a path (handle both absolute and relative)
fn resolve_path(path: &str, base_dir: &str) -> String {
    let path_obj = Path::new(path);
    if path_obj.is_absolute() {
        path.to_string()
    } else {
        Path::new(base_dir)
            .join(path)
            .to_string_lossy()
            .to_string()
    }
}

/// Read a text file with line numbers
fn read_text_file(resolved_path: &str, original_path: &str) -> Result<Vec<ContentBlock>> {
    let content = fs::read_to_string(resolved_path)?;

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let end = total_lines.min(DEFAULT_LINE_LIMIT);

    let mut result = format!("File: {}\n\n", original_path);

    for (i, line) in lines[..end].iter().enumerate() {
        let line_num = i + 1;
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
            "\n... ({} more lines)\n",
            total_lines - end
        ));
    }

    Ok(vec![ContentBlock::Text {
        text: result,
        cache_control: None,
    }])
}

/// Read an image file
fn read_image(resolved_path: &str, original_path: &str) -> Result<Vec<ContentBlock>> {
    // Check file size first
    let metadata = fs::metadata(resolved_path)?;

    if metadata.len() > MAX_IMAGE_SIZE {
        anyhow::bail!(
            "Image file too large: {} bytes (max: {} bytes)",
            metadata.len(),
            MAX_IMAGE_SIZE
        );
    }

    // Read the file as bytes
    let data = fs::read(resolved_path)?;

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

    // Encode to base64
    use base64::Engine;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

    tracing::info!(
        "[Attachments] Read image: {} ({} bytes, type: {})",
        original_path,
        data.len(),
        media_type
    );

    Ok(vec![ContentBlock::image(
        base64_data,
        media_type.to_string(),
    )])
}

/// Read a PDF file
fn read_pdf(resolved_path: &str, original_path: &str) -> Result<Vec<ContentBlock>> {
    // Check file size first
    let metadata = fs::metadata(resolved_path)?;

    if metadata.len() > MAX_PDF_SIZE {
        anyhow::bail!(
            "PDF file too large: {} bytes (max: {} bytes)",
            metadata.len(),
            MAX_PDF_SIZE
        );
    }

    // Read the file as bytes
    let data = fs::read(resolved_path)?;

    // Encode to base64
    use base64::Engine;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

    tracing::info!(
        "[Attachments] Read PDF: {} ({} bytes)",
        original_path,
        data.len()
    );

    Ok(vec![ContentBlock::document(
        base64_data,
        "application/pdf".to_string(),
    )])
}

/// Read a directory and list its contents
fn read_directory(resolved_path: &str, original_path: &str) -> Result<Vec<ContentBlock>> {
    let entries = fs::read_dir(resolved_path)?;

    let mut result = format!("Directory: {}\n\n", original_path);
    let mut items: Vec<(String, bool, u64)> = Vec::new(); // (name, is_dir, size)

    // Collect all entries
    for entry in entries {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = metadata.is_dir();
        let size = metadata.len();

        items.push((name, is_dir, size));
    }

    // Sort: directories first, then files, alphabetically within each group
    items.sort_by(|a, b| {
        match (a.1, b.1) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.0.cmp(&b.0),
        }
    });

    let item_count = items.len();

    // Format output
    if items.is_empty() {
        result.push_str("(empty directory)\n");
    } else {
        for (name, is_dir, size) in items {
            let type_marker = if is_dir { "DIR " } else { "    " };
            let size_str = if is_dir {
                String::new()
            } else if size < 1024 {
                format!("{:>8} B", size)
            } else if size < 1024 * 1024 {
                format!("{:>7.1} KB", size as f64 / 1024.0)
            } else if size < 1024 * 1024 * 1024 {
                format!("{:>7.1} MB", size as f64 / 1024.0 / 1024.0)
            } else {
                format!("{:>7.1} GB", size as f64 / 1024.0 / 1024.0 / 1024.0)
            };

            result.push_str(&format!("{} {:>12}  {}\n", type_marker, size_str, name));
        }
    }

    tracing::info!(
        "[Attachments] Listed directory: {} ({} items)",
        original_path,
        item_count
    );

    Ok(vec![ContentBlock::Text {
        text: result,
        cache_control: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attachment_tag_parsing() {
        let input = "Hello <vibe-work-attachment>/path/to/file.txt</vibe-work-attachment> world!";
        let re = Regex::new(r"<vibe-work-attachment>(.*?)</vibe-work-attachment>").unwrap();

        let paths: Vec<String> = re
            .captures_iter(input)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();

        assert_eq!(paths, vec!["/path/to/file.txt"]);
    }

    #[test]
    fn test_multiple_attachments() {
        let input = "First <vibe-work-attachment>file1.txt</vibe-work-attachment> and <vibe-work-attachment>file2.png</vibe-work-attachment>";
        let re = Regex::new(r"<vibe-work-attachment>(.*?)</vibe-work-attachment>").unwrap();

        let paths: Vec<String> = re
            .captures_iter(input)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();

        assert_eq!(paths, vec!["file1.txt", "file2.png"]);
    }

    #[test]
    fn test_resolve_path() {
        // Absolute path
        let abs = resolve_path("/absolute/path", "/base");
        assert_eq!(abs, "/absolute/path");

        // Relative path
        let rel = resolve_path("relative/path", "/base");
        assert_eq!(rel, "/base/relative/path");
    }
}
