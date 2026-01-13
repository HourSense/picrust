//! Context providers for dynamic context injection
//!
//! Context providers generate dynamic context based on the current
//! state of the conversation or the environment.

use anyhow::Result;
use async_trait::async_trait;
use std::process::Command;

use crate::llm::Message;

/// Trait for context providers
///
/// Context providers generate context that can be injected into
/// the system prompt or as hidden user messages.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Get the name of this context provider
    fn name(&self) -> &str;

    /// Get context based on the current conversation
    ///
    /// Returns None if no context should be injected.
    async fn get_context(&self, messages: &[Message]) -> Result<Option<String>>;

    /// Whether to inject this context as a user message rather than
    /// part of the system prompt
    fn inject_as_message(&self) -> bool {
        false
    }
}

/// Provider for file structure context
///
/// Provides a tree-like view of the project structure.
pub struct FileStructureProvider {
    base_dir: String,
    max_depth: usize,
    exclude_patterns: Vec<String>,
}

impl FileStructureProvider {
    /// Create a new file structure provider
    pub fn new(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            max_depth: 3,
            exclude_patterns: vec![
                ".git".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "__pycache__".to_string(),
                ".venv".to_string(),
            ],
        }
    }

    /// Set the maximum depth for directory traversal
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Add patterns to exclude
    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns = patterns;
        self
    }

    /// Generate a tree structure of the directory
    fn generate_tree(&self) -> Result<String> {
        // Use the tree command if available, otherwise fall back to a simple listing
        let output = Command::new("tree")
            .arg("-L")
            .arg(self.max_depth.to_string())
            .arg("--noreport")
            .args(self.exclude_patterns.iter().flat_map(|p| vec!["-I", p]))
            .current_dir(&self.base_dir)
            .output();

        match output {
            Ok(output) if output.status.success() => {
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
            _ => {
                // Fall back to simple ls-based listing
                let output = Command::new("find")
                    .arg(".")
                    .arg("-maxdepth")
                    .arg(self.max_depth.to_string())
                    .arg("-type")
                    .arg("f")
                    .current_dir(&self.base_dir)
                    .output()?;

                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
        }
    }
}

#[async_trait]
impl ContextProvider for FileStructureProvider {
    fn name(&self) -> &str {
        "file_structure"
    }

    async fn get_context(&self, _messages: &[Message]) -> Result<Option<String>> {
        let tree = self.generate_tree()?;
        if tree.is_empty() {
            Ok(None)
        } else {
            Ok(Some(format!("Project structure:\n```\n{}\n```", tree)))
        }
    }
}

/// Provider for git status context
///
/// Provides current git status and recent commits.
pub struct GitStatusProvider {
    base_dir: String,
    include_diff: bool,
    recent_commits: usize,
}

impl GitStatusProvider {
    /// Create a new git status provider
    pub fn new(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            include_diff: false,
            recent_commits: 5,
        }
    }

    /// Include the git diff in the context
    pub fn with_diff(mut self) -> Self {
        self.include_diff = true;
        self
    }

    /// Set the number of recent commits to include
    pub fn with_recent_commits(mut self, count: usize) -> Self {
        self.recent_commits = count;
        self
    }

    fn get_git_status(&self) -> Result<String> {
        let status = Command::new("git")
            .args(["status", "--short"])
            .current_dir(&self.base_dir)
            .output()?;

        if !status.status.success() {
            return Ok("Not a git repository".to_string());
        }

        let mut result = String::from("Git status:\n");
        let status_output = String::from_utf8_lossy(&status.stdout);
        if status_output.is_empty() {
            result.push_str("Working tree clean\n");
        } else {
            result.push_str(&status_output);
        }

        // Get current branch
        let branch = Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&self.base_dir)
            .output()?;

        if branch.status.success() {
            result.push_str(&format!(
                "\nCurrent branch: {}",
                String::from_utf8_lossy(&branch.stdout).trim()
            ));
        }

        // Get recent commits
        if self.recent_commits > 0 {
            let log = Command::new("git")
                .args([
                    "log",
                    "--oneline",
                    &format!("-{}", self.recent_commits),
                ])
                .current_dir(&self.base_dir)
                .output()?;

            if log.status.success() {
                result.push_str("\n\nRecent commits:\n");
                result.push_str(&String::from_utf8_lossy(&log.stdout));
            }
        }

        // Include diff if requested
        if self.include_diff {
            let diff = Command::new("git")
                .args(["diff", "--stat"])
                .current_dir(&self.base_dir)
                .output()?;

            if diff.status.success() {
                let diff_output = String::from_utf8_lossy(&diff.stdout);
                if !diff_output.is_empty() {
                    result.push_str("\n\nUnstaged changes:\n");
                    result.push_str(&diff_output);
                }
            }
        }

        Ok(result)
    }
}

#[async_trait]
impl ContextProvider for GitStatusProvider {
    fn name(&self) -> &str {
        "git_status"
    }

    async fn get_context(&self, _messages: &[Message]) -> Result<Option<String>> {
        match self.get_git_status() {
            Ok(status) => Ok(Some(status)),
            Err(_) => Ok(None),
        }
    }
}

/// A simple static context provider
///
/// Always returns the same context string.
pub struct StaticContextProvider {
    name: String,
    context: String,
    as_message: bool,
}

impl StaticContextProvider {
    /// Create a new static context provider
    pub fn new(name: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            context: context.into(),
            as_message: false,
        }
    }

    /// Set whether to inject as a message
    pub fn inject_as_message(mut self, as_message: bool) -> Self {
        self.as_message = as_message;
        self
    }
}

#[async_trait]
impl ContextProvider for StaticContextProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn get_context(&self, _messages: &[Message]) -> Result<Option<String>> {
        Ok(Some(self.context.clone()))
    }

    fn inject_as_message(&self) -> bool {
        self.as_message
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_provider() {
        let provider = StaticContextProvider::new("test", "Test context");
        let context = provider.get_context(&[]).await.unwrap();
        assert_eq!(context, Some("Test context".to_string()));
    }

    #[tokio::test]
    async fn test_file_structure_provider() {
        let provider = FileStructureProvider::new(".");
        let context = provider.get_context(&[]).await.unwrap();
        assert!(context.is_some());
    }
}
