//! Session storage helpers
//!
//! Handles reading and writing session data to disk.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::core::FrameworkResult;
use crate::core::error::FrameworkError;
use crate::llm::Message;

use super::metadata::SessionMetadata;

/// Default directory for session storage
const SESSIONS_DIR: &str = "sessions";

/// Session storage manager
#[derive(Debug, Clone)]
pub struct SessionStorage {
    base_dir: PathBuf,
}

impl SessionStorage {
    /// Create a new session storage with the default directory
    pub fn new() -> Self {
        Self {
            base_dir: PathBuf::from(SESSIONS_DIR),
        }
    }

    /// Create a new session storage with a custom directory
    pub fn with_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: dir.into(),
        }
    }

    /// Get the directory path for a session
    pub fn session_dir(&self, session_id: &str) -> PathBuf {
        self.base_dir.join(session_id)
    }

    /// Get the metadata file path for a session
    pub fn metadata_path(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id).join("metadata.json")
    }

    /// Get the history file path for a session
    pub fn history_path(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id).join("history.jsonl")
    }

    /// Get the system prompt file path for a session
    pub fn system_prompt_path(&self, session_id: &str) -> PathBuf {
        self.session_dir(session_id).join("system_prompt.md")
    }

    /// Save the system prompt to disk
    pub fn save_system_prompt(&self, session_id: &str, prompt: &str) -> FrameworkResult<()> {
        self.ensure_session_dir(session_id)?;
        let path = self.system_prompt_path(session_id);
        fs::write(&path, prompt)?;
        Ok(())
    }

    /// Load the system prompt from disk
    pub fn load_system_prompt(&self, session_id: &str) -> FrameworkResult<String> {
        let path = self.system_prompt_path(session_id);
        if !path.exists() {
            return Err(crate::core::error::FrameworkError::Other(
                format!("system_prompt.md not found for session '{}'", session_id),
            ));
        }
        Ok(fs::read_to_string(&path)?)
    }

    /// Create the session directory if it doesn't exist
    pub fn ensure_session_dir(&self, session_id: &str) -> FrameworkResult<PathBuf> {
        let dir = self.session_dir(session_id);
        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }
        Ok(dir)
    }

    /// Save session metadata
    pub fn save_metadata(&self, metadata: &SessionMetadata) -> FrameworkResult<()> {
        self.ensure_session_dir(&metadata.session_id)?;
        let path = self.metadata_path(&metadata.session_id);

        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, metadata)?;

        Ok(())
    }

    /// Load session metadata
    pub fn load_metadata(&self, session_id: &str) -> FrameworkResult<SessionMetadata> {
        let path = self.metadata_path(session_id);

        if !path.exists() {
            return Err(FrameworkError::SessionNotFound(session_id.to_string()));
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let metadata: SessionMetadata = serde_json::from_reader(reader)?;

        Ok(metadata)
    }

    /// Append a message to the history file
    pub fn append_message(&self, session_id: &str, message: &Message) -> FrameworkResult<()> {
        self.ensure_session_dir(session_id)?;
        let path = self.history_path(session_id);

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let json = serde_json::to_string(message)?;
        writeln!(file, "{}", json)?;

        Ok(())
    }

    /// Load all messages from the history file
    pub fn load_messages(&self, session_id: &str) -> FrameworkResult<Vec<Message>> {
        let path = self.history_path(session_id);

        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let message: Message = serde_json::from_str(&line)?;
            messages.push(message);
        }

        Ok(messages)
    }

    /// Save all messages (overwrites existing history)
    pub fn save_messages(&self, session_id: &str, messages: &[Message]) -> FrameworkResult<()> {
        self.ensure_session_dir(session_id)?;
        let path = self.history_path(session_id);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        for message in messages {
            let json = serde_json::to_string(message)?;
            writeln!(writer, "{}", json)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Check if a session exists
    pub fn session_exists(&self, session_id: &str) -> bool {
        self.metadata_path(session_id).exists()
    }

    /// List all session IDs
    pub fn list_sessions(&self) -> FrameworkResult<Vec<String>> {
        if !self.base_dir.exists() {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        // Check if it has a metadata file
                        if self.metadata_path(name_str).exists() {
                            sessions.push(name_str.to_string());
                        }
                    }
                }
            }
        }

        Ok(sessions)
    }

    /// List session IDs with optional filtering
    ///
    /// If `top_level_only` is true, only returns sessions that are not subagents
    /// (i.e., sessions with no parent_session_id).
    pub fn list_sessions_filtered(&self, top_level_only: bool) -> FrameworkResult<Vec<String>> {
        let all_sessions = self.list_sessions()?;

        if !top_level_only {
            return Ok(all_sessions);
        }

        // Filter to only top-level sessions
        let mut filtered = Vec::new();
        for session_id in all_sessions {
            if let Ok(metadata) = self.load_metadata(&session_id) {
                if !metadata.is_subagent() {
                    filtered.push(session_id);
                }
            }
        }

        Ok(filtered)
    }

    /// List only top-level session IDs (sessions that are not subagents)
    pub fn list_top_level_sessions(&self) -> FrameworkResult<Vec<String>> {
        self.list_sessions_filtered(true)
    }

    /// List all sessions with their metadata
    ///
    /// Returns tuples of (session_id, metadata) for all valid sessions.
    /// If `top_level_only` is true, only includes sessions that are not subagents.
    pub fn list_sessions_with_metadata(
        &self,
        top_level_only: bool,
    ) -> FrameworkResult<Vec<(String, SessionMetadata)>> {
        let session_ids = self.list_sessions()?;
        let mut result = Vec::new();

        for session_id in session_ids {
            if let Ok(metadata) = self.load_metadata(&session_id) {
                if !top_level_only || !metadata.is_subagent() {
                    result.push((session_id, metadata));
                }
            }
        }

        Ok(result)
    }

    /// Delete a session
    pub fn delete_session(&self, session_id: &str) -> FrameworkResult<()> {
        let dir = self.session_dir(session_id);
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

impl Default for SessionStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_storage() -> (SessionStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = SessionStorage::with_dir(temp_dir.path());
        (storage, temp_dir)
    }

    #[test]
    fn test_save_load_metadata() {
        let (storage, _temp) = create_test_storage();

        let meta = SessionMetadata::new("test_session", "coder", "Test", "Testing");
        storage.save_metadata(&meta).unwrap();

        let loaded = storage.load_metadata("test_session").unwrap();
        assert_eq!(loaded.session_id, "test_session");
        assert_eq!(loaded.agent_type, "coder");
    }

    #[test]
    fn test_append_load_messages() {
        let (storage, _temp) = create_test_storage();

        // Create session dir
        storage.ensure_session_dir("test_session").unwrap();

        // Append messages
        let msg1 = Message::user("Hello");
        let msg2 = Message::assistant("Hi there");

        storage.append_message("test_session", &msg1).unwrap();
        storage.append_message("test_session", &msg2).unwrap();

        // Load messages
        let messages = storage.load_messages("test_session").unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_session_exists() {
        let (storage, _temp) = create_test_storage();

        assert!(!storage.session_exists("nonexistent"));

        let meta = SessionMetadata::new("test_session", "coder", "Test", "Testing");
        storage.save_metadata(&meta).unwrap();

        assert!(storage.session_exists("test_session"));
    }

    #[test]
    fn test_list_sessions() {
        let (storage, _temp) = create_test_storage();

        // Create a few sessions
        storage.save_metadata(&SessionMetadata::new("session1", "coder", "S1", "D1")).unwrap();
        storage.save_metadata(&SessionMetadata::new("session2", "researcher", "S2", "D2")).unwrap();

        let sessions = storage.list_sessions().unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains(&"session1".to_string()));
        assert!(sessions.contains(&"session2".to_string()));
    }

    #[test]
    fn test_delete_session() {
        let (storage, _temp) = create_test_storage();

        let meta = SessionMetadata::new("to_delete", "coder", "Test", "Testing");
        storage.save_metadata(&meta).unwrap();
        assert!(storage.session_exists("to_delete"));

        storage.delete_session("to_delete").unwrap();
        assert!(!storage.session_exists("to_delete"));
    }

    #[test]
    fn test_list_sessions_filtered() {
        let (storage, _temp) = create_test_storage();

        // Create a top-level session
        storage
            .save_metadata(&SessionMetadata::new("parent1", "main", "Parent", "A parent"))
            .unwrap();

        // Create another top-level session
        storage
            .save_metadata(&SessionMetadata::new("parent2", "main", "Parent 2", "Another parent"))
            .unwrap();

        // Create a subagent session
        storage
            .save_metadata(&SessionMetadata::new_subagent(
                "child1",
                "researcher",
                "Child",
                "A child",
                "parent1",
                "tool_123",
            ))
            .unwrap();

        // List all sessions
        let all = storage.list_sessions().unwrap();
        assert_eq!(all.len(), 3);

        // List with filter = false (all sessions)
        let all_filtered = storage.list_sessions_filtered(false).unwrap();
        assert_eq!(all_filtered.len(), 3);

        // List only top-level sessions
        let top_level = storage.list_sessions_filtered(true).unwrap();
        assert_eq!(top_level.len(), 2);
        assert!(top_level.contains(&"parent1".to_string()));
        assert!(top_level.contains(&"parent2".to_string()));
        assert!(!top_level.contains(&"child1".to_string()));

        // Also test the convenience method
        let top_level2 = storage.list_top_level_sessions().unwrap();
        assert_eq!(top_level2.len(), 2);
    }

    #[test]
    fn test_list_sessions_with_metadata() {
        let (storage, _temp) = create_test_storage();

        // Create sessions
        storage
            .save_metadata(&SessionMetadata::new("main1", "coder", "Main 1", "First main"))
            .unwrap();
        storage
            .save_metadata(&SessionMetadata::new_subagent(
                "sub1",
                "helper",
                "Sub 1",
                "First sub",
                "main1",
                "tool_1",
            ))
            .unwrap();

        // List all with metadata
        let all = storage.list_sessions_with_metadata(false).unwrap();
        assert_eq!(all.len(), 2);

        // List top-level only with metadata
        let top_level = storage.list_sessions_with_metadata(true).unwrap();
        assert_eq!(top_level.len(), 1);
        assert_eq!(top_level[0].0, "main1");
        assert_eq!(top_level[0].1.agent_type, "coder");
    }
}
