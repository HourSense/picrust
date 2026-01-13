use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::llm::Message;

const CONVERSATIONS_DIR: &str = "conversations";

/// Metadata for a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMetadata {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub model_provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Represents a conversation with history stored in JSONL format
pub struct Conversation {
    metadata: ConversationMetadata,
    path: PathBuf,
}

impl Conversation {
    /// Create a new conversation
    pub fn new() -> Result<Self> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        tracing::info!("Creating new conversation with ID: {}", id);

        let metadata = ConversationMetadata {
            id: id.clone(),
            created_at: now,
            updated_at: now,
            model_provider: "anthropic".to_string(),
            title: None,
        };

        // Create conversation directory
        let path = PathBuf::from(CONVERSATIONS_DIR).join(&id);
        fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create conversation directory: {:?}", path))?;

        tracing::debug!("Conversation directory created: {:?}", path);

        let conversation = Self { metadata, path };

        // Save initial metadata
        conversation.save_metadata()?;

        // Create empty history file
        conversation.ensure_history_file()?;

        tracing::info!("Conversation created successfully: {}", id);

        Ok(conversation)
    }

    /// Load an existing conversation by ID
    pub fn load(id: &str) -> Result<Self> {
        tracing::info!("Loading conversation: {}", id);

        let path = PathBuf::from(CONVERSATIONS_DIR).join(id);

        if !path.exists() {
            anyhow::bail!("Conversation not found: {}", id);
        }

        let metadata_path = path.join("metadata.json");
        let metadata_content = fs::read_to_string(&metadata_path)
            .with_context(|| format!("Failed to read metadata: {:?}", metadata_path))?;

        let metadata: ConversationMetadata = serde_json::from_str(&metadata_content)
            .context("Failed to parse metadata")?;

        tracing::info!("Conversation loaded: {}", id);

        Ok(Self { metadata, path })
    }

    /// Get conversation ID
    pub fn id(&self) -> &str {
        &self.metadata.id
    }

    /// Get conversation metadata
    pub fn metadata(&self) -> &ConversationMetadata {
        &self.metadata
    }

    /// Add a message to the conversation history (raw, Anthropic format)
    pub fn add_message_raw(&mut self, message: &Message) -> Result<()> {
        tracing::debug!(
            "Adding {} message to conversation {}",
            message.role,
            self.metadata.id
        );

        // Append to history.jsonl
        let history_path = self.path.join("history.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&history_path)
            .with_context(|| format!("Failed to open history file: {:?}", history_path))?;

        let json = serde_json::to_string(message).context("Failed to serialize message")?;
        writeln!(file, "{}", json)
            .with_context(|| format!("Failed to write to history file: {:?}", history_path))?;

        // Update metadata timestamp
        self.metadata.updated_at = Utc::now();
        self.save_metadata()?;

        tracing::debug!("Message added successfully");

        Ok(())
    }

    /// Add a user message (simple text)
    pub fn add_user_message(&mut self, content: impl Into<String>) -> Result<()> {
        self.add_message_raw(&Message::user(content))
    }

    /// Add an assistant message (simple text)
    pub fn add_assistant_message(&mut self, content: impl Into<String>) -> Result<()> {
        self.add_message_raw(&Message::assistant(content))
    }

    /// Get all messages from the conversation history
    pub fn get_messages(&self) -> Result<Vec<Message>> {
        let history_path = self.path.join("history.jsonl");

        if !history_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&history_path)
            .with_context(|| format!("Failed to open history file: {:?}", history_path))?;

        let reader = BufReader::new(file);
        let mut messages = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.context("Failed to read line from history file")?;
            if line.trim().is_empty() {
                continue;
            }

            let message: Message = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse message on line {}", line_num + 1))?;

            messages.push(message);
        }

        Ok(messages)
    }

    /// Get the number of messages in the conversation
    pub fn message_count(&self) -> Result<usize> {
        Ok(self.get_messages()?.len())
    }

    /// Set the conversation title
    pub fn set_title(&mut self, title: impl Into<String>) -> Result<()> {
        self.metadata.title = Some(title.into());
        self.save_metadata()?;
        Ok(())
    }

    /// Save metadata to metadata.json
    fn save_metadata(&self) -> Result<()> {
        let metadata_path = self.path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .context("Failed to serialize metadata")?;

        fs::write(&metadata_path, metadata_json)
            .with_context(|| format!("Failed to write metadata: {:?}", metadata_path))?;

        tracing::debug!("Metadata saved for conversation {}", self.metadata.id);

        Ok(())
    }

    /// Ensure history.jsonl exists
    fn ensure_history_file(&self) -> Result<()> {
        let history_path = self.path.join("history.jsonl");
        if !history_path.exists() {
            File::create(&history_path)
                .with_context(|| format!("Failed to create history file: {:?}", history_path))?;
        }
        Ok(())
    }

    /// List all conversation IDs
    pub fn list_all() -> Result<Vec<String>> {
        let conversations_path = Path::new(CONVERSATIONS_DIR);

        if !conversations_path.exists() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::new();

        for entry in fs::read_dir(conversations_path)
            .context("Failed to read conversations directory")?
        {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(id) = path.file_name().and_then(|n| n.to_str()) {
                    ids.push(id.to_string());
                }
            }
        }

        Ok(ids)
    }

    /// Delete this conversation
    pub fn delete(self) -> Result<()> {
        tracing::warn!("Deleting conversation: {}", self.metadata.id);

        fs::remove_dir_all(&self.path)
            .with_context(|| format!("Failed to delete conversation directory: {:?}", self.path))?;

        tracing::info!("Conversation deleted: {}", self.metadata.id);

        Ok(())
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new().expect("Failed to create default conversation")
    }
}
