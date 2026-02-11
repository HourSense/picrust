//! Session metadata types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Metadata for an agent session
///
/// This is persisted separately from the message history for quick access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    // --- Identity ---
    /// Unique session ID
    pub session_id: String,

    /// Type of agent (e.g., "coder", "researcher")
    pub agent_type: String,

    /// Human-readable name for this agent
    pub name: String,

    /// Description of what this agent does
    pub description: String,

    /// Auto-generated name for this conversation based on content
    /// This is typically set after the first turn by a conversation namer helper
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_name: Option<String>,

    // --- Lineage ---
    /// Parent session ID (if this is a subagent)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,

    /// The tool_use_id that spawned this agent (if subagent)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_tool_use_id: Option<String>,

    /// Session IDs of child agents spawned by this session
    #[serde(default)]
    pub child_session_ids: Vec<String>,

    // --- LLM Configuration ---
    /// Model being used
    pub model: String,

    /// Provider (e.g., "anthropic")
    pub provider: String,

    // --- Timestamps ---
    /// When the session was created
    pub created_at: DateTime<Utc>,

    /// When the session was last updated
    pub updated_at: DateTime<Utc>,

    // --- Custom Metadata ---
    /// Extensible metadata
    #[serde(default)]
    pub custom: HashMap<String, Value>,
}

impl SessionMetadata {
    /// Create new metadata for a root agent session
    pub fn new(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.into(),
            agent_type: agent_type.into(),
            name: name.into(),
            description: description.into(),
            conversation_name: None,
            parent_session_id: None,
            parent_tool_use_id: None,
            child_session_ids: Vec::new(),
            model: String::new(),
            provider: String::new(),
            created_at: now,
            updated_at: now,
            custom: HashMap::new(),
        }
    }

    /// Create new metadata for a subagent session
    pub fn new_subagent(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_tool_use_id: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.into(),
            agent_type: agent_type.into(),
            name: name.into(),
            description: description.into(),
            conversation_name: None,
            parent_session_id: Some(parent_session_id.into()),
            parent_tool_use_id: Some(parent_tool_use_id.into()),
            child_session_ids: Vec::new(),
            model: String::new(),
            provider: String::new(),
            created_at: now,
            updated_at: now,
            custom: HashMap::new(),
        }
    }

    /// Check if this is a subagent session
    pub fn is_subagent(&self) -> bool {
        self.parent_session_id.is_some()
    }

    /// Update the updated_at timestamp
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the provider
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Add a child session ID
    pub fn add_child(&mut self, child_session_id: impl Into<String>) {
        self.child_session_ids.push(child_session_id.into());
        self.touch();
    }

    /// Set the conversation name
    ///
    /// This is typically called by a conversation namer helper after the first
    /// turn to give the conversation a descriptive name based on its content.
    pub fn set_conversation_name(&mut self, name: impl Into<String>) {
        self.conversation_name = Some(name.into());
        self.touch();
    }

    /// Get the conversation name
    pub fn conversation_name(&self) -> Option<&str> {
        self.conversation_name.as_deref()
    }

    /// Check if the conversation has been named
    pub fn has_conversation_name(&self) -> bool {
        self.conversation_name.is_some()
    }

    /// Set custom metadata
    pub fn set_custom(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.custom.insert(key.into(), value.into());
        self.touch();
    }

    /// Get custom metadata
    pub fn get_custom(&self, key: &str) -> Option<&Value> {
        self.custom.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_metadata() {
        let meta = SessionMetadata::new(
            "session_123",
            "coder",
            "My Coder",
            "A coding agent",
        );

        assert_eq!(meta.session_id, "session_123");
        assert_eq!(meta.agent_type, "coder");
        assert_eq!(meta.name, "My Coder");
        assert!(!meta.is_subagent());
        assert!(meta.child_session_ids.is_empty());
    }

    #[test]
    fn test_subagent_metadata() {
        let meta = SessionMetadata::new_subagent(
            "sub_123",
            "researcher",
            "Research Helper",
            "Finds information",
            "parent_456",
            "tool_789",
        );

        assert!(meta.is_subagent());
        assert_eq!(meta.parent_session_id, Some("parent_456".into()));
        assert_eq!(meta.parent_tool_use_id, Some("tool_789".into()));
    }

    #[test]
    fn test_add_child() {
        let mut meta = SessionMetadata::new("session", "test", "Test", "Testing");

        meta.add_child("child_1");
        meta.add_child("child_2");

        assert_eq!(meta.child_session_ids.len(), 2);
        assert!(meta.child_session_ids.contains(&"child_1".to_string()));
        assert!(meta.child_session_ids.contains(&"child_2".to_string()));
    }

    #[test]
    fn test_custom_metadata() {
        let mut meta = SessionMetadata::new("session", "test", "Test", "Testing");

        meta.set_custom("key1", "value1");
        meta.set_custom("key2", serde_json::json!(42));

        assert_eq!(
            meta.get_custom("key1").and_then(|v| v.as_str()),
            Some("value1")
        );
        assert_eq!(
            meta.get_custom("key2").and_then(|v| v.as_i64()),
            Some(42)
        );
    }

    #[test]
    fn test_conversation_name() {
        let mut meta = SessionMetadata::new("session", "test", "Test", "Testing");

        // Initially no conversation name
        assert!(!meta.has_conversation_name());
        assert!(meta.conversation_name().is_none());

        // Set conversation name
        meta.set_conversation_name("Help with Rust code");

        assert!(meta.has_conversation_name());
        assert_eq!(meta.conversation_name(), Some("Help with Rust code"));
    }

    #[test]
    fn test_conversation_name_serialization() {
        let mut meta = SessionMetadata::new("session", "test", "Test", "Testing");
        meta.set_conversation_name("Debugging session");

        // Serialize
        let json = serde_json::to_string(&meta).unwrap();

        // Deserialize
        let loaded: SessionMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.conversation_name(), Some("Debugging session"));
    }

    #[test]
    fn test_conversation_name_skipped_when_none() {
        let meta = SessionMetadata::new("session", "test", "Test", "Testing");

        // Serialize without conversation_name
        let json = serde_json::to_string(&meta).unwrap();

        // conversation_name should not be in the JSON when None
        assert!(!json.contains("conversation_name"));
    }
}
