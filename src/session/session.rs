//! Agent session management
//!
//! The `AgentSession` struct combines metadata and message history,
//! providing a complete view of an agent's conversation state.

use crate::core::FrameworkResult;
use crate::llm::Message;

use super::metadata::SessionMetadata;
use super::storage::SessionStorage;

/// An agent session that tracks conversation history and metadata
///
/// Each agent has its own session, identified by a unique session_id.
/// Sessions can be linked via parent/child relationships for subagent tracking.
#[derive(Debug)]
pub struct AgentSession {
    /// Session metadata (identity, lineage, timestamps)
    pub metadata: SessionMetadata,

    /// Conversation history
    pub messages: Vec<Message>,

    /// The agent's system prompt — loaded from/persisted to system_prompt.md
    system_prompt: String,

    /// Storage backend for persistence
    storage: SessionStorage,
}

impl AgentSession {
    /// Create a new root agent session
    ///
    /// This creates a new session with the given metadata and empty history.
    /// The system prompt is written to `system_prompt.md` in the session folder immediately.
    pub fn new(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> FrameworkResult<Self> {
        let storage = SessionStorage::new();
        let metadata = SessionMetadata::new(session_id, agent_type, name, description);
        let system_prompt = system_prompt.into();

        storage.save_metadata(&metadata)?;
        storage.save_system_prompt(&metadata.session_id, &system_prompt)?;

        Ok(Self {
            metadata,
            messages: Vec::new(),
            system_prompt,
            storage,
        })
    }

    /// Create a new root agent session with custom storage
    pub fn new_with_storage(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
        storage: SessionStorage,
    ) -> FrameworkResult<Self> {
        let metadata = SessionMetadata::new(session_id, agent_type, name, description);
        let system_prompt = system_prompt.into();

        storage.save_metadata(&metadata)?;
        storage.save_system_prompt(&metadata.session_id, &system_prompt)?;

        Ok(Self {
            metadata,
            messages: Vec::new(),
            system_prompt,
            storage,
        })
    }

    /// Create a new subagent session
    ///
    /// This creates a session that is linked to a parent session.
    /// The parent session is automatically updated to track this child.
    pub fn new_subagent(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_tool_use_id: impl Into<String>,
    ) -> FrameworkResult<Self> {
        let storage = SessionStorage::new();
        let parent_id = parent_session_id.into();
        let system_prompt = system_prompt.into();

        let metadata = SessionMetadata::new_subagent(
            session_id,
            agent_type,
            name,
            description,
            &parent_id,
            parent_tool_use_id,
        );

        storage.save_metadata(&metadata)?;
        storage.save_system_prompt(&metadata.session_id, &system_prompt)?;

        // Update parent to track this child
        if let Ok(mut parent_meta) = storage.load_metadata(&parent_id) {
            parent_meta.add_child(&metadata.session_id);
            storage.save_metadata(&parent_meta)?;
        }

        Ok(Self {
            metadata,
            messages: Vec::new(),
            system_prompt,
            storage,
        })
    }

    /// Create a new subagent session with custom storage
    pub fn new_subagent_with_storage(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_tool_use_id: impl Into<String>,
        storage: SessionStorage,
    ) -> FrameworkResult<Self> {
        let parent_id = parent_session_id.into();
        let system_prompt = system_prompt.into();

        let metadata = SessionMetadata::new_subagent(
            session_id,
            agent_type,
            name,
            description,
            &parent_id,
            parent_tool_use_id,
        );

        storage.save_metadata(&metadata)?;
        storage.save_system_prompt(&metadata.session_id, &system_prompt)?;

        // Update parent to track this child
        if let Ok(mut parent_meta) = storage.load_metadata(&parent_id) {
            parent_meta.add_child(&metadata.session_id);
            storage.save_metadata(&parent_meta)?;
        }

        Ok(Self {
            metadata,
            messages: Vec::new(),
            system_prompt,
            storage,
        })
    }

    /// Load an existing session from storage
    pub fn load(session_id: &str) -> FrameworkResult<Self> {
        let storage = SessionStorage::new();
        Self::load_with_storage(session_id, storage)
    }

    /// Load an existing session with custom storage
    pub fn load_with_storage(session_id: &str, storage: SessionStorage) -> FrameworkResult<Self> {
        let metadata = storage.load_metadata(session_id)?;
        let messages = storage.load_messages(session_id)?;
        let system_prompt = storage.load_system_prompt(session_id)?;

        Ok(Self {
            metadata,
            messages,
            system_prompt,
            storage,
        })
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.metadata.session_id
    }

    /// Get the agent type
    pub fn agent_type(&self) -> &str {
        &self.metadata.agent_type
    }

    /// Get the agent name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get the agent description
    pub fn description(&self) -> &str {
        &self.metadata.description
    }

    /// Get the current system prompt
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Update the system prompt in memory and persist to system_prompt.md
    pub fn update_system_prompt(&mut self, prompt: impl Into<String>) -> FrameworkResult<()> {
        self.system_prompt = prompt.into();
        self.storage.save_system_prompt(&self.metadata.session_id, &self.system_prompt)?;
        Ok(())
    }

    /// Check if this is a subagent session
    pub fn is_subagent(&self) -> bool {
        self.metadata.is_subagent()
    }

    /// Get the parent session ID (if this is a subagent)
    pub fn parent_session_id(&self) -> Option<&str> {
        self.metadata.parent_session_id.as_deref()
    }

    /// Get child session IDs
    pub fn child_session_ids(&self) -> &[String] {
        &self.metadata.child_session_ids
    }

    /// Add a message to the conversation history
    ///
    /// The message is immediately persisted to disk.
    pub fn add_message(&mut self, message: Message) -> FrameworkResult<()> {
        self.storage
            .append_message(&self.metadata.session_id, &message)?;
        self.messages.push(message);
        self.metadata.touch();
        self.storage.save_metadata(&self.metadata)?;
        Ok(())
    }

    /// Get the conversation history
    pub fn history(&self) -> &[Message] {
        &self.messages
    }

    /// Get a mutable reference to the conversation history
    ///
    /// Note: Changes made directly to this vector are not automatically persisted.
    /// Call `save()` after making changes.
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    /// Save the entire session (metadata and messages)
    ///
    /// This overwrites the existing history file.
    pub fn save(&mut self) -> FrameworkResult<()> {
        self.metadata.touch();
        self.storage.save_metadata(&self.metadata)?;
        self.storage
            .save_messages(&self.metadata.session_id, &self.messages)?;
        Ok(())
    }

    /// Reload the session from storage
    ///
    /// This discards any unsaved changes and reloads from disk.
    pub fn reload(&mut self) -> FrameworkResult<()> {
        self.metadata = self.storage.load_metadata(&self.metadata.session_id)?;
        self.messages = self.storage.load_messages(&self.metadata.session_id)?;
        self.system_prompt = self.storage.load_system_prompt(&self.metadata.session_id)?;
        Ok(())
    }

    /// Delete this session from storage
    ///
    /// Warning: This permanently deletes the session data.
    pub fn delete(self) -> FrameworkResult<()> {
        self.storage.delete_session(&self.metadata.session_id)
    }

    /// Get the storage backend
    pub fn storage(&self) -> &SessionStorage {
        &self.storage
    }

    /// Set the model for this session
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.metadata.model = model.into();
        self.metadata.touch();
    }

    /// Get the model for this session
    pub fn model(&self) -> &str {
        &self.metadata.model
    }

    /// Set the provider for this session
    pub fn set_provider(&mut self, provider: impl Into<String>) {
        self.metadata.provider = provider.into();
        self.metadata.touch();
    }

    /// Get the provider for this session
    pub fn provider(&self) -> &str {
        &self.metadata.provider
    }

    /// Set the conversation name
    ///
    /// This is typically called by a conversation namer helper after the first
    /// turn to give the conversation a descriptive name based on its content.
    /// The name is persisted to disk immediately.
    pub fn set_conversation_name(&mut self, name: impl Into<String>) -> FrameworkResult<()> {
        self.metadata.set_conversation_name(name);
        self.storage.save_metadata(&self.metadata)?;
        Ok(())
    }

    /// Get the conversation name
    pub fn conversation_name(&self) -> Option<&str> {
        self.metadata.conversation_name()
    }

    /// Check if the conversation has been named
    pub fn has_conversation_name(&self) -> bool {
        self.metadata.has_conversation_name()
    }

    /// Set custom metadata
    pub fn set_custom<T: Into<serde_json::Value>>(&mut self, key: impl Into<String>, value: T) {
        self.metadata.set_custom(key, value);
    }

    /// Get custom metadata
    pub fn get_custom(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get_custom(key)
    }

    /// List all sessions in storage
    pub fn list_all() -> FrameworkResult<Vec<String>> {
        SessionStorage::new().list_sessions()
    }

    /// List all sessions with custom storage
    pub fn list_all_with_storage(storage: &SessionStorage) -> FrameworkResult<Vec<String>> {
        storage.list_sessions()
    }

    /// List sessions with optional filtering
    ///
    /// If `top_level_only` is true, only returns sessions that are not subagents.
    pub fn list_filtered(top_level_only: bool) -> FrameworkResult<Vec<String>> {
        SessionStorage::new().list_sessions_filtered(top_level_only)
    }

    /// List sessions with optional filtering and custom storage
    pub fn list_filtered_with_storage(
        top_level_only: bool,
        storage: &SessionStorage,
    ) -> FrameworkResult<Vec<String>> {
        storage.list_sessions_filtered(top_level_only)
    }

    /// List only top-level sessions (not subagents)
    pub fn list_top_level() -> FrameworkResult<Vec<String>> {
        SessionStorage::new().list_top_level_sessions()
    }

    /// List only top-level sessions with custom storage
    pub fn list_top_level_with_storage(storage: &SessionStorage) -> FrameworkResult<Vec<String>> {
        storage.list_top_level_sessions()
    }

    /// List sessions with their metadata
    ///
    /// Returns tuples of (session_id, metadata) for all valid sessions.
    /// If `top_level_only` is true, only includes sessions that are not subagents.
    pub fn list_with_metadata(
        top_level_only: bool,
    ) -> FrameworkResult<Vec<(String, SessionMetadata)>> {
        SessionStorage::new().list_sessions_with_metadata(top_level_only)
    }

    /// List sessions with metadata using custom storage
    pub fn list_with_metadata_and_storage(
        top_level_only: bool,
        storage: &SessionStorage,
    ) -> FrameworkResult<Vec<(String, SessionMetadata)>> {
        storage.list_sessions_with_metadata(top_level_only)
    }

    /// Get conversation history for a session by ID
    ///
    /// This is a convenience method that loads only the messages without
    /// loading the full session object.
    pub fn get_history(session_id: &str) -> FrameworkResult<Vec<Message>> {
        SessionStorage::new().load_messages(session_id)
    }

    /// Get conversation history with custom storage
    pub fn get_history_with_storage(
        session_id: &str,
        storage: &SessionStorage,
    ) -> FrameworkResult<Vec<Message>> {
        storage.load_messages(session_id)
    }

    /// Get session metadata by ID
    ///
    /// This is a convenience method that loads only the metadata without
    /// loading the full session object.
    pub fn get_metadata(session_id: &str) -> FrameworkResult<SessionMetadata> {
        SessionStorage::new().load_metadata(session_id)
    }

    /// Get session metadata with custom storage
    pub fn get_metadata_with_storage(
        session_id: &str,
        storage: &SessionStorage,
    ) -> FrameworkResult<SessionMetadata> {
        storage.load_metadata(session_id)
    }

    /// Check if a session exists
    pub fn exists(session_id: &str) -> bool {
        SessionStorage::new().session_exists(session_id)
    }

    /// Check if a session exists with custom storage
    pub fn exists_with_storage(session_id: &str, storage: &SessionStorage) -> bool {
        storage.session_exists(session_id)
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
    fn test_new_session() {
        let (storage, _temp) = create_test_storage();

        let session =
            AgentSession::new_with_storage("test_session", "coder", "Test Coder", "A test agent", "You are a coder.", storage).unwrap();

        assert_eq!(session.session_id(), "test_session");
        assert_eq!(session.agent_type(), "coder");
        assert_eq!(session.name(), "Test Coder");
        assert_eq!(session.description(), "A test agent");
        assert!(!session.is_subagent());
        assert!(session.history().is_empty());
    }

    #[test]
    fn test_subagent_session() {
        let (storage, _temp) = create_test_storage();

        // Create parent first
        let _parent =
            AgentSession::new_with_storage("parent", "main", "Main Agent", "Parent agent", "Test system prompt.", storage.clone())
                .unwrap();

        // Create subagent
        let subagent = AgentSession::new_subagent_with_storage(
            "sub_session",
            "researcher",
            "Research Agent",
            "Finds info",
            "Test system prompt.",
            "parent",
            "tool_123",
            storage.clone(),
        )
        .unwrap();

        assert!(subagent.is_subagent());
        assert_eq!(subagent.parent_session_id(), Some("parent"));

        // Verify parent was updated
        let reloaded_parent = AgentSession::load_with_storage("parent", storage).unwrap();
        assert!(reloaded_parent
            .child_session_ids()
            .contains(&"sub_session".to_string()));
    }

    #[test]
    fn test_add_and_get_messages() {
        let (storage, _temp) = create_test_storage();

        let mut session =
            AgentSession::new_with_storage("msg_session", "coder", "Test", "Testing", "Test system prompt.", storage.clone())
                .unwrap();

        // Add messages
        session.add_message(Message::user("Hello")).unwrap();
        session.add_message(Message::assistant("Hi there")).unwrap();

        assert_eq!(session.history().len(), 2);

        // Reload and verify persistence
        let reloaded = AgentSession::load_with_storage("msg_session", storage).unwrap();
        assert_eq!(reloaded.history().len(), 2);
    }

    #[test]
    fn test_save_and_reload() {
        let (storage, _temp) = create_test_storage();

        let mut session =
            AgentSession::new_with_storage("save_test", "coder", "Test", "Testing", "Test system prompt.", storage.clone())
                .unwrap();

        // Add messages directly (bypassing append)
        session.messages.push(Message::user("Direct add"));
        session.save().unwrap();

        // Reload and verify
        let reloaded = AgentSession::load_with_storage("save_test", storage).unwrap();
        assert_eq!(reloaded.history().len(), 1);
    }

    #[test]
    fn test_delete_session() {
        let (storage, _temp) = create_test_storage();

        let session =
            AgentSession::new_with_storage("to_delete", "coder", "Test", "Testing", "Test system prompt.", storage.clone())
                .unwrap();

        assert!(AgentSession::exists_with_storage("to_delete", &storage));

        session.delete().unwrap();

        assert!(!AgentSession::exists_with_storage("to_delete", &storage));
    }

    #[test]
    fn test_list_sessions() {
        let (storage, _temp) = create_test_storage();

        let _s1 =
            AgentSession::new_with_storage("session1", "coder", "S1", "D1", "Test system prompt.", storage.clone()).unwrap();
        let _s2 =
            AgentSession::new_with_storage("session2", "researcher", "S2", "D2", "Test system prompt.", storage.clone())
                .unwrap();

        let sessions = AgentSession::list_all_with_storage(&storage).unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains(&"session1".to_string()));
        assert!(sessions.contains(&"session2".to_string()));
    }

    #[test]
    fn test_custom_metadata() {
        let (storage, _temp) = create_test_storage();

        let mut session =
            AgentSession::new_with_storage("custom_test", "coder", "Test", "Testing", "Test system prompt.", storage.clone())
                .unwrap();

        session.set_custom("key1", "value1");
        session.set_custom("count", serde_json::json!(42));
        session.save().unwrap();

        // Reload and verify
        let reloaded = AgentSession::load_with_storage("custom_test", storage).unwrap();
        assert_eq!(
            reloaded.get_custom("key1").and_then(|v| v.as_str()),
            Some("value1")
        );
        assert_eq!(
            reloaded.get_custom("count").and_then(|v| v.as_i64()),
            Some(42)
        );
    }

    #[test]
    fn test_model_and_provider() {
        let (storage, _temp) = create_test_storage();

        let mut session =
            AgentSession::new_with_storage("model_test", "coder", "Test", "Testing", "Test system prompt.", storage).unwrap();

        session.set_model("claude-opus-4-5-20251101");
        session.set_provider("anthropic");

        assert_eq!(session.model(), "claude-opus-4-5-20251101");
        assert_eq!(session.provider(), "anthropic");
    }

    #[test]
    fn test_conversation_name() {
        let (storage, _temp) = create_test_storage();

        let mut session = AgentSession::new_with_storage(
            "conv_name_test",
            "coder",
            "Test",
            "Testing",
            "Test system prompt.",
            storage.clone(),
        )
        .unwrap();

        // Initially no conversation name
        assert!(!session.has_conversation_name());
        assert!(session.conversation_name().is_none());

        // Set conversation name
        session
            .set_conversation_name("Help with Rust code")
            .unwrap();

        assert!(session.has_conversation_name());
        assert_eq!(session.conversation_name(), Some("Help with Rust code"));

        // Reload and verify persistence
        let reloaded = AgentSession::load_with_storage("conv_name_test", storage).unwrap();
        assert_eq!(reloaded.conversation_name(), Some("Help with Rust code"));
    }

    #[test]
    fn test_list_filtered() {
        let (storage, _temp) = create_test_storage();

        // Create top-level sessions
        let _parent1 =
            AgentSession::new_with_storage("parent1", "main", "Parent 1", "First parent", "Test system prompt.", storage.clone())
                .unwrap();
        let _parent2 =
            AgentSession::new_with_storage("parent2", "main", "Parent 2", "Second parent", "Test system prompt.", storage.clone())
                .unwrap();

        // Create a subagent
        let _child = AgentSession::new_subagent_with_storage(
            "child1",
            "helper",
            "Child 1",
            "A child agent",
            "Test system prompt.",
            "parent1",
            "tool_123",
            storage.clone(),
        )
        .unwrap();

        // List all
        let all = AgentSession::list_filtered_with_storage(false, &storage).unwrap();
        assert_eq!(all.len(), 3);

        // List top-level only
        let top_level = AgentSession::list_filtered_with_storage(true, &storage).unwrap();
        assert_eq!(top_level.len(), 2);
        assert!(top_level.contains(&"parent1".to_string()));
        assert!(top_level.contains(&"parent2".to_string()));
        assert!(!top_level.contains(&"child1".to_string()));

        // Test convenience method
        let top_level2 = AgentSession::list_top_level_with_storage(&storage).unwrap();
        assert_eq!(top_level2.len(), 2);
    }

    #[test]
    fn test_list_with_metadata() {
        let (storage, _temp) = create_test_storage();

        let _main = AgentSession::new_with_storage(
            "main_agent",
            "coder",
            "Main Agent",
            "Main agent description",
            "Test system prompt.",
            storage.clone(),
        )
        .unwrap();

        let _sub = AgentSession::new_subagent_with_storage(
            "sub_agent",
            "researcher",
            "Sub Agent",
            "Sub agent description",
            "Test system prompt.",
            "main_agent",
            "tool_456",
            storage.clone(),
        )
        .unwrap();

        // All sessions with metadata
        let all = AgentSession::list_with_metadata_and_storage(false, &storage).unwrap();
        assert_eq!(all.len(), 2);

        // Top-level only with metadata
        let top_level = AgentSession::list_with_metadata_and_storage(true, &storage).unwrap();
        assert_eq!(top_level.len(), 1);
        assert_eq!(top_level[0].0, "main_agent");
        assert_eq!(top_level[0].1.name, "Main Agent");
    }

    #[test]
    fn test_get_history() {
        let (storage, _temp) = create_test_storage();

        let mut session = AgentSession::new_with_storage(
            "history_test",
            "coder",
            "Test",
            "Testing",
            "Test system prompt.",
            storage.clone(),
        )
        .unwrap();

        // Add some messages
        session.add_message(Message::user("Hello")).unwrap();
        session.add_message(Message::assistant("Hi there!")).unwrap();
        session.add_message(Message::user("How are you?")).unwrap();

        // Get history using static method
        let history = AgentSession::get_history_with_storage("history_test", &storage).unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_get_metadata() {
        let (storage, _temp) = create_test_storage();

        let _session = AgentSession::new_with_storage(
            "meta_test",
            "researcher",
            "Research Agent",
            "Finds information",
            "Test system prompt.",
            storage.clone(),
        )
        .unwrap();

        // Get metadata using static method
        let metadata = AgentSession::get_metadata_with_storage("meta_test", &storage).unwrap();
        assert_eq!(metadata.session_id, "meta_test");
        assert_eq!(metadata.agent_type, "researcher");
        assert_eq!(metadata.name, "Research Agent");
        assert!(!metadata.is_subagent());
    }
}
