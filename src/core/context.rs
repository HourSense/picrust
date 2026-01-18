//! Agent context - hidden state passed to tools

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// ============================================================================
// ResourceMap - Type-safe container for agent-specific resources
// ============================================================================

/// Type-safe container for agent-specific resources
///
/// This allows tools to access agent-specific state like TodoManager,
/// FileWatcher, or any other Rust object that the agent needs.
///
/// # Example
///
/// ```ignore
/// // Insert a resource
/// ctx.resources.insert(TodoManager::new());
///
/// // Get a resource in a tool
/// let todo = ctx.resources.get::<TodoManager>()
///     .expect("TodoManager not available");
/// todo.add_task("Fix bug");
/// ```
#[derive(Default, Clone)]
pub struct ResourceMap {
    map: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl ResourceMap {
    /// Create a new empty resource map
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Insert a resource by type
    ///
    /// If a resource of this type already exists, it will be replaced.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// Insert an Arc-wrapped resource by type
    ///
    /// Use this when you already have an Arc and want to share it.
    pub fn insert_arc<T: Send + Sync + 'static>(&mut self, value: Arc<T>) {
        self.map.insert(TypeId::of::<T>(), value);
    }

    /// Get a resource by type
    ///
    /// Returns `Some(Arc<T>)` if the resource exists, `None` otherwise.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|arc| arc.clone().downcast::<T>().ok())
    }

    /// Check if a resource of the given type exists
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    /// Remove a resource by type
    ///
    /// Returns the removed resource if it existed.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<Arc<T>> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|arc| arc.downcast::<T>().ok())
    }

    /// Get the number of resources stored
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the resource map is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear all resources
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

impl fmt::Debug for ResourceMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceMap")
            .field("count", &self.map.len())
            .finish()
    }
}

// ============================================================================
// AgentContext - Hidden state passed to tools
// ============================================================================

/// Hidden context passed to tools during execution
///
/// This context is NOT exposed in the tool's JSON schema to the LLM.
/// It provides tools with access to agent state, session info, lineage,
/// and agent-specific resources.
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentContext {
    // --- Identity ---
    /// Unique session ID for this agent
    pub session_id: String,

    /// Type of agent (e.g., "coder", "researcher", "main")
    pub agent_type: String,

    /// Human-readable name for this agent (set at runtime)
    pub name: String,

    /// Description of what this agent does (set at runtime)
    pub description: String,

    // --- Lineage (for subagents) ---
    /// Parent agent's session ID (if this is a subagent)
    pub parent_session_id: Option<String>,

    /// The tool_use_id that spawned this agent (if subagent)
    pub parent_tool_use_id: Option<String>,

    // --- Current Execution State ---
    /// Current turn number (increments each LLM call)
    pub current_turn: usize,

    /// Current tool_use_id being executed (set during tool execution)
    pub current_tool_use_id: Option<String>,

    // --- Extensible Metadata (JSON-serializable) ---
    /// Custom metadata that can be set by agent logic (JSON values only)
    #[serde(default)]
    pub metadata: HashMap<String, Value>,

    // --- Agent Resources (runtime objects) ---
    /// Agent-specific resources like TodoManager, FileWatcher, etc.
    /// These are NOT serialized - they exist only at runtime.
    #[serde(skip)]
    pub resources: ResourceMap,
}

impl fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentContext")
            .field("session_id", &self.session_id)
            .field("agent_type", &self.agent_type)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parent_session_id", &self.parent_session_id)
            .field("parent_tool_use_id", &self.parent_tool_use_id)
            .field("current_turn", &self.current_turn)
            .field("current_tool_use_id", &self.current_tool_use_id)
            .field("metadata", &self.metadata)
            .field("resources", &self.resources)
            .finish()
    }
}

impl AgentContext {
    /// Create a new context for a root agent (not a subagent)
    pub fn new(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            agent_type: agent_type.into(),
            name: name.into(),
            description: description.into(),
            parent_session_id: None,
            parent_tool_use_id: None,
            current_turn: 0,
            current_tool_use_id: None,
            metadata: HashMap::new(),
            resources: ResourceMap::new(),
        }
    }

    /// Create a new context for a subagent
    pub fn new_subagent(
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_tool_use_id: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            agent_type: agent_type.into(),
            name: name.into(),
            description: description.into(),
            parent_session_id: Some(parent_session_id.into()),
            parent_tool_use_id: Some(parent_tool_use_id.into()),
            current_turn: 0,
            current_tool_use_id: None,
            metadata: HashMap::new(),
            resources: ResourceMap::new(),
        }
    }

    /// Check if this agent is a subagent
    pub fn is_subagent(&self) -> bool {
        self.parent_session_id.is_some()
    }

    /// Increment the turn counter
    pub fn next_turn(&mut self) {
        self.current_turn += 1;
    }

    /// Create a copy with the current tool_use_id set
    ///
    /// Used when executing a tool so the tool knows its own ID.
    /// Note: Resources are cloned (Arc references are shared).
    pub fn with_tool_use_id(&self, tool_use_id: impl Into<String>) -> Self {
        let mut ctx = self.clone();
        ctx.current_tool_use_id = Some(tool_use_id.into());
        ctx
    }

    /// Clear the current tool_use_id
    pub fn clear_tool_use_id(&mut self) {
        self.current_tool_use_id = None;
    }

    // --- Metadata Methods (JSON-serializable data) ---

    /// Set a metadata value (must be JSON-serializable)
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Get a metadata value as a string
    pub fn get_metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Remove a metadata value
    pub fn remove_metadata(&mut self, key: &str) -> Option<Value> {
        self.metadata.remove(key)
    }

    /// Check if a metadata key exists
    pub fn has_metadata(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }

    // --- Resource Methods (runtime objects) ---

    /// Insert a resource by type
    ///
    /// # Example
    /// ```ignore
    /// ctx.insert_resource(TodoManager::new());
    /// ```
    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) {
        self.resources.insert(value);
    }

    /// Insert an already Arc-wrapped resource by type
    ///
    /// Use this when the resource is already wrapped in Arc (e.g., shared with other components).
    ///
    /// # Example
    /// ```ignore
    /// let manager = Arc::new(TodoManager::new());
    /// ctx.insert_resource_arc(manager.clone());
    /// ```
    pub fn insert_resource_arc<T: Send + Sync + 'static>(&mut self, value: Arc<T>) {
        self.resources.insert_arc(value);
    }

    /// Get a resource by type
    ///
    /// # Example
    /// ```ignore
    /// let todo = ctx.get_resource::<TodoManager>();
    /// ```
    pub fn get_resource<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.resources.get::<T>()
    }

    /// Check if a resource of the given type exists
    pub fn has_resource<T: Send + Sync + 'static>(&self) -> bool {
        self.resources.contains::<T>()
    }

    /// Remove a resource by type
    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<Arc<T>> {
        self.resources.remove::<T>()
    }
}

impl Default for AgentContext {
    fn default() -> Self {
        Self::new("unknown", "unknown", "Unnamed Agent", "No description")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_context() {
        let ctx = AgentContext::new(
            "session_123",
            "coder",
            "My Coder Agent",
            "An agent that writes code",
        );
        assert_eq!(ctx.session_id, "session_123");
        assert_eq!(ctx.agent_type, "coder");
        assert_eq!(ctx.name, "My Coder Agent");
        assert_eq!(ctx.description, "An agent that writes code");
        assert!(!ctx.is_subagent());
        assert_eq!(ctx.current_turn, 0);
        assert!(ctx.current_tool_use_id.is_none());
    }

    #[test]
    fn test_subagent_context() {
        let ctx = AgentContext::new_subagent(
            "sub_session",
            "researcher",
            "Research Assistant",
            "Finds information",
            "parent_session",
            "tool_456",
        );
        assert!(ctx.is_subagent());
        assert_eq!(ctx.name, "Research Assistant");
        assert_eq!(ctx.parent_session_id, Some("parent_session".into()));
        assert_eq!(ctx.parent_tool_use_id, Some("tool_456".into()));
    }

    #[test]
    fn test_with_tool_use_id() {
        let ctx = AgentContext::new("session", "test", "Test Agent", "For testing");
        let ctx_with_tool = ctx.with_tool_use_id("tool_789");

        // Original unchanged
        assert!(ctx.current_tool_use_id.is_none());
        // New has tool ID
        assert_eq!(ctx_with_tool.current_tool_use_id, Some("tool_789".into()));
    }

    #[test]
    fn test_metadata() {
        let mut ctx = AgentContext::new("session", "test", "Test", "Test agent");

        ctx.set_metadata("key1", "value1");
        ctx.set_metadata("key2", serde_json::json!(42));

        assert_eq!(ctx.get_metadata_str("key1"), Some("value1"));
        assert_eq!(
            ctx.get_metadata("key2").and_then(|v| v.as_i64()),
            Some(42)
        );
        assert!(ctx.has_metadata("key1"));
        assert!(!ctx.has_metadata("nonexistent"));

        ctx.remove_metadata("key1");
        assert!(!ctx.has_metadata("key1"));
    }

    #[test]
    fn test_turn_counter() {
        let mut ctx = AgentContext::new("session", "test", "Test", "Test agent");
        assert_eq!(ctx.current_turn, 0);

        ctx.next_turn();
        assert_eq!(ctx.current_turn, 1);

        ctx.next_turn();
        assert_eq!(ctx.current_turn, 2);
    }

    // --- ResourceMap Tests ---

    #[derive(Debug, Clone, PartialEq)]
    struct TestResource {
        value: i32,
    }

    #[derive(Debug)]
    struct AnotherResource {
        name: String,
    }

    #[test]
    fn test_resource_map_insert_get() {
        let mut resources = ResourceMap::new();

        resources.insert(TestResource { value: 42 });
        resources.insert(AnotherResource {
            name: "test".to_string(),
        });

        let test_res = resources.get::<TestResource>().unwrap();
        assert_eq!(test_res.value, 42);

        let another_res = resources.get::<AnotherResource>().unwrap();
        assert_eq!(another_res.name, "test");
    }

    #[test]
    fn test_resource_map_missing() {
        let resources = ResourceMap::new();

        assert!(resources.get::<TestResource>().is_none());
        assert!(!resources.contains::<TestResource>());
    }

    #[test]
    fn test_resource_map_replace() {
        let mut resources = ResourceMap::new();

        resources.insert(TestResource { value: 1 });
        resources.insert(TestResource { value: 2 }); // Replace

        let res = resources.get::<TestResource>().unwrap();
        assert_eq!(res.value, 2);
    }

    #[test]
    fn test_resource_map_remove() {
        let mut resources = ResourceMap::new();

        resources.insert(TestResource { value: 42 });
        assert!(resources.contains::<TestResource>());

        let removed = resources.remove::<TestResource>().unwrap();
        assert_eq!(removed.value, 42);
        assert!(!resources.contains::<TestResource>());
    }

    #[test]
    fn test_context_resources() {
        let mut ctx = AgentContext::new("session", "test", "Test", "Test agent");

        // Insert resources
        ctx.insert_resource(TestResource { value: 100 });

        // Get resources
        assert!(ctx.has_resource::<TestResource>());
        let res = ctx.get_resource::<TestResource>().unwrap();
        assert_eq!(res.value, 100);

        // Resources are shared via Arc when context is cloned
        let ctx2 = ctx.clone();
        let res2 = ctx2.get_resource::<TestResource>().unwrap();
        assert_eq!(res2.value, 100);

        // They point to the same Arc
        assert!(Arc::ptr_eq(&res, &res2));
    }

    #[test]
    fn test_resource_map_len() {
        let mut resources = ResourceMap::new();
        assert!(resources.is_empty());
        assert_eq!(resources.len(), 0);

        resources.insert(TestResource { value: 1 });
        assert!(!resources.is_empty());
        assert_eq!(resources.len(), 1);

        resources.insert(AnotherResource {
            name: "x".to_string(),
        });
        assert_eq!(resources.len(), 2);

        resources.clear();
        assert!(resources.is_empty());
    }

    #[test]
    fn test_context_serialization_skips_resources() {
        let mut ctx = AgentContext::new("session", "test", "Test", "Test agent");
        ctx.insert_resource(TestResource { value: 42 });
        ctx.set_metadata("key", "value");

        // Serialize
        let json = serde_json::to_string(&ctx).unwrap();

        // Deserialize
        let ctx2: AgentContext = serde_json::from_str(&json).unwrap();

        // Metadata is preserved
        assert_eq!(ctx2.get_metadata_str("key"), Some("value"));

        // Resources are NOT preserved (skipped during serialization)
        assert!(!ctx2.has_resource::<TestResource>());
        assert!(ctx2.resources.is_empty());
    }
}
