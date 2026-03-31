//! SubAgentManager - Tracks subagents spawned by a parent agent
//!
//! The `SubAgentManager` is stored in an agent's context and provides:
//! - Registration of subagent handles when spawned
//! - Access to subagent handles by session ID
//! - Tracking of completed subagents and their results
//!
//! # Example
//!
//! ```ignore
//! // In parent agent, get the manager from context
//! let manager = internals.context.get_resource::<SubAgentManager>().unwrap();
//!
//! // Access a subagent's handle
//! if let Some(handle) = manager.get("subagent-session-id") {
//!     // Subscribe to subagent output
//!     let rx = handle.subscribe();
//!     // Or check state
//!     let state = handle.state().await;
//! }
//!
//! // List all active subagents
//! for (session_id, handle) in manager.active_subagents() {
//!     println!("Subagent: {}", session_id);
//! }
//! ```

use std::collections::HashMap;
use std::sync::RwLock;

use super::handle::AgentHandle;

/// Tracks subagents spawned by a parent agent
///
/// This manager is automatically created and added to an agent's context
/// when it spawns subagents. It allows the parent agent to:
/// - Access subagent handles for monitoring
/// - Subscribe to subagent output streams
/// - Track completed subagents and their results
#[derive(Default)]
pub struct SubAgentManager {
    /// Active subagent handles keyed by session ID
    active: RwLock<HashMap<String, AgentHandle>>,

    /// Completed subagents with their final results
    completed: RwLock<HashMap<String, CompletedSubAgent>>,
}

/// Information about a completed subagent
#[derive(Debug, Clone)]
pub struct CompletedSubAgent {
    /// The session ID of the subagent
    pub session_id: String,

    /// The type/name of the subagent
    pub agent_type: String,

    /// Final result or summary (if any)
    pub result: Option<String>,

    /// Whether the subagent completed successfully
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

impl SubAgentManager {
    /// Create a new empty SubAgentManager
    pub fn new() -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            completed: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new subagent
    ///
    /// Called automatically when a subagent is spawned via the runtime.
    pub fn register(&self, session_id: impl Into<String>, handle: AgentHandle) {
        let session_id = session_id.into();
        tracing::debug!("[SubAgentManager] Registering subagent: {}", session_id);
        self.active.write().unwrap().insert(session_id, handle);
    }

    /// Get a subagent handle by session ID
    ///
    /// Returns None if the subagent doesn't exist or has completed.
    pub fn get(&self, session_id: &str) -> Option<AgentHandle> {
        self.active.read().unwrap().get(session_id).cloned()
    }

    /// Check if a subagent exists (active or completed)
    pub fn exists(&self, session_id: &str) -> bool {
        self.active.read().unwrap().contains_key(session_id)
            || self.completed.read().unwrap().contains_key(session_id)
    }

    /// Check if a subagent is still active
    pub fn is_active(&self, session_id: &str) -> bool {
        self.active.read().unwrap().contains_key(session_id)
    }

    /// Get all active subagent session IDs
    pub fn active_session_ids(&self) -> Vec<String> {
        self.active.read().unwrap().keys().cloned().collect()
    }

    /// Get all active subagents as (session_id, handle) pairs
    pub fn active_subagents(&self) -> Vec<(String, AgentHandle)> {
        self.active
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get the number of active subagents
    pub fn active_count(&self) -> usize {
        self.active.read().unwrap().len()
    }

    /// Mark a subagent as completed
    ///
    /// Moves the subagent from active to completed status.
    pub fn mark_completed(
        &self,
        session_id: &str,
        agent_type: impl Into<String>,
        result: Option<String>,
        success: bool,
        error: Option<String>,
    ) {
        // Remove from active
        self.active.write().unwrap().remove(session_id);

        // Add to completed
        let completed = CompletedSubAgent {
            session_id: session_id.to_string(),
            agent_type: agent_type.into(),
            result,
            success,
            error,
        };
        self.completed
            .write()
            .unwrap()
            .insert(session_id.to_string(), completed);

        tracing::debug!(
            "[SubAgentManager] Subagent {} marked as completed (success={})",
            session_id,
            success
        );
    }

    /// Get information about a completed subagent
    pub fn get_completed(&self, session_id: &str) -> Option<CompletedSubAgent> {
        self.completed.read().unwrap().get(session_id).cloned()
    }

    /// Get all completed subagents
    pub fn completed_subagents(&self) -> Vec<CompletedSubAgent> {
        self.completed.read().unwrap().values().cloned().collect()
    }

    /// Get the total number of subagents (active + completed)
    pub fn total_count(&self) -> usize {
        self.active.read().unwrap().len() + self.completed.read().unwrap().len()
    }

    /// Remove a subagent from tracking entirely
    ///
    /// Use this to clean up completed subagents when no longer needed.
    pub fn remove(&self, session_id: &str) {
        self.active.write().unwrap().remove(session_id);
        self.completed.write().unwrap().remove(session_id);
    }

    /// Clear all completed subagents
    pub fn clear_completed(&self) {
        self.completed.write().unwrap().clear();
    }
}

impl std::fmt::Debug for SubAgentManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let active_count = self.active.read().unwrap().len();
        let completed_count = self.completed.read().unwrap().len();
        f.debug_struct("SubAgentManager")
            .field("active_count", &active_count)
            .field("completed_count", &completed_count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AgentState;
    use crate::runtime::channels::create_agent_channels;
    use crate::session::{AgentSession, SessionStorage};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::RwLock as TokioRwLock;

    fn create_test_handle(session_id: &str) -> (AgentHandle, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = SessionStorage::with_dir(temp_dir.path());
        let session = AgentSession::new_with_storage(
            session_id,
            "test-agent",
            "Test",
            "Test",
            "",
            storage,
        )
        .unwrap();
        let session = Arc::new(TokioRwLock::new(session));

        let (input_tx, _input_rx, output_tx) = create_agent_channels();
        let state = Arc::new(TokioRwLock::new(AgentState::Idle));
        let handle = AgentHandle::new(session_id.to_string(), session, input_tx, output_tx, state);
        (handle, temp_dir)
    }

    #[test]
    fn test_register_and_get() {
        let manager = SubAgentManager::new();
        let (handle, _temp) = create_test_handle("sub-1");

        manager.register("sub-1", handle.clone());

        assert!(manager.exists("sub-1"));
        assert!(manager.is_active("sub-1"));
        assert!(manager.get("sub-1").is_some());
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_mark_completed() {
        let manager = SubAgentManager::new();
        let (handle, _temp) = create_test_handle("sub-1");

        manager.register("sub-1", handle);
        manager.mark_completed("sub-1", "test-agent", Some("Done".to_string()), true, None);

        assert!(!manager.is_active("sub-1"));
        assert!(manager.exists("sub-1"));
        assert!(manager.get("sub-1").is_none()); // No longer in active

        let completed = manager.get_completed("sub-1").unwrap();
        assert_eq!(completed.result, Some("Done".to_string()));
        assert!(completed.success);
    }

    #[test]
    fn test_active_session_ids() {
        let manager = SubAgentManager::new();
        let (handle1, _temp1) = create_test_handle("sub-1");
        manager.register("sub-1", handle1);
        let (handle2, _temp2) = create_test_handle("sub-2");
        manager.register("sub-2", handle2);

        let ids = manager.active_session_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"sub-1".to_string()));
        assert!(ids.contains(&"sub-2".to_string()));
    }

    #[test]
    fn test_remove() {
        let manager = SubAgentManager::new();
        let (handle1, _temp1) = create_test_handle("sub-1");
        manager.register("sub-1", handle1);
        manager.mark_completed("sub-1", "test", None, true, None);

        assert!(manager.exists("sub-1"));
        manager.remove("sub-1");
        assert!(!manager.exists("sub-1"));
    }
}
