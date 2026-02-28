//! AgentRuntime - Spawns and manages agent tasks
//!
//! The `AgentRuntime` is responsible for:
//! - Spawning agents as tokio tasks
//! - Creating channels and returning handles
//! - Tracking running agents
//! - Providing shutdown methods
//! - Sharing global permissions across all agents

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::{AgentContext, AgentState, FrameworkError, FrameworkResult};
use crate::permissions::{GlobalPermissions, PermissionManager, PermissionRule};
use crate::session::AgentSession;

use super::channels::create_agent_channels;
use super::handle::AgentHandle;
use super::internals::AgentInternals;
use super::subagent_manager::SubAgentManager;

/// Runtime for spawning and managing agents
///
/// The runtime maintains a registry of running agents and provides
/// methods to spawn, query, and shutdown agents.
///
/// All agents spawned by this runtime share the same `GlobalPermissions`,
/// so permission rules added to global scope are immediately visible to all agents.
#[derive(Clone)]
pub struct AgentRuntime {
    /// Map of session_id -> AgentHandle for running agents
    agents: Arc<RwLock<HashMap<String, AgentHandle>>>,
    /// Shared global permissions for all agents
    global_permissions: Arc<GlobalPermissions>,
}

impl AgentRuntime {
    /// Create a new agent runtime
    pub fn new() -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            global_permissions: Arc::new(GlobalPermissions::new()),
        }
    }

    /// Create a runtime with initial global permission rules
    pub fn with_global_rules(rules: Vec<PermissionRule>) -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            global_permissions: Arc::new(GlobalPermissions::with_rules(rules)),
        }
    }

    /// Get a reference to the global permissions
    ///
    /// This can be used to add rules that apply to all agents.
    pub fn global_permissions(&self) -> &Arc<GlobalPermissions> {
        &self.global_permissions
    }

    /// Spawn a new agent task
    ///
    /// The `agent_fn` receives `AgentInternals` and runs the agent logic.
    /// Returns an `AgentHandle` for external interaction.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = AgentRuntime::new();
    /// let session = AgentSession::new("my-session", "coder", "Coder", "A coding agent")?;
    ///
    /// let handle = runtime.spawn(session, |mut internals| async move {
    ///     loop {
    ///         match internals.receive().await {
    ///             Some(InputMessage::UserInput(text)) => {
    ///                 internals.send_text(format!("You said: {}", text));
    ///                 internals.send_done();
    ///             }
    ///             Some(InputMessage::Shutdown) | None => break,
    ///             _ => {}
    ///         }
    ///     }
    ///     Ok(())
    /// }).await;
    /// ```
    pub async fn spawn<F, Fut>(&self, session: AgentSession, agent_fn: F) -> AgentHandle
    where
        F: FnOnce(AgentInternals) -> Fut + Send + 'static,
        Fut: Future<Output = FrameworkResult<()>> + Send + 'static,
    {
        self.spawn_with_local_rules(session, Vec::new(), agent_fn).await
    }

    /// Spawn a new agent task with local permission rules
    ///
    /// Similar to `spawn`, but allows specifying agent-specific permission rules.
    pub async fn spawn_with_local_rules<F, Fut>(
        &self,
        session: AgentSession,
        local_rules: Vec<PermissionRule>,
        agent_fn: F,
    ) -> AgentHandle
    where
        F: FnOnce(AgentInternals) -> Fut + Send + 'static,
        Fut: Future<Output = FrameworkResult<()>> + Send + 'static,
    {
        let session_id = session.session_id().to_string();
        let agent_type = session.agent_type().to_string();

        // Wrap session in Arc<RwLock> for shared access
        let session = Arc::new(RwLock::new(session));

        // Create channels
        let (input_tx, input_rx, output_tx) = create_agent_channels();

        // Create shared state
        let state = Arc::new(RwLock::new(AgentState::Idle));

        // Create context from session
        let session_read = session.read().await;
        let mut context = AgentContext::new(
            session_read.session_id(),
            session_read.agent_type(),
            session_read.name(),
            session_read.description(),
        );
        drop(session_read); // Release the lock

        // Add SubAgentManager to context for tracking spawned subagents
        context.insert_resource(SubAgentManager::new());

        // Store runtime reference so agents can spawn subagents
        context.insert_resource(self.clone());

        // Create permission manager with shared global + local rules
        let permissions = PermissionManager::with_local_rules(
            self.global_permissions.clone(),
            &agent_type,
            local_rules,
        );

        // Create internals for the agent
        let internals = AgentInternals::new(
            session.clone(),
            context,
            permissions,
            input_rx,
            output_tx.clone(),
            state.clone(),
        );

        // Create handle for external use
        let handle = AgentHandle::new(
            session_id.clone(),
            session,
            input_tx,
            output_tx,
            state,
        );

        // Store handle in registry
        {
            let mut agents = self.agents.write().await;
            agents.insert(session_id.clone(), handle.clone());
        }

        // Spawn the agent task
        let agents_ref = self.agents.clone();
        let session_id_clone = session_id.clone();

        tokio::spawn(async move {
            // Run the agent function
            let result = agent_fn(internals).await;

            // Log errors (but don't panic)
            if let Err(e) = result {
                tracing::error!(session_id = %session_id_clone, error = %e, "Agent task errored");
            }

            // Remove from registry when done
            let mut agents = agents_ref.write().await;
            agents.remove(&session_id_clone);

            tracing::debug!(session_id = %session_id_clone, "Agent task completed");
        });

        handle
    }

    /// Spawn a subagent
    ///
    /// Similar to `spawn`, but creates a subagent session linked to a parent.
    pub async fn spawn_subagent<F, Fut>(
        &self,
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_tool_use_id: impl Into<String>,
        agent_fn: F,
    ) -> FrameworkResult<AgentHandle>
    where
        F: FnOnce(AgentInternals) -> Fut + Send + 'static,
        Fut: Future<Output = FrameworkResult<()>> + Send + 'static,
    {
        let session = AgentSession::new_subagent(
            session_id,
            agent_type,
            name,
            description,
            system_prompt,
            parent_session_id,
            parent_tool_use_id,
        )?;

        Ok(self.spawn(session, agent_fn).await)
    }

    /// Get a handle to a running agent
    pub async fn get(&self, session_id: &str) -> Option<AgentHandle> {
        let agents = self.agents.read().await;
        agents.get(session_id).cloned()
    }

    /// Check if an agent is running
    pub async fn is_running(&self, session_id: &str) -> bool {
        let agents = self.agents.read().await;
        agents.contains_key(session_id)
    }

    /// Get the number of running agents
    pub async fn count(&self) -> usize {
        let agents = self.agents.read().await;
        agents.len()
    }

    /// List all running session IDs
    pub async fn list_running(&self) -> Vec<String> {
        let agents = self.agents.read().await;
        agents.keys().cloned().collect()
    }

    /// Shutdown a specific agent
    ///
    /// Sends a shutdown message to the agent.
    pub async fn shutdown(&self, session_id: &str) -> FrameworkResult<()> {
        let handle = {
            let agents = self.agents.read().await;
            agents.get(session_id).cloned()
        };

        match handle {
            Some(h) => h.shutdown().await,
            None => Err(FrameworkError::AgentNotRunning(session_id.to_string())),
        }
    }

    /// Interrupt a specific agent
    ///
    /// Sends an interrupt message to the agent.
    pub async fn interrupt(&self, session_id: &str) -> FrameworkResult<()> {
        let handle = {
            let agents = self.agents.read().await;
            agents.get(session_id).cloned()
        };

        match handle {
            Some(h) => h.interrupt().await,
            None => Err(FrameworkError::AgentNotRunning(session_id.to_string())),
        }
    }

    /// Shutdown all running agents
    pub async fn shutdown_all(&self) -> Vec<(String, FrameworkResult<()>)> {
        let session_ids = self.list_running().await;
        let mut results = Vec::new();

        for session_id in session_ids {
            let result = self.shutdown(&session_id).await;
            results.push((session_id, result));
        }

        results
    }

    /// Wait for a specific agent to complete
    pub async fn wait_for(&self, session_id: &str) -> FrameworkResult<()> {
        loop {
            if !self.is_running(session_id).await {
                return Ok(());
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }

    /// Wait for all agents to complete
    pub async fn wait_all(&self) {
        loop {
            if self.count().await == 0 {
                return;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }
}

impl Default for AgentRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AgentRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRuntime").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{InputMessage, OutputChunk};
    use crate::session::SessionStorage;
    use tempfile::TempDir;

    fn create_test_session(name: &str) -> (AgentSession, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let storage = SessionStorage::with_dir(temp_dir.path());
        let session = AgentSession::new_with_storage(
            name,
            "test-agent",
            "Test Agent",
            "A test agent",
            storage,
        )
        .unwrap();
        (session, temp_dir)
    }

    #[tokio::test]
    async fn test_spawn_agent() {
        let runtime = AgentRuntime::new();
        let (session, _temp) = create_test_session("spawn-test");

        let handle = runtime
            .spawn(session, |internals| async move {
                // Simple agent that exits immediately
                internals.set_done().await;
                Ok(())
            })
            .await;

        assert_eq!(handle.session_id(), "spawn-test");

        // Wait for agent to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_agent_communication() {
        let runtime = AgentRuntime::new();
        let (session, _temp) = create_test_session("comm-test");

        let handle = runtime
            .spawn(session, |mut internals| async move {
                loop {
                    match internals.receive().await {
                        Some(InputMessage::UserInput(text)) => {
                            internals.send_text(format!("Echo: {}", text));
                            internals.send_done();
                        }
                        Some(InputMessage::Shutdown) | None => {
                            internals.set_done().await;
                            break;
                        }
                        _ => {}
                    }
                }
                Ok(())
            })
            .await;

        // Subscribe to output
        let mut rx = handle.subscribe();

        // Send input
        handle.send_input("Hello").await.unwrap();

        // Receive output
        let chunk = rx.recv().await.unwrap();
        assert!(matches!(chunk, OutputChunk::TextDelta(s) if s == "Echo: Hello"));

        let done = rx.recv().await.unwrap();
        assert!(matches!(done, OutputChunk::Done));

        // Shutdown
        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_runtime_registry() {
        let runtime = AgentRuntime::new();
        let (session1, _temp1) = create_test_session("agent-1");
        let (session2, _temp2) = create_test_session("agent-2");

        // Spawn two agents that wait for shutdown
        let _handle1 = runtime
            .spawn(session1, |mut internals| async move {
                loop {
                    match internals.receive().await {
                        Some(InputMessage::Shutdown) | None => break,
                        _ => {}
                    }
                }
                Ok(())
            })
            .await;

        let _handle2 = runtime
            .spawn(session2, |mut internals| async move {
                loop {
                    match internals.receive().await {
                        Some(InputMessage::Shutdown) | None => break,
                        _ => {}
                    }
                }
                Ok(())
            })
            .await;

        // Check registry
        assert_eq!(runtime.count().await, 2);
        assert!(runtime.is_running("agent-1").await);
        assert!(runtime.is_running("agent-2").await);
        assert!(!runtime.is_running("agent-3").await);

        let running = runtime.list_running().await;
        assert!(running.contains(&"agent-1".to_string()));
        assert!(running.contains(&"agent-2".to_string()));

        // Get handle
        let handle = runtime.get("agent-1").await;
        assert!(handle.is_some());
        assert_eq!(handle.unwrap().session_id(), "agent-1");

        // Shutdown all
        runtime.shutdown_all().await;

        // Wait for cleanup
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert_eq!(runtime.count().await, 0);
    }

    #[tokio::test]
    async fn test_shutdown_nonexistent() {
        let runtime = AgentRuntime::new();

        let result = runtime.shutdown("nonexistent").await;
        assert!(matches!(result, Err(FrameworkError::AgentNotRunning(_))));
    }

    #[tokio::test]
    async fn test_interrupt() {
        let runtime = AgentRuntime::new();
        let (session, _temp) = create_test_session("interrupt-test");

        let handle = runtime
            .spawn(session, |mut internals| async move {
                loop {
                    match internals.receive().await {
                        Some(InputMessage::Interrupt) => {
                            internals.send_status("Interrupted");
                            internals.set_done().await;
                            break;
                        }
                        Some(InputMessage::Shutdown) | None => break,
                        _ => {}
                    }
                }
                Ok(())
            })
            .await;

        let mut rx = handle.subscribe();

        // Interrupt
        runtime.interrupt("interrupt-test").await.unwrap();

        // Check we got the status
        let chunk = rx.recv().await.unwrap();
        assert!(matches!(chunk, OutputChunk::Status(s) if s == "Interrupted"));
    }

    #[tokio::test]
    async fn test_agent_auto_cleanup() {
        let runtime = AgentRuntime::new();
        let (session, _temp) = create_test_session("cleanup-test");

        let _handle = runtime
            .spawn(session, |_internals| async move {
                // Agent exits immediately
                Ok(())
            })
            .await;

        // Initially registered
        assert!(runtime.is_running("cleanup-test").await);

        // Wait for agent to complete and clean up
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Should be removed from registry
        assert!(!runtime.is_running("cleanup-test").await);
    }

    #[tokio::test]
    async fn test_wait_for() {
        let runtime = AgentRuntime::new();
        let (session, _temp) = create_test_session("wait-test");

        let handle = runtime
            .spawn(session, |internals| async move {
                // Wait a bit then exit
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                internals.set_done().await;
                Ok(())
            })
            .await;

        assert!(runtime.is_running("wait-test").await);

        // Wait for completion
        runtime.wait_for("wait-test").await.unwrap();

        // Should be done
        assert!(!runtime.is_running("wait-test").await);

        // Handle state should be done
        assert!(handle.is_done().await);
    }

    #[tokio::test]
    async fn test_clone_runtime() {
        let runtime1 = AgentRuntime::new();
        let runtime2 = runtime1.clone();

        let (session, _temp) = create_test_session("clone-test");

        // Spawn from runtime1
        let _handle = runtime1
            .spawn(session, |mut internals| async move {
                loop {
                    match internals.receive().await {
                        Some(InputMessage::Shutdown) | None => break,
                        _ => {}
                    }
                }
                Ok(())
            })
            .await;

        // Should be visible from runtime2 (same underlying data)
        assert!(runtime2.is_running("clone-test").await);

        // Shutdown from runtime2
        runtime2.shutdown("clone-test").await.unwrap();
    }
}
