//! AgentInternals - Internal state passed to agent functions
//!
//! The `AgentInternals` struct is what the agent function receives when spawned.
//! It provides methods to:
//! - Receive input from the handle
//! - Send output chunks to subscribers
//! - Update and query agent state
//! - Check and manage permissions

use std::sync::Arc;
use tokio::sync::RwLock;

use std::collections::HashMap;

use crate::core::{AgentContext, AgentState, FrameworkError, FrameworkResult, InputMessage, OutputChunk};
use crate::core::output::UserQuestion;
use crate::permissions::{CheckResult, PermissionManager, PermissionRule, PermissionScope};
use crate::session::AgentSession;

use super::channels::{InputReceiver, OutputSender};

/// Internal state and channels for an agent
///
/// This is passed to the agent function when spawned by `AgentRuntime`.
/// The agent uses this to receive input and send output.
pub struct AgentInternals {
    /// The agent's session (history, metadata)
    pub session: AgentSession,

    /// The agent's context (passed to tools)
    pub context: AgentContext,

    /// Permission manager for this agent
    pub permissions: PermissionManager,

    /// Receiver for input messages
    input_rx: InputReceiver,

    /// Sender for output chunks
    output_tx: OutputSender,

    /// Current agent state (shared with AgentHandle)
    state: Arc<RwLock<AgentState>>,
}

impl AgentInternals {
    /// Create new agent internals
    ///
    /// This is typically called by `AgentRuntime::spawn()`, not directly.
    pub fn new(
        session: AgentSession,
        context: AgentContext,
        permissions: PermissionManager,
        input_rx: InputReceiver,
        output_tx: OutputSender,
        state: Arc<RwLock<AgentState>>,
    ) -> Self {
        Self {
            session,
            context,
            permissions,
            input_rx,
            output_tx,
            state,
        }
    }

    // =========================================================================
    // Input Methods
    // =========================================================================

    /// Receive the next input message
    ///
    /// Blocks until an input message is available.
    /// Returns `None` if the input channel is closed (handle dropped).
    pub async fn receive(&mut self) -> Option<InputMessage> {
        self.input_rx.recv().await
    }

    /// Receive the next input message, returning an error if channel closed
    pub async fn receive_or_err(&mut self) -> FrameworkResult<InputMessage> {
        self.input_rx
            .recv()
            .await
            .ok_or(FrameworkError::ChannelClosed)
    }

    /// Try to receive input without blocking
    ///
    /// Returns `None` if no message is available.
    pub fn try_receive(&mut self) -> Option<InputMessage> {
        self.input_rx.try_recv().ok()
    }

    // =========================================================================
    // Output Methods
    // =========================================================================

    /// Send an output chunk to all subscribers
    ///
    /// Returns the number of subscribers that received the message.
    /// Returns 0 if there are no subscribers (which is not an error).
    pub fn send(&self, chunk: OutputChunk) -> usize {
        self.output_tx.send(chunk).unwrap_or(0)
    }

    /// Send a text delta
    pub fn send_text(&self, text: impl Into<String>) -> usize {
        self.send(OutputChunk::TextDelta(text.into()))
    }

    /// Send a thinking delta
    pub fn send_thinking(&self, text: impl Into<String>) -> usize {
        self.send(OutputChunk::ThinkingDelta(text.into()))
    }

    /// Send thinking complete signal
    pub fn send_thinking_complete(&self, full_text: impl Into<String>) -> usize {
        self.send(OutputChunk::ThinkingComplete(full_text.into()))
    }

    /// Send text complete signal
    pub fn send_text_complete(&self, full_text: impl Into<String>) -> usize {
        self.send(OutputChunk::TextComplete(full_text.into()))
    }

    /// Send a status update
    pub fn send_status(&self, status: impl Into<String>) -> usize {
        self.send(OutputChunk::Status(status.into()))
    }

    /// Send an error
    pub fn send_error(&self, error: impl Into<String>) -> usize {
        self.send(OutputChunk::Error(error.into()))
    }

    /// Send done signal
    pub fn send_done(&self) -> usize {
        self.send(OutputChunk::Done)
    }

    /// Send a tool start notification
    pub fn send_tool_start(
        &self,
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> usize {
        self.send(OutputChunk::ToolStart {
            id: id.into(),
            name: name.into(),
            input,
        })
    }

    /// Send a tool end notification
    pub fn send_tool_end(&self, id: impl Into<String>, result: crate::tools::ToolResult) -> usize {
        self.send(OutputChunk::ToolEnd {
            id: id.into(),
            result,
        })
    }

    /// Send a permission request
    pub fn send_permission_request(
        &self,
        tool_name: impl Into<String>,
        action: impl Into<String>,
        input: impl Into<String>,
        details: Option<String>,
    ) -> usize {
        self.send(OutputChunk::PermissionRequest {
            tool_name: tool_name.into(),
            action: action.into(),
            input: input.into(),
            details,
        })
    }

    /// Get the number of current subscribers
    pub fn subscriber_count(&self) -> usize {
        self.output_tx.receiver_count()
    }

    // =========================================================================
    // State Methods
    // =========================================================================

    /// Set the current agent state
    pub async fn set_state(&self, new_state: AgentState) {
        let mut state = self.state.write().await;
        *state = new_state.clone();
        // Notify subscribers of state change
        let _ = self.output_tx.send(OutputChunk::StateChange(new_state));
    }

    /// Set state without notifying subscribers
    pub async fn set_state_silent(&self, new_state: AgentState) {
        let mut state = self.state.write().await;
        *state = new_state;
    }

    /// Get the current agent state
    pub async fn state(&self) -> AgentState {
        self.state.read().await.clone()
    }

    /// Set state to Idle
    pub async fn set_idle(&self) {
        self.set_state(AgentState::Idle).await;
    }

    /// Set state to Processing
    pub async fn set_processing(&self) {
        self.set_state(AgentState::Processing).await;
    }

    /// Set state to Done
    pub async fn set_done(&self) {
        self.set_state(AgentState::Done).await;
    }

    /// Set state to Error
    pub async fn set_error(&self, message: impl Into<String>) {
        self.set_state(AgentState::Error {
            message: message.into(),
        })
        .await;
    }

    /// Set state to WaitingForPermission
    pub async fn set_waiting_for_permission(&self) {
        self.set_state(AgentState::WaitingForPermission).await;
    }

    /// Set state to ExecutingTool
    pub async fn set_executing_tool(&self, tool_name: impl Into<String>, tool_use_id: impl Into<String>) {
        self.set_state(AgentState::ExecutingTool {
            tool_name: tool_name.into(),
            tool_use_id: tool_use_id.into(),
        })
        .await;
    }

    /// Set state to WaitingForSubAgent
    pub async fn set_waiting_for_subagent(&self, session_id: impl Into<String>) {
        self.set_state(AgentState::WaitingForSubAgent {
            session_id: session_id.into(),
        })
        .await;
    }

    /// Set state to WaitingForUserInput
    pub async fn set_waiting_for_user_input(&self, request_id: impl Into<String>) {
        self.set_state(AgentState::WaitingForUserInput {
            request_id: request_id.into(),
        })
        .await;
    }

    // =========================================================================
    // Context Methods
    // =========================================================================

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.context.session_id
    }

    /// Get the agent type
    pub fn agent_type(&self) -> &str {
        &self.context.agent_type
    }

    /// Increment the turn counter
    pub fn next_turn(&mut self) {
        self.context.next_turn();
    }

    /// Get a context with the current tool_use_id set
    ///
    /// Use this when executing a tool so it knows its own ID.
    pub fn context_for_tool(&self, tool_use_id: impl Into<String>) -> AgentContext {
        self.context.with_tool_use_id(tool_use_id)
    }

    // =========================================================================
    // Permission Methods
    // =========================================================================

    /// Check if a tool action is allowed
    ///
    /// Returns `CheckResult::Allowed` if a rule matches, `CheckResult::AskUser`
    /// if user confirmation is needed, or `CheckResult::Denied` in non-interactive mode.
    pub fn check_permission(&self, tool_name: &str, input: &str) -> CheckResult {
        self.permissions.check(tool_name, input)
    }

    /// Check permission and request user approval if needed
    ///
    /// This is a convenience method that:
    /// 1. Checks existing rules
    /// 2. If no rule matches, sends a PermissionRequest and waits for response
    /// 3. Processes the response (adding rules if "Always" was selected)
    ///
    /// Returns `Ok(true)` if allowed, `Ok(false)` if denied, or an error if
    /// the channel closed while waiting.
    pub async fn request_permission(
        &mut self,
        tool_name: &str,
        action_description: &str,
        input: &str,
    ) -> FrameworkResult<bool> {
        match self.permissions.check(tool_name, input) {
            CheckResult::Allowed => Ok(true),
            CheckResult::Denied => Ok(false),
            CheckResult::AskUser => {
                // Send permission request
                self.send_permission_request(tool_name, action_description, input, None);
                self.set_waiting_for_permission().await;

                // Wait for response
                match self.receive().await {
                    Some(InputMessage::PermissionResponse {
                        tool_name: resp_tool,
                        allowed,
                        remember,
                    }) => {
                        if resp_tool == tool_name {
                            if remember && allowed {
                                // Add to session rules (could also be global based on UI)
                                self.permissions.add_rule(
                                    PermissionRule::allow_tool(tool_name),
                                    PermissionScope::Session,
                                );
                            }
                            Ok(allowed)
                        } else {
                            // Mismatched tool name - shouldn't happen
                            tracing::warn!(
                                "Permission response for {} but expected {}",
                                resp_tool,
                                tool_name
                            );
                            Ok(false)
                        }
                    }
                    Some(InputMessage::Shutdown) => Err(FrameworkError::Shutdown),
                    Some(InputMessage::Interrupt) => Err(FrameworkError::Interrupted),
                    None => Err(FrameworkError::ChannelClosed),
                    _ => {
                        // Unexpected message while waiting for permission
                        Ok(false)
                    }
                }
            }
        }
    }

    /// Add a permission rule
    ///
    /// Use this to programmatically add rules (e.g., from configuration).
    pub fn add_permission_rule(&mut self, rule: PermissionRule, scope: PermissionScope) {
        self.permissions.add_rule(rule, scope);
    }

    // =========================================================================
    // User Question Methods
    // =========================================================================

    /// Ask the user a set of questions and wait for their response
    ///
    /// This method:
    /// 1. Sends an `AskUserQuestion` output chunk with the questions
    /// 2. Sets state to `WaitingForUserInput`
    /// 3. Waits for a `UserQuestionResponse` input message
    /// 4. Returns the answers map (header -> selected answer)
    ///
    /// Handles `Interrupt`, `Shutdown`, and channel close gracefully.
    pub async fn ask_user_question(
        &mut self,
        request_id: impl Into<String>,
        questions: Vec<UserQuestion>,
    ) -> FrameworkResult<HashMap<String, String>> {
        let request_id = request_id.into();

        // Send the question request
        self.send(OutputChunk::AskUserQuestion {
            request_id: request_id.clone(),
            questions,
        });

        // Set state to waiting for user input
        self.set_waiting_for_user_input(&request_id).await;

        // Wait for response
        match self.receive().await {
            Some(InputMessage::UserQuestionResponse { request_id: resp_id, answers }) => {
                if resp_id == request_id {
                    Ok(answers)
                } else {
                    // Mismatched request ID - shouldn't happen
                    tracing::warn!(
                        "Question response for {} but expected {}",
                        resp_id,
                        request_id
                    );
                    Err(FrameworkError::Other(format!(
                        "Mismatched request ID: expected {}, got {}",
                        request_id, resp_id
                    )))
                }
            }
            Some(InputMessage::Interrupt) => Err(FrameworkError::Interrupted),
            Some(InputMessage::Shutdown) => Err(FrameworkError::Shutdown),
            None => Err(FrameworkError::ChannelClosed),
            _ => Err(FrameworkError::Other("Unexpected message while waiting for user response".into())),
        }
    }

    /// Check if running in interactive mode
    pub fn is_interactive(&self) -> bool {
        self.permissions.is_interactive()
    }

    /// Set interactive mode
    ///
    /// When false, permission checks that would require user input will be denied.
    pub fn set_interactive(&mut self, interactive: bool) {
        self.permissions.set_interactive(interactive);
    }

    // =========================================================================
    // SubAgent Methods
    // =========================================================================

    /// Spawn a subagent and register it with this agent's SubAgentManager
    ///
    /// This is the preferred way to spawn subagents from within an agent,
    /// as it automatically:
    /// 1. Creates the subagent with proper parent linkage
    /// 2. Registers the handle with this agent's SubAgentManager
    /// 3. Sends a SubAgentSpawned notification to subscribers
    ///
    /// # Example
    ///
    /// ```ignore
    /// let handle = internals.spawn_subagent(
    ///     "sub-1",
    ///     "researcher",
    ///     "Research Agent",
    ///     "Researches topics",
    ///     "tool_123",
    ///     |sub_internals| async move {
    ///         // Subagent logic here
    ///         Ok(())
    ///     },
    /// ).await?;
    /// ```
    pub async fn spawn_subagent<F, Fut>(
        &self,
        session_id: impl Into<String>,
        agent_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        tool_use_id: impl Into<String>,
        agent_fn: F,
    ) -> FrameworkResult<super::AgentHandle>
    where
        F: FnOnce(AgentInternals) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = FrameworkResult<()>> + Send + 'static,
    {
        let session_id = session_id.into();
        let agent_type = agent_type.into();
        let name_str = name.into();
        let description_str = description.into();

        // Get the runtime from context
        let runtime = self
            .context
            .get_resource::<super::AgentRuntime>()
            .ok_or_else(|| FrameworkError::Other("Runtime not found in context".into()))?;

        // Spawn the subagent
        let handle = runtime
            .spawn_subagent(
                &session_id,
                &agent_type,
                &name_str,
                &description_str,
                self.session_id(),
                tool_use_id,
                agent_fn,
            )
            .await?;

        // Register with our SubAgentManager
        if let Some(manager) = self.context.get_resource::<super::SubAgentManager>() {
            manager.register(&session_id, handle.clone());
        }

        // Notify subscribers
        self.send(OutputChunk::SubAgentSpawned {
            session_id: session_id.clone(),
            agent_type: agent_type.clone(),
        });

        tracing::info!(
            "[{}] Spawned subagent: {} ({})",
            self.session_id(),
            session_id,
            agent_type
        );

        Ok(handle)
    }

    /// Get the SubAgentManager for this agent
    ///
    /// Returns None if no subagents have been spawned yet.
    pub fn subagent_manager(&self) -> Option<std::sync::Arc<super::SubAgentManager>> {
        self.context.get_resource::<super::SubAgentManager>()
    }

    /// Get a subagent's handle by session ID
    pub fn get_subagent(&self, session_id: &str) -> Option<super::AgentHandle> {
        self.subagent_manager()
            .and_then(|m| m.get(session_id))
    }

    /// List all active subagent session IDs
    pub fn active_subagents(&self) -> Vec<String> {
        self.subagent_manager()
            .map(|m| m.active_session_ids())
            .unwrap_or_default()
    }

    /// Mark a subagent as completed
    ///
    /// Call this when a subagent finishes to track its result.
    pub fn mark_subagent_completed(
        &self,
        session_id: &str,
        result: Option<String>,
        success: bool,
        error: Option<String>,
    ) {
        if let Some(manager) = self.subagent_manager() {
            // Get agent type from handle before marking complete
            let agent_type = manager
                .get(session_id)
                .map(|_| "unknown") // We don't store agent_type in handle
                .unwrap_or("unknown");

            manager.mark_completed(session_id, agent_type, result.clone(), success, error);

            // Notify subscribers
            self.send(OutputChunk::SubAgentComplete {
                session_id: session_id.to_string(),
                result,
            });
        }
    }
}

impl std::fmt::Debug for AgentInternals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentInternals")
            .field("session_id", &self.context.session_id)
            .field("agent_type", &self.context.agent_type)
            .field("subscriber_count", &self.output_tx.receiver_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::GlobalPermissions;
    use crate::runtime::channels::create_agent_channels;
    use crate::session::SessionStorage;
    use tempfile::TempDir;

    fn create_test_internals() -> (AgentInternals, super::super::channels::InputSender, super::super::channels::OutputReceiver) {
        let (input_tx, input_rx, output_tx) = create_agent_channels();
        let output_rx = output_tx.subscribe();
        let state = Arc::new(RwLock::new(AgentState::Idle));

        let temp_dir = TempDir::new().unwrap();
        let storage = SessionStorage::with_dir(temp_dir.path());
        let session = AgentSession::new_with_storage(
            "test-session",
            "test-agent",
            "Test Agent",
            "A test agent",
            storage,
        )
        .unwrap();

        let context = AgentContext::new(
            "test-session",
            "test-agent",
            "Test Agent",
            "A test agent",
        );

        let global_permissions = Arc::new(GlobalPermissions::new());
        let permissions = PermissionManager::new(global_permissions, "test-agent");

        let internals = AgentInternals::new(session, context, permissions, input_rx, output_tx, state);

        (internals, input_tx, output_rx)
    }

    #[tokio::test]
    async fn test_receive() {
        let (mut internals, input_tx, _output_rx) = create_test_internals();

        input_tx
            .send(InputMessage::UserInput("Hello".into()))
            .await
            .unwrap();

        let msg = internals.receive().await.unwrap();
        assert!(matches!(msg, InputMessage::UserInput(s) if s == "Hello"));
    }

    #[tokio::test]
    async fn test_send() {
        let (internals, _input_tx, mut output_rx) = create_test_internals();

        let count = internals.send_text("Hello");
        assert_eq!(count, 1); // One subscriber

        let chunk = output_rx.recv().await.unwrap();
        assert!(matches!(chunk, OutputChunk::TextDelta(s) if s == "Hello"));
    }

    #[tokio::test]
    async fn test_state() {
        let (internals, _input_tx, mut output_rx) = create_test_internals();

        // Initial state
        assert!(matches!(internals.state().await, AgentState::Idle));

        // Change state
        internals.set_processing().await;
        assert!(matches!(internals.state().await, AgentState::Processing));

        // Should have sent state change notification
        let chunk = output_rx.recv().await.unwrap();
        assert!(matches!(chunk, OutputChunk::StateChange(AgentState::Processing)));
    }

    #[tokio::test]
    async fn test_set_state_silent() {
        let (internals, _input_tx, mut output_rx) = create_test_internals();

        internals.set_state_silent(AgentState::Processing).await;
        assert!(matches!(internals.state().await, AgentState::Processing));

        // Should NOT have sent notification - verify by trying to receive with timeout
        let result = tokio::time::timeout(
            tokio::time::Duration::from_millis(10),
            output_rx.recv(),
        )
        .await;
        assert!(result.is_err()); // Timeout means no message
    }

    #[tokio::test]
    async fn test_context_for_tool() {
        let (internals, _input_tx, _output_rx) = create_test_internals();

        let ctx = internals.context_for_tool("tool_123");
        assert_eq!(ctx.current_tool_use_id, Some("tool_123".into()));
        assert_eq!(ctx.session_id, "test-session");
    }

    #[tokio::test]
    async fn test_next_turn() {
        let (mut internals, _input_tx, _output_rx) = create_test_internals();

        assert_eq!(internals.context.current_turn, 0);
        internals.next_turn();
        assert_eq!(internals.context.current_turn, 1);
    }

    #[tokio::test]
    async fn test_send_done() {
        let (internals, _input_tx, mut output_rx) = create_test_internals();

        internals.send_done();

        let chunk = output_rx.recv().await.unwrap();
        assert!(matches!(chunk, OutputChunk::Done));
    }

    #[tokio::test]
    async fn test_channel_close() {
        let (mut internals, input_tx, _output_rx) = create_test_internals();

        // Drop the sender
        drop(input_tx);

        // Receive should return None
        let msg = internals.receive().await;
        assert!(msg.is_none());

        // receive_or_err should return error
        // (Need to recreate since we already consumed the None)
    }
}
