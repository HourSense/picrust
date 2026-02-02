//! Hook Types
//!
//! Core types for the hooks system:
//! - `HookEvent` - The type of hook event
//! - `HookContext` - Mutable context passed to hooks
//! - `HookResult` - Result returned from hooks
//! - `PermissionDecision` - Permission decision for PreToolUse hooks

use std::collections::HashMap;

use serde_json::Value;

use crate::llm::{ContentBlock, Message, StopReason};
use crate::runtime::AgentInternals;
use crate::tools::ToolResult;

/// Hook event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEvent {
    /// Before a tool is executed - can block, allow, or modify
    PreToolUse,
    /// After a tool successfully executes
    PostToolUse,
    /// After a tool fails
    PostToolUseFailure,
    /// When user submits a prompt
    UserPromptSubmit,
    /// After assistant generates a response
    PostAssistantResponse,
}

impl std::fmt::Display for HookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookEvent::PreToolUse => write!(f, "PreToolUse"),
            HookEvent::PostToolUse => write!(f, "PostToolUse"),
            HookEvent::PostToolUseFailure => write!(f, "PostToolUseFailure"),
            HookEvent::UserPromptSubmit => write!(f, "UserPromptSubmit"),
            HookEvent::PostAssistantResponse => write!(f, "PostAssistantResponse"),
        }
    }
}

/// Mutable context passed to hooks
///
/// Hooks can read and modify anything here:
/// - Access agent internals (session, context, permissions)
/// - Modify tool input
/// - Modify messages via `messages_mut()`
/// - Modify user prompt
pub struct HookContext<'a> {
    /// The hook event type
    pub event: HookEvent,

    /// Full agent internals - access to session, context, permissions
    pub internals: &'a mut AgentInternals,

    // === Tool-specific (populated for tool hooks) ===
    /// Tool name being called
    pub tool_name: Option<String>,

    /// Tool input - can be modified by hook
    pub tool_input: Option<Value>,

    /// Tool use ID
    pub tool_use_id: Option<String>,

    // === Results (for post hooks) ===
    /// Tool result (for PostToolUse)
    pub tool_result: Option<ToolResult>,

    /// Error message (for PostToolUseFailure)
    pub error: Option<String>,

    // === User input (for UserPromptSubmit) ===
    /// User prompt - can be modified by hook
    pub user_prompt: Option<String>,

    // === Assistant response (for PostAssistantResponse) ===
    /// Assistant's response content blocks
    pub assistant_content: Option<Vec<ContentBlock>>,

    /// Stop reason for the assistant's response
    pub stop_reason: Option<StopReason>,
}

impl<'a> HookContext<'a> {
    /// Create context for PreToolUse hook
    pub fn pre_tool_use(
        internals: &'a mut AgentInternals,
        tool_name: &str,
        tool_input: &Value,
        tool_use_id: &str,
    ) -> Self {
        Self {
            event: HookEvent::PreToolUse,
            internals,
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_use_id: Some(tool_use_id.to_string()),
            tool_result: None,
            error: None,
            user_prompt: None,
            assistant_content: None,
            stop_reason: None,
        }
    }

    /// Create context for PostToolUse hook
    pub fn post_tool_use(
        internals: &'a mut AgentInternals,
        tool_name: &str,
        tool_input: &Value,
        tool_use_id: &str,
        result: &ToolResult,
    ) -> Self {
        Self {
            event: HookEvent::PostToolUse,
            internals,
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_use_id: Some(tool_use_id.to_string()),
            tool_result: Some(result.clone()),
            error: None,
            user_prompt: None,
            assistant_content: None,
            stop_reason: None,
        }
    }

    /// Create context for PostToolUseFailure hook
    pub fn post_tool_use_failure(
        internals: &'a mut AgentInternals,
        tool_name: &str,
        tool_input: &Value,
        tool_use_id: &str,
        error: &str,
    ) -> Self {
        Self {
            event: HookEvent::PostToolUseFailure,
            internals,
            tool_name: Some(tool_name.to_string()),
            tool_input: Some(tool_input.clone()),
            tool_use_id: Some(tool_use_id.to_string()),
            tool_result: None,
            error: Some(error.to_string()),
            user_prompt: None,
            assistant_content: None,
            stop_reason: None,
        }
    }

    /// Create context for UserPromptSubmit hook
    pub fn user_prompt_submit(internals: &'a mut AgentInternals, prompt: &str) -> Self {
        Self {
            event: HookEvent::UserPromptSubmit,
            internals,
            tool_name: None,
            tool_input: None,
            tool_use_id: None,
            tool_result: None,
            error: None,
            user_prompt: Some(prompt.to_string()),
            assistant_content: None,
            stop_reason: None,
        }
    }

    /// Create context for PostAssistantResponse hook
    pub fn post_assistant_response(
        internals: &'a mut AgentInternals,
        content_blocks: &[ContentBlock],
        stop_reason: Option<StopReason>,
    ) -> Self {
        Self {
            event: HookEvent::PostAssistantResponse,
            internals,
            tool_name: None,
            tool_input: None,
            tool_use_id: None,
            tool_result: None,
            error: None,
            user_prompt: None,
            assistant_content: Some(content_blocks.to_vec()),
            stop_reason,
        }
    }

    // === Convenience methods ===

    /// Get conversation history
    ///
    /// Note: This returns a cloned copy of the messages since the session
    /// is now behind an Arc<RwLock>. If you need to modify messages, use
    /// the session directly through async access.
    pub fn messages(&self) -> Vec<Message> {
        let session = self.internals.session.blocking_read();
        session.history().to_vec()
    }

    /// Get conversation history (deprecated - hooks should not mutate history directly)
    ///
    /// This method is kept for backwards compatibility but will block.
    /// Consider using async access to the session if you need to modify history.
    #[deprecated(note = "Use async session access instead of synchronous mutation")]
    pub fn messages_mut(&mut self) -> Vec<Message> {
        let session = self.internals.session.blocking_read();
        session.history().to_vec()
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        self.internals.session_id()
    }

    /// Get agent type
    pub fn agent_type(&self) -> &str {
        self.internals.agent_type()
    }

    /// Get current turn number
    pub fn current_turn(&self) -> usize {
        self.internals.context.current_turn
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.internals.context.get_metadata(key)
    }

    /// Set metadata value
    pub fn set_metadata(&mut self, key: &str, value: Value) {
        self.internals.context.set_metadata(key, value);
    }

    /// Get all metadata
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.internals.context.metadata
    }
}

/// Permission decision for PreToolUse hooks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Allow the tool call, skip user permission check
    Allow,
    /// Deny the tool call, return error to LLM
    Deny,
    /// Use normal permission flow (ask user if needed)
    Ask,
}

/// Result returned from a hook
///
/// For most hooks, just return `HookResult::none()` or `HookResult::default()`.
/// For PreToolUse hooks that want to control permissions, use `allow()`, `deny()`, or `ask()`.
#[derive(Debug, Clone, Default)]
pub struct HookResult {
    /// Permission decision (mainly for PreToolUse)
    pub decision: Option<PermissionDecision>,

    /// Reason for the decision (shown in error message if denied)
    pub reason: Option<String>,
}

impl HookResult {
    /// Allow the operation (skip permission check)
    pub fn allow() -> Self {
        Self {
            decision: Some(PermissionDecision::Allow),
            reason: None,
        }
    }

    /// Deny the operation with a reason
    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            decision: Some(PermissionDecision::Deny),
            reason: Some(reason.into()),
        }
    }

    /// Use normal permission flow
    pub fn ask() -> Self {
        Self {
            decision: Some(PermissionDecision::Ask),
            reason: None,
        }
    }

    /// No decision - continue with default behavior
    pub fn none() -> Self {
        Self::default()
    }

    /// Add a reason to an existing result
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}
