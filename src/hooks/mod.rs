//! Hooks Module
//!
//! Intercept and control agent behavior at key execution points.
//!
//! # Overview
//!
//! Hooks let you:
//! - Block dangerous operations before they execute
//! - Modify tool arguments (e.g., rewrite paths for remote filesystem)
//! - Auto-approve certain tools
//! - Log and audit tool calls
//! - Filter or modify conversation history
//!
//! **Important:** ALL matching hooks run to completion (no short-circuiting).
//! This ensures security hooks can't be bypassed and monitoring hooks always fire.
//!
//! # Example
//!
//! ```ignore
//! use shadow_agent_sdk::hooks::{HookRegistry, HookEvent, HookResult};
//!
//! let mut hooks = HookRegistry::new();
//!
//! // Block dangerous commands
//! hooks.add_with_pattern(HookEvent::PreToolUse, "Bash", |ctx| async move {
//!     let cmd = ctx.tool_input.as_ref()
//!         .and_then(|v| v.get("command"))
//!         .and_then(|v| v.as_str())
//!         .unwrap_or("");
//!
//!     if cmd.contains("rm -rf") {
//!         HookResult::deny("Dangerous command blocked")
//!     } else {
//!         HookResult::none()
//!     }
//! })?;
//!
//! // Auto-approve read-only tools
//! hooks.add_with_pattern(HookEvent::PreToolUse, "Read|Glob|Grep", |_ctx| async move {
//!     HookResult::allow()
//! })?;
//!
//! // Log all assistant responses
//! hooks.add(HookEvent::PostAssistantResponse, |ctx| {
//!     if let Some(ref content) = ctx.assistant_content {
//!         let text_blocks = content.iter().filter(|b| matches!(b, ContentBlock::Text { .. })).count();
//!         let tool_blocks = content.iter().filter(|b| matches!(b, ContentBlock::ToolUse { .. })).count();
//!         tracing::info!(
//!             "Assistant response: {} text blocks, {} tool calls, stop_reason: {:?}",
//!             text_blocks, tool_blocks, ctx.stop_reason
//!         );
//!     }
//!     HookResult::none()
//! });
//!
//! // Use with agent config
//! let config = AgentConfig::new("You are helpful")
//!     .with_hooks(hooks);
//! ```
//!
//! # Hook Events
//!
//! | Event | When | Can modify |
//! |-------|------|------------|
//! | `PreToolUse` | Before tool executes | `tool_input`, messages, permission |
//! | `PostToolUse` | After tool succeeds | messages (for logging) |
//! | `PostToolUseFailure` | After tool fails | messages (for logging) |
//! | `UserPromptSubmit` | When user sends prompt | `user_prompt`, messages |
//! | `PostAssistantResponse` | After assistant generates response | messages (for logging) |
//!
//! # HookResult
//!
//! Hooks return a `HookResult` that controls behavior:
//!
//! | Method | Effect |
//! |--------|--------|
//! | `HookResult::none()` | Continue normally |
//! | `HookResult::allow()` | Skip permission check, execute tool |
//! | `HookResult::deny("reason")` | Block tool, return error to LLM |
//! | `HookResult::ask()` | Use normal permission flow |
//!
//! # Result Combination
//!
//! When multiple hooks run, results are combined with priority: **Deny > Allow > Ask > None**
//!
//! - If ANY hook returns `Deny` → Tool is blocked (most restrictive wins)
//! - Else if ANY hook returns `Allow` → Skip permissions
//! - Else if ANY hook returns `Ask` → Use normal permission flow
//! - Else (all returned `None`) → Continue normal flow
//!
//! This ensures security hooks can't be bypassed by earlier allow hooks.

mod registry;
mod types;

pub use registry::{ArcHook, Hook, HookMatcher, HookRegistry};
pub use types::{HookContext, HookEvent, HookResult, PermissionDecision};
