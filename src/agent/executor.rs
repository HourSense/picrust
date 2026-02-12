//! Tool Executor
//!
//! Handles permission-aware tool execution with optional debug logging and hooks.

use serde_json::Value;

use crate::core::InputMessage;
use crate::helpers::Debugger;
use crate::hooks::{HookContext, HookRegistry, PermissionDecision};
use crate::permissions::{CheckResult, PermissionRule, PermissionScope};
use crate::runtime::AgentInternals;
use crate::tools::{ToolRegistry, ToolResult};

/// Handles tool execution with permission checking and hooks
pub struct ToolExecutor;

impl ToolExecutor {
    /// Execute a tool with permission checking and hooks
    ///
    /// This handles the full flow:
    /// 1. Run PreToolUse hooks (can block, allow, or modify input)
    /// 2. Check if permission exists (unless hook already decided)
    /// 3. If not, ask user (via output channel)
    /// 4. Wait for response
    /// 5. Execute if allowed, return error if denied
    /// 6. Run PostToolUse or PostToolUseFailure hooks
    pub async fn execute_with_permission(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        hooks: Option<&HookRegistry>,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
        hook_short_circuit: bool,
    ) -> ToolResult {
        let mut current_input = input.clone();

        // === Run PreToolUse hooks ===
        if let Some(hooks) = hooks {
            let mut ctx = HookContext::pre_tool_use(
                internals,
                tool_name,
                &current_input,
                tool_id,
                hook_short_circuit,
            );
            let result = hooks.run(&mut ctx);

            // Hook may have modified tool_input
            if let Some(modified_input) = ctx.tool_input {
                current_input = modified_input;
            }

            // Handle permission decision from hooks
            match result.decision {
                Some(PermissionDecision::Deny) => {
                    let reason = result
                        .reason
                        .unwrap_or_else(|| "Blocked by hook".to_string());
                    tracing::info!("[Executor] Hook denied {}: {}", tool_name, reason);
                    return ToolResult::error(format!("Hook denied: {}", reason));
                }
                Some(PermissionDecision::Allow) => {
                    // Skip permission check, execute directly
                    tracing::info!("[Executor] Hook allowed {} (skipping permission check)", tool_name);
                    return Self::execute_with_hooks(
                        internals,
                        tools,
                        Some(hooks),
                        tool_name,
                        tool_id,
                        &current_input,
                        hook_short_circuit,
                    )
                    .await;
                }
                Some(PermissionDecision::Ask) | None => {
                    // Fall through to normal permission check
                }
            }
        }

        let input_str = current_input.to_string();

        // Get tool info for better permission prompts
        let tool_info = tools.get_tool_info(tool_name, &current_input);
        let action_desc = tool_info
            .as_ref()
            .map(|i| i.action_description.clone())
            .unwrap_or_else(|| format!("Execute {}", tool_name));

        // Check permission
        match internals.check_permission(tool_name, &input_str) {
            CheckResult::Allowed => {
                tracing::info!("[Executor] Permission allowed for {}", tool_name);
                Self::execute_with_hooks(
                    internals,
                    tools,
                    hooks,
                    tool_name,
                    tool_id,
                    &current_input,
                    hook_short_circuit,
                )
                .await
            }

            CheckResult::Denied => {
                tracing::info!("[Executor] Permission denied for {}", tool_name);
                ToolResult::error(format!("Permission denied for tool: {}", tool_name))
            }

            CheckResult::AskUser => {
                tracing::info!("[Executor] Asking user for permission: {}", tool_name);
                Self::ask_and_execute(
                    internals,
                    tools,
                    hooks,
                    tool_name,
                    tool_id,
                    &current_input,
                    &action_desc,
                    tool_info.and_then(|i| i.details),
                    hook_short_circuit,
                )
                .await
            }
        }
    }

    /// Ask user for permission and execute if granted
    async fn ask_and_execute(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        hooks: Option<&HookRegistry>,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
        action_desc: &str,
        details: Option<String>,
        hook_short_circuit: bool,
    ) -> ToolResult {
        let input_str = input.to_string();

        // Send permission request
        internals.send_permission_request(tool_name, action_desc, &input_str, details);
        internals.set_waiting_for_permission().await;

        // Wait for response
        match internals.receive().await {
            Some(InputMessage::PermissionResponse {
                tool_name: resp_tool,
                allowed,
                remember,
            }) => {
                if resp_tool != tool_name {
                    tracing::warn!(
                        "[Executor] Permission response mismatch: expected {}, got {}",
                        tool_name,
                        resp_tool
                    );
                    return ToolResult::error("Permission response mismatch");
                }

                if remember && allowed {
                    tracing::info!("[Executor] Adding 'Always Allow' rule for {}", tool_name);
                    internals.add_permission_rule(
                        PermissionRule::allow_tool(tool_name),
                        PermissionScope::Session,
                    );
                }

                if allowed {
                    tracing::info!("[Executor] User allowed {}", tool_name);
                    Self::execute_with_hooks(
                        internals,
                        tools,
                        hooks,
                        tool_name,
                        tool_id,
                        input,
                        hook_short_circuit,
                    )
                    .await
                } else {
                    tracing::info!("[Executor] User denied {}", tool_name);
                    ToolResult::error(format!("User denied permission for: {}", tool_name))
                }
            }

            Some(InputMessage::Interrupt) => {
                tracing::info!("[Executor] Interrupted while waiting for permission");
                ToolResult::error("Interrupted")
            }

            Some(InputMessage::Shutdown) => {
                tracing::info!("[Executor] Shutdown while waiting for permission");
                ToolResult::error("Shutdown")
            }

            None => {
                tracing::info!("[Executor] Channel closed while waiting for permission");
                ToolResult::error("Channel closed")
            }

            _ => {
                tracing::warn!("[Executor] Unexpected message while waiting for permission");
                ToolResult::error("Unexpected message during permission request")
            }
        }
    }

    /// Execute a tool with post-execution hooks
    async fn execute_with_hooks(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        hooks: Option<&HookRegistry>,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
        hook_short_circuit: bool,
    ) -> ToolResult {
        // Set the current tool_use_id on context so tools can access it
        internals.context.current_tool_use_id = Some(tool_id.to_string());

        // Update state
        internals.set_executing_tool(tool_name, tool_id).await;

        // Log tool call if debugger is enabled
        if let Some(debugger) = internals.context.get_resource::<Debugger>() {
            if let Err(e) = debugger.log_tool_call(tool_name, tool_id, input) {
                tracing::warn!("[Executor] Failed to log tool call: {}", e);
            }
        }

        // Send tool start notification
        internals.send_tool_start(tool_name, tool_name, input.clone());

        // Execute
        let result = match tools.execute(tool_name, input, internals).await {
            Ok(result) => {
                // Run PostToolUse hooks
                if let Some(hooks) = hooks {
                    let mut ctx = HookContext::post_tool_use(
                        internals,
                        tool_name,
                        input,
                        tool_id,
                        &result,
                        hook_short_circuit,
                    );
                    let _hook_result = hooks.run(&mut ctx);
                    // PostToolUse hooks are for logging/observation, we don't act on the result
                }
                result
            }
            Err(e) => {
                let error_msg = format!("Tool execution failed: {}", e);

                // Run PostToolUseFailure hooks
                if let Some(hooks) = hooks {
                    let mut ctx = HookContext::post_tool_use_failure(
                        internals,
                        tool_name,
                        input,
                        tool_id,
                        &error_msg,
                        hook_short_circuit,
                    );
                    let _hook_result = hooks.run(&mut ctx);
                }

                ToolResult::error(error_msg)
            }
        };

        // Log tool result if debugger is enabled
        if let Some(debugger) = internals.context.get_resource::<Debugger>() {
            if let Err(e) = debugger.log_tool_result(tool_name, tool_id, &result) {
                tracing::warn!("[Executor] Failed to log tool result: {}", e);
            }
        }

        // Send tool end notification
        internals.send_tool_end(tool_name, result.clone());

        // Clear the current tool_use_id
        internals.context.current_tool_use_id = None;

        result
    }

    /// Execute a tool without hooks (for backwards compatibility)
    pub async fn execute(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
    ) -> ToolResult {
        Self::execute_with_hooks(internals, tools, None, tool_name, tool_id, input, false).await
    }
}
