//! Tool Executor
//!
//! Handles permission-aware tool execution with optional debug logging.

use serde_json::Value;

use crate::core::InputMessage;
use crate::helpers::Debugger;
use crate::permissions::{CheckResult, PermissionRule, PermissionScope};
use crate::runtime::AgentInternals;
use crate::tools::{ToolRegistry, ToolResult};

/// Handles tool execution with permission checking
pub struct ToolExecutor;

impl ToolExecutor {
    /// Execute a tool with permission checking
    ///
    /// This handles the full permission flow:
    /// 1. Check if permission exists
    /// 2. If not, ask user (via output channel)
    /// 3. Wait for response
    /// 4. Execute if allowed, return error if denied
    pub async fn execute_with_permission(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
    ) -> ToolResult {
        let input_str = input.to_string();

        // Get tool info for better permission prompts
        let tool_info = tools.get_tool_info(tool_name, input);
        let action_desc = tool_info
            .as_ref()
            .map(|i| i.action_description.clone())
            .unwrap_or_else(|| format!("Execute {}", tool_name));

        // Check permission
        match internals.check_permission(tool_name, &input_str) {
            CheckResult::Allowed => {
                tracing::info!("[Executor] Permission allowed for {}", tool_name);
                Self::execute(internals, tools, tool_name, tool_id, input).await
            }

            CheckResult::Denied => {
                tracing::info!("[Executor] Permission denied for {}", tool_name);
                ToolResult::error(format!("Permission denied for tool: {}", tool_name))
            }

            CheckResult::AskUser => {
                tracing::info!("[Executor] Asking user for permission: {}", tool_name);
                Self::ask_and_execute(internals, tools, tool_name, tool_id, input, &action_desc, tool_info.and_then(|i| i.details)).await
            }
        }
    }

    /// Ask user for permission and execute if granted
    async fn ask_and_execute(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
        action_desc: &str,
        details: Option<String>,
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
                    Self::execute(internals, tools, tool_name, tool_id, input).await
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

    /// Execute a tool (permission already granted)
    pub async fn execute(
        internals: &mut AgentInternals,
        tools: &ToolRegistry,
        tool_name: &str,
        tool_id: &str,
        input: &Value,
    ) -> ToolResult {
        // Update state
        internals.set_executing_tool(tool_name, "").await;

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
            Ok(result) => result,
            Err(e) => ToolResult::error(format!("Tool execution failed: {}", e)),
        };

        // Log tool result if debugger is enabled
        if let Some(debugger) = internals.context.get_resource::<Debugger>() {
            if let Err(e) = debugger.log_tool_result(tool_name, tool_id, &result) {
                tracing::warn!("[Executor] Failed to log tool result: {}", e);
            }
        }

        // Send tool end notification
        internals.send_tool_end(tool_name, result.clone());

        result
    }
}
