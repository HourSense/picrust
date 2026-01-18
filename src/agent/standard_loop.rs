//! Standard Agent Loop
//!
//! The main agent implementation that handles:
//! - Input → LLM → Tools → Output cycle
//! - Context injection before LLM calls
//! - Session persistence
//! - Debug logging (when enabled)

use std::sync::Arc;

use anyhow::Result;

use crate::core::{FrameworkResult, InputMessage};
use crate::helpers::Debugger;
use crate::llm::{AnthropicProvider, ContentBlock, Message, StopReason};
use crate::runtime::AgentInternals;
use crate::tools::ToolResult;

use super::config::AgentConfig;
use super::executor::ToolExecutor;

/// Standard agent that handles the full agent loop
///
/// # Example
///
/// ```ignore
/// let config = AgentConfig::new("You are helpful")
///     .with_tools(tools);
///
/// let agent = StandardAgent::new(config, llm);
///
/// let handle = runtime.spawn(session, |internals| {
///     agent.run(internals)
/// }).await;
/// ```
pub struct StandardAgent {
    config: AgentConfig,
    llm: Arc<AnthropicProvider>,
}

impl StandardAgent {
    /// Create a new standard agent
    pub fn new(config: AgentConfig, llm: Arc<AnthropicProvider>) -> Self {
        Self { config, llm }
    }

    /// Run the agent loop
    ///
    /// This is the main entry point - pass this to `runtime.spawn()`.
    pub async fn run(self, mut internals: AgentInternals) -> FrameworkResult<()> {
        tracing::info!("[StandardAgent] Started, waiting for input...");

        // Initialize debugger if enabled
        if self.config.debug_enabled {
            let session_dir = internals
                .session
                .storage()
                .session_dir(internals.session.session_id());

            match Debugger::new(&session_dir) {
                Ok(debugger) => {
                    tracing::info!(
                        "[StandardAgent] Debug logging enabled at {:?}",
                        debugger.dir()
                    );
                    internals.context.insert_resource(debugger);
                }
                Err(e) => {
                    tracing::warn!("[StandardAgent] Failed to initialize debugger: {}", e);
                }
            }
        }

        loop {
            // Signal we're ready for input
            internals.set_idle().await;

            // Wait for next message
            match internals.receive().await {
                Some(InputMessage::UserInput(text)) => {
                    tracing::info!("[StandardAgent] Received: {}", text);
                    internals.set_processing().await;

                    // Process the user message
                    if let Err(e) = self.process_turn(&mut internals, &text).await {
                        tracing::error!("[StandardAgent] Error processing turn: {}", e);
                        internals.send_error(format!("Error: {}", e));
                    }

                    // Signal turn complete
                    internals.send_done();

                    // Persist session if configured
                    if self.config.auto_save_session {
                        if let Err(e) = internals.session.save() {
                            tracing::error!("[StandardAgent] Failed to save session: {}", e);
                        }
                    }
                }

                Some(InputMessage::Interrupt) => {
                    tracing::info!("[StandardAgent] Interrupted");
                    internals.send_status("Interrupted");
                    internals.set_done().await;
                    break;
                }

                Some(InputMessage::Shutdown) | None => {
                    tracing::info!("[StandardAgent] Shutting down");
                    internals.set_done().await;
                    break;
                }

                _ => {
                    // Ignore other message types
                }
            }

            internals.next_turn();
        }

        Ok(())
    }

    /// Process a single user turn (may involve multiple LLM calls for tool use)
    async fn process_turn(&self, internals: &mut AgentInternals, user_input: &str) -> Result<()> {
        // Add user message to history
        internals.session.add_message(Message::user(user_input))?;

        // Get tool definitions
        let tool_definitions = self.config.tool_definitions();

        let mut iterations = 0;

        // LLM loop - continues until no more tool calls
        loop {
            iterations += 1;
            if iterations > self.config.max_tool_iterations {
                tracing::warn!(
                    "[StandardAgent] Max tool iterations ({}) reached",
                    self.config.max_tool_iterations
                );
                internals.send_status("Max tool iterations reached");
                break;
            }

            // Get messages and apply context injections
            let mut messages = internals.session.history().to_vec();
            messages = self.config.injections.apply(internals, messages);

            tracing::info!(
                "[StandardAgent] Calling LLM with {} messages (iteration {})",
                messages.len(),
                iterations
            );

            // Log API request if debugger is enabled
            if let Some(debugger) = internals.context.get_resource::<Debugger>() {
                let tool_defs: Vec<serde_json::Value> = tool_definitions
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect();

                if let Err(e) = debugger.log_api_request(
                    &messages,
                    Some(&self.config.system_prompt),
                    Some(&tool_defs),
                ) {
                    tracing::warn!("[StandardAgent] Failed to log API request: {}", e);
                }
            }

            // Call LLM
            let response = self
                .llm
                .send_with_tools(
                    messages,
                    Some(&self.config.system_prompt),
                    tool_definitions.clone(),
                    None,
                    None,
                )
                .await?;

            tracing::info!(
                "[StandardAgent] LLM response: stop_reason={:?}",
                response.stop_reason
            );

            // Log API response if debugger is enabled
            if let Some(debugger) = internals.context.get_resource::<Debugger>() {
                if let Ok(response_json) = serde_json::to_value(&response) {
                    if let Err(e) = debugger.log_api_response(&response_json) {
                        tracing::warn!("[StandardAgent] Failed to log API response: {}", e);
                    }
                }
            }

            // Process response content blocks
            let mut tool_results: Vec<(String, ToolResult)> = Vec::new();

            for block in &response.content {
                match block {
                    ContentBlock::Text { text } => {
                        internals.send_text(text);
                    }

                    ContentBlock::Thinking { thinking, .. } => {
                        internals.send_thinking(thinking);
                    }

                    ContentBlock::ToolUse { id, name, input } => {
                        tracing::info!("[StandardAgent] Tool use: {} ({})", name, id);

                        // Execute tool with permission check (if tools configured)
                        let result = if let Some(ref tools) = self.config.tools {
                            ToolExecutor::execute_with_permission(internals, tools, name, id, input)
                                .await
                        } else {
                            ToolResult::error(format!("No tools configured, cannot execute: {}", name))
                        };

                        tool_results.push((id.clone(), result));
                    }

                    _ => {}
                }
            }

            // Add assistant message to history
            internals
                .session
                .add_message(Message::assistant_with_blocks(response.content.clone()))?;

            // If there were tool calls, add results and continue loop
            if !tool_results.is_empty() {
                // Add tool results as a message
                let tool_result_blocks: Vec<ContentBlock> = tool_results
                    .into_iter()
                    .map(|(id, result)| {
                        ContentBlock::tool_result(&id, &result.output, result.is_error)
                    })
                    .collect();

                internals
                    .session
                    .add_message(Message::user_with_blocks(tool_result_blocks))?;

                // Continue to next LLM call
                continue;
            }

            // No tool calls - check if we should stop
            match response.stop_reason {
                Some(StopReason::EndTurn) | Some(StopReason::StopSequence) | None => {
                    // Done with this turn
                    break;
                }
                Some(StopReason::ToolUse) => {
                    // Shouldn't happen if tool_results is empty, but continue just in case
                    continue;
                }
                Some(StopReason::MaxTokens) => {
                    internals.send_status("Response truncated (max tokens)");
                    break;
                }
                Some(StopReason::PauseTurn) => {
                    // Model paused, wait for next input
                    break;
                }
                Some(StopReason::Refusal) => {
                    internals.send_status("Model refused to respond");
                    break;
                }
            }
        }

        Ok(())
    }
}
