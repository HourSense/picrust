//! Console Renderer - Subscribes to an agent and renders output to terminal
//!
//! The `ConsoleRenderer` is an opt-in component that:
//! - Subscribes to an agent's output stream
//! - Renders streaming text, thinking, tool calls to the terminal
//! - Handles user input and permission requests
//! - Is completely decoupled from the agent logic
//!
//! This can be replaced with other renderers (Tauri UI, Web UI, etc.)

use std::io::{self, Write};
use std::sync::Arc;

use crate::core::{InputMessage, OutputChunk};
use crate::helpers::TodoListManager;
use crate::permissions::PermissionDecision;
use crate::runtime::AgentHandle;

use super::console::Console;

/// Console renderer that subscribes to an agent and handles terminal I/O
///
/// # Example
///
/// ```ignore
/// let handle = runtime.spawn(session, agent_fn).await;
/// let renderer = ConsoleRenderer::new(handle);
/// renderer.run().await?;
/// ```
pub struct ConsoleRenderer {
    /// The agent handle to communicate with
    handle: AgentHandle,

    /// The console for formatted output
    console: Console,

    /// Whether to show thinking blocks
    show_thinking: bool,

    /// Whether to show tool execution details
    show_tools: bool,
}

impl ConsoleRenderer {
    /// Create a new console renderer for an agent
    pub fn new(handle: AgentHandle) -> Self {
        Self {
            handle,
            console: Console::new(),
            show_thinking: true,
            show_tools: true,
        }
    }

    /// Create a renderer with a custom console
    pub fn with_console(handle: AgentHandle, console: Console) -> Self {
        Self {
            handle,
            console,
            show_thinking: true,
            show_tools: true,
        }
    }

    /// Set whether to show thinking blocks
    pub fn show_thinking(mut self, show: bool) -> Self {
        self.show_thinking = show;
        self
    }

    /// Set whether to show tool execution details
    pub fn show_tools(mut self, show: bool) -> Self {
        self.show_tools = show;
        self
    }

    /// Set the todo manager for displaying task progress
    pub fn with_todo_manager(mut self, manager: Arc<TodoListManager>) -> Self {
        self.console.set_todo_manager(manager);
        self
    }

    /// Run the console renderer
    ///
    /// This starts the main loop that:
    /// 1. Reads user input
    /// 2. Sends it to the agent
    /// 3. Renders streaming output
    /// 4. Handles permission requests
    ///
    /// Returns when the user types "exit" or the agent shuts down.
    pub async fn run(&self) -> io::Result<()> {
        self.console.print_banner();

        loop {
            // Read user input
            let input = self.console.read_input()?;

            // Check for exit commands
            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                self.console.print_system("Shutting down...");
                let _ = self.handle.shutdown().await;
                break;
            }

            // Skip empty input
            if input.trim().is_empty() {
                continue;
            }

            // Send input to agent
            if let Err(e) = self.handle.send_input(&input).await {
                self.console.print_error(&format!("Failed to send input: {}", e));
                continue;
            }

            // Render the response
            if let Err(e) = self.render_response().await {
                self.console.print_error(&format!("Render error: {}", e));
            }

            self.console.println();
        }

        Ok(())
    }

    /// Run a single turn - send input and render response
    ///
    /// Use this for programmatic interaction instead of the full loop.
    pub async fn run_turn(&self, input: &str) -> io::Result<()> {
        // Send input to agent
        if let Err(e) = self.handle.send_input(input).await {
            self.console.print_error(&format!("Failed to send input: {}", e));
            return Ok(());
        }

        // Render the response
        self.render_response().await
    }

    /// Render the agent's response until Done or Error
    async fn render_response(&self) -> io::Result<()> {
        let mut rx = self.handle.subscribe();
        let mut in_text = false;
        let mut in_thinking = false;

        loop {
            match rx.recv().await {
                Ok(chunk) => {
                    match chunk {
                        // Text streaming
                        OutputChunk::TextDelta(text) => {
                            if !in_text {
                                self.console.print_assistant_prefix();
                                in_text = true;
                            }
                            self.console.print_assistant_chunk(&text);
                        }
                        OutputChunk::TextComplete(_) => {
                            if in_text {
                                self.console.println();
                                in_text = false;
                            }
                        }

                        // Thinking - stream in real-time
                        OutputChunk::ThinkingDelta(text) => {
                            if self.show_thinking {
                                if !in_thinking {
                                    self.console.print_thinking_prefix();
                                    in_thinking = true;
                                }
                                self.console.print_thinking_chunk(&text);
                            }
                        }
                        OutputChunk::ThinkingComplete(_) => {
                            if self.show_thinking && in_thinking {
                                self.console.print_thinking_suffix();
                                in_thinking = false;
                            }
                        }

                        // Tool execution
                        OutputChunk::ToolStart { name, .. } => {
                            if in_text {
                                self.console.println();
                                in_text = false;
                            }
                            if self.show_tools {
                                self.console.print_tool_action(&name, "executing...");
                            }
                        }
                        OutputChunk::ToolProgress { output, .. } => {
                            if self.show_tools {
                                print!("{}", output);
                                io::stdout().flush()?;
                            }
                        }
                        OutputChunk::ToolEnd { result, .. } => {
                            if self.show_tools {
                                use crate::tools::ToolResultData;
                                let output_text = match &result.content {
                                    ToolResultData::Text(text) => text.clone(),
                                    ToolResultData::Image { data, media_type } => {
                                        format!("Image ({}, {} bytes)", media_type, data.len())
                                    }
                                    ToolResultData::Document { description, data, media_type } => {
                                        format!("{} ({}, {} bytes)", description, media_type, data.len())
                                    }
                                };
                                self.console.print_tool_result(&output_text, result.is_error);
                            }
                        }

                        // Permission requests
                        OutputChunk::PermissionRequest { tool_name, action, input, details } => {
                            if in_text {
                                self.console.println();
                                in_text = false;
                            }

                            // Create a permission request for the console
                            let request = crate::permissions::PermissionRequest {
                                tool_name: tool_name.clone(),
                                action_description: action,
                                input,
                                details,
                            };

                            // Ask user
                            let decision = self.console.ask_permission(&request)?;

                            // Convert decision to allowed/remember
                            let (allowed, remember) = match decision {
                                PermissionDecision::Allow => (true, false),
                                PermissionDecision::Deny => (false, false),
                                PermissionDecision::AlwaysAllow => (true, true),
                                PermissionDecision::AlwaysDeny => (false, true),
                            };

                            // Send response back to agent
                            let _ = self.handle.send_permission_response(&tool_name, allowed, remember).await;
                        }

                        // User questions
                        OutputChunk::AskUserQuestion { request_id, questions } => {
                            if in_text {
                                self.console.println();
                                in_text = false;
                            }

                            // Display questions and collect answers
                            let mut answers = std::collections::HashMap::new();
                            for q in &questions {
                                self.console.print_system(&format!("[{}] {}", q.header, q.question));
                                for (i, opt) in q.options.iter().enumerate() {
                                    self.console.print_system(&format!("  {}. {} - {}", i + 1, opt.label, opt.description));
                                }
                                // For CLI, just use first option as default for now
                                // A full implementation would prompt user for input
                                if let Some(first_option) = q.options.first() {
                                    answers.insert(q.header.clone(), first_option.label.clone());
                                }
                            }

                            // Send response back to agent
                            let _ = self.handle.send(InputMessage::UserQuestionResponse {
                                request_id,
                                answers,
                            }).await;
                        }

                        // Status updates
                        OutputChunk::Status(status) => {
                            self.console.print_system(&status);
                        }
                        OutputChunk::StateChange(state) => {
                            // Could show state changes if desired
                            tracing::debug!("Agent state: {:?}", state);
                        }

                        // Completion
                        OutputChunk::Done => {
                            if in_text {
                                self.console.println();
                            }
                            break;
                        }
                        OutputChunk::Error(e) => {
                            if in_text {
                                self.console.println();
                            }
                            self.console.print_error(&e);
                            break;
                        }

                        // Subagent events (could render differently)
                        OutputChunk::SubAgentSpawned { session_id, agent_type } => {
                            self.console.print_system(&format!(
                                "Spawned subagent: {} ({})", agent_type, session_id
                            ));
                        }
                        OutputChunk::SubAgentComplete { session_id, result } => {
                            self.console.print_system(&format!(
                                "Subagent {} completed: {:?}", session_id, result
                            ));
                        }
                        OutputChunk::SubAgentOutput { chunk, .. } => {
                            // Could recursively render subagent output
                            tracing::debug!("Subagent output: {:?}", chunk);
                        }
                    }
                }
                Err(e) => {
                    // Channel closed or lagged
                    tracing::warn!("Output channel error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get the underlying agent handle
    pub fn handle(&self) -> &AgentHandle {
        &self.handle
    }

    /// Get the underlying console
    pub fn console(&self) -> &Console {
        &self.console
    }
}
