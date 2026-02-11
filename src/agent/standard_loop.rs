//! Standard Agent Loop
//!
//! The main agent implementation that handles:
//! - Input → LLM → Tools → Output cycle
//! - Context injection before LLM calls
//! - Session persistence
//! - Debug logging (when enabled)
//! - Streaming responses (when enabled)
//! - Automatic conversation naming (after first turn)

use std::sync::Arc;

use anyhow::Result;
use futures::StreamExt;
use serde_json::Value;

use crate::core::{FrameworkResult, InputMessage};
use crate::helpers::{process_attachments, ConversationNamer, Debugger};
use crate::hooks::HookContext;
use crate::llm::{
    CacheControl, ContentBlock, ContentBlockStart, ContentDelta, LlmProvider, Message,
    StopReason, StreamEvent, SystemBlock, SystemPrompt,
};
use crate::runtime::AgentInternals;
use crate::tools::{ToolResult, ToolResultData};

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
    llm: Arc<dyn LlmProvider>,
}

impl StandardAgent {
    /// Create a new standard agent
    pub fn new(config: AgentConfig, llm: Arc<dyn LlmProvider>) -> Self {
        Self { config, llm }
    }

    /// Run the agent loop
    ///
    /// This is the main entry point - pass this to `runtime.spawn()`.
    pub async fn run(self, mut internals: AgentInternals) -> FrameworkResult<()> {
        tracing::info!("[StandardAgent] Started, waiting for input...");

        // Write initial model/provider info into session metadata
        {
            let mut session = internals.session.write().await;
            session.set_model(self.llm.model());
            session.set_provider(self.llm.provider_name());
        }

        // Initialize debugger if enabled
        if self.config.debug_enabled {
            let session = internals.session.read().await;
            let session_dir = session.storage().session_dir(session.session_id());
            drop(session);

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

                    // Run UserPromptSubmit hooks
                    let mut current_text = text.clone();
                    let mut should_process = true;

                    if let Some(ref hooks) = self.config.hooks {
                        let mut ctx = HookContext::user_prompt_submit(&mut internals, &text);
                        let result = hooks.run(&mut ctx);

                        // Hook may have modified the prompt
                        if let Some(modified) = ctx.user_prompt {
                            current_text = modified;
                        }

                        // Check if hook denied the prompt
                        if let Some(crate::hooks::PermissionDecision::Deny) = result.decision {
                            let reason = result.reason.unwrap_or_else(|| "Blocked by hook".to_string());
                            tracing::info!("[StandardAgent] UserPromptSubmit hook denied: {}", reason);
                            internals.send_error(format!("Prompt blocked: {}", reason));
                            should_process = false;
                        }
                    }

                    tracing::info!("[StandardAgent] Should process: {}", should_process);

                    // Process the user message (if not blocked by hook)
                    if should_process {
                        if let Err(e) = self.process_turn(&mut internals, &current_text).await {
                            tracing::error!("[StandardAgent] Error processing turn: {}", e);
                            internals.send_error(format!("Error: {}", e));
                        }

                        if self.config.auto_name_conversation && internals.context.current_turn == 0
                        {
                            let session_id = {
                                let session = internals.session.read().await;
                                session.session_id().to_string()
                            };
                            let has_name = internals.session.read().await.has_conversation_name();
                            if !has_name {
                                self.generate_conversation_name(&mut internals, Some(&session_id)).await;
                            }
                        }
                    }
                    // Signal turn complete
                    internals.send_done();

                    // Persist session if configured
                    if self.config.auto_save_session {
                        if let Err(e) = internals.session.write().await.save() {
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

    /// Generate a conversation name using the ConversationNamer helper
    async fn generate_conversation_name(&self, internals: &mut AgentInternals, session_id: Option<&str>) {
        tracing::debug!("[StandardAgent] Generating conversation name...");

        // Use naming LLM if configured, otherwise fall back to main LLM
        let naming_llm = self.config.naming_llm.clone().unwrap_or_else(|| self.llm.clone());
        let namer = ConversationNamer::new(naming_llm);
        let history = {
            let session = internals.session.read().await;
            session.history().to_vec()
        };

        match namer.generate_name(&history, session_id).await {
            Ok(name) => {
                tracing::info!("[StandardAgent] Generated conversation name: {}", name);
                let mut session = internals.session.write().await;
                if let Err(e) = session.set_conversation_name(&name) {
                    tracing::warn!(
                        "[StandardAgent] Failed to save conversation name: {}",
                        e
                    );
                }
            }
            Err(e) => {
                tracing::warn!("[StandardAgent] Failed to generate conversation name: {}", e);
            }
        }
    }

    /// Process a single user turn (may involve multiple LLM calls for tool use)
    async fn process_turn(&self, internals: &mut AgentInternals, user_input: &str) -> Result<()> {
        // Check if input contains attachment tags and process them
        let user_message = if user_input.contains("<vibe-work-attachment>") {
            tracing::info!("[StandardAgent] Processing attachments in user input");

            // Get base directory from current working directory
            let base_dir = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .to_string_lossy()
                .to_string();

            // Process attachments
            let attachment_blocks = process_attachments(user_input, &base_dir);

            // Build message blocks: original text first, then attachments
            let mut blocks = vec![ContentBlock::Text {
                text: user_input.to_string(),
                cache_control: None,
            }];
            blocks.extend(attachment_blocks);

            Message::user_with_blocks(blocks)
        } else {
            // No attachments, use simple text message
            Message::user(user_input)
        };

        // Add user message to history
        internals.session.write().await.add_message(user_message)?;

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

            // Get messages from history
            let messages = {
                let session = internals.session.read().await;
                session.history().to_vec()
            };

            // IMPORTANT: Apply cache control BEFORE injections
            // This ensures we cache the stable message content (without dynamic injections)
            // The injections will be added AFTER the cache breakpoint, so they're sent but not cached
            // This allows the cache to match across turns even though injections are dynamic
            let (tools_with_cache, system_with_cache, mut messages_with_cache) =
                self.apply_cache_control(tool_definitions.to_vec(), messages);

            // Apply context injections AFTER cache control
            messages_with_cache = self.config.injections.apply(internals, messages_with_cache);

            // Update session metadata with current model/provider (may change via SwappableLlmProvider)
            {
                let mut session = internals.session.write().await;
                session.set_model(self.llm.model());
                session.set_provider(self.llm.provider_name());
            }

            tracing::info!(
                "[StandardAgent] Calling LLM with {} messages (iteration {})",
                messages_with_cache.len(),
                iterations
            );

            // Log API request if debugger is enabled (with cache_control included)
            if let Some(debugger) = internals.context.get_resource::<Debugger>() {
                let tool_defs: Vec<serde_json::Value> = tools_with_cache
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect();

                // Convert SystemPrompt to string for logging (or serialize as-is)
                let system_str = match &system_with_cache {
                    Some(SystemPrompt::Text(s)) => Some(s.as_str()),
                    Some(SystemPrompt::Blocks(_)) => {
                        // For blocks, we'll serialize them so cache_control is visible
                        None // Will serialize full structure below
                    }
                    None => None,
                };

                // If we have system blocks, we need to log them differently
                if let Some(SystemPrompt::Blocks(_)) = &system_with_cache {
                    // Log the full request with SystemPrompt blocks
                    if let Err(e) = debugger.log_api_request_full(
                        &messages_with_cache,
                        system_with_cache.clone(),
                        Some(&tool_defs),
                    ) {
                        tracing::warn!("[StandardAgent] Failed to log API request: {}", e);
                    }
                } else {
                    // Legacy path for simple string system prompt
                    if let Err(e) = debugger.log_api_request(
                        &messages_with_cache,
                        system_str,
                        Some(&tool_defs),
                    ) {
                        tracing::warn!("[StandardAgent] Failed to log API request: {}", e);
                    }
                }
            }

            // Choose streaming or non-streaming based on config
            // Pass the already-cache-controlled data
            let (content_blocks, stop_reason) = if self.config.streaming_enabled {
                self.call_llm_streaming_with_cache(
                    internals,
                    messages_with_cache,
                    tools_with_cache,
                    system_with_cache,
                )
                .await?
            } else {
                self.call_llm_non_streaming_with_cache(
                    internals,
                    messages_with_cache,
                    tools_with_cache,
                    system_with_cache,
                )
                .await?
            };

            tracing::info!(
                "[StandardAgent] LLM response: stop_reason={:?}",
                stop_reason
            );

            // Process tool use blocks and execute tools
            let mut tool_results: Vec<(String, ToolResult)> = Vec::new();

            for (index, block) in content_blocks.iter().enumerate() {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    tracing::info!("[StandardAgent] Tool use: {} ({})", name, id);

                    // Execute tool with permission check (if tools configured)
                    let result = if let Some(ref tools) = self.config.tools {
                        let hooks = self.config.hooks.as_deref();
                        ToolExecutor::execute_with_permission(internals, tools, hooks, name, id, input)
                            .await
                    } else {
                        ToolResult::error(format!(
                            "No tools configured, cannot execute: {}",
                            name
                        ))
                    };

                    tool_results.push((id.clone(), result));

                    // Check if user interrupted after tool execution (non-blocking check)
                    // Use tokio::select with immediate timeout to check without blocking
                    let interrupt_check = tokio::time::timeout(
                        std::time::Duration::from_millis(0),
                        internals.receive()
                    );

                    if let Ok(Some(InputMessage::Interrupt)) = interrupt_check.await {
                        tracing::info!("[StandardAgent] Interrupt detected after tool execution");

                        // For all remaining tools that haven't executed, add "Interrupted" error
                        for remaining_block in content_blocks.iter().skip(index + 1) {
                            if let ContentBlock::ToolUse { id: remaining_id, .. } = remaining_block {
                                tool_results.push((remaining_id.clone(), ToolResult::error("Interrupted")));
                            }
                        }

                        break;
                    }
                }
            }

            // Add assistant message to history
            internals
                .session
                .write()
                .await
                .add_message(Message::assistant_with_blocks(content_blocks.clone()))?;

            // Run PostAssistantResponse hooks
            if let Some(ref hooks) = self.config.hooks {
                let mut ctx = HookContext::post_assistant_response(
                    internals,
                    &content_blocks,
                    stop_reason.clone(),
                );
                let _result = hooks.run(&mut ctx);
                // Hook can access/modify history via internals if needed
                // Primarily used for logging and monitoring
            }

            // Check if any tool was interrupted
            let has_interrupt = tool_results.iter().any(|(_, result)| {
                result.is_error && matches!(&result.content, ToolResultData::Text(text) if text == "Interrupted")
            });

            if has_interrupt {
                tracing::info!("[StandardAgent] Tool execution interrupted, ending turn");
                // Add the interrupt results to history
                let tool_result_blocks: Vec<ContentBlock> = tool_results
                    .into_iter()
                    .flat_map(|(id, result)| {
                        match result.content {
                            ToolResultData::Text(text) => {
                                vec![ContentBlock::tool_result(&id, &text, result.is_error)]
                            }
                            _ => vec![]
                        }
                    })
                    .collect();

                internals
                    .session
                    .write()
                    .await
                    .add_message(Message::user_with_blocks(tool_result_blocks))?;

                // Add system message indicating the interrupt
                internals
                    .session
                    .write()
                    .await
                    .add_message(Message::assistant("<vibe-working-agent-system>User interrupted this message</vibe-working-agent-system>"))?;

                // Break out of the loop
                break;
            }

            // If there were tool calls, add results and continue loop
            if !tool_results.is_empty() {
                // Add tool results as a message (WITHOUT cache_control)
                // Cache control will be applied dynamically in apply_cache_control()
                let tool_result_blocks: Vec<ContentBlock> = tool_results
                    .into_iter()
                    .flat_map(|(id, result)| {
                        match result.content {
                            ToolResultData::Text(text) => {
                                vec![ContentBlock::tool_result(&id, &text, result.is_error)]
                            }
                            ToolResultData::Image { data, media_type } => {
                                // Encode image data to base64
                                use base64::Engine;
                                let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

                                vec![ContentBlock::ToolResult {
                                    tool_use_id: id,
                                    content: None,
                                    is_error: if result.is_error { Some(true) } else { None },
                                    cache_control: None,
                                }, ContentBlock::image(base64_data, media_type)]
                            }
                            ToolResultData::Document {
                                data,
                                media_type,
                                description,
                            } => {
                                // Encode document data to base64
                                use base64::Engine;
                                let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);

                                // For PDFs: two separate blocks as per API spec
                                vec![
                                    ContentBlock::tool_result(&id, &description, result.is_error),
                                    ContentBlock::document(base64_data, media_type),
                                ]
                            }
                        }
                    })
                    .collect();

                internals
                    .session
                    .write()
                    .await
                    .add_message(Message::user_with_blocks(tool_result_blocks))?;

                // Continue to next LLM call
                continue;
            }

            // No tool calls - check if we should stop
            match stop_reason {
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

    /// Apply cache control to tools, system prompt, and messages (if enabled)
    fn apply_cache_control(
        &self,
        mut tool_definitions: Vec<crate::llm::ToolDefinition>,
        mut messages: Vec<Message>,
    ) -> (Vec<crate::llm::ToolDefinition>, Option<SystemPrompt>, Vec<Message>) {
        if !self.config.enable_prompt_caching {
            // Caching disabled - return system prompt as simple text
            return (
                tool_definitions,
                Some(SystemPrompt::Text(self.config.system_prompt.clone())),
                messages,
            );
        }

        // IMPORTANT: Strip ALL existing cache_control from messages first
        // This ensures we don't accidentally create duplicate cache breakpoints
        for message in &mut messages {
            if let crate::llm::MessageContent::Blocks(blocks) = &mut message.content {
                for block in blocks {
                    match block {
                        ContentBlock::Text { cache_control, .. } => {
                            *cache_control = None;
                        }
                        ContentBlock::ToolResult { cache_control, .. } => {
                            *cache_control = None;
                        }
                        _ => {}
                    }
                }
            }
        }

        // 1. Add cache control to last tool definition (caches all tools)
        if let Some(last_tool) = tool_definitions.last_mut() {
            *last_tool = last_tool.clone().with_cache_control(CacheControl::ephemeral());
        }

        // 2. Create system prompt with cache control
        let system_prompt = Some(SystemPrompt::Blocks(vec![SystemBlock::new(
            self.config.system_prompt.clone(),
        )
        .with_cache_control(CacheControl::ephemeral())]));

        // 3. Add cache control to the last content block of the LAST message
        // This caches everything including the current user input, creating a stable growing cache
        // Next request will have this content cached, allowing prefix matching
        if let Some(last_message) = messages.last_mut() {
            match &mut last_message.content {
                crate::llm::MessageContent::Text(text) => {
                    // Convert to blocks format with cache control on the text
                    last_message.content = crate::llm::MessageContent::Blocks(vec![
                        ContentBlock::Text {
                            text: text.clone(),
                            cache_control: Some(CacheControl::ephemeral()),
                        },
                    ]);
                }
                crate::llm::MessageContent::Blocks(blocks) => {
                    // Add cache control to the last block
                    if let Some(last_block) = blocks.last_mut() {
                        *last_block = last_block.clone().with_cache_control(CacheControl::ephemeral());
                    }
                }
            }
        }

        (tool_definitions, system_prompt, messages)
    }

    /// Call LLM without streaming (with pre-applied cache control)
    async fn call_llm_non_streaming_with_cache(
        &self,
        internals: &mut AgentInternals,
        messages: Vec<Message>,
        tools: Vec<crate::llm::ToolDefinition>,
        system: Option<SystemPrompt>,
    ) -> Result<(Vec<ContentBlock>, Option<StopReason>)> {
        // Get session ID
        let session_id = {
            let session = internals.session.read().await;
            session.session_id().to_string()
        };

        let response = self
            .llm
            .send_with_tools_and_system(
                messages,
                system,
                tools,
                None,
                self.config.thinking.clone(),
                Some(&session_id),
            )
            .await?;

        // Log API response if debugger is enabled
        if let Some(debugger) = internals.context.get_resource::<Debugger>() {
            if let Ok(response_json) = serde_json::to_value(&response) {
                if let Err(e) = debugger.log_api_response(&response_json) {
                    tracing::warn!("[StandardAgent] Failed to log API response: {}", e);
                }
            }
        }

        // Send text and thinking content to output
        for block in &response.content {
            match block {
                ContentBlock::Text { text, .. } => {
                    internals.send_text(text);
                    internals.send_text_complete(text);
                }
                ContentBlock::Thinking { thinking, .. } => {
                    internals.send_thinking(thinking);
                    internals.send_thinking_complete(thinking);
                }
                _ => {}
            }
        }

        Ok((response.content, response.stop_reason))
    }

    /// Call LLM with streaming (with pre-applied cache control) - sends deltas in real-time
    async fn call_llm_streaming_with_cache(
        &self,
        internals: &mut AgentInternals,
        messages: Vec<Message>,
        tools: Vec<crate::llm::ToolDefinition>,
        system: Option<SystemPrompt>,
    ) -> Result<(Vec<ContentBlock>, Option<StopReason>)> {
        // Get session ID
        let session_id = {
            let session = internals.session.read().await;
            session.session_id().to_string()
        };

        let mut stream = self
            .llm
            .stream_with_tools_and_system(
                messages,
                system,
                tools,
                None,
                self.config.thinking.clone(),
                Some(&session_id),
            )
            .await?;

        // Track content blocks as they're built
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut current_block_index: Option<usize> = None;
        let mut stop_reason: Option<StopReason> = None;

        // Track message metadata for logging
        let mut message_id: Option<String> = None;
        let mut model: Option<String> = None;
        let mut initial_usage: Option<crate::llm::Usage> = None;
        let mut output_tokens: u32 = 0;

        // Accumulators for building content blocks
        let mut text_accum = String::new();
        let mut thinking_accum = String::new();
        let mut thinking_signature = String::new();
        let mut tool_input_accum = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();

        loop {
            tokio::select! {
                event_result = stream.next() => {
                    let event_result = match event_result {
                        Some(result) => result,
                        None => break, // Stream ended
                    };

                    match event_result {
                        Ok(event) => {
                            match event {
                                StreamEvent::MessageStart(msg_start) => {
                                    tracing::debug!("[StandardAgent] Stream started");
                                    // Capture message metadata for logging
                                    message_id = Some(msg_start.message.id.clone());
                                    model = Some(msg_start.message.model.clone());
                                    initial_usage = Some(msg_start.message.usage.clone());
                                }

                        StreamEvent::ContentBlockStart(block_start) => {
                            current_block_index = Some(block_start.index);

                            match &block_start.content_block {
                                ContentBlockStart::Text { .. } => {
                                    text_accum.clear();
                                }
                                ContentBlockStart::Thinking { .. } => {
                                    thinking_accum.clear();
                                    thinking_signature.clear();
                                }
                                ContentBlockStart::ToolUse { id, name, .. } => {
                                    tool_input_accum.clear();
                                    current_tool_id = id.clone();
                                    current_tool_name = name.clone();
                                }
                            }
                        }

                        StreamEvent::ContentBlockDelta(delta) => {
                            match &delta.delta {
                                ContentDelta::TextDelta { text } => {
                                    text_accum.push_str(text);
                                    // Stream text to output immediately
                                    internals.send_text(text);
                                }
                                ContentDelta::ThinkingDelta { thinking } => {
                                    thinking_accum.push_str(thinking);
                                    // Stream thinking to output immediately
                                    internals.send_thinking(thinking);
                                }
                                ContentDelta::SignatureDelta { signature } => {
                                    thinking_signature.push_str(signature);
                                }
                                ContentDelta::InputJsonDelta { partial_json } => {
                                    tool_input_accum.push_str(partial_json);
                                }
                            }
                        }

                        StreamEvent::ContentBlockStop(block_stop) => {
                            if current_block_index == Some(block_stop.index) {
                                // Finalize the content block
                                if !text_accum.is_empty() {
                                    // Send text complete signal to CLI
                                    internals.send_text_complete(&text_accum);
                                    content_blocks.push(ContentBlock::Text {
                                        text: text_accum.clone(),
                                        cache_control: None,
                                    });
                                    text_accum.clear();
                                } else if !thinking_accum.is_empty() {
                                    // Send thinking complete signal to CLI
                                    internals.send_thinking_complete(&thinking_accum);
                                    content_blocks.push(ContentBlock::Thinking {
                                        thinking: thinking_accum.clone(),
                                        signature: thinking_signature.clone(),
                                    });
                                    thinking_accum.clear();
                                    thinking_signature.clear();
                                } else if !tool_input_accum.is_empty()
                                    || !current_tool_name.is_empty()
                                {
                                    // Parse accumulated JSON
                                    let input: Value =
                                        serde_json::from_str(&tool_input_accum).unwrap_or_default();
                                    content_blocks.push(ContentBlock::ToolUse {
                                        id: current_tool_id.clone(),
                                        name: current_tool_name.clone(),
                                        input,
                                    });
                                    tool_input_accum.clear();
                                    current_tool_id.clear();
                                    current_tool_name.clear();
                                }
                                current_block_index = None;
                            }
                        }

                        StreamEvent::MessageDelta(msg_delta) => {
                            stop_reason = msg_delta.delta.stop_reason;
                            // Capture final output tokens
                            output_tokens = msg_delta.usage.output_tokens;
                        }

                        StreamEvent::MessageStop => {
                            tracing::debug!("[StandardAgent] Stream complete");
                        }

                        StreamEvent::Ping => {
                            tracing::trace!("[StandardAgent] Ping");
                        }

                        StreamEvent::Error(err) => {
                            tracing::error!(
                                "[StandardAgent] Stream error: {}: {}",
                                err.error.error_type,
                                err.error.message
                            );
                            internals.send_error(format!(
                                "Stream error: {}",
                                err.error.message
                            ));
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("[StandardAgent] Stream error: {}", e);
                    return Err(e);
                }
            }
                }

                // Check for interrupt messages
                msg = internals.receive() => {
                    if let Some(InputMessage::Interrupt) = msg {
                        tracing::info!("[StandardAgent] Interrupt received");

                        // Finalize any in-progress text content block
                        if !text_accum.is_empty() {
                            content_blocks.push(ContentBlock::Text {
                                text: text_accum.clone(),
                                cache_control: None,
                            });
                        }
                        // Discard incomplete thinking blocks (signature may be incomplete)
                        // Discard partial tool calls (don't add them to content_blocks)

                        // Remove all ToolUse blocks from content_blocks (discard all tool calls)
                        content_blocks.retain(|block| !matches!(block, ContentBlock::ToolUse { .. }));

                        // Append interrupt notification to the assistant's content blocks
                        content_blocks.push(ContentBlock::Text {
                            text: "<vibe-working-agent-system>User interrupted this message</vibe-working-agent-system>".to_string(),
                            cache_control: None,
                        });

                        break;
                    }
                }
            }
        }

        // Log the assembled response if debugger is enabled
        if let Some(debugger) = internals.context.get_resource::<Debugger>() {
            // Construct a response object similar to MessageResponse for logging
            let mut response_for_logging = serde_json::json!({
                "id": message_id.unwrap_or_else(|| "unknown".to_string()),
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": model.unwrap_or_else(|| "streamed".to_string()),
                "stop_reason": stop_reason,
            });

            // Add usage information if we captured it
            if let Some(usage) = initial_usage {
                let usage_obj = serde_json::json!({
                    "input_tokens": usage.input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                    "cache_read_input_tokens": usage.cache_read_input_tokens,
                });
                response_for_logging["usage"] = usage_obj;
            }

            if let Err(e) = debugger.log_api_response(&response_for_logging) {
                tracing::warn!("[StandardAgent] Failed to log streaming API response: {}", e);
            }
        }

        Ok((content_blocks, stop_reason))
    }
}
