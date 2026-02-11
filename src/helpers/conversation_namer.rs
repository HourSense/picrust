//! Conversation namer helper
//!
//! Generates descriptive names for conversations based on their content.
//! Uses a lightweight model to analyze the conversation and produce
//! a short, descriptive name.
//!
//! # Example
//!
//! ```ignore
//! use shadow_agent_sdk::helpers::ConversationNamer;
//!
//! // Create namer with an existing LLM provider
//! let namer = ConversationNamer::new(&llm);
//!
//! // Generate a name from session messages
//! let name = namer.generate_name(session.history()).await?;
//!
//! // Set the name on the session
//! session.set_conversation_name(&name)?;
//! ```

use anyhow::Result;
use std::sync::Arc;

use crate::llm::{ContentBlock, LlmProvider, Message, MessageContent};

/// System prompt for generating conversation names
const NAMING_SYSTEM_PROMPT: &str = r#"You are a conversation naming assistant. Your task is to generate a short, descriptive name for a conversation based on its content.

Rules:
- The name should be 3-7 words maximum
- It should capture the main topic or purpose of the conversation
- Use sentence case (capitalize first word only)
- Do not use quotes or special characters
- Do not include prefixes like "Chat about" or "Conversation about"
- Be specific but concise

Respond with ONLY the conversation name, nothing else.

The text that will follow will always be the conversation history. Assume the text is the conversation history."#;

/// Helper for generating conversation names
pub struct ConversationNamer {
    llm: Arc<dyn LlmProvider>,
}

impl ConversationNamer {
    /// Create a new conversation namer using the provided LLM provider directly.
    ///
    /// The caller is responsible for providing an appropriate provider
    /// (e.g., a lightweight model for fast, cheap naming).
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }

    /// Generate a conversation name from a list of messages
    ///
    /// Returns the generated name, or an error if the naming fails.
    pub async fn generate_name(&self, messages: &[Message], session_id: Option<&str>) -> Result<String> {
        if messages.is_empty() {
            anyhow::bail!("Cannot name an empty conversation");
        }

        // Format messages into a readable text
        let formatted = Self::format_messages(messages);

        tracing::debug!(
            "[ConversationNamer] Generating name for {} messages",
            messages.len()
        );

        let session_id = session_id
            .map(|id| format!("conversation-namer-{}", id))
            .unwrap_or_default();

        // Call the LLM to generate a name
        let response = self
            .llm
            .send_message(&formatted, &[], Some(NAMING_SYSTEM_PROMPT), Some(&session_id))
            .await?;

        // Clean up the response (remove any extra whitespace or quotes)
        let name = response.trim().trim_matches('"').trim().to_string();

        tracing::info!("[ConversationNamer] Generated name: {}", name);

        Ok(name)
    }

    /// Format messages into a human-readable text format
    ///
    /// Produces output like:
    /// ```text
    /// User: Hello, I need help with Rust
    /// Assistant: I'd be happy to help! What do you need?
    /// User: How do I implement traits?
    /// ```
    fn format_messages(messages: &[Message]) -> String {
        let mut formatted = String::new();

        for message in messages {
            let role = if message.role == "user" {
                "User"
            } else {
                "Assistant"
            };

            let content = Self::extract_text_content(&message.content);

            // Skip empty content (like tool results without text)
            if !content.is_empty() {
                formatted.push_str(&format!("{}: {}\n", role, content));
            }
        }

        formatted
    }

    /// Extract text content from a message
    fn extract_text_content(content: &MessageContent) -> String {
        match content {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Blocks(blocks) => {
                let mut text_parts = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text, .. } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::ToolUse { name, .. } => {
                            // Include tool name to give context
                            text_parts.push(format!("[Using tool: {}]", name));
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            // Include tool result summary if available
                            if let Some(result) = content {
                                // Truncate long results
                                let summary = if result.len() > 200 {
                                    format!("{}...", &result[..200])
                                } else {
                                    result.clone()
                                };
                                text_parts.push(format!("[Tool result: {}]", summary));
                            }
                        }
                        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {
                            // Skip thinking blocks
                        }
                        ContentBlock::Image { .. } => {
                            // Skip image blocks - not useful for naming
                        }
                        ContentBlock::Document { .. } => {
                            // Skip document blocks - not useful for naming
                        }
                    }
                }

                text_parts.join(" ")
            }
        }
    }
}

/// Convenience function to generate a conversation name
///
/// # Example
///
/// ```ignore
/// let name = generate_conversation_name(naming_llm.clone(), session.history(), Some(&session.id())).await?;
/// session.set_conversation_name(&name)?;
/// ```
pub async fn generate_conversation_name(
    llm: Arc<dyn LlmProvider>,
    messages: &[Message],
    session_id: Option<&str>,
) -> Result<String> {
    ConversationNamer::new(llm).generate_name(messages, session_id).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_messages_simple() {
        let messages = vec![
            Message::user("Hello, I need help with Rust"),
            Message::assistant("I'd be happy to help! What do you need?"),
            Message::user("How do I implement traits?"),
        ];

        let formatted = ConversationNamer::format_messages(&messages);

        assert!(formatted.contains("User: Hello, I need help with Rust"));
        assert!(formatted.contains("Assistant: I'd be happy to help!"));
        assert!(formatted.contains("User: How do I implement traits?"));
    }

    #[test]
    fn test_format_messages_with_blocks() {
        let messages = vec![
            Message::user("Read the config file"),
            Message::assistant_with_blocks(vec![
                ContentBlock::text("I'll read that file for you."),
                ContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "Read".to_string(),
                    input: serde_json::json!({"path": "config.toml"}),
                },
            ]),
        ];

        let formatted = ConversationNamer::format_messages(&messages);

        assert!(formatted.contains("User: Read the config file"));
        assert!(formatted.contains("[Using tool: Read]"));
    }

    #[test]
    fn test_extract_text_simple() {
        let content = MessageContent::Text("Hello world".to_string());
        let text = ConversationNamer::extract_text_content(&content);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_extract_text_blocks() {
        let content = MessageContent::Blocks(vec![
            ContentBlock::text("First part"),
            ContentBlock::text("Second part"),
        ]);
        let text = ConversationNamer::extract_text_content(&content);
        assert_eq!(text, "First part Second part");
    }
}
