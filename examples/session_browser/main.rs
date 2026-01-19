//! Session Browser Example
//!
//! Demonstrates the session listing and history API:
//! - List all top-level sessions (not subagents)
//! - Show conversation history for a specific session
//!
//! Run with: cargo run --example session_browser

use anyhow::Result;
use shadow_agent_sdk::llm::{ContentBlock, MessageContent};
use shadow_agent_sdk::session::AgentSession;

fn main() -> Result<()> {
    println!("=== Session Browser ===\n");

    // --- List all top-level sessions ---
    println!("Top-level sessions (not subagents):");
    println!("{}", "-".repeat(60));

    let top_level = AgentSession::list_with_metadata(true)?;

    if top_level.is_empty() {
        println!("  (no sessions found)");
    } else {
        for (session_id, metadata) in &top_level {
            println!("  Session: {}", session_id);
            println!("    Type: {}", metadata.agent_type);
            println!("    Name: {}", metadata.name);
            println!("    Description: {}", metadata.description);
            println!(
                "    Created: {}",
                metadata.created_at.format("%Y-%m-%d %H:%M:%S")
            );
            println!(
                "    Updated: {}",
                metadata.updated_at.format("%Y-%m-%d %H:%M:%S")
            );
            if !metadata.child_session_ids.is_empty() {
                println!("    Children: {:?}", metadata.child_session_ids);
            }
            println!();
        }
    }

    // --- Show conversation history for test-agent-session ---
    let target_session = "test-agent-session";
    println!("\n{}", "=".repeat(60));
    println!("Conversation history for: {}", target_session);
    println!("{}", "-".repeat(60));

    if AgentSession::exists(target_session) {
        let history = AgentSession::get_history(target_session)?;

        if history.is_empty() {
            println!("  (no messages in history)");
        } else {
            println!("  Total messages: {}\n", history.len());

            for (i, message) in history.iter().enumerate() {
                println!("  [{}] {}:", i + 1, message.role.to_uppercase());

                // Handle MessageContent enum
                match &message.content {
                    MessageContent::Text(text) => {
                        let display_text = if text.len() > 300 {
                            format!("{}...", &text[..300])
                        } else {
                            text.clone()
                        };
                        println!("      {}", display_text.replace('\n', "\n      "));
                    }
                    MessageContent::Blocks(blocks) => {
                        for block in blocks {
                            match block {
                                ContentBlock::Text { text } => {
                                    let display_text = if text.len() > 300 {
                                        format!("{}...", &text[..300])
                                    } else {
                                        text.clone()
                                    };
                                    println!("      {}", display_text.replace('\n', "\n      "));
                                }
                                ContentBlock::ToolUse { id, name, input } => {
                                    println!("      [Tool Use] {} ({})", name, id);
                                    let input_str = input.to_string();
                                    let display_input = if input_str.len() > 100 {
                                        format!("{}...", &input_str[..100])
                                    } else {
                                        input_str
                                    };
                                    println!("      Input: {}", display_input);
                                }
                                ContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                    is_error,
                                } => {
                                    let error_str = if is_error.unwrap_or(false) {
                                        " (ERROR)"
                                    } else {
                                        ""
                                    };
                                    println!(
                                        "      [Tool Result]{} for {}",
                                        error_str, tool_use_id
                                    );
                                    if let Some(c) = content {
                                        let display_content = if c.len() > 100 {
                                            format!("{}...", &c[..100])
                                        } else {
                                            c.clone()
                                        };
                                        println!("      Result: {}", display_content);
                                    }
                                }
                                ContentBlock::Thinking { thinking, .. } => {
                                    let display = if thinking.len() > 100 {
                                        format!("{}...", &thinking[..100])
                                    } else {
                                        thinking.clone()
                                    };
                                    println!("      [Thinking] {}", display);
                                }
                                ContentBlock::RedactedThinking { .. } => {
                                    println!("      [Redacted Thinking]");
                                }
                            }
                        }
                    }
                }
                println!();
            }
        }
    } else {
        println!("  Session '{}' not found.", target_session);
        println!("  Run the test_agent example first to create it:");
        println!("    cargo run --example test_agent");
    }

    Ok(())
}
