//! Display Streaming Example
//!
//! Demonstrates the streaming capability of the Anthropic API client.
//! Accepts a message from the user and streams the response in real-time.
//!
//! Run with:
//!   cargo run --example display_streaming
//!
//! Or with a custom message:
//!   cargo run --example display_streaming -- "Your message here"

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;
use std::env;
use std::io::{self, Write};

use picrust::llm::{
    AnthropicProvider, ContentBlockStart, ContentDelta, StreamEvent,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (optional, set RUST_LOG=debug to see details)
    tracing_subscriber::fmt()
        .with_env_filter("display_streaming=info,picrust=warn")
        .init();

    println!("{}", "=== Streaming Demo ===".bold().cyan());
    println!("This example demonstrates real-time streaming from the Anthropic API.\n");

    // Get user message from command line args or prompt
    let args: Vec<String> = env::args().collect();
    let user_message = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        // Interactive prompt
        print!("{} ", "Enter your message:".bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    };

    if user_message.is_empty() {
        println!("{}", "No message provided. Exiting.".yellow());
        return Ok(());
    }

    println!("\n{} {}\n", "You:".bold().green(), user_message);

    // Create the Anthropic provider
    let provider = AnthropicProvider::from_env()?;
    println!(
        "{} {} (streaming)\n",
        "Using model:".dimmed(),
        provider.model().dimmed()
    );

    // Start streaming
    print!("{} ", "Assistant:".bold().blue());
    io::stdout().flush()?;

    let mut stream = provider
        .stream_message(&user_message, &[], None)
        .await?;

    // Track state for display
    let mut _current_block_type: Option<String> = None;
    let mut in_thinking = false;

    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => {
                match event {
                    StreamEvent::MessageStart(msg_start) => {
                        tracing::debug!("Message started: {}", msg_start.message.id);
                    }
                    StreamEvent::ContentBlockStart(block_start) => {
                        match &block_start.content_block {
                            ContentBlockStart::Text { .. } => {
                                _current_block_type = Some("text".to_string());
                                if in_thinking {
                                    // End thinking section
                                    println!("\n{}", "</thinking>".dimmed());
                                    in_thinking = false;
                                    print!("{} ", "Assistant:".bold().blue());
                                    io::stdout().flush()?;
                                }
                            }
                            ContentBlockStart::Thinking { .. } => {
                                _current_block_type = Some("thinking".to_string());
                                if !in_thinking {
                                    println!("\n{}", "<thinking>".dimmed());
                                    in_thinking = true;
                                }
                            }
                            ContentBlockStart::ToolUse { name, .. } => {
                                _current_block_type = Some("tool_use".to_string());
                                print!("\n{} ", format!("[Tool: {}]", name).yellow());
                                io::stdout().flush()?;
                            }
                        }
                    }
                    StreamEvent::ContentBlockDelta(delta_event) => {
                        match &delta_event.delta {
                            ContentDelta::TextDelta { text } => {
                                print!("{}", text);
                                io::stdout().flush()?;
                            }
                            ContentDelta::ThinkingDelta { thinking } => {
                                // Display thinking in dimmed color
                                print!("{}", thinking.dimmed());
                                io::stdout().flush()?;
                            }
                            ContentDelta::InputJsonDelta { partial_json } => {
                                // Display tool input as it arrives
                                print!("{}", partial_json.dimmed());
                                io::stdout().flush()?;
                            }
                            ContentDelta::SignatureDelta { .. } => {
                                // Signature is internal, don't display
                            }
                        }
                    }
                    StreamEvent::ContentBlockStop(_) => {
                        _current_block_type = None;
                    }
                    StreamEvent::MessageDelta(msg_delta) => {
                        if let Some(stop_reason) = &msg_delta.delta.stop_reason {
                            tracing::debug!("Stop reason: {:?}", stop_reason);
                        }
                    }
                    StreamEvent::MessageStop => {
                        if in_thinking {
                            println!("\n{}", "</thinking>".dimmed());
                        }
                        println!("\n");
                        tracing::debug!("Stream complete");
                    }
                    StreamEvent::Ping => {
                        tracing::trace!("Ping received");
                    }
                    StreamEvent::Error(err) => {
                        eprintln!(
                            "\n{} {}: {}",
                            "Stream error:".bold().red(),
                            err.error.error_type,
                            err.error.message
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("\n{} {}", "Error:".bold().red(), e);
                break;
            }
        }
    }

    println!("{}", "Stream complete.".dimmed());
    Ok(())
}
