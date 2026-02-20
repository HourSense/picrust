// Example: Using OpenAI provider with Picrust
//
// This example shows how to use the OpenAI provider for basic chat and tool calling.
//
// Set these environment variables before running:
// export OPENAI_API_KEY="sk-..."
// export OPENAI_MODEL="gpt-4o-mini"

use anyhow::Result;
use picrust::llm::{define_tool, Message, OpenAiProvider, ToolChoice};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug logs
    tracing_subscriber::fmt::init();

    // Example 1: Basic conversation
    println!("=== Example 1: Basic Conversation ===\n");
    basic_conversation().await?;

    // Example 2: Tool calling
    println!("\n=== Example 2: Tool Calling ===\n");
    tool_calling().await?;

    // Example 3: Streaming response
    println!("\n=== Example 3: Streaming Response ===\n");
    streaming_response().await?;

    Ok(())
}

async fn basic_conversation() -> Result<()> {
    // Create provider from environment variables
    let provider = OpenAiProvider::from_env()?;

    // Send a simple message
    let response = provider
        .send_message(
            "What is the capital of France?",
            &[],
            Some("You are a helpful geography assistant."),
            None,
        )
        .await?;

    println!("Response: {}", response);

    Ok(())
}

async fn tool_calling() -> Result<()> {
    let provider = OpenAiProvider::from_env()?;

    // Define a tool
    let tools = vec![define_tool(
        "get_weather",
        "Get the current weather for a location",
        json!({
            "location": {
                "type": "string",
                "description": "City name, e.g., 'Tokyo'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        }),
        vec!["location".to_string()],
    )];

    // Send message with tools
    let messages = vec![Message::user("What's the weather in Tokyo?")];

    let response = provider
        .send_with_tools_and_system(
            messages,
            Some("You are a weather assistant.".into()),
            tools,
            Some(ToolChoice::Auto {
                disable_parallel_tool_use: None,
            }),
            None, // thinking config
            None, // session_id
        )
        .await?;

    println!("Model used: {}", response.model);
    println!("Stop reason: {:?}", response.stop_reason);

    // Check for tool calls
    for block in &response.content {
        match block {
            picrust::llm::ContentBlock::Text { text, .. } => {
                println!("Text: {}", text);
            }
            picrust::llm::ContentBlock::ToolUse { id, name, input, .. } => {
                println!("Tool call:");
                println!("  ID: {}", id);
                println!("  Name: {}", name);
                println!("  Input: {}", serde_json::to_string_pretty(input)?);
            }
            _ => {}
        }
    }

    Ok(())
}

async fn streaming_response() -> Result<()> {
    use futures::StreamExt;
    use picrust::llm::{ContentDelta, StreamEvent};

    let provider = OpenAiProvider::from_env()?;

    let messages = vec![Message::user("Write a haiku about AI")];

    let mut stream = provider
        .stream_with_tools_and_system(
            messages,
            Some("You are a creative poet.".into()),
            vec![],
            None,
            None,
            None,
        )
        .await?;

    print!("Response: ");
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::MessageStart(start) => {
                println!("(Model: {})", start.message.model);
                print!("  ");
            }
            StreamEvent::ContentBlockDelta(delta) => {
                if let ContentDelta::TextDelta { text } = delta.delta {
                    print!("{}", text);
                    use std::io::Write;
                    std::io::stdout().flush()?;
                }
            }
            StreamEvent::MessageDelta(delta) => {
                println!("\n\nTokens used: {}", delta.usage.output_tokens);
            }
            StreamEvent::MessageStop => {
                println!("\n\nStream complete!");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
