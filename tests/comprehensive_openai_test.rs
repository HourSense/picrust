//! Comprehensive OpenAI Provider Test Suite
//!
//! Tests all features of the OpenAI provider:
//! - Basic message sending
//! - Tool calling
//! - Streaming
//! - Error handling
//! - Reasoning model support (gpt-5.2)

use anyhow::Result;
use futures::StreamExt;
use picrust::llm::{define_tool, ContentBlock, ContentDelta, LlmProvider, Message, OpenAiProvider, StreamEvent, ToolChoice};
use serde_json::json;

fn get_api_key() -> String {
    std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set for tests")
}

#[tokio::test]
async fn comprehensive_openai_test() -> Result<()> {

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║   OpenAI Provider Comprehensive Test Suite            ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    let mut tests_passed = 0;
    let mut tests_failed = 0;

    // Test 1: Basic Message with gpt-5.2 (Reasoning Model)
    println!("📝 Test 1: Basic message with gpt-5.2 (reasoning model)");
    println!("{}", "─".repeat(60));
    match test_basic_message().await {
        Ok(_) => {
            println!("✅ PASSED\n");
            tests_passed += 1;
        }
        Err(e) => {
            println!("❌ FAILED: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 2: System Prompt
    println!("📝 Test 2: System prompt handling");
    println!("{}", "─".repeat(60));
    match test_system_prompt().await {
        Ok(_) => {
            println!("✅ PASSED\n");
            tests_passed += 1;
        }
        Err(e) => {
            println!("❌ FAILED: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 3: Tool Calling
    println!("📝 Test 3: Tool calling (function calling)");
    println!("{}", "─".repeat(60));
    match test_tool_calling().await {
        Ok(_) => {
            println!("✅ PASSED\n");
            tests_passed += 1;
        }
        Err(e) => {
            println!("❌ FAILED: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 4: Streaming
    println!("📝 Test 4: Streaming responses");
    println!("{}", "─".repeat(60));
    match test_streaming().await {
        Ok(_) => {
            println!("✅ PASSED\n");
            tests_passed += 1;
        }
        Err(e) => {
            println!("❌ FAILED: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 5: Conversation History
    println!("📝 Test 5: Conversation history");
    println!("{}", "─".repeat(60));
    match test_conversation_history().await {
        Ok(_) => {
            println!("✅ PASSED\n");
            tests_passed += 1;
        }
        Err(e) => {
            println!("❌ FAILED: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 6: Provider Traits
    println!("📝 Test 6: LlmProvider trait implementation");
    println!("{}", "─".repeat(60));
    match test_provider_traits().await {
        Ok(_) => {
            println!("✅ PASSED\n");
            tests_passed += 1;
        }
        Err(e) => {
            println!("❌ FAILED: {}\n", e);
            tests_failed += 1;
        }
    }

    // Summary
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║                   TEST SUMMARY                         ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!("✅ Passed: {}", tests_passed);
    println!("❌ Failed: {}", tests_failed);
    println!("📊 Total:  {}", tests_passed + tests_failed);
    println!();

    if tests_failed == 0 {
        println!("🎉 All tests passed! OpenAI provider is working correctly.");
        Ok(())
    } else {
        anyhow::bail!("{} test(s) failed", tests_failed)
    }
}

async fn test_basic_message() -> Result<()> {
    let provider = OpenAiProvider::new(&get_api_key())?
        .with_model("gpt-5.2")
        .with_max_tokens(100);

    let response = provider
        .send_message(
            "Say exactly 'Test successful' and nothing else.",
            &[],
            None,
            None,
        )
        .await?;

    println!("  Response: {}", response);
    
    if response.to_lowercase().contains("test successful") {
        Ok(())
    } else {
        anyhow::bail!("Response did not contain expected text")
    }
}

async fn test_system_prompt() -> Result<()> {
    let provider = OpenAiProvider::new(&get_api_key())?
        .with_model("gpt-5.2")
        .with_max_tokens(100);

    let response = provider
        .send_message(
            "What is your role?",
            &[],
            Some("You are a helpful math tutor specializing in algebra."),
            None,
        )
        .await?;

    println!("  Response: {}", response);
    
    if response.to_lowercase().contains("math") 
        || response.to_lowercase().contains("tutor") 
        || response.to_lowercase().contains("algebra") {
        Ok(())
    } else {
        anyhow::bail!("Response did not reflect system prompt")
    }
}

async fn test_tool_calling() -> Result<()> {
    let provider = OpenAiProvider::new(&get_api_key())?
        .with_model("gpt-5.2")
        .with_max_tokens(500);

    let tools = vec![define_tool(
        "get_weather",
        "Get current weather for a location",
        json!({
            "location": {
                "type": "string",
                "description": "City name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        }),
        vec!["location".to_string()],
    )];

    let messages = vec![Message::user("What's the weather in Paris, France?")];

    let response = provider
        .send_with_tools_and_system(
            messages,
            None,
            tools,
            Some(ToolChoice::Auto { disable_parallel_tool_use: None }),
            None,
            None,
        )
        .await?;

    println!("  Stop reason: {:?}", response.stop_reason);
    
    let mut found_tool_use = false;
    for block in &response.content {
        if let ContentBlock::ToolUse { name, input, .. } = block {
            println!("  Tool called: {}", name);
            println!("  Input: {}", serde_json::to_string_pretty(input)?);
            found_tool_use = true;
            
            if name == "get_weather" {
                // Check that location was extracted
                if let Some(location) = input.get("location") {
                    println!("  ✓ Location extracted: {}", location);
                } else {
                    anyhow::bail!("Tool call missing location parameter");
                }
            }
        }
    }

    if found_tool_use {
        Ok(())
    } else {
        anyhow::bail!("No tool use found in response")
    }
}

async fn test_streaming() -> Result<()> {
    let provider = OpenAiProvider::new(&get_api_key())?
        .with_model("gpt-5.2")
        .with_max_tokens(100);

    let messages = vec![Message::user("Count from 1 to 3")];

    let mut stream = provider
        .stream_with_tools_and_system(
            messages,
            None,
            vec![],
            None,
            None,
            None,
        )
        .await?;

    let mut received_text = String::new();
    let mut message_started = false;
    let mut message_stopped = false;

    print!("  Streaming: ");
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::MessageStart(_) => {
                message_started = true;
            }
            StreamEvent::ContentBlockDelta(delta) => {
                if let ContentDelta::TextDelta { text } = delta.delta {
                    print!("{}", text);
                    received_text.push_str(&text);
                    use std::io::Write;
                    std::io::stdout().flush()?;
                }
            }
            StreamEvent::MessageStop => {
                message_stopped = true;
                break;
            }
            _ => {}
        }
    }
    println!();

    if !message_started {
        anyhow::bail!("MessageStart event not received");
    }
    if !message_stopped {
        anyhow::bail!("MessageStop event not received");
    }
    if received_text.is_empty() {
        anyhow::bail!("No text received in stream");
    }

    println!("  ✓ Received {} characters", received_text.len());
    Ok(())
}

async fn test_conversation_history() -> Result<()> {
    let provider = OpenAiProvider::new(&get_api_key())?
        .with_model("gpt-5.2")
        .with_max_tokens(100);

    let history = vec![
        Message::user("My favorite color is blue."),
        Message::assistant("Nice! Blue is a calming color."),
    ];

    let response = provider
        .send_message(
            "What's my favorite color?",
            &history,
            None,
            None,
        )
        .await?;

    println!("  Response: {}", response);
    
    if response.to_lowercase().contains("blue") {
        Ok(())
    } else {
        anyhow::bail!("Did not remember favorite color from conversation history")
    }
}

async fn test_provider_traits() -> Result<()> {
    let provider = OpenAiProvider::new(&get_api_key())?
        .with_model("gpt-5.2");

    // Test provider_name
    if provider.provider_name() != "openai" {
        anyhow::bail!("Provider name incorrect: {}", provider.provider_name());
    }
    println!("  ✓ provider_name() = \"openai\"");

    // Test model
    if provider.model() != "gpt-5.2" {
        anyhow::bail!("Model name incorrect: {}", provider.model());
    }
    println!("  ✓ model() = \"gpt-5.2\"");

    // Test create_variant
    let variant = provider.create_variant("gpt-4o", 2000);
    if variant.model() != "gpt-4o" {
        anyhow::bail!("Variant model incorrect: {}", variant.model());
    }
    println!("  ✓ create_variant() works correctly");

    Ok(())
}
