//! OpenAI Test Agent - Verification Script
//!
//! Tests the OpenAI provider implementation with the actual API

use anyhow::Result;
use picrust::llm::{LlmProvider, OpenAiProvider};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== OpenAI Provider Test ===\n");

    // Create provider with API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    println!("Creating OpenAI provider...");
    let provider = OpenAiProvider::new(api_key)?
        .with_model("gpt-5.2") // Testing with gpt-5.2 as requested
        .with_max_tokens(1000);

    println!("Model: {}", provider.model());
    println!("Provider: {}\n", provider.provider_name());

    // Test 1: Simple message
    println!("Test 1: Simple message");
    println!("{}", "-".repeat(50));
    
    match provider.send_message(
        "Say 'Hello from OpenAI!' and nothing else.",
        &[],
        None,
        None,
    ).await {
        Ok(response) => {
            println!("✅ Success!");
            println!("Response: {}\n", response);
        }
        Err(e) => {
            eprintln!("❌ Error: {}\n", e);
            return Err(e);
        }
    }

    println!("\n=== All tests passed! ===");
    Ok(())
}
