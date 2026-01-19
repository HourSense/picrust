//! Integration Test - Main Agent with FileSummaryAgent subagent
//!
//! This example demonstrates:
//! - A main agent with console attachment
//! - A custom SummarizeFile tool that spawns a subagent
//! - Communication between parent and child agents
//!
//! Run with: cargo run --example integration_test

mod file_summary_agent;
mod summarize_file_tool;

use std::sync::Arc;

use anyhow::Result;

use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    llm::AnthropicProvider,
    runtime::AgentRuntime,
    session::AgentSession,
    tools::ToolRegistry,
};

use summarize_file_tool::SummarizeFileTool;

/// System prompt for the main agent
const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to file tools.

You have access to:
- SummarizeFile: Spawns a subagent to read and summarize any file

When the user asks about a file's contents, use SummarizeFile to get a summary.
Be helpful and concise."#;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("integration_test=info,shadow_agent_sdk=info")
        .init();

    println!("=== Integration Test: Main Agent + Subagent ===");
    println!("This agent can spawn a FileSummaryAgent subagent.\n");

    // --- Step 1: Create LLM provider ---
    println!("[Setup] Creating LLM provider...");
    let llm = Arc::new(AnthropicProvider::from_env()?);
    println!("[Setup] Model: {}", llm.model());

    // --- Step 2: Create runtime ---
    let runtime = AgentRuntime::new();
    println!("[Setup] Runtime created");

    // --- Step 3: Create tool registry with SummarizeFileTool ---
    let mut tools = ToolRegistry::new();
    tools.register(SummarizeFileTool::new(llm.clone(), runtime.clone()));
    let tools = Arc::new(tools);
    println!("[Setup] Tools registered: {:?}", tools.tool_names());

    // --- Step 4: Create session ---
    let session = AgentSession::new(
        "integration-test-main",
        "main-agent",
        "Main Agent",
        "Main agent that can spawn FileSummaryAgent subagents",
    )?;
    println!("[Setup] Session: {}", session.session_id());

    // --- Step 5: Configure and create agent ---
    let config = AgentConfig::new(SYSTEM_PROMPT)
        .with_tools(tools)
        .with_debug(true);

    let agent = StandardAgent::new(config, llm);
    println!("[Setup] Agent configured");

    // --- Step 6: Spawn the agent ---
    println!("[Setup] Spawning agent...");
    let handle = runtime
        .spawn(session, move |internals| agent.run(internals))
        .await;
    println!("[Setup] Agent spawned!");

    // --- Step 7: Create and run console ---
    println!();
    println!("Try: \"Summarize the file src/lib.rs\"");
    println!("Type 'exit' or 'quit' to stop.\n");

    let renderer = ConsoleRenderer::new(handle)
        .show_thinking(true)
        .show_tools(true);

    renderer.run().await?;

    // --- Cleanup ---
    println!("\n[Cleanup] Shutting down runtime...");
    runtime.shutdown_all().await;
    println!("[Cleanup] Done.");

    Ok(())
}
