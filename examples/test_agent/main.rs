//! Test Agent Example - Using StandardAgent
//!
//! Demonstrates the standardized agent framework:
//! - AgentConfig for configuration
//! - StandardAgent for the agent loop
//! - Context injections for dynamic message modification
//! - TodoListManager for task tracking
//!
//! Read operations are pre-allowed, others will prompt the user.
//!
//! Run with: cargo run --example test_agent

mod tools;

use anyhow::Result;
use std::sync::Arc;

use singapore_project::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    helpers::{inject_system_reminder, TodoListManager},
    llm::AnthropicProvider,
    permissions::PermissionRule,
    runtime::AgentRuntime,
    session::{AgentSession, SessionStorage},
};

/// System prompt for the test agent
const SYSTEM_PROMPT: &str = r#"You are a helpful coding assistant with access to tools.

You have the following tools available:
- Read: Read file contents
- Write: Write or create files
- Bash: Execute shell commands
- TodoWrite: Track tasks you need to perform

When the user asks you to do something, use the appropriate tools.
Use TodoWrite to track multi-step tasks and show progress.
Be concise in your responses."#;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("test_agent=info,singapore_project=warn")
        .init();

    println!("=== Test Agent (StandardAgent) ===");
    println!("This agent uses the standardized agent framework.");
    println!("Read operations are pre-allowed. Others will require permission.\n");

    // --- Step 1: Create LLM provider ---
    println!("[Setup] Creating LLM provider...");
    let llm = Arc::new(AnthropicProvider::from_env()?);
    println!("[Setup] Model: {}", llm.model());

    // --- Step 2: Create runtime with global Read permission ---
    let runtime = AgentRuntime::new();
    runtime
        .global_permissions()
        .add_rule(PermissionRule::allow_tool("Read"));
    println!("[Setup] Runtime created (Read tool globally allowed)");

    // --- Step 3: Create tool registry ---
    let tools = Arc::new(tools::create_registry()?);
    println!("[Setup] Tools registered: {:?}", tools.tool_names());

    // --- Step 4: Create TodoListManager (shared between agent and console) ---
    let todo_manager = Arc::new(TodoListManager::new());
    println!("[Setup] TodoListManager created");

    // --- Step 5: Create session with persistent storage ---
    let storage = SessionStorage::with_dir("./sessions");
    let session = AgentSession::new_with_storage(
        "test-agent-session",
        "test-agent",
        "Test Agent",
        "A test agent demonstrating the StandardAgent framework",
        storage,
    )?;
    println!("[Setup] Session: {}", session.session_id());

    // --- Step 6: Configure the agent ---
    // Clone todo_manager for the injection closure
    let todo_for_injection = todo_manager.clone();

    let config = AgentConfig::new(SYSTEM_PROMPT)
        .with_tools(tools)
        .with_debug(true) // Enable debug logging
        .with_injection_fn("todo_status", move |_internals, mut messages| {
            // Only inject reminder if todo list is empty
            if todo_for_injection.is_empty() {
                inject_system_reminder(
                    &mut messages,
                    "The TodoWrite tool hasn't been used yet. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Only use it if it's relevant to the current work.",
                );
            }
            messages
        });

    println!("[Setup] AgentConfig created with debug logging and todo reminder injection");

    // --- Step 7: Create StandardAgent ---
    let agent = StandardAgent::new(config, llm);

    // --- Step 8: Spawn the agent ---
    println!("[Setup] Spawning agent...");
    let todo_for_context = todo_manager.clone();
    let handle = runtime
        .spawn(session, move |mut internals| {
            // Insert TodoListManager into context for TodoWriteTool
            // Use insert_resource_arc since todo_for_context is already Arc-wrapped
            internals.context.insert_resource_arc(todo_for_context);
            agent.run(internals)
        })
        .await;
    println!("[Setup] Agent spawned!");

    // --- Step 9: Create and run the console renderer ---
    println!("[Setup] Starting console renderer...");
    println!();
    println!("Type your requests below. Read is pre-allowed, others will ask for permission.");
    println!("Type 'exit' or 'quit' to stop.\n");

    let renderer = ConsoleRenderer::new(handle)
        .show_thinking(true)
        .show_tools(true)
        .with_todo_manager(todo_manager);

    // Run the console - this blocks until user types "exit"
    renderer.run().await?;

    // --- Cleanup ---
    println!("\n[Cleanup] Shutting down runtime...");
    runtime.shutdown_all().await;

    println!("[Cleanup] Done.");
    Ok(())
}
