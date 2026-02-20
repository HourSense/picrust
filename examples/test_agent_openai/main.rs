//! OpenAI Test Agent - Using StandardAgent with OpenAI
//!
//! Demonstrates the OpenAI provider with the standardized agent framework:
//! - AgentConfig for configuration
//! - StandardAgent for the agent loop
//! - OpenAiProvider for LLM backend
//! - Tool calling and function execution
//! - Streaming responses (optional)
//!
//! Read operations are pre-allowed, others will prompt the user.
//!
//! Run with:
//!   cargo run --example test_agent_openai                     # New session
//!   cargo run --example test_agent_openai -- --stream         # With streaming

mod tools;

use anyhow::{bail, Result};
use std::env;
use std::sync::Arc;

use picrust::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    helpers::{inject_system_reminder, TodoListManager},
    hooks::{HookContext, HookEvent, HookRegistry, HookResult},
    llm::{LlmProvider, OpenAiProvider},
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
        .with_env_filter("test_agent_openai=info,picrust=info")
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let resume = args.iter().any(|a| a == "--resume" || a == "-r");

    // Generate session ID with timestamp
    let session_id = format!(
        "test-agent-openai-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );

    println!("=== OpenAI Test Agent (StandardAgent) ===");
    println!("This agent uses OpenAI via the standardized agent framework.");
    println!("Read operations are pre-allowed. Others will require permission.");
    println!("Use --stream/-s flag to enable streaming responses.\n");

    // --- Step 1: Create OpenAI provider ---
    println!("[Setup] Creating OpenAI provider...");

    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");

    let llm = Arc::new(
        OpenAiProvider::new(api_key)?
            .with_model("gpt-5.2")
            .with_max_tokens(4000),
    );
    println!("[Setup] Model: {} (provider: {})", llm.model(), llm.provider_name());

    // --- Step 2: Create runtime with global Read permission ---
    let runtime = AgentRuntime::new();
    runtime.global_permissions();
    println!("[Setup] Runtime created (Read tool globally allowed)");

    // --- Step 3: Create tool registry ---
    let tools = Arc::new(tools::create_registry()?);
    println!("[Setup] Tools registered: {:?}", tools.tool_names());

    // --- Step 4: Create TodoListManager (shared between agent and console) ---
    let todo_manager = Arc::new(TodoListManager::new());
    println!("[Setup] TodoListManager created");

    // --- Step 5: Create hooks ---
    let mut hooks = HookRegistry::new();

    // Block dangerous Bash commands
    hooks
        .add_with_pattern(HookEvent::PreToolUse, "Bash", |ctx: &mut HookContext| {
            println!("PreToolUse hook called with context: {:?}", ctx.tool_input.as_ref().map(|v| v.to_string()));
            let cmd = ctx
                .tool_input
                .as_ref()
                .and_then(|v| v.get("command"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Block dangerous patterns
            if cmd.contains("rm ") {
                HookResult::deny("Dangerous command blocked by safety hook")
            } else {
                HookResult::none() // Continue with normal permission flow
            }
        })
        .expect("Invalid regex pattern");

    // Auto-approve read-only tools (skip permission prompts)
    hooks
        .add_with_pattern(HookEvent::PreToolUse, "^(Read|Glob|Grep)$", |_ctx: &mut HookContext| {
            HookResult::allow()
        })
        .expect("Invalid regex pattern");

    println!("[Setup] Hooks configured: dangerous command blocker, read-only auto-approve");

    // --- Step 6: Create or load session ---
    let storage = SessionStorage::with_dir("./sessions");
    let session = if resume {
        // Resume existing session
        if !AgentSession::exists_with_storage(&session_id, &storage) {
            bail!(
                "Cannot resume: session '{}' does not exist. Run without --resume to create a new session.",
                session_id
            );
        }
        let session = AgentSession::load_with_storage(&session_id, storage)?;
        println!("[Setup] Resumed session: {} ({} messages in history)",
            session.session_id(),
            session.history().len()
        );
        session
    } else {
        // Create new session
        let session = AgentSession::new_with_storage(
            &session_id,
            "test-agent-openai",
            "OpenAI Test Agent",
            "A test agent demonstrating OpenAI provider with StandardAgent framework",
            storage,
        )?;
        println!("[Setup] New session: {}", session.session_id());
        session
    };

    // --- Step 7: Configure the agent ---
    // Clone todo_manager for the injection closure
    let todo_for_injection = todo_manager.clone();

    // Check if streaming is requested via command line
    let streaming = args.iter().any(|a| a == "--stream" || a == "-s");

    let config = AgentConfig::new(SYSTEM_PROMPT)
        .with_tools(tools)
        .with_hooks(hooks) // Add hooks for safety and auto-approval
        .with_debug(true) // Enable debug logging
        .with_streaming(streaming) // Enable streaming if --stream flag is passed
        .with_prompt_caching(false) // OpenAI doesn't support prompt caching like Anthropic
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

    println!(
        "[Setup] AgentConfig created with debug logging and hooks{}",
        if streaming { ", streaming enabled" } else { "" }
    );

    // --- Step 8: Create StandardAgent ---
    let agent = StandardAgent::new(config, llm);

    // --- Step 9: Spawn the agent ---
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

    // --- Step 10: Create and run the console renderer ---
    println!("[Setup] Starting console renderer...");
    println!();
    println!("Type your requests below. Read/Glob/Grep are auto-approved by hooks.");
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
