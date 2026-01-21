//! Test Agent Example - Using StandardAgent
//!
//! Demonstrates the standardized agent framework:
//! - AgentConfig for configuration
//! - StandardAgent for the agent loop
//! - Context injections for dynamic message modification
//! - TodoListManager for task tracking
//! - Streaming responses (optional)
//! - Prompt caching for cost savings (enabled by default)
//!
//! Read operations are pre-allowed, others will prompt the user.
//!
//! Run with:
//!   cargo run --example test_agent                     # New session (with caching)
//!   cargo run --example test_agent -- --resume         # Resume existing session
//!   cargo run --example test_agent -- --stream         # New session with streaming
//!   cargo run --example test_agent -- --stream --resume # Resume with streaming
//!   cargo run --example test_agent -- --think          # Enable extended thinking
//!   cargo run --example test_agent -- --stream --think # Streaming with thinking
//!   cargo run --example test_agent -- --no-cache       # Disable prompt caching

mod tools;

use anyhow::{bail, Result};
use std::env;
use std::sync::Arc;

use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    helpers::{inject_system_reminder, TodoListManager},
    hooks::{HookContext, HookEvent, HookRegistry, HookResult},
    llm::{AnthropicProvider, AuthConfig},
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
        .with_env_filter("test_agent=info,shadow_agent_sdk=warn")
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let resume = args.iter().any(|a| a == "--resume" || a == "-r");

    // Generate session ID with timestamp
    let session_id = format!(
        "test-agent-session-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );

    println!("=== Test Agent (StandardAgent) ===");
    println!("This agent uses the standardized agent framework.");
    println!("Read operations are pre-allowed. Others will require permission.");
    println!("Use --stream/-s flag to enable streaming responses.");
    println!("Use --think/-t flag to enable extended thinking.");
    println!("Prompt caching is enabled by default (use --no-cache to disable).\n");

    // --- Step 1: Create LLM provider with dynamic auth ---
    println!("[Setup] Creating LLM provider with dynamic auth...");

    // Using dynamic auth provider - callback is called before each API request
    // This demonstrates the pattern for JWT tokens that expire frequently
    let llm = Arc::new(
        AnthropicProvider::with_auth_provider(|| async {
            // Read API key from ANTHROPIC_KEY (not ANTHROPIC_API_KEY)
            // In production, this would fetch a fresh JWT from your auth service
            let api_key = env::var("ANTHROPIC_KEY")
                .map_err(|_| anyhow::anyhow!("ANTHROPIC_KEY environment variable not set"))?;

            // Return auth config with Anthropic's default base URL
            Ok(AuthConfig::with_base_url(
                api_key,
                "https://api.anthropic.com/v1/messages",
            ))
        })
        .with_model(
            env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-5-20250929".to_string()),
        )
        .with_max_tokens(32000),
    );
    println!("[Setup] Model: {} (using dynamic auth)", llm.model());

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
            "test-agent",
            "Test Agent",
            "A test agent demonstrating the StandardAgent framework",
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
    // Check if extended thinking is requested via command line
    let thinking = args.iter().any(|a| a == "--think" || a == "-t");
    // Check if caching should be disabled (enabled by default)
    let no_cache = args.iter().any(|a| a == "--no-cache");
    let caching = !no_cache;

    let mut config = AgentConfig::new(SYSTEM_PROMPT)
        .with_tools(tools)
        .with_hooks(hooks) // Add hooks for safety and auto-approval
        .with_debug(true) // Enable debug logging
        .with_streaming(streaming) // Enable streaming if --stream flag is passed
        .with_prompt_caching(caching); // Enable/disable prompt caching

    // Enable extended thinking if --think flag is passed
    if thinking {
        config = config.with_thinking(16000); // 16k token budget for thinking
    }

    let config = config.with_injection_fn("todo_status", move |_internals, mut messages| {
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
        "[Setup] AgentConfig created with debug logging, hooks{}{}{} and todo reminder injection",
        if streaming { ", streaming enabled" } else { "" },
        if thinking { ", extended thinking enabled" } else { "" },
        if caching { ", prompt caching enabled" } else { ", prompt caching disabled" }
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
    if caching {
        println!("💰 Prompt caching enabled: 90% cost savings on repeated content!");
        println!("   (Tools, system prompt, and conversation history are automatically cached)");
    } else {
        println!("⚠️  Prompt caching disabled. To enable: run without --no-cache flag");
    }
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
