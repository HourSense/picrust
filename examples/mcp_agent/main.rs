//! MCP Agent Example - Using StandardAgent with MCP Tools
//!
//! Demonstrates MCP (Model Context Protocol) integration:
//! - MCPServerManager for managing MCP connections
//! - MCPToolProvider for exposing MCP tools
//! - Custom transport configuration
//! - Tool namespacing (server_id:tool_name)
//!
//! Run with:
//!   cargo run --example mcp_agent                     # New session
//!   cargo run --example mcp_agent -- --resume         # Resume existing session
//!   cargo run --example mcp_agent -- --stream         # New session with streaming
//!   cargo run --example mcp_agent -- --think          # Enable extended thinking

use anyhow::{bail, Result};
use std::env;
use std::sync::Arc;

use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    helpers::TodoListManager,
    hooks::{HookContext, HookEvent, HookRegistry, HookResult},
    llm::{AnthropicProvider, AuthConfig},
    mcp::{MCPServerManager, MCPToolProvider},
    runtime::AgentRuntime,
    session::{AgentSession, SessionStorage},
    tools::ToolRegistry,
};

use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;

/// System prompt for the MCP agent
const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to MCP tools.

All tools are provided via MCP (Model Context Protocol) and are namespaced with their server ID.
For example: filesystem__read_file, filesystem__write_file, etc.

When the user asks you to do something, use the appropriate MCP tools.
Be concise in your responses."#;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("mcp_agent=info,shadow_agent_sdk=info")
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let resume = args.iter().any(|a| a == "--resume" || a == "-r");

    // Generate session ID with timestamp
    let session_id = format!(
        "mcp-agent-session-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );

    println!("=== MCP Agent (StandardAgent) ===");
    println!("This agent demonstrates MCP (Model Context Protocol) integration.");
    println!("All tools are provided via MCP server at http://localhost:8005/mcp");
    println!("Prompt caching is enabled by default.\n");

    // --- Step 1: Create LLM provider with dynamic auth ---
    println!("[Setup] Creating LLM provider...");

    let llm = Arc::new(
        AnthropicProvider::with_auth_provider(|| async {
            let api_key = env::var("ANTHROPIC_KEY")
                .map_err(|_| anyhow::anyhow!("ANTHROPIC_KEY environment variable not set"))?;

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
    println!("[Setup] Model: {}", llm.model());

    // --- Step 2: Create runtime ---
    let runtime = AgentRuntime::new();
    println!("[Setup] Runtime created");

    // --- Step 3: Set up MCP connection ---
    println!("[Setup] Connecting to MCP server at http://localhost:8005/mcp...");

    // Create transport and service
    let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp");
    let service = ().serve(transport).await?;

    // Create MCP manager and add the service
    let mcp_manager = Arc::new(MCPServerManager::new());
    mcp_manager.add_service("filesystem", service).await?;

    println!("[Setup] Connected to MCP server 'filesystem'");

    // --- Step 4: Create tool registry with MCP provider ---
    let mut tool_registry = ToolRegistry::new();
    let mcp_provider = Arc::new(MCPToolProvider::new(mcp_manager));
    tool_registry.add_provider(mcp_provider).await?;

    let tools = Arc::new(tool_registry);
    println!("[Setup] MCP tools registered: {:?}", tools.tool_names());

    // --- Step 5: Create TodoListManager ---
    let todo_manager = Arc::new(TodoListManager::new());
    println!("[Setup] TodoListManager created");

    // --- Step 6: Create hooks ---
    let mut hooks = HookRegistry::new();

    // Auto-approve all MCP tools (they're already marked as requiring permission)
    // This is just for demo - in production you'd want more granular control
    hooks
        .add_with_pattern(HookEvent::PreToolUse, ".*__.*", |ctx: &mut HookContext| {
            // Allow all namespaced MCP tools (server_id__tool_name)
            if let Some(tool_name) = &ctx.tool_name {
                println!("Auto-approving MCP tool: {}", tool_name);
            }
            HookResult::allow()
        })
        .expect("Invalid regex pattern");

    println!("[Setup] Hooks configured: auto-approve MCP tools");

    // --- Step 7: Create or load session ---
    let storage = SessionStorage::with_dir("./sessions");
    let session = if resume {
        if !AgentSession::exists_with_storage(&session_id, &storage) {
            bail!(
                "Cannot resume: session '{}' does not exist. Run without --resume to create a new session.",
                session_id
            );
        }
        let session = AgentSession::load_with_storage(&session_id, storage)?;
        println!(
            "[Setup] Resumed session: {} ({} messages in history)",
            session.session_id(),
            session.history().len()
        );
        session
    } else {
        let session = AgentSession::new_with_storage(
            &session_id,
            "mcp-agent",
            "MCP Agent",
            "An agent demonstrating MCP integration",
            storage,
        )?;
        println!("[Setup] New session: {}", session.session_id());
        session
    };

    // --- Step 8: Configure the agent ---
    let streaming = args.iter().any(|a| a == "--stream" || a == "-s");
    let thinking = args.iter().any(|a| a == "--think" || a == "-t");
    let no_cache = args.iter().any(|a| a == "--no-cache");
    let caching = !no_cache;

    let mut config = AgentConfig::new(SYSTEM_PROMPT)
        .with_tools(tools)
        .with_hooks(hooks)
        .with_debug(true)
        .with_streaming(streaming)
        .with_prompt_caching(caching);

    if thinking {
        config = config.with_thinking(16000);
    }

    println!(
        "[Setup] AgentConfig created with debug logging, MCP tools{}{}{}",
        if streaming { ", streaming enabled" } else { "" },
        if thinking { ", extended thinking enabled" } else { "" },
        if caching {
            ", prompt caching enabled"
        } else {
            ", prompt caching disabled"
        }
    );

    // --- Step 9: Create StandardAgent ---
    let agent = StandardAgent::new(config, llm);

    // --- Step 10: Spawn the agent ---
    println!("[Setup] Spawning agent...");
    let todo_for_context = todo_manager.clone();
    let handle = runtime
        .spawn(session, move |mut internals| {
            internals.context.insert_resource_arc(todo_for_context);
            agent.run(internals)
        })
        .await;
    println!("[Setup] Agent spawned!");

    // --- Step 11: Create and run the console renderer ---
    println!("[Setup] Starting console renderer...");
    println!();
    println!("Type your requests below. MCP tools are auto-approved for this demo.");
    if caching {
        println!("💰 Prompt caching enabled: 90% cost savings on repeated content!");
    }
    println!("Type 'exit' or 'quit' to stop.\n");

    let renderer = ConsoleRenderer::new(handle)
        .show_thinking(true)
        .show_tools(true)
        .with_todo_manager(todo_manager);

    renderer.run().await?;

    // --- Cleanup ---
    println!("\n[Cleanup] Shutting down runtime...");
    runtime.shutdown_all().await;

    println!("[Cleanup] Done.");
    Ok(())
}
