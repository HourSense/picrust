//! Gemini Test Agent - Non-interactive test using GeminiProvider
//!
//! This example demonstrates using GeminiProvider with the StandardAgent framework,
//! including SwappableLlmProvider for runtime model switching and dynamic auth via proxy.
//!
//! Run with:
//!   cargo run --example gemini_test_agent
//!   cargo run --example gemini_test_agent -- --tools       # Test with tool use
//!   cargo run --example gemini_test_agent -- --stream      # Test with streaming
//!   cargo run --example gemini_test_agent -- --interactive  # Interactive mode

mod tools;

use anyhow::Result;
use std::env;
use std::sync::Arc;

use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    hooks::{HookContext, HookEvent, HookRegistry, HookResult},
    llm::{AuthConfig, GeminiProvider, LlmProvider, SwappableLlmProvider},
    runtime::AgentRuntime,
    session::{AgentSession, SessionStorage},
};

const SYSTEM_PROMPT: &str = r#"You are a helpful coding assistant with access to tools.

You have the following tools available:
- Read: Read file contents
- Write: Write or create files
- Bash: Execute shell commands
- Grep: Search file contents
- Glob: Find files by pattern

When the user asks you to do something, use the appropriate tools.
Be concise in your responses."#;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "gemini_test_agent=info,shadow_agent_sdk=info".to_string()),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    let use_tools = args.iter().any(|a| a == "--tools" || a == "-t");
    let use_streaming = args.iter().any(|a| a == "--stream" || a == "-s");
    let interactive = args.iter().any(|a| a == "--interactive" || a == "-i");

    println!("=== Gemini Test Agent ===");
    println!("Using GeminiProvider via proxy + SwappableLlmProvider\n");

    // --- Create LLM provider with auth provider (proxy) ---
    let model = env::var("GEMINI_MODEL")
        .unwrap_or_else(|_| "gemini-3-flash-preview".to_string());

    let gemini = Arc::new(
        GeminiProvider::with_auth_provider(|| async {
            // Any key works with the local proxy
            Ok(AuthConfig::with_base_url(
                "test-key",
                "http://localhost:8000/api/ai/gemini/v1/v1beta",
            ))
        })
        .with_model(&model)
        .with_max_tokens(8192),
    );

    // Wrap in SwappableLlmProvider for runtime switching
    let swappable = SwappableLlmProvider::new(gemini);
    let _llm_handle = swappable.handle(); // Keep handle for potential model switching
    let llm: Arc<dyn LlmProvider> = Arc::new(swappable);

    println!("[Setup] Model: {}", model);
    println!("[Setup] Proxy: http://localhost:8000/api/ai/gemini/v1/v1beta");

    // --- Create runtime ---
    let runtime = AgentRuntime::new();
    runtime.global_permissions();

    // --- Create tool registry ---
    let tools = Arc::new(tools::create_registry()?);
    println!("[Setup] Tools registered: {:?}", tools.tool_names());

    // --- Create hooks ---
    let mut hooks = HookRegistry::new();

    // Auto-approve all tools for non-interactive test
    hooks
        .add(HookEvent::PreToolUse, |_ctx: &mut HookContext| {
            HookResult::allow()
        });

    // --- Create session ---
    let session_id = format!(
        "gemini-test-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );
    let storage = SessionStorage::with_dir("./sessions");
    let session = AgentSession::new_with_storage(
        &session_id,
        "gemini-test-agent",
        "Gemini Test Agent",
        "Non-interactive test for GeminiProvider",
        storage,
    )?;

    // --- Configure agent ---
    let mut config = AgentConfig::new(SYSTEM_PROMPT)
        .with_hooks(hooks)
        .with_streaming(use_streaming)
        .with_prompt_caching(false) // Gemini doesn't use Anthropic-style caching
        .with_auto_name(false); // Skip naming for non-interactive

    if use_tools {
        config = config.with_tools(tools);
    }

    let agent = StandardAgent::new(config, llm);

    // --- Spawn agent ---
    let handle = runtime
        .spawn(session, move |internals| agent.run(internals))
        .await;

    if interactive {
        // Interactive mode - use console renderer
        println!("Type your requests below. All tools are auto-approved.");
        println!("Type 'exit' or 'quit' to stop.\n");

        let renderer = ConsoleRenderer::new(handle)
            .show_thinking(true)
            .show_tools(true);

        renderer.run().await?;
    } else {
        // Non-interactive mode - send a test message and wait for response
        let test_message = if use_tools {
            "List the files in the current directory using the Bash tool. Just run 'ls -la' and show me what you find."
        } else {
            "Explain what Rust's ownership system is in 2-3 sentences."
        };

        println!("[Test] Sending: {}", test_message);
        println!("[Test] Streaming: {}", use_streaming);
        println!("[Test] Tools: {}", use_tools);
        println!("---");

        handle.send_input(test_message).await.expect("Failed to send input");

        // Subscribe and collect output
        let mut output_rx = handle.subscribe();
        let mut full_response = String::new();

        loop {
            match output_rx.recv().await {
                Ok(chunk) => {
                    use shadow_agent_sdk::core::OutputChunk;
                    match chunk {
                        OutputChunk::TextDelta(text) => {
                            print!("{}", text);
                            full_response.push_str(&text);
                        }
                        OutputChunk::TextComplete(_) => {
                            // Already printed via deltas
                        }
                        OutputChunk::ThinkingDelta(text) => {
                            print!("[thinking] {}", text);
                        }
                        OutputChunk::ToolStart { name, input, .. } => {
                            println!("\n[Tool: {}] Input: {}", name, input);
                        }
                        OutputChunk::ToolEnd { id: _, result } => {
                            let result_text = match &result.content {
                                shadow_agent_sdk::tools::ToolResultData::Text(t) => t.clone(),
                                _ => "(non-text result)".to_string(),
                            };
                            let truncated = if result_text.len() > 200 {
                                format!("{}...", &result_text[..200])
                            } else {
                                result_text
                            };
                            println!("[Tool result] {}", truncated);
                        }
                        OutputChunk::Error(err) => {
                            eprintln!("\n[Error] {}", err);
                        }
                        OutputChunk::Done => {
                            println!("\n---");
                            println!("[Test] Complete!");
                            break;
                        }
                        _ => {}
                    }
                }
                Err(_) => break,
            }
        }
    }

    // Cleanup
    runtime.shutdown_all().await;
    println!("[Cleanup] Done.");
    Ok(())
}
