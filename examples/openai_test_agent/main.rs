//! OpenAI Test Agent
//!
//! Tests the OpenAIProvider: streaming, tool calling, tool results, interactive mode.
//!
//! Run with:
//!   OPENAI_API_KEY=sk-... cargo run --example openai_test_agent
//!   OPENAI_API_KEY=sk-... cargo run --example openai_test_agent -- --stream
//!   OPENAI_API_KEY=sk-... cargo run --example openai_test_agent -- --tools
//!   OPENAI_API_KEY=sk-... cargo run --example openai_test_agent -- --stream --tools
//!   OPENAI_API_KEY=sk-... cargo run --example openai_test_agent -- --interactive
//!   OPENAI_API_KEY=sk-... cargo run --example openai_test_agent -- --interactive --stream --tools

mod tools;

use anyhow::Result;
use std::env;
use std::sync::Arc;

use picrust::{
    agent::{AgentConfig, StandardAgent},
    cli::ConsoleRenderer,
    hooks::{HookContext, HookEvent, HookRegistry, HookResult},
    llm::{LlmProvider, OpenAIProvider, SwappableLlmProvider},
    runtime::AgentRuntime,
    session::{AgentSession, SessionStorage},
};

const SYSTEM_PROMPT: &str = r#"You are a helpful coding assistant with access to tools.

You have the following tools available:
- Read: Read file contents
- Bash: Execute shell commands
- Grep: Search file contents
- Glob: Find files by pattern

When the user asks you to do something, use the appropriate tools.
Be concise in your responses."#;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "openai_test_agent=info,picrust=info".to_string()),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    let use_tools = args.iter().any(|a| a == "--tools" || a == "-t");
    let use_streaming = args.iter().any(|a| a == "--stream" || a == "-s");
    let interactive = args.iter().any(|a| a == "--interactive" || a == "-i");

    println!("=== OpenAI Test Agent ===");
    println!(
        "Mode: {} | Streaming: {} | Tools: {}\n",
        if interactive { "interactive" } else { "non-interactive" },
        use_streaming,
        use_tools
    );

    // --- Create OpenAI provider ---
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY not set"))?;
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let base_url = env::var("OPENAI_BASE_URL").ok();

    let mut provider = OpenAIProvider::new(api_key)?
        .with_model(&model)
        .with_max_tokens(4096);
    if let Some(url) = &base_url {
        provider = provider.with_base_url(url);
    }
    let openai = Arc::new(provider);

    // Wrap in SwappableLlmProvider (same pattern as Gemini example)
    let swappable = SwappableLlmProvider::new(openai);
    let _llm_handle = swappable.handle();
    let llm: Arc<dyn LlmProvider> = Arc::new(swappable);

    println!("[Setup] Model: {}", model);
    println!("[Setup] Provider: openai (direct API)");

    // --- Runtime ---
    let runtime = AgentRuntime::new();
    runtime.global_permissions();

    // --- Tools ---
    let tools = Arc::new(tools::create_registry()?);
    println!("[Setup] Tools: {:?}", tools.tool_names());

    // --- Hooks: auto-approve everything ---
    let mut hooks = HookRegistry::new();
    hooks.add(HookEvent::PreToolUse, |_ctx: &mut HookContext| {
        HookResult::allow()
    });

    // --- Session ---
    let session_id = format!(
        "openai-test-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    );
    let storage = SessionStorage::with_dir("./sessions");
    let session = AgentSession::new_with_storage(
        &session_id,
        "openai-test-agent",
        "OpenAI Test Agent",
        "Tests for OpenAIProvider",
        SYSTEM_PROMPT,
        storage,
    )?;

    // --- Agent config ---
    let mut config = AgentConfig::new()
        .with_hooks(hooks)
        .with_streaming(use_streaming)
        .with_prompt_caching(false) // OpenAI Responses API doesn't use Anthropic-style caching
        .with_auto_name(false);

    if use_tools {
        config = config.with_tools(tools);
    }

    let agent = StandardAgent::new(config, llm);

    // --- Spawn ---
    let handle = runtime
        .spawn(session, move |internals| agent.run(internals))
        .await;

    if interactive {
        println!("Type your requests below. All tools are auto-approved.");
        println!("Type 'exit' or 'quit' to stop.\n");

        let renderer = ConsoleRenderer::new(handle).show_tools(true);
        renderer.run().await?;
    } else {
        // Non-interactive: run through a sequence of tests
        let tests: Vec<(&str, &str)> = if use_tools {
            vec![
                // 1. Simple text response
                ("1. Basic text", "Say 'Hello from OpenAI!' and nothing else."),
                // 2. Tool call — bash ls
                ("2. Tool call (Bash)", "Run 'echo hello_from_openai_tool' using the Bash tool and show me the output."),
                // 3. Multi-turn with tool result
                ("3. Glob tool", "Use the Glob tool to find all *.toml files in the current directory (pattern: *.toml). List what you find."),
            ]
        } else {
            vec![
                ("1. Basic", "Say 'Hello from OpenAI!' and nothing else."),
                ("2. Short explanation", "What is Rust's borrow checker in one sentence?"),
                ("3. List", "Name three programming languages in a bullet list."),
            ]
        };

        for (label, prompt) in tests {
            println!("\n[Test {}]", label);
            println!("  Prompt: {}", prompt);
            println!("  ---");

            handle.send_input(prompt).await.expect("Failed to send input");

            let mut output_rx = handle.subscribe();
            let mut response_text = String::new();

            loop {
                match output_rx.recv().await {
                    Ok(chunk) => {
                        use picrust::core::OutputChunk;
                        match chunk {
                            OutputChunk::TextDelta(text) => {
                                print!("{}", text);
                                response_text.push_str(&text);
                            }
                            OutputChunk::TextComplete(_) => {}
                            OutputChunk::ToolStart { name, input, .. } => {
                                println!("\n  [Tool call: {}]", name);
                                println!("  Input: {}", input);
                            }
                            OutputChunk::ToolEnd { result, .. } => {
                                let result_text = match &result.content {
                                    picrust::tools::ToolResultData::Text(t) => t.clone(),
                                    _ => "(non-text result)".to_string(),
                                };
                                let truncated = if result_text.len() > 300 {
                                    format!("{}...", &result_text[..300])
                                } else {
                                    result_text
                                };
                                println!("  [Tool result]: {}", truncated);
                            }
                            OutputChunk::Error(err) => {
                                eprintln!("\n  [Error] {}", err);
                                break;
                            }
                            OutputChunk::Done => {
                                println!("\n  --- Done ---");
                                break;
                            }
                            _ => {}
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        println!("\n=== All tests complete ===");
    }

    runtime.shutdown_all().await;
    println!("[Cleanup] Done.");
    Ok(())
}
