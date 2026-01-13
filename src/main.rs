use singapore_project::agent::{Agent, default_system_prompt};
use singapore_project::cli::Console;
use singapore_project::context::ContextManager;
use singapore_project::llm::AnthropicProvider;
use singapore_project::logging;
use singapore_project::tools::{new_todo_list, BashTool, FileEditTool, TodoTool, ToolRegistry};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging system
    logging::init_logging()?;

    tracing::info!("=== Coding Agent Starting ===");

    // Create console for terminal I/O
    let console = Console::new();

    // Create Anthropic LLM provider from environment
    let llm_provider = AnthropicProvider::from_env()?;

    // Create tool registry with available tools
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(BashTool::new()?);
    tool_registry.register(FileEditTool::new()?);

    // Create shared todo list and register TODO tool
    let todo_list = new_todo_list();
    tool_registry.register(TodoTool::new(todo_list));

    tracing::info!("Registered {} tools", tool_registry.len());

    // Create context manager with system prompt
    let context_manager = ContextManager::new(default_system_prompt());

    // Create agent with all components
    let mut agent = Agent::new(console, llm_provider, tool_registry, context_manager)?;

    tracing::info!(
        "Agent initialized with conversation ID: {}",
        agent.conversation_id()
    );

    // Run the agent loop
    agent.run().await?;

    tracing::info!("=== Coding Agent Shutting Down ===");

    Ok(())
}
