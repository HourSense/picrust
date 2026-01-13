use singapore_project::agent::Agent;
use singapore_project::cli::Console;
use singapore_project::llm::AnthropicProvider;
use singapore_project::logging;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging system
    logging::init_logging()?;

    tracing::info!("=== Coding Agent Starting ===");

    // Create console for terminal I/O
    let console = Console::new();

    // Create Anthropic LLM provider from environment
    let llm_provider = AnthropicProvider::from_env()?;

    // Create agent with console and LLM provider
    let agent = Agent::new(console, llm_provider)
        .with_system_prompt("You are a helpful coding assistant.");

    // Run the agent loop
    agent.run().await?;

    tracing::info!("=== Coding Agent Shutting Down ===");

    Ok(())
}
