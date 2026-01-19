//! FileSummaryAgent - A subagent that summarizes file contents
//!
//! This agent:
//! - Has access only to the Read tool
//! - Has local permission to always use Read (no prompts)
//! - Takes a file path and returns a summary
//! - No console attachment - runs autonomously

use std::sync::Arc;

use anyhow::Result;

use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    llm::AnthropicProvider,
    permissions::PermissionRule,
    runtime::{AgentHandle, AgentRuntime},
    session::AgentSession,
    tools::ToolRegistry,
};

/// System prompt for the FileSummaryAgent
const SYSTEM_PROMPT: &str = r#"You are a file summarization agent. Your only job is to:

1. Read the file you are given using the Read tool
2. Provide a concise summary of its contents (max 100 words)

Be direct and factual. Focus on:
- What the file is (code, config, docs, etc.)
- Key contents or purpose
- Important details

Do NOT ask questions. Just read and summarize."#;

/// Creates the tool registry for FileSummaryAgent (Read only)
pub fn create_tools() -> Result<ToolRegistry> {
    use shadow_agent_sdk::tools::common::ReadTool;

    let mut registry = ToolRegistry::new();
    registry.register(ReadTool::new()?);

    Ok(registry)
}

/// Configuration for spawning a FileSummaryAgent
pub struct FileSummaryAgentConfig {
    /// The LLM provider to use
    pub llm: Arc<AnthropicProvider>,
    /// The runtime to spawn on
    pub runtime: AgentRuntime,
}

impl FileSummaryAgentConfig {
    pub fn new(llm: Arc<AnthropicProvider>, runtime: AgentRuntime) -> Self {
        Self { llm, runtime }
    }
}

/// Spawns a FileSummaryAgent as a subagent
///
/// # Arguments
/// * `config` - The agent configuration
/// * `parent_session_id` - The parent agent's session ID
/// * `parent_tool_use_id` - The tool_use_id that triggered this spawn
/// * `file_path` - The file to summarize
///
/// # Returns
/// The agent handle for communication
pub async fn spawn_file_summary_agent(
    config: &FileSummaryAgentConfig,
    parent_session_id: &str,
    parent_tool_use_id: &str,
    file_path: &str,
) -> Result<AgentHandle> {
    // Generate unique session ID for this subagent
    let session_id = format!(
        "file-summary-{}-{}",
        parent_tool_use_id,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    // Create subagent session linked to parent
    let session = AgentSession::new_subagent(
        &session_id,
        "file-summary-agent",
        "File Summary Agent",
        &format!("Summarizes file: {}", file_path),
        parent_session_id,
        parent_tool_use_id,
    )?;

    // Create tool registry with Read only
    let tools = Arc::new(create_tools()?);

    // Create agent config - no debug logging for subagents
    let agent_config = AgentConfig::new(SYSTEM_PROMPT)
        .with_tools(tools);

    // Create the agent
    let agent = StandardAgent::new(agent_config, config.llm.clone());

    // Spawn with local Read permission (always allowed)
    let handle = config
        .runtime
        .spawn_with_local_rules(
            session,
            vec![PermissionRule::allow_tool("Read")],
            move |internals| agent.run(internals),
        )
        .await;

    // Send the initial task to the agent
    handle
        .send_input(format!("Summarize this file: {}", file_path))
        .await?;

    Ok(handle)
}
