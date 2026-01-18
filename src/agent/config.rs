//! Agent Configuration
//!
//! Configuration options for the StandardAgent.

use std::sync::Arc;

use crate::helpers::InjectionChain;
use crate::tools::ToolRegistry;

/// Configuration for a StandardAgent
///
/// Use the builder pattern to configure the agent:
///
/// ```ignore
/// let config = AgentConfig::new("You are a helpful assistant")
///     .with_tools(tools)
///     .with_injection_chain(injections)
///     .with_max_tool_iterations(50)
///     .with_debug(true);
/// ```
pub struct AgentConfig {
    /// System prompt for the LLM
    pub system_prompt: String,

    /// Tool registry (optional - agent can work without tools)
    pub tools: Option<Arc<ToolRegistry>>,

    /// Context injection chain (applied before each LLM call)
    pub injections: InjectionChain,

    /// Maximum number of tool iterations per turn (prevents infinite loops)
    pub max_tool_iterations: usize,

    /// Whether to auto-save session after each turn
    pub auto_save_session: bool,

    /// Whether to enable debug logging (API calls, tool calls)
    pub debug_enabled: bool,
}

impl AgentConfig {
    /// Create a new agent configuration with a system prompt
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            tools: None,
            injections: InjectionChain::new(),
            max_tool_iterations: 100,
            auto_save_session: true,
            debug_enabled: false,
        }
    }

    /// Set the tool registry
    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the context injection chain
    pub fn with_injection_chain(mut self, injections: InjectionChain) -> Self {
        self.injections = injections;
        self
    }

    /// Add a single injection to the chain
    pub fn with_injection<I: crate::helpers::ContextInjection + 'static>(
        mut self,
        injection: I,
    ) -> Self {
        self.injections.add(injection);
        self
    }

    /// Add a function-based injection
    pub fn with_injection_fn<F>(mut self, name: impl Into<String>, func: F) -> Self
    where
        F: Fn(&crate::runtime::AgentInternals, Vec<crate::llm::Message>) -> Vec<crate::llm::Message>
            + Send
            + Sync
            + 'static,
    {
        self.injections.add_fn(name, func);
        self
    }

    /// Set maximum tool iterations per turn
    pub fn with_max_tool_iterations(mut self, max: usize) -> Self {
        self.max_tool_iterations = max;
        self
    }

    /// Set whether to auto-save session after each turn
    pub fn with_auto_save(mut self, auto_save: bool) -> Self {
        self.auto_save_session = auto_save;
        self
    }

    /// Enable or disable debug logging
    ///
    /// When enabled, the agent will log all API requests/responses and tool
    /// calls to a `debugger/` folder in the session directory.
    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.debug_enabled = enabled;
        self
    }

    /// Get tool definitions (empty vec if no tools)
    pub fn tool_definitions(&self) -> Vec<crate::llm::ToolDefinition> {
        self.tools
            .as_ref()
            .map(|t| t.get_definitions())
            .unwrap_or_default()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self::new("You are a helpful assistant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_defaults() {
        let config = AgentConfig::default();
        assert!(!config.debug_enabled);
        assert!(config.auto_save_session);
        assert_eq!(config.max_tool_iterations, 100);
    }

    #[test]
    fn test_agent_config_with_debug() {
        let config = AgentConfig::new("Test").with_debug(true);
        assert!(config.debug_enabled);
    }
}
