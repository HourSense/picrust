//! Agent Configuration
//!
//! Configuration options for the StandardAgent.

use std::sync::Arc;

use crate::helpers::InjectionChain;
use crate::hooks::HookRegistry;
use crate::llm::{LlmProvider, ThinkingConfig};
use crate::tools::ToolRegistry;

/// Configuration for a StandardAgent
///
/// Use the builder pattern to configure the agent:
///
/// ```ignore
/// let config = AgentConfig::new()
///     .with_tools(tools)
///     .with_injection_chain(injections)
///     .with_max_tool_iterations(50)
///     .with_thinking(16000)  // Enable extended thinking with 16k token budget
///     .with_streaming(true)
///     .with_debug(true);
/// ```
///
/// The system prompt is no longer part of AgentConfig — it lives in the session's
/// `system_prompt.md` file and is passed to `AgentSession::new()`.
pub struct AgentConfig {
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

    /// Whether to enable streaming responses from the LLM
    pub streaming_enabled: bool,

    /// Extended thinking configuration (optional)
    /// When enabled, Claude will show its step-by-step reasoning process.
    pub thinking: Option<ThinkingConfig>,

    /// Hooks for intercepting agent behavior
    /// Use hooks to block dangerous operations, modify tool inputs, auto-approve tools, etc.
    pub hooks: Option<Arc<HookRegistry>>,

    /// Whether to automatically generate a conversation name after the first turn
    /// Uses Haiku to create a short, descriptive name based on conversation content.
    pub auto_name_conversation: bool,

    /// Whether to enable prompt caching
    /// When enabled, the agent will automatically add cache_control breakpoints to:
    /// - The last tool definition (caches all tools)
    /// - The system prompt (caches static instructions)
    /// - The last message block before each LLM call (caches conversation history)
    /// This significantly reduces costs and latency for multi-turn conversations.
    pub enable_prompt_caching: bool,

    /// Optional LLM provider for conversation naming.
    /// If set, this provider is used for auto-naming conversations (typically a
    /// lightweight/fast model). If not set, the main agent LLM is used.
    pub naming_llm: Option<Arc<dyn LlmProvider>>,

    /// Whether to enable hook short-circuiting.
    ///
    /// When enabled (false by default), if a hook returns `Deny`, subsequent hooks
    /// will not run. This provides a performance optimization but means:
    /// - Later security hooks might not run
    /// - Logging/monitoring hooks might not fire
    ///
    /// **Default: false** (safer - all hooks run, Deny wins at the end)
    ///
    /// Set to true only if you need the performance optimization and understand
    /// that security/monitoring hooks may be bypassed.
    pub hook_short_circuit: bool,

    /// **DANGEROUS:** Skip all permission checks.
    ///
    /// When enabled, tools execute without asking for user permission.
    /// Hooks still run and can block operations, but the permission system is bypassed.
    ///
    /// **Default: false** (safe - permissions are enforced)
    ///
    /// **Use cases:**
    /// - Automated workflows where user cannot approve
    /// - Testing environments
    /// - Trusted agent scenarios
    ///
    /// **Security implications:**
    /// - Tools execute without user approval
    /// - Global/local/session permission rules are ignored
    /// - Hooks are your only safety mechanism
    ///
    /// This can be changed at runtime via `AgentHandle::set_dangerous_skip_permissions()`.
    pub dangerous_skip_permissions: bool,
}

impl AgentConfig {
    /// Create a new agent configuration
    ///
    /// The system prompt is not set here — it is stored in the session's `system_prompt.md`
    /// and managed via `AgentSession::new()` or `AgentSession::update_system_prompt()`.
    pub fn new() -> Self {
        Self {
            tools: None,
            injections: InjectionChain::new(),
            max_tool_iterations: 100,
            auto_save_session: true,
            debug_enabled: false,
            streaming_enabled: false,
            thinking: None,
            hooks: None,
            auto_name_conversation: true,
            enable_prompt_caching: true,
            naming_llm: None,
            hook_short_circuit: false, // Safe default: all hooks run
            dangerous_skip_permissions: false, // Safe default: permissions enforced
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

    /// Enable or disable streaming responses
    ///
    /// When enabled, the agent will stream LLM responses in real-time,
    /// sending text deltas as they arrive. This provides a better user
    /// experience for interactive applications.
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming_enabled = enabled;
        self
    }

    /// Enable extended thinking with a token budget
    ///
    /// Extended thinking gives Claude enhanced reasoning capabilities for complex tasks.
    /// Claude will show its step-by-step thought process before delivering its final answer.
    ///
    /// The `budget_tokens` parameter determines the maximum tokens Claude can use for thinking.
    /// Minimum is 1024 tokens. Larger budgets can improve response quality for complex problems.
    ///
    /// Note: When thinking is enabled, temperature is automatically set to 1 (required by API).
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig::enabled(budget_tokens));
        self
    }

    /// Set a custom thinking configuration
    pub fn with_thinking_config(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Set the hook registry for intercepting agent behavior
    ///
    /// Hooks allow you to:
    /// - Block dangerous operations before they execute
    /// - Modify tool arguments (e.g., rewrite paths for remote filesystem)
    /// - Auto-approve certain tools
    /// - Log and audit tool calls
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut hooks = HookRegistry::new();
    /// hooks.add_with_pattern(HookEvent::PreToolUse, "Bash", |ctx| {
    ///     if ctx.tool_input.as_ref()
    ///         .and_then(|v| v.get("command"))
    ///         .and_then(|v| v.as_str())
    ///         .map(|c| c.contains("rm -rf"))
    ///         .unwrap_or(false)
    ///     {
    ///         HookResult::deny("Dangerous command blocked")
    ///     } else {
    ///         HookResult::none()
    ///     }
    /// })?;
    ///
    /// let config = AgentConfig::new("...").with_hooks(hooks);
    /// ```
    pub fn with_hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = Some(Arc::new(hooks));
        self
    }

    /// Enable or disable automatic conversation naming
    ///
    /// When enabled (default), the agent will automatically generate a short,
    /// descriptive name for the conversation after the first turn completes.
    /// Uses the naming LLM if set (see [`with_naming_llm`](Self::with_naming_llm)),
    /// otherwise uses the main agent LLM.
    pub fn with_auto_name(mut self, enabled: bool) -> Self {
        self.auto_name_conversation = enabled;
        self
    }

    /// Set a separate LLM provider for conversation naming
    ///
    /// This allows using a lightweight/fast model for naming (e.g., Haiku or Flash)
    /// while the main agent uses a more capable model.
    ///
    /// If not set, the main agent LLM is used for naming.
    pub fn with_naming_llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.naming_llm = Some(llm);
        self
    }

    /// Enable or disable prompt caching
    ///
    /// When enabled (default), the agent automatically adds cache_control breakpoints to:
    /// - The last tool definition (caches all tools)
    /// - The system prompt (caches static instructions)
    /// - The last message block (caches conversation history)
    ///
    /// This provides significant cost savings (90% discount on cached tokens) and
    /// improved latency for multi-turn conversations.
    ///
    /// See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    pub fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.enable_prompt_caching = enabled;
        self
    }

    /// Enable or disable hook short-circuiting
    ///
    /// **Default: false** (safer - all hooks run, security can't be bypassed)
    ///
    /// When false (default):
    /// - ALL hooks run to completion
    /// - If ANY hook returns `Deny`, final result is `Deny`
    /// - Security hooks can't be bypassed by earlier `Allow` hooks
    /// - Logging/monitoring hooks always fire
    ///
    /// When true:
    /// - First hook to return `Deny` stops execution
    /// - Later hooks don't run (performance optimization)
    /// - Security hooks might not run if bypassed by earlier hooks
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Safe mode (default): all hooks run
    /// let config = AgentConfig::new("...")
    ///     .with_hooks(hooks);
    ///
    /// // Performance mode: short-circuit on first Deny
    /// let config = AgentConfig::new("...")
    ///     .with_hooks(hooks)
    ///     .with_hook_short_circuit(true);
    /// ```
    pub fn with_hook_short_circuit(mut self, enabled: bool) -> Self {
        self.hook_short_circuit = enabled;
        self
    }

    /// **DANGEROUS:** Skip all permission checks
    ///
    /// **Default: false** (safe - permissions are enforced)
    ///
    /// When enabled:
    /// - Tools execute without asking for user permission
    /// - Global/local/session permission rules are bypassed
    /// - Hooks still run and can block operations
    /// - Can be changed at runtime via `AgentHandle::set_dangerous_skip_permissions()`
    ///
    /// **Only enable if:**
    /// - Running in automated/testing environment
    /// - You trust the agent completely
    /// - You have hooks configured for security
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Automated workflow mode
    /// let config = AgentConfig::new("...")
    ///     .with_dangerous_skip_permissions(true)
    ///     .with_hooks(security_hooks);  // Still use hooks for safety!
    /// ```
    pub fn with_dangerous_skip_permissions(mut self, enabled: bool) -> Self {
        self.dangerous_skip_permissions = enabled;
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
        Self::new()
    }
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("tools", &self.tools.as_ref().map(|t| t.tool_names()))
            .field("max_tool_iterations", &self.max_tool_iterations)
            .field("auto_save_session", &self.auto_save_session)
            .field("debug_enabled", &self.debug_enabled)
            .field("streaming_enabled", &self.streaming_enabled)
            .field("thinking", &self.thinking)
            .field("hooks", &self.hooks.as_ref().map(|h| format!("{:?}", h)))
            .field("auto_name_conversation", &self.auto_name_conversation)
            .field("enable_prompt_caching", &self.enable_prompt_caching)
            .field("naming_llm", &self.naming_llm.as_ref().map(|l| l.model()))
            .field("hook_short_circuit", &self.hook_short_circuit)
            .field("dangerous_skip_permissions", &self.dangerous_skip_permissions)
            .finish()
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
        assert!(config.auto_name_conversation);
        assert_eq!(config.max_tool_iterations, 100);
    }

    #[test]
    fn test_agent_config_with_debug() {
        let config = AgentConfig::new().with_debug(true);
        assert!(config.debug_enabled);
    }
}
