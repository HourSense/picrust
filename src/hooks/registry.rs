//! Hook Registry
//!
//! Contains:
//! - `Hook` trait - for implementing hooks
//! - `HookMatcher` - matches tools by regex pattern
//! - `HookRegistry` - stores and runs hooks

use std::collections::HashMap;
use std::sync::Arc;

use regex::Regex;

use super::types::{HookContext, HookEvent, HookResult, PermissionDecision};

/// Trait for hook implementations
///
/// Hooks are synchronous for simplicity. If you need async operations
/// (like HTTP calls), spawn a task and don't block.
pub trait Hook: Send + Sync {
    /// Execute the hook with the given context
    fn call(&self, ctx: &mut HookContext<'_>) -> HookResult;
}

/// Implement Hook for closures
///
/// Uses Higher-Ranked Trait Bounds (HRTB) to ensure the closure works
/// with any lifetime of HookContext.
impl<F> Hook for F
where
    F: for<'a> Fn(&mut HookContext<'a>) -> HookResult + Send + Sync,
{
    fn call(&self, ctx: &mut HookContext<'_>) -> HookResult {
        (self)(ctx)
    }
}

/// Type alias for stored hooks
pub type ArcHook = Arc<dyn Hook>;

/// Matches tools by name pattern and executes a hook
pub struct HookMatcher {
    /// Regex pattern to match tool names (None = match all)
    pattern: Option<Regex>,

    /// The hook to execute
    hook: ArcHook,
}

impl HookMatcher {
    /// Create a matcher that matches all tools
    pub fn new<H: Hook + 'static>(hook: H) -> Self {
        Self {
            pattern: None,
            hook: Arc::new(hook),
        }
    }

    /// Create a matcher with a regex pattern
    ///
    /// Pattern examples:
    /// - `"Bash"` - match only Bash tool
    /// - `"Read|Write|Edit"` - match file tools
    /// - `"^mcp__"` - match all MCP tools
    pub fn with_pattern<H: Hook + 'static>(pattern: &str, hook: H) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: Some(Regex::new(pattern)?),
            hook: Arc::new(hook),
        })
    }

    /// Check if this matcher applies to a tool name
    pub fn matches(&self, tool_name: &str) -> bool {
        match &self.pattern {
            Some(regex) => regex.is_match(tool_name),
            None => true, // No pattern = match all
        }
    }

    /// Run the hook with the given context
    pub fn run(&self, ctx: &mut HookContext<'_>) -> HookResult {
        self.hook.call(ctx)
    }
}

impl std::fmt::Debug for HookMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookMatcher")
            .field("pattern", &self.pattern.as_ref().map(|r| r.as_str()))
            .finish()
    }
}

/// Central registry for all hooks
///
/// # Example
///
/// ```ignore
/// let mut hooks = HookRegistry::new();
///
/// // Block dangerous commands
/// hooks.add_with_pattern(
///     HookEvent::PreToolUse,
///     "Bash",
///     |ctx| {
///         let cmd = ctx.tool_input.as_ref()
///             .and_then(|v| v.get("command"))
///             .and_then(|v| v.as_str())
///             .unwrap_or("");
///
///         if cmd.contains("rm -rf") {
///             HookResult::deny("Dangerous command blocked")
///         } else {
///             HookResult::none()
///         }
///     },
/// )?;
///
/// // Auto-approve read-only tools
/// hooks.add_with_pattern(
///     HookEvent::PreToolUse,
///     "Read|Glob|Grep",
///     |_| HookResult::allow(),
/// )?;
/// ```
#[derive(Default)]
pub struct HookRegistry {
    hooks: HashMap<HookEvent, Vec<HookMatcher>>,
}

impl HookRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a hook that matches all tools
    pub fn add<H: Hook + 'static>(&mut self, event: HookEvent, hook: H) -> &mut Self {
        self.hooks
            .entry(event)
            .or_default()
            .push(HookMatcher::new(hook));
        self
    }

    /// Add a hook with a tool name pattern
    pub fn add_with_pattern<H: Hook + 'static>(
        &mut self,
        event: HookEvent,
        pattern: &str,
        hook: H,
    ) -> Result<&mut Self, regex::Error> {
        self.hooks
            .entry(event)
            .or_default()
            .push(HookMatcher::with_pattern(pattern, hook)?);
        Ok(self)
    }

    /// Add a pre-built matcher
    pub fn add_matcher(&mut self, event: HookEvent, matcher: HookMatcher) -> &mut Self {
        self.hooks.entry(event).or_default().push(matcher);
        self
    }

    /// Check if there are any hooks for an event
    pub fn has_hooks(&self, event: HookEvent) -> bool {
        self.hooks
            .get(&event)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Get the number of hooks for an event
    pub fn hook_count(&self, event: HookEvent) -> usize {
        self.hooks.get(&event).map(|v| v.len()).unwrap_or(0)
    }

    /// Run all matching hooks for an event
    ///
    /// For tool hooks, filters by tool name.
    /// For non-tool hooks (like UserPromptSubmit), runs all hooks.
    ///
    /// **Hook execution behavior:**
    /// - By default (short_circuit_on_deny = false): ALL hooks run to completion
    ///   - Security hooks can't be bypassed
    ///   - Logging/monitoring hooks always fire
    ///   - Multiple hooks compose properly
    ///
    /// - When short_circuit_on_deny = true: Stops on first Deny
    ///   - Performance optimization
    ///   - ⚠️ Later hooks may not run
    ///
    /// Results are combined with priority:
    /// - If ANY hook said Deny → DENY (most restrictive wins)
    /// - Else if ANY hook said Allow → ALLOW
    /// - Else if ANY hook said Ask → ASK
    /// - Else (all said None) → NONE (continue normal flow)
    pub fn run(&self, ctx: &mut HookContext<'_>) -> HookResult {
        let event = ctx.event;
        let tool_name = ctx.tool_name.clone();
        let short_circuit = ctx.short_circuit_on_deny;

        let matchers = match self.hooks.get(&event) {
            Some(matchers) => matchers,
            None => return HookResult::none(),
        };

        let mut combined = HookResult::none();

        for matcher in matchers {
            // For tool hooks, check if matcher applies to this tool
            let should_run = match (&tool_name, event) {
                (
                    Some(name),
                    HookEvent::PreToolUse | HookEvent::PostToolUse | HookEvent::PostToolUseFailure,
                ) => matcher.matches(name),
                _ => true, // Non-tool hooks always run
            };

            if !should_run {
                continue;
            }

            // Run the hook
            let result = matcher.run(ctx);

            // Combine results (Deny > Allow > Ask > None)
            combined = combine_results(combined, result);

            // Optionally short-circuit on Deny (if enabled in config)
            if short_circuit && combined.decision == Some(PermissionDecision::Deny) {
                tracing::debug!("[HookRegistry] Short-circuiting on Deny (remaining hooks skipped)");
                break;
            }
        }

        combined
    }
}

/// Combine two hook results
///
/// Priority: Deny > Allow > Ask > None
fn combine_results(a: HookResult, b: HookResult) -> HookResult {
    match (a.decision, b.decision) {
        // Deny always wins
        (Some(PermissionDecision::Deny), _) => a,
        (_, Some(PermissionDecision::Deny)) => b,

        // Allow beats Ask and None
        (Some(PermissionDecision::Allow), _) => a,
        (_, Some(PermissionDecision::Allow)) => b,

        // Ask beats None
        (Some(PermissionDecision::Ask), _) => a,
        (_, Some(PermissionDecision::Ask)) => b,

        // Both None
        _ => HookResult::none(),
    }
}

impl std::fmt::Debug for HookRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (event, matchers) in &self.hooks {
            map.entry(event, &matchers.len());
        }
        map.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_matcher_pattern() {
        let matcher =
            HookMatcher::with_pattern("Bash|Shell", |_ctx: &mut HookContext| HookResult::none()).unwrap();

        assert!(matcher.matches("Bash"));
        assert!(matcher.matches("Shell"));
        assert!(!matcher.matches("Read"));
        assert!(!matcher.matches("Write"));
    }

    #[test]
    fn test_hook_matcher_no_pattern() {
        let matcher = HookMatcher::new(|_ctx: &mut HookContext| HookResult::none());

        assert!(matcher.matches("Bash"));
        assert!(matcher.matches("Read"));
        assert!(matcher.matches("anything"));
    }

    #[test]
    fn test_combine_results() {
        // Deny wins
        assert_eq!(
            combine_results(HookResult::deny("x"), HookResult::allow()).decision,
            Some(PermissionDecision::Deny)
        );
        assert_eq!(
            combine_results(HookResult::allow(), HookResult::deny("x")).decision,
            Some(PermissionDecision::Deny)
        );

        // Allow beats Ask
        assert_eq!(
            combine_results(HookResult::allow(), HookResult::ask()).decision,
            Some(PermissionDecision::Allow)
        );

        // Ask beats None
        assert_eq!(
            combine_results(HookResult::ask(), HookResult::none()).decision,
            Some(PermissionDecision::Ask)
        );
    }

    #[test]
    fn test_registry_add() {
        let mut registry = HookRegistry::new();

        registry.add(HookEvent::PreToolUse, |_ctx: &mut HookContext| HookResult::none());
        registry
            .add_with_pattern(HookEvent::PreToolUse, "Bash", |_ctx: &mut HookContext| {
                HookResult::deny("blocked")
            })
            .unwrap();

        assert!(registry.has_hooks(HookEvent::PreToolUse));
        assert_eq!(registry.hook_count(HookEvent::PreToolUse), 2);
        assert!(!registry.has_hooks(HookEvent::PostToolUse));
    }

    #[test]
    fn test_combine_deny_wins_over_allow() {
        // Deny should win over Allow
        let result = combine_results(HookResult::allow(), HookResult::deny("blocked"));
        assert_eq!(result.decision, Some(PermissionDecision::Deny));
        assert_eq!(result.reason, Some("blocked".to_string()));

        // Order shouldn't matter
        let result = combine_results(HookResult::deny("blocked"), HookResult::allow());
        assert_eq!(result.decision, Some(PermissionDecision::Deny));
    }

    #[test]
    fn test_combine_deny_wins_over_all() {
        // Deny > Allow
        assert_eq!(
            combine_results(HookResult::deny("x"), HookResult::allow()).decision,
            Some(PermissionDecision::Deny)
        );

        // Deny > Ask
        assert_eq!(
            combine_results(HookResult::deny("x"), HookResult::ask()).decision,
            Some(PermissionDecision::Deny)
        );

        // Deny > None
        assert_eq!(
            combine_results(HookResult::deny("x"), HookResult::none()).decision,
            Some(PermissionDecision::Deny)
        );
    }

    #[test]
    fn test_combine_allow_wins_over_ask_and_none() {
        // Allow > Ask
        assert_eq!(
            combine_results(HookResult::allow(), HookResult::ask()).decision,
            Some(PermissionDecision::Allow)
        );

        // Allow > None
        assert_eq!(
            combine_results(HookResult::allow(), HookResult::none()).decision,
            Some(PermissionDecision::Allow)
        );
    }
}
