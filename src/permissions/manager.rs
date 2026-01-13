//! Permission manager implementation
//!
//! Handles permission requests for tool execution with support for
//! "always allow" and "always deny" patterns.

use std::collections::HashSet;

/// A request for permission to execute a tool
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    /// Name of the tool
    pub tool_name: String,
    /// Human-readable description of the action
    pub action_description: String,
    /// Optional details about the action
    pub details: Option<String>,
}

impl PermissionRequest {
    /// Create a new permission request
    pub fn new(
        tool_name: impl Into<String>,
        action_description: impl Into<String>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            action_description: action_description.into(),
            details: None,
        }
    }

    /// Add details to the request
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// The user's decision on a permission request
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Allow this action
    Allow,
    /// Deny this action
    Deny,
    /// Always allow this tool
    AlwaysAllow,
    /// Always deny this tool
    AlwaysDeny,
}

/// A permission rule for matching tool actions
#[derive(Debug, Clone)]
pub struct Permission {
    /// Tool name pattern (exact match for now)
    pub tool_name: String,
    /// Whether to allow or deny
    pub allow: bool,
}

impl Permission {
    /// Create a new permission rule
    pub fn new(tool_name: impl Into<String>, allow: bool) -> Self {
        Self {
            tool_name: tool_name.into(),
            allow,
        }
    }
}

/// Manages permissions for tool execution
pub struct PermissionManager {
    /// Tools that are always allowed
    always_allow: HashSet<String>,
    /// Tools that are always denied
    always_deny: HashSet<String>,
    /// Permission rules (for future extensibility)
    rules: Vec<Permission>,
}

impl PermissionManager {
    /// Create a new permission manager
    pub fn new() -> Self {
        Self {
            always_allow: HashSet::new(),
            always_deny: HashSet::new(),
            rules: Vec::new(),
        }
    }

    /// Add a permission rule
    pub fn add_rule(&mut self, rule: Permission) {
        self.rules.push(rule);
    }

    /// Set a tool to always be allowed
    pub fn always_allow_tool(&mut self, tool_name: impl Into<String>) {
        let name = tool_name.into();
        tracing::info!("Setting tool to always allow: {}", name);
        self.always_deny.remove(&name);
        self.always_allow.insert(name);
    }

    /// Set a tool to always be denied
    pub fn always_deny_tool(&mut self, tool_name: impl Into<String>) {
        let name = tool_name.into();
        tracing::info!("Setting tool to always deny: {}", name);
        self.always_allow.remove(&name);
        self.always_deny.insert(name);
    }

    /// Clear all "always allow" and "always deny" settings
    pub fn clear_auto_decisions(&mut self) {
        self.always_allow.clear();
        self.always_deny.clear();
    }

    /// Check if a tool should be automatically allowed or denied
    ///
    /// Returns:
    /// - Some(true) if the tool should be automatically allowed
    /// - Some(false) if the tool should be automatically denied
    /// - None if the user should be asked
    pub fn check_auto_decision(&self, tool_name: &str) -> Option<bool> {
        if self.always_allow.contains(tool_name) {
            return Some(true);
        }
        if self.always_deny.contains(tool_name) {
            return Some(false);
        }
        None
    }

    /// Process a permission decision and update rules accordingly
    pub fn process_decision(&mut self, tool_name: &str, decision: PermissionDecision) {
        match decision {
            PermissionDecision::AlwaysAllow => {
                self.always_allow_tool(tool_name);
            }
            PermissionDecision::AlwaysDeny => {
                self.always_deny_tool(tool_name);
            }
            PermissionDecision::Allow | PermissionDecision::Deny => {
                // One-time decision, no persistent change
            }
        }
    }

    /// Check if a permission request should be allowed based on current rules
    ///
    /// This is for future extensibility with pattern matching rules.
    pub fn evaluate_rules(&self, _request: &PermissionRequest) -> Option<bool> {
        // For now, just use the simple always_allow/always_deny sets
        // Future: implement pattern matching on action descriptions, etc.
        None
    }

    /// Get a list of tools that are always allowed
    pub fn get_always_allowed(&self) -> Vec<&str> {
        self.always_allow.iter().map(|s| s.as_str()).collect()
    }

    /// Get a list of tools that are always denied
    pub fn get_always_denied(&self) -> Vec<&str> {
        self.always_deny.iter().map(|s| s.as_str()).collect()
    }
}

impl Default for PermissionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_decision() {
        let mut manager = PermissionManager::new();

        assert_eq!(manager.check_auto_decision("bash"), None);

        manager.always_allow_tool("bash");
        assert_eq!(manager.check_auto_decision("bash"), Some(true));

        manager.always_deny_tool("bash");
        assert_eq!(manager.check_auto_decision("bash"), Some(false));
    }

    #[test]
    fn test_process_decision() {
        let mut manager = PermissionManager::new();

        manager.process_decision("bash", PermissionDecision::AlwaysAllow);
        assert_eq!(manager.check_auto_decision("bash"), Some(true));

        manager.process_decision("bash", PermissionDecision::AlwaysDeny);
        assert_eq!(manager.check_auto_decision("bash"), Some(false));
    }

    #[test]
    fn test_clear_decisions() {
        let mut manager = PermissionManager::new();

        manager.always_allow_tool("bash");
        manager.always_deny_tool("file_edit");

        manager.clear_auto_decisions();

        assert_eq!(manager.check_auto_decision("bash"), None);
        assert_eq!(manager.check_auto_decision("file_edit"), None);
    }
}
