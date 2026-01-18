//! Context Injection System
//!
//! Allows programmers to modify messages before each LLM call without
//! writing the entire agent loop themselves.
//!
//! # Overview
//!
//! Context injections are functions/traits that transform the message history
//! right before it's sent to the LLM. This enables:
//!
//! - Adding system reminders (e.g., "you haven't updated your todo list")
//! - Augmenting the first user message with detailed instructions
//! - Injecting dynamic context based on agent state
//! - Any message transformation based on `AgentInternals`
//!
//! # Example
//!
//! ```ignore
//! // Create a todo reminder injection
//! let todo_injection = FnInjection::new("todo_reminder", |internals, mut messages| {
//!     let manager = internals.context.get_resource::<TodoListManager>();
//!     if let Some(manager) = manager {
//!         let turns_since_update = internals.context.current_turn - manager.last_updated_turn();
//!         if turns_since_update > 20 {
//!             // Inject a reminder into the last message
//!             inject_system_reminder(&mut messages, "Consider updating your todo list");
//!         }
//!     }
//!     messages
//! });
//!
//! // Add to agent's injection list
//! agent.add_injection(todo_injection);
//! ```

use crate::llm::Message;
use crate::runtime::AgentInternals;

/// Trait for context injection implementations
///
/// Implement this trait to create reusable, stateful injections.
/// For simple closures, use `FnInjection` instead.
pub trait ContextInjection: Send + Sync {
    /// Name of this injection (for logging/debugging)
    fn name(&self) -> &str;

    /// Inject context into messages before LLM call
    ///
    /// Called with the current agent internals and message history.
    /// Returns the (potentially modified) messages to send to the LLM.
    ///
    /// # Arguments
    /// * `internals` - Read access to agent state, context, session, etc.
    /// * `messages` - The current message history
    ///
    /// # Returns
    /// The modified message history to send to the LLM
    fn inject(&self, internals: &AgentInternals, messages: Vec<Message>) -> Vec<Message>;
}

/// A context injection created from a closure
///
/// Use this for simple, stateless injections:
///
/// ```ignore
/// let injection = FnInjection::new("my_injection", |internals, messages| {
///     // modify messages
///     messages
/// });
/// ```
pub struct FnInjection<F>
where
    F: Fn(&AgentInternals, Vec<Message>) -> Vec<Message> + Send + Sync,
{
    name: String,
    func: F,
}

impl<F> FnInjection<F>
where
    F: Fn(&AgentInternals, Vec<Message>) -> Vec<Message> + Send + Sync,
{
    /// Create a new function-based injection
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

impl<F> ContextInjection for FnInjection<F>
where
    F: Fn(&AgentInternals, Vec<Message>) -> Vec<Message> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn inject(&self, internals: &AgentInternals, messages: Vec<Message>) -> Vec<Message> {
        (self.func)(internals, messages)
    }
}

/// Boxed context injection for storing in collections
pub type BoxedInjection = Box<dyn ContextInjection>;

/// Arc-wrapped injection for sharing across threads
pub type SharedInjection = std::sync::Arc<dyn ContextInjection>;

/// A chain of context injections that are applied in order
///
/// This collects multiple injections and applies them sequentially:
/// `messages = injection1(messages)`
/// `messages = injection2(messages)`
/// etc.
pub struct InjectionChain {
    injections: Vec<SharedInjection>,
}

impl InjectionChain {
    /// Create a new empty injection chain
    pub fn new() -> Self {
        Self {
            injections: Vec::new(),
        }
    }

    /// Add an injection to the chain
    pub fn add<I: ContextInjection + 'static>(&mut self, injection: I) {
        self.injections.push(std::sync::Arc::new(injection));
    }

    /// Add a shared injection to the chain
    pub fn add_shared(&mut self, injection: SharedInjection) {
        self.injections.push(injection);
    }

    /// Add a function-based injection to the chain
    pub fn add_fn<F>(&mut self, name: impl Into<String>, func: F)
    where
        F: Fn(&AgentInternals, Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    {
        self.add(FnInjection::new(name, func));
    }

    /// Apply all injections in order
    ///
    /// Each injection receives the output of the previous one.
    pub fn apply(&self, internals: &AgentInternals, mut messages: Vec<Message>) -> Vec<Message> {
        for injection in &self.injections {
            tracing::debug!("Applying context injection: {}", injection.name());
            messages = injection.inject(internals, messages);
        }
        messages
    }

    /// Get the number of injections in the chain
    pub fn len(&self) -> usize {
        self.injections.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.injections.is_empty()
    }

    /// Get the names of all injections in the chain
    pub fn names(&self) -> Vec<&str> {
        self.injections.iter().map(|i| i.name()).collect()
    }
}

impl Default for InjectionChain {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper functions for common injection patterns
// ============================================================================

/// Inject a system reminder into the last assistant or user message
///
/// System reminders use the format `<system-reminder>...</system-reminder>`
/// and are typically added to provide hints without being too intrusive.
///
/// # Example
/// ```ignore
/// inject_system_reminder(&mut messages, "Remember to update your todo list");
/// ```
pub fn inject_system_reminder(messages: &mut Vec<Message>, reminder: &str) {
    if let Some(last_msg) = messages.last_mut() {
        let reminder_text = format!("\n<system-reminder>\n{}\n</system-reminder>", reminder);
        last_msg.append_text(&reminder_text);
    }
}

/// Prepend text to the first user message
///
/// Useful for adding detailed instructions that should appear at the start.
///
/// # Example
/// ```ignore
/// prepend_to_first_user_message(&mut messages, "Context: You are working on project X\n\n");
/// ```
pub fn prepend_to_first_user_message(messages: &mut Vec<Message>, text: &str) {
    for msg in messages.iter_mut() {
        if msg.role.as_str() == "user" {
            msg.prepend_text(text);
            break;
        }
    }
}

/// Append text to the last message
///
/// # Example
/// ```ignore
/// append_to_last_message(&mut messages, "\n\nRemember: Be concise.");
/// ```
pub fn append_to_last_message(messages: &mut Vec<Message>, text: &str) {
    if let Some(last_msg) = messages.last_mut() {
        last_msg.append_text(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_injection_chain_empty() {
        let chain = InjectionChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_injection_chain_names() {
        let mut chain = InjectionChain::new();
        chain.add_fn("first", |_, m| m);
        chain.add_fn("second", |_, m| m);

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.names(), vec!["first", "second"]);
    }
}
