//! Context manager for handling system prompts and hidden context
//!
//! The ContextManager aggregates context from multiple providers
//! and injects it into the conversation as hidden prompts.

use anyhow::Result;
use std::sync::Arc;

use super::providers::ContextProvider;
use crate::llm::Message;

/// Manages context injection for the agent
pub struct ContextManager {
    /// The base system prompt
    system_prompt: String,
    /// Context providers for dynamic context
    providers: Vec<Arc<dyn ContextProvider>>,
    /// Static hidden prompts (always included)
    static_prompts: Vec<String>,
}

impl ContextManager {
    /// Create a new context manager with a system prompt
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            providers: Vec::new(),
            static_prompts: Vec::new(),
        }
    }

    /// Add a context provider
    pub fn add_provider<P: ContextProvider + 'static>(&mut self, provider: P) {
        self.providers.push(Arc::new(provider));
    }

    /// Add a static hidden prompt
    pub fn add_static_prompt(&mut self, prompt: impl Into<String>) {
        self.static_prompts.push(prompt.into());
    }

    /// Get the base system prompt
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Set the system prompt
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Build the full system prompt with all context
    ///
    /// This combines the base system prompt with static prompts
    /// and dynamic context from providers.
    pub async fn build_full_system_prompt(&self, messages: &[Message]) -> Result<String> {
        let mut parts = vec![self.system_prompt.clone()];

        // Add static prompts
        for prompt in &self.static_prompts {
            parts.push(prompt.clone());
        }

        // Get dynamic context from providers
        for provider in &self.providers {
            if let Some(context) = provider.get_context(messages).await? {
                parts.push(format!(
                    "\n<context name=\"{}\">\n{}\n</context>",
                    provider.name(),
                    context
                ));
            }
        }

        Ok(parts.join("\n\n"))
    }

    /// Inject hidden context as a user message
    ///
    /// This creates a user message containing hidden context that
    /// will be prepended to the conversation. Useful for providing
    /// context that should appear as part of the conversation rather
    /// than the system prompt.
    pub async fn create_hidden_context_message(&self, messages: &[Message]) -> Result<Option<Message>> {
        let mut context_parts = Vec::new();

        // Gather context from providers that want to inject as messages
        for provider in &self.providers {
            if provider.inject_as_message() {
                if let Some(context) = provider.get_context(messages).await? {
                    context_parts.push(format!(
                        "<hidden-context name=\"{}\">\n{}\n</hidden-context>",
                        provider.name(),
                        context
                    ));
                }
            }
        }

        if context_parts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Message::user(context_parts.join("\n\n"))))
        }
    }

    /// Clear all providers and static prompts
    pub fn clear(&mut self) {
        self.providers.clear();
        self.static_prompts.clear();
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_system_prompt() {
        let manager = ContextManager::new("You are a helpful assistant.");
        let prompt = manager.build_full_system_prompt(&[]).await.unwrap();
        assert!(prompt.contains("You are a helpful assistant."));
    }

    #[tokio::test]
    async fn test_static_prompts() {
        let mut manager = ContextManager::new("Base prompt");
        manager.add_static_prompt("Additional context");

        let prompt = manager.build_full_system_prompt(&[]).await.unwrap();
        assert!(prompt.contains("Base prompt"));
        assert!(prompt.contains("Additional context"));
    }
}
