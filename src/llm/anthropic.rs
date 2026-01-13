use anyhow::{Context, Result};
use anthropic_sdk::{Anthropic, MessageCreateBuilder};

/// Anthropic LLM provider wrapper
pub struct AnthropicProvider {
    client: Anthropic,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from environment variables
    pub fn from_env() -> Result<Self> {
        tracing::info!("Creating Anthropic provider from environment");

        let client = Anthropic::from_env()
            .context("Failed to create Anthropic client. Make sure ANTHROPIC_API_KEY is set")?;

        tracing::info!("Anthropic client created successfully");
        tracing::info!("Using model: claude-haiku-4-5");
        tracing::info!("Max tokens: 4096");

        Ok(Self {
            client,
            model: "claude-haiku-4-5".to_string(),
            max_tokens: 4096,
        })
    }

    /// Create a new Anthropic provider with custom API key
    pub fn new(api_key: &str) -> Result<Self> {
        let client = Anthropic::new(api_key)
            .context("Failed to create Anthropic client")?;

        Ok(Self {
            client,
            model: "claude-haiku-4-5".to_string(),
            max_tokens: 4096,
        })
    }

    /// Set the model to use
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the max tokens for responses
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    // TODO: Implement streaming once we figure out the correct SDK API
    // For now, we'll use send_message which gets the complete response

    /// Send a message and get the complete response
    pub async fn send_message(
        &self,
        user_message: &str,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        tracing::info!("Sending message to Anthropic API");
        tracing::debug!("User message: {}", user_message);
        tracing::debug!("System prompt: {:?}", system_prompt);
        tracing::debug!("Model: {}", self.model);
        tracing::debug!("Max tokens: {}", self.max_tokens);

        let mut builder = MessageCreateBuilder::new(&self.model, self.max_tokens)
            .user(user_message);

        if let Some(system) = system_prompt {
            builder = builder.system(system);
        }

        tracing::debug!("Calling Anthropic API...");

        let response = self
            .client
            .messages()
            .create(builder.build())
            .await
            .map_err(|e| {
                tracing::error!("Anthropic API error: {:?}", e);
                e
            })
            .context("Failed to send message")?;

        tracing::info!("Received response from Anthropic API");
        tracing::debug!("Response ID: {}", response.id);
        tracing::debug!("Content blocks: {}", response.content.len());

        // Extract text from content blocks
        let text = response
            .content
            .iter()
            .filter_map(|block| {
                if let anthropic_sdk::ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        tracing::info!("Extracted text response, length: {} chars", text.len());

        Ok(text)
    }
}
