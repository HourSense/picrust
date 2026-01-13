use crate::cli::Console;
use crate::llm::AnthropicProvider;
use anyhow::Result;

/// Main agent that orchestrates the conversation loop
pub struct Agent {
    console: Console,
    llm_provider: AnthropicProvider,
    system_prompt: Option<String>,
}

impl Agent {
    /// Create a new Agent with a console and LLM provider
    pub fn new(console: Console, llm_provider: AnthropicProvider) -> Self {
        tracing::info!("Creating new Agent");
        Self {
            console,
            llm_provider,
            system_prompt: None,
        }
    }

    /// Set a system prompt for the agent
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Get a reference to the console
    pub fn console(&self) -> &Console {
        &self.console
    }

    /// Run the main agent loop
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Starting agent loop");
        self.console.print_banner();

        loop {
            // Read user input
            let user_input = match self.console.read_input() {
                Ok(input) => {
                    tracing::debug!("User input received: {}", input);
                    input
                }
                Err(e) => {
                    tracing::error!("Failed to read user input: {}", e);
                    self.console.print_error(&format!("Failed to read input: {}", e));
                    continue;
                }
            };

            // Check for exit commands
            if user_input.to_lowercase() == "exit" || user_input.to_lowercase() == "quit" {
                tracing::info!("User requested exit");
                self.console.print_system("Goodbye!");
                break;
            }

            // Skip empty input
            if user_input.trim().is_empty() {
                tracing::debug!("Empty input, skipping");
                continue;
            }

            // Print separator for readability
            self.console.println();

            // Process the message
            tracing::info!("Processing user message");
            if let Err(e) = self.process_message(&user_input).await {
                tracing::error!("Error processing message: {:?}", e);
                self.console.print_error(&format!("Error processing message: {}", e));
            }

            // Print separator after response
            self.console.println();
            self.console.print_separator();
        }

        tracing::info!("Agent loop ended");
        Ok(())
    }

    /// Process a single user message
    async fn process_message(&self, user_message: &str) -> Result<()> {
        tracing::debug!("Processing message: {}", user_message);

        // Get the complete response (TODO: add streaming support later)
        let response = self
            .llm_provider
            .send_message(user_message, self.system_prompt.as_deref())
            .await
            .map_err(|e| {
                tracing::error!("Failed to get LLM response: {:?}", e);
                e
            })?;

        tracing::debug!("Response received, length: {} chars", response.len());

        // Print the response
        self.console.print_assistant(&response);

        tracing::info!("Message processing complete");

        Ok(())
    }
}
