// Tests for OpenAI provider
//
// Note: These tests require a valid OPENAI_API_KEY environment variable.
// Run with: cargo test --test openai_provider_test -- --nocapture

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use picrust::llm::{define_tool, Message, OpenAiProvider, ToolChoice};
    use serde_json::json;

    fn setup_provider() -> Result<OpenAiProvider> {
        // Use gpt-4o-mini for faster, cheaper tests
        OpenAiProvider::new(
            std::env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY must be set for tests"),
        )?
        .with_model("gpt-4o-mini")
        .with_max_tokens(1000)
        .into()
    }

    #[tokio::test]
    async fn test_simple_message() -> Result<()> {
        let provider = setup_provider()?;

        let response = provider
            .send_message(
                "Say exactly: 'test passed'",
                &[],
                None,
                None,
            )
            .await?;

        assert!(response.to_lowercase().contains("test passed"));
        Ok(())
    }

    #[tokio::test]
    async fn test_with_system_prompt() -> Result<()> {
        let provider = setup_provider()?;

        let response = provider
            .send_message(
                "What is your role?",
                &[],
                Some("You are a helpful math tutor."),
                None,
            )
            .await?;

        // Response should mention math or tutor
        assert!(
            response.to_lowercase().contains("math") 
            || response.to_lowercase().contains("tutor")
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_conversation_history() -> Result<()> {
        let provider = setup_provider()?;

        let history = vec![
            Message::user("My name is Alice."),
            Message::assistant("Nice to meet you, Alice!"),
        ];

        let response = provider
            .send_message(
                "What's my name?",
                &history,
                None,
                None,
            )
            .await?;

        assert!(response.to_lowercase().contains("alice"));
        Ok(())
    }

    #[tokio::test]
    async fn test_tool_calling() -> Result<()> {
        let provider = setup_provider()?;

        let tools = vec![define_tool(
            "get_weather",
            "Get current weather for a location",
            json!({
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            }),
            vec!["location".to_string()],
        )];

        let messages = vec![Message::user("What's the weather in Paris?")];

        let response = provider
            .send_with_tools_and_system(
                messages,
                None,
                tools,
                Some(ToolChoice::Auto {
                    disable_parallel_tool_use: None,
                }),
                None,
                None,
            )
            .await?;

        // Should contain a tool use
        let has_tool_use = response
            .content
            .iter()
            .any(|block| matches!(block, picrust::llm::ContentBlock::ToolUse { .. }));

        assert!(has_tool_use, "Expected tool use in response");

        // Extract tool name
        for block in &response.content {
            if let picrust::llm::ContentBlock::ToolUse { name, .. } = block {
                assert_eq!(name, "get_weather");
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_streaming() -> Result<()> {
        use futures::StreamExt;
        use picrust::llm::{ContentDelta, StreamEvent};

        let provider = setup_provider()?;
        let messages = vec![Message::user("Count to 3")];

        let mut stream = provider
            .stream_with_tools_and_system(
                messages,
                None,
                vec![],
                None,
                None,
                None,
            )
            .await?;

        let mut received_text = String::new();
        let mut message_started = false;
        let mut message_stopped = false;

        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::MessageStart(_) => {
                    message_started = true;
                }
                StreamEvent::ContentBlockDelta(delta) => {
                    if let ContentDelta::TextDelta { text } = delta.delta {
                        received_text.push_str(&text);
                    }
                }
                StreamEvent::MessageStop => {
                    message_stopped = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(message_started, "MessageStart event not received");
        assert!(message_stopped, "MessageStop event not received");
        assert!(!received_text.is_empty(), "No text received in stream");

        Ok(())
    }

    #[tokio::test]
    async fn test_provider_name() -> Result<()> {
        use picrust::llm::LlmProvider;

        let provider = setup_provider()?;
        assert_eq!(provider.provider_name(), "openai");
        Ok(())
    }

    #[tokio::test]
    async fn test_model_name() -> Result<()> {
        use picrust::llm::LlmProvider;

        let provider = setup_provider()?;
        assert_eq!(provider.model(), "gpt-4o-mini");
        Ok(())
    }

    #[tokio::test]
    async fn test_create_variant() -> Result<()> {
        use picrust::llm::LlmProvider;

        let provider = setup_provider()?;
        let variant = provider.create_variant("gpt-3.5-turbo", 500);

        assert_eq!(variant.model(), "gpt-3.5-turbo");
        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling_invalid_key() -> Result<()> {
        let provider = OpenAiProvider::new("invalid-key")?
            .with_model("gpt-4o-mini");

        let result = provider
            .send_message("test", &[], None, None)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("401") || err.to_string().contains("Incorrect API key"));

        Ok(())
    }

    #[tokio::test]
    async fn test_from_env() -> Result<()> {
        // Set test environment variables
        std::env::set_var("OPENAI_API_KEY", "sk-test123");
        std::env::set_var("OPENAI_MODEL", "gpt-4o");
        std::env::set_var("OPENAI_MAX_TOKENS", "2000");

        let provider = OpenAiProvider::from_env()?;

        use picrust::llm::LlmProvider;
        assert_eq!(provider.model(), "gpt-4o");

        // Clean up
        std::env::remove_var("OPENAI_MODEL");
        std::env::remove_var("OPENAI_MAX_TOKENS");

        Ok(())
    }
}
