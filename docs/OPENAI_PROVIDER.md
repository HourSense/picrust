# OpenAI Provider for Picrust

This document describes the OpenAI LLM provider implementation for the Picrust framework.

## Overview

The `OpenAiProvider` integrates OpenAI's Chat Completions API with Picrust's SwappableLlmProvider architecture. It translates between:
- **Internal format**: Anthropic-style messages (used throughout Picrust)
- **Wire format**: OpenAI API format (used for API requests/responses)

This allows seamless switching between OpenAI, Anthropic, and Gemini providers at runtime.

## Features

- ✅ Full support for OpenAI Chat Completions API
- ✅ Tool/function calling support
- ✅ Streaming responses with SSE
- ✅ Static and dynamic authentication
- ✅ Custom base URL support (for proxies/Azure OpenAI)
- ✅ Message format translation (Anthropic ↔ OpenAI)
- ✅ SwappableLlmProvider integration

## Quick Start

### Basic Usage

```rust
use picrust::llm::OpenAiProvider;

// From environment variables
let provider = OpenAiProvider::from_env()?;

// With explicit API key
let provider = OpenAiProvider::new("sk-...")? 
    .with_model("gpt-4o")
    .with_max_tokens(4096);
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"

# Optional
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Default
export OPENAI_MAX_TOKENS="4096"                      # Default
```

### Supported Models

- **GPT-4 Turbo**: `gpt-4-turbo`, `gpt-4-turbo-preview`
- **GPT-4o**: `gpt-4o`, `gpt-4o-mini`
- **GPT-4**: `gpt-4`, `gpt-4-32k`
- **GPT-3.5 Turbo**: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`

## Message Format Translation

### Internal Format (Anthropic-style)

Picrust uses Anthropic's message format internally:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Hello"},
    {"type": "tool_use", "id": "tool_1", "name": "search", "input": {...}}
  ]
}
```

### OpenAI Format (Wire)

The provider translates to OpenAI's format:

```json
{
  "role": "user",
  "content": "Hello",
  "tool_calls": [
    {
      "id": "tool_1",
      "type": "function",
      "function": {"name": "search", "arguments": "{...}"}
    }
  ]
}
```

### Key Differences Handled

| Feature | Anthropic (Internal) | OpenAI (Wire) | Translation |
|---------|---------------------|---------------|-------------|
| System prompt | Separate field | First message with role="system" | Prepended as first message |
| Tool use | ContentBlock in assistant message | `tool_calls` array | Mapped to tool_calls |
| Tool results | ContentBlock in user message | Separate message with role="tool" | Separate tool messages |
| Thinking blocks | Native ContentBlock | Not supported | Formatted as text with marker |
| Images | ContentBlock with base64 | Content parts with data URLs | Converted to data URLs |

## Tool Calling

### Defining Tools

```rust
use picrust::llm::define_tool;
use serde_json::json;

let tools = vec![
    define_tool(
        "search",
        "Search the web for information",
        json!({
            "query": {
                "type": "string",
                "description": "Search query"
            }
        }),
        vec!["query".to_string()],
    ),
];
```

### Using Tools

```rust
use picrust::llm::{Message, ToolChoice};

let messages = vec![
    Message::user("What's the weather in Tokyo?")
];

let response = provider.send_with_tools_and_system(
    messages,
    None, // system prompt
    tools,
    Some(ToolChoice::Auto { disable_parallel_tool_use: None }),
    None, // thinking config
    None, // session_id
).await?;

// Check for tool calls
for block in &response.content {
    if let ContentBlock::ToolUse { name, input, .. } = block {
        println!("Tool: {}, Input: {}", name, input);
    }
}
```

## Streaming

```rust
use futures::StreamExt;

let mut stream = provider.stream_with_tools_and_system(
    messages,
    None,
    tools,
    None,
    None,
    None,
).await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::ContentBlockDelta(delta) => {
            match delta.delta {
                ContentDelta::TextDelta { text } => print!("{}", text),
                ContentDelta::InputJsonDelta { partial_json } => {
                    println!("Tool input: {}", partial_json);
                }
                _ => {}
            }
        }
        StreamEvent::MessageStop => break,
        _ => {}
    }
}
```

## Advanced Usage

### Dynamic Authentication

For rotating API keys or proxy servers:

```rust
use picrust::llm::{OpenAiProvider, AuthConfig};

let provider = OpenAiProvider::with_auth_provider(|| async {
    // Fetch fresh credentials
    let token = my_auth_service.get_openai_token().await?;
    Ok(AuthConfig::new(token))
});
```

### Custom Base URL

For Azure OpenAI or proxies:

```rust
// Via environment
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/gpt-4"

// Or programmatically
let provider = OpenAiProvider::new("sk-...")?
    .with_model("gpt-4");
// Note: Set base_url via AuthConfig for dynamic providers
```

### SwappableLlmProvider

Runtime model switching:

```rust
use picrust::llm::{OpenAiProvider, SwappableLlmProvider};
use std::sync::Arc;

// Start with GPT-4o-mini
let fast = Arc::new(
    OpenAiProvider::new("sk-...")?.with_model("gpt-4o-mini")
);
let swappable = SwappableLlmProvider::new(fast);
let handle = swappable.handle();

// Use in agent
let llm: Arc<dyn LlmProvider> = Arc::new(swappable);

// Later, switch to GPT-4o
let pro = Arc::new(
    OpenAiProvider::new("sk-...")?.with_model("gpt-4o")
);
handle.set_provider(pro).await;
```

## Error Handling

```rust
match provider.send_message("Hello", &[], None, None).await {
    Ok(response) => println!("Response: {}", response),
    Err(e) => {
        // API errors include status code and body
        eprintln!("OpenAI API error: {}", e);
    }
}
```

Common errors:
- `OPENAI_API_KEY environment variable not set` - Missing API key
- `OpenAI API error (401)` - Invalid API key
- `OpenAI API error (429)` - Rate limit exceeded
- `OpenAI API error (500)` - Server error

## Limitations

1. **Thinking Blocks**: OpenAI doesn't support native thinking blocks like Anthropic. Internal reasoning is formatted as text with markers `[Internal reasoning: ...]`.

2. **Extended Thinking**: The `ThinkingConfig` parameter is accepted but ignored since OpenAI doesn't have an equivalent feature.

3. **Document Blocks**: Not directly supported by OpenAI. These are skipped during translation.

4. **Prompt Caching**: OpenAI doesn't support prompt caching like Anthropic. Cache control directives are ignored.

## Testing

```rust
#[tokio::test]
async fn test_openai_basic() -> Result<()> {
    let provider = OpenAiProvider::new("sk-test")?
        .with_model("gpt-4o-mini");
    
    let response = provider.send_message(
        "Say 'test successful'",
        &[],
        None,
        None,
    ).await?;
    
    assert!(response.contains("test successful"));
    Ok(())
}
```

## Implementation Details

### Architecture

```
┌─────────────────────────────────────────┐
│         StandardAgent                    │
│  (Uses Anthropic-style messages)         │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      LlmProvider Trait                   │
│  (Anthropic-style interface)             │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      OpenAiProvider                      │
│  ┌─────────────────────────────────┐    │
│  │ Format Translation Layer         │    │
│  │ - Messages: Anthropic → OpenAI   │    │
│  │ - Tools: Custom → Functions      │    │
│  │ - Responses: OpenAI → Anthropic  │    │
│  └─────────────────────────────────┘    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│   OpenAI Chat Completions API            │
│   (OpenAI wire format)                   │
└─────────────────────────────────────────┘
```

### File Structure

- `src/llm/openai.rs` - Main provider implementation
- `src/llm/mod.rs` - Module exports
- `src/llm/swappable.rs` - Updated to support OpenAI

## Comparison with Other Providers

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| Tool calling | ✅ Functions | ✅ Tools | ✅ Functions |
| Streaming | ✅ SSE | ✅ SSE | ✅ SSE |
| System prompt | First message | Separate field | Separate field |
| Thinking | ❌ | ✅ Native | ✅ Native |
| Images | ✅ Data URLs | ✅ Base64 | ✅ Inline data |
| Prompt caching | ❌ | ✅ | ❌ |
| Max tokens | Model-dependent | 200K+ | 2M+ |

## Migration Guide

### From Anthropic to OpenAI

```rust
// Before (Anthropic)
let provider = AnthropicProvider::from_env()?;

// After (OpenAI)
let provider = OpenAiProvider::from_env()?;

// That's it! The rest of your code remains the same.
```

### From Gemini to OpenAI

```rust
// Before (Gemini)
let provider = GeminiProvider::from_env()?;

// After (OpenAI)
let provider = OpenAiProvider::from_env()?;

// Update environment variables:
// GEMINI_API_KEY → OPENAI_API_KEY
// GEMINI_MODEL → OPENAI_MODEL
```

## Contributing

When extending the OpenAI provider:

1. **Message Translation**: Update `convert_messages()` and `convert_openai_message_to_blocks()`
2. **New Features**: Check OpenAI API docs first, then map to internal types
3. **Testing**: Add tests for both sync and streaming modes
4. **Documentation**: Update this file with examples

## References

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic Message Format](https://docs.anthropic.com/claude/reference/messages)
- [Picrust SwappableLlmProvider](../src/llm/swappable.rs)
