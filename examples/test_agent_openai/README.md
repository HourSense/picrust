# Test Agent Example

This example demonstrates the StandardAgent framework with all features including prompt caching.

## Features Demonstrated

- **StandardAgent**: Standardized agent loop with configuration
- **Tool Registry**: Custom tools (Read, Write, Bash, etc.)
- **Hooks**: Safety hooks and auto-approval patterns
- **Context Injections**: Dynamic message modification
- **TodoListManager**: Task tracking
- **Streaming**: Real-time response streaming
- **Extended Thinking**: Deep reasoning for complex tasks
- **Prompt Caching**: Automatic cost optimization (NEW!)

## Running the Example

### Basic Usage

```bash
# New session with default settings (caching enabled)
cargo run --example test_agent

# Resume an existing session
cargo run --example test_agent -- --resume
```

### Advanced Options

```bash
# Enable streaming responses
cargo run --example test_agent -- --stream

# Enable extended thinking (16k token budget)
cargo run --example test_agent -- --think

# Disable prompt caching (for comparison)
cargo run --example test_agent -- --no-cache

# Combine options
cargo run --example test_agent -- --stream --think --resume
```

## Prompt Caching

**Prompt caching is enabled by default** to provide automatic cost savings and improved latency.

### What Gets Cached?

The agent automatically caches:
1. **Tool definitions** - Static tool schemas
2. **System prompt** - Your agent's instructions
3. **Conversation history** - Previous turns

### Cost Savings

Example 3-turn conversation:
- **Without caching**: ~21,000 tokens
- **With caching**: ~11,450 tokens
- **Savings: 46%** (increases with longer conversations!)

### Cache Behavior

- **Cache lifetime**: 5 minutes (refreshed on each use)
- **Cache discount**: 90% off regular input token price
- **Cache write cost**: 25% premium on first use
- **Automatic**: No manual intervention needed

### Viewing Cache Metrics

When debug mode is enabled (always on in this example), you'll see cache metrics in the logs:

```
Cache creation tokens: 5000
Cache read tokens: 12000
```

These show:
- **Cache creation**: Tokens being written to cache (first turn)
- **Cache read**: Tokens being read from cache (subsequent turns)

### Disabling Caching

To compare performance with/without caching:

```bash
# Without caching
cargo run --example test_agent -- --no-cache

# With caching (default)
cargo run --example test_agent
```

## Example Interaction

```
=== Test Agent (StandardAgent) ===
This agent uses the standardized agent framework.
Read operations are pre-allowed. Others will require permission.
Use --stream/-s flag to enable streaming responses.
Use --think/-t flag to enable extended thinking.
Prompt caching is enabled by default (use --no-cache to disable).

[Setup] Creating LLM provider with dynamic auth...
[Setup] Model: claude-sonnet-4-5-20250929 (using dynamic auth)
[Setup] Runtime created (Read tool globally allowed)
[Setup] Tools registered: ["Read", "Write", "Bash", "TodoWrite"]
[Setup] TodoListManager created
[Setup] Hooks configured: dangerous command blocker, read-only auto-approve
[Setup] New session: test-agent-session
[Setup] AgentConfig created with debug logging, hooks, prompt caching enabled and todo reminder injection
[Setup] Spawning agent...
[Setup] Agent spawned!
[Setup] Starting console renderer...

Type your requests below. Read/Glob/Grep are auto-approved by hooks.
💰 Prompt caching enabled: 90% cost savings on repeated content!
   (Tools, system prompt, and conversation history are automatically cached)
Type 'exit' or 'quit' to stop.

You: Write a hello world program in Python to hello.py
```

## Environment Variables

Required:
- `ANTHROPIC_KEY`: Your Anthropic API key

Optional:
- `ANTHROPIC_MODEL`: Model to use (default: claude-sonnet-4-5-20250929)

## Command Line Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--resume` | `-r` | Resume existing session |
| `--stream` | `-s` | Enable streaming responses |
| `--think` | `-t` | Enable extended thinking (16k budget) |
| `--no-cache` | - | Disable prompt caching |

## Tips

1. **Multi-turn conversations**: Let caching run for several turns to see maximum savings
2. **Check logs**: Debug mode shows cache metrics for each API call
3. **Compare costs**: Try with and without `--no-cache` to see the difference
4. **Streaming + Caching**: Works seamlessly together for best experience

## Understanding Cache Metrics

In debug logs, you'll see:

```
Usage: 150 input, 200 output tokens
Cache creation tokens: 5000
Cache read tokens: 0
```

This means:
- **First turn**: 5000 tokens written to cache + 150 new tokens = 5150 total input
- **Cost**: (5000 × 1.25) + (150 × 1.0) = 6400 token cost

On the second turn:

```
Usage: 200 input, 250 output tokens
Cache creation tokens: 200
Cache read tokens: 5000
```

This means:
- **Second turn**: 5000 tokens from cache + 200 new tokens cached + 200 new input
- **Cost**: (5000 × 0.10) + (200 × 1.25) + (200 × 1.0) = 950 token cost

**Savings**: 6400 vs 950 = 85% reduction on second turn!

## More Information

See the full documentation:
- [Prompt Caching Guide](../../docs/PROMPT_CACHING.md)
- [Implementation Details](../../docs/implementation_summary_prompt_caching.md)
