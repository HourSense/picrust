# Logging Guide

## Overview

The agent now has comprehensive logging that writes to the `logs/` folder. This helps debug issues and track what the agent is doing.

**Note**: Logs are written ONLY to files, not to the console, to keep the CLI interface clean.

## Log Locations

All logs are written to:
- **Directory**: `logs/`
- **File**: `agent.log` (rotates daily)
- **Old logs**: Named with date suffix (e.g., `agent.log.2026-01-14`)

## Log Levels

The system supports different verbosity levels:

| Level | What It Shows |
|-------|---------------|
| `error` | Only errors |
| `warn` | Errors + warnings |
| `info` | Errors + warnings + important events (default) |
| `debug` | All of above + detailed debugging info |
| `trace` | Everything, maximum verbosity |

## Running with Different Log Levels

### Default (INFO level)
```bash
cargo run
```

### Debug Level (Recommended for troubleshooting)
```bash
RUST_LOG=debug cargo run
```

### Trace Level (Maximum detail)
```bash
RUST_LOG=trace cargo run
```

### Specific Module Logging
```bash
# Only debug the LLM provider
RUST_LOG=shadow_agent_sdk::llm=debug cargo run

# Debug LLM and agent, info for everything else
RUST_LOG=shadow_agent_sdk::llm=debug,shadow_agent_sdk::agent=debug cargo run
```

## Viewing Logs

### View the latest log file
```bash
cat logs/agent.log
```

### Follow logs in real-time
```bash
tail -f logs/agent.log
```

### Search logs for errors
```bash
grep ERROR logs/agent.log
```

### Search logs for a specific message
```bash
grep "Anthropic API" logs/agent.log
```

### View logs from a specific date
```bash
cat logs/agent.log.2026-01-14
```

## What Gets Logged

### INFO Level (Default)
- Agent startup/shutdown
- User input received
- Message processing start/end
- LLM provider creation
- API calls to Anthropic
- Response received

### DEBUG Level
- Full user messages
- System prompts
- Model and parameter details
- API request details
- Response content length

### ERROR Level
- API failures
- Network errors
- Input/output errors
- Processing failures

## Log Format

Logs include:
- **Timestamp**: When the event occurred
- **Level**: ERROR, WARN, INFO, DEBUG, or TRACE
- **Target**: Which module logged it
- **Thread ID**: Which thread (for async debugging)
- **Line Number**: Source code location
- **Message**: The log message

Example:
```
2026-01-14T10:30:45.123456Z  INFO shadow_agent_sdk::agent: Starting agent loop
2026-01-14T10:30:50.234567Z DEBUG shadow_agent_sdk::llm: User message: hi
2026-01-14T10:30:50.345678Z DEBUG shadow_agent_sdk::llm: Calling Anthropic API...
2026-01-14T10:30:51.456789Z ERROR shadow_agent_sdk::llm: Anthropic API error: ...
```

## Debugging Your Error

When you see an error like:
```
Error processing message: Failed to send message
```

Run with debug logging:
```bash
RUST_LOG=debug cargo run
```

Then check `logs/agent.log` for:
1. The full error message and stack trace
2. What parameters were sent to the API
3. What the API response was
4. Network or configuration issues

## Common Issues and Log Signatures

### Missing API Key
```
ERROR Failed to create Anthropic client. Make sure ANTHROPIC_API_KEY is set
```
**Solution**: Set your API key in `.env` file

### Invalid API Key
```
ERROR Anthropic API error: 401 Unauthorized
```
**Solution**: Check your API key is correct

### Network Issues
```
ERROR Network error: Connection timeout
```
**Solution**: Check internet connection

### Rate Limiting
```
ERROR Anthropic API error: 429 Too Many Requests
```
**Solution**: Wait and retry, or use a different API key

## Log Rotation

Logs rotate daily automatically:
- Old logs are renamed with date suffix
- New logs start fresh each day
- Prevents log files from growing too large

## Enabling Console Logging (Optional)

By default, logs go ONLY to files to keep the CLI clean. If you want to also see logs in the console (for development), edit `src/logging.rs`:

```rust
// Add this back before .init():
let stdout_layer = fmt::layer()
    .with_writer(std::io::stdout)
    .with_target(false)
    .with_ansi(true);

// Then add it to the registry:
tracing_subscriber::registry()
    .with(env_filter)
    .with(file_layer)
    .with(stdout_layer)  // Add this line
    .init();
```

Then rebuild:
```bash
cargo build
```

## Performance Impact

Logging has minimal performance impact:
- INFO level: ~1-2% overhead
- DEBUG level: ~3-5% overhead
- TRACE level: ~5-10% overhead

For production use, stick with INFO or WARN level.
