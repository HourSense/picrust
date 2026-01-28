# MCP Agent Example

This example demonstrates how to integrate MCP (Model Context Protocol) servers with the Shadow Agent Framework.

## Prerequisites

1. **MCP Server**: You need a running MCP server. This example connects to `http://localhost:8005/mcp`
2. **Environment Variable**: Set your Anthropic API key:
   ```bash
   export ANTHROPIC_KEY="your-api-key-here"
   ```

## Running the Example

```bash
# Run with default settings (prompt caching enabled)
cargo run --example mcp_agent

# Run with streaming enabled
cargo run --example mcp_agent -- --stream

# Run with extended thinking
cargo run --example mcp_agent -- --think

# Resume an existing session
cargo run --example mcp_agent -- --resume

# Disable prompt caching
cargo run --example mcp_agent -- --no-cache
```

## Features

- **MCP Integration**: Connects to an MCP server and exposes its tools to the agent
- **Tool Namespacing**: MCP tools are namespaced as `server_id__tool_name` (e.g., `filesystem__read_file`)
- **Auto-Approval**: All MCP tools are auto-approved by hooks (for demo purposes)
- **Prompt Caching**: Enabled by default for 90% cost savings
- **Debug Logging**: Full visibility into agent operations
- **Session Persistence**: Conversations are saved in `./sessions/`

## How It Works

1. Creates an MCP transport to connect to `http://localhost:8005/mcp`
2. Serves the transport to create a running service
3. Adds the service to `MCPServerManager` with ID `"filesystem"`
4. Creates an `MCPToolProvider` to expose MCP tools
5. Adds the provider to the tool registry
6. All tools from the MCP server are now available to the agent with `filesystem__` prefix

## Customization

To add authentication headers or customize the transport:

```rust
let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp")
    .with_header("Authorization", "Bearer your-token")
    .with_header("X-Custom", "value");

let service = ().serve(transport).await?;
mcp_manager.add_service("filesystem", service).await?;
```

## Tool Namespacing

MCP tools are automatically namespaced to avoid conflicts:
- **Original tool**: `read_file`
- **Exposed to agent**: `filesystem__read_file`
- **Pattern**: `{server_id}__{tool_name}`

The double underscore is used instead of colon because Anthropic's API only allows `[a-zA-Z0-9_-]` in tool names.
