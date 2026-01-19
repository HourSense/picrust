# Coding Agent Architecture

A Rust-based coding assistant with tool calling, permission management, and conversation persistence.

## Project Structure

```
shadow-agent-sdk/
├── Cargo.toml                    # Dependencies and project config
├── Cargo.lock                    # Locked dependency versions
├── .gitignore                    # Git ignore rules
├── .env.example                  # Environment variable template
├── CLAUDE.md                     # Instructions for Claude
│
├── src/
│   ├── lib.rs                    # Library exports
│   ├── main.rs                   # Entry point
│   │
│   ├── agent/                    # Agent orchestration
│   │   ├── mod.rs
│   │   ├── agent_loop.rs         # Main agent loop with tool calling
│   │   └── system_prompt.rs      # Default system prompt
│   │
│   ├── cli/                      # Terminal I/O
│   │   ├── mod.rs
│   │   └── console.rs            # Colored output, input, permissions
│   │
│   ├── context/                  # Context injection
│   │   ├── mod.rs
│   │   ├── manager.rs            # ContextManager
│   │   └── providers.rs          # FileStructure, GitStatus providers
│   │
│   ├── conversation/             # Conversation persistence
│   │   ├── mod.rs
│   │   ├── conversation.rs       # Conversation storage
│   │   └── message.rs            # Re-exports from llm::types
│   │
│   ├── llm/                      # LLM integration
│   │   ├── mod.rs
│   │   ├── anthropic.rs          # HTTP client for Anthropic API
│   │   └── types.rs              # Message, ContentBlock, Tool types
│   │
│   ├── logging.rs                # Tracing/logging setup
│   │
│   ├── debugger/                 # Debug logging
│   │   └── mod.rs                # Request/response/tool logging
│   │
│   ├── permissions/              # Permission system
│   │   ├── mod.rs
│   │   └── manager.rs            # PermissionManager
│   │
│   └── tools/                    # Tool implementations
│       ├── mod.rs
│       ├── tool.rs               # Tool trait
│       ├── registry.rs           # ToolRegistry
│       ├── bash.rs               # Bash command execution
│       ├── read_tool.rs          # Read file contents
│       ├── edit_tool.rs          # Edit files (str_replace)
│       ├── write_tool.rs         # Write/create files
│       ├── glob_tool.rs          # File pattern matching
│       ├── grep_tool.rs          # Content search (ripgrep)
│       └── todo.rs               # TodoWrite task tracking
│
├── conversations/                # Conversation storage (gitignored)
│   └── <uuid>/
│       ├── metadata.json
│       └── history.jsonl
│
├── logs/                         # Log files (gitignored)
│   └── agent.log
│
├── debugger/                     # Debug session logs (gitignored)
│   └── <YYYYMMDD_HHMMSS>/        # Session folder
│       ├── events.jsonl          # All events in sequence
│       ├── 000001_api_request.json
│       ├── 000002_api_response.json
│       ├── 000003_tool_call.json
│       └── 000004_tool_result.json
│
└── docs/                         # Documentation
    ├── architecture.md           # This file
    ├── anthropic-curl-sdk-doc.md # API reference
    ├── conversation-system.md    # Conversation storage details
    ├── logging-guide.md          # Logging configuration
    └── session-notes.md          # Development session history
```

## Core Components

### 1. Agent Loop (`src/agent/agent_loop.rs`)

The agent operates in two nested loops:

**Outer Loop (User Conversation)**
```
User Input → Process Turn → Display Output → Repeat
```

**Inner Loop (Tool Execution)**
```
Send to LLM → Check for tool_use → Ask Permission → Execute Tool →
Add tool_result → Send back to LLM → Repeat until end_turn
```

```rust
pub struct Agent {
    console: Console,           // Terminal I/O
    llm_provider: AnthropicProvider,  // API client
    conversation: Conversation, // Message storage
    tool_registry: ToolRegistry, // Available tools
    permission_manager: PermissionManager, // Permission tracking
    context_manager: ContextManager, // System prompt + context
}
```

### 2. LLM Integration (`src/llm/`)

**HTTP Client (`anthropic.rs`)**
- Direct HTTP calls to `https://api.anthropic.com/v1/messages`
- Headers: `x-api-key`, `anthropic-version: 2023-06-01`
- Default model: `claude-sonnet-4-5-20250929`
- Default max_tokens: 16000 (increased for extended thinking)
- Methods:
  - `send_message()`: Simple text conversations
  - `send_with_tools()`: Full tool calling support with optional extended thinking

**Extended Thinking**
- Enabled with 10,000 budget tokens per request
- Thinking blocks are displayed to the user in a formatted box
- Configuration: `ThinkingConfig::enabled(10000)`
- Thinking content appears before the assistant's main response

**Type System (`types.rs`)**
- Matches Anthropic API exactly for correct serialization

```rust
// Message with flexible content
pub struct Message {
    pub role: String,  // "user" or "assistant"
    pub content: MessageContent,  // String or Vec<ContentBlock>
}

// Content block types
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: Option<String>, is_error: Option<bool> },
    Thinking { thinking: String, signature: String },
    RedactedThinking { data: String },
}

// Tool definition with JSON schema
pub struct CustomTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: ToolInputSchema,
}
```

### 3. Tool System (`src/tools/`)

**Tool Trait**
```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn definition(&self) -> ToolDefinition;
    fn get_info(&self, input: &Value) -> ToolInfo;
    async fn execute(&self, input: &Value) -> Result<ToolResult>;
    fn requires_permission(&self) -> bool { true }
}
```

**Available Tools**

| Tool | Parameters | Description |
|------|------------|-------------|
| `Bash` | command, timeout?, description? | Execute shell commands with optional timeout |
| `Read` | file_path, offset?, limit? | Read file contents with line numbers |
| `Edit` | file_path, old_string, new_string, replace_all? | Perform string replacements in files |
| `Write` | file_path, content | Write/create files |
| `Glob` | pattern, path? | Fast file pattern matching |
| `Grep` | pattern, path?, glob?, output_mode?, etc. | Content search using ripgrep |
| `TodoWrite` | todos[] | Task tracking with persisted state |

**Bash Tool**
- Executes shell commands with optional timeout (default 2 min, max 10 min)
- Optional description for logging
- Output truncated at 30KB

**Read Tool**
- Reads files with line numbers (cat -n format)
- Default limit: 2000 lines
- Supports offset and limit for large files
- No permission required (read-only)

**Edit Tool**
- Exact string replacement in files
- Fails if old_string is not unique (unless replace_all=true)
- Requires permission

**Write Tool**
- Creates or overwrites files
- Creates parent directories automatically
- Requires permission

**Glob Tool**
- Fast file pattern matching (e.g., `**/*.rs`)
- Results sorted by modification time
- No permission required (read-only)

**Grep Tool**
- Uses ripgrep for fast content search
- Output modes: `content`, `files_with_matches`, `count`
- Supports context lines (-A, -B, -C), case insensitive, multiline
- No permission required (read-only)

**TodoWrite Tool**
- Maintains a shared todo list across the agent session
- Uses `Arc<RwLock<Vec<TodoItem>>>` for thread-safe shared state
- Does not require permission
- Input format:
```json
{
  "todos": [
    {"content": "First task", "status": "pending", "activeForm": "Working on first task"},
    {"content": "Second task", "status": "completed", "activeForm": "Completing second task"}
  ]
}
```


### 4. Permission System (`src/permissions/`)

```rust
pub enum PermissionDecision {
    Allow,       // Allow this action
    Deny,        // Deny this action
    AlwaysAllow, // Always allow this tool
    AlwaysDeny,  // Always deny this tool
}
```

**Flow**
1. Tool use requested by LLM
2. Check `PermissionManager` for auto-decision
3. If no auto-decision, prompt user via Console
4. Store decision if "always" variant
5. Execute or skip tool based on decision

### 5. Context Manager (`src/context/`)

Manages system prompt and dynamic context injection.

```rust
pub struct ContextManager {
    system_prompt: String,
    providers: Vec<Arc<dyn ContextProvider>>,
    static_prompts: Vec<String>,
}
```

**Context Providers**
- `FileStructureProvider`: Project directory tree
- `GitStatusProvider`: Git status, branch, recent commits
- `StaticContextProvider`: Static text context

Context is injected into the system prompt in XML-like tags:
```xml
<context name="file_structure">
Project structure:
├── src/
│   ├── main.rs
...
</context>
```

### 6. Conversation Storage (`src/conversation/`)

**Directory Structure**
```
conversations/
└── 550e8400-e29b-41d4-a716-446655440000/
    ├── metadata.json
    └── history.jsonl
```

**metadata.json**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-01-14T10:30:45.123456Z",
  "updated_at": "2026-01-14T10:35:22.654321Z",
  "model_provider": "anthropic",
  "title": null
}
```

**history.jsonl**
Each line is a complete JSON message in Anthropic format:
```jsonl
{"role":"user","content":"list files"}
{"role":"assistant","content":[{"type":"text","text":"I'll list the files."},{"type":"tool_use","id":"toolu_123","name":"bash","input":{"command":"ls"}}]}
{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_123","content":"file1.txt\nfile2.txt"}]}
{"role":"assistant","content":"Here are the files:\n- file1.txt\n- file2.txt"}
```

### 7. Console (`src/cli/console.rs`)

**Color Scheme**
- User input: Cyan
- Assistant output: Green
- Tools: Magenta
- System messages: Yellow
- Errors: Red

**Methods**
- `print_user()`, `print_assistant()`, `print_system()`, `print_error()`
- `print_tool_action()`, `print_tool_result()`
- `print_thinking_block()`: Display extended thinking in a formatted box
- `ask_permission()`: Interactive permission prompt
- `read_input()`: Read user input with prompt

### 8. Logging (`src/logging.rs`)

- File-only logging (no console output to avoid CLI interference)
- Daily rotation in `logs/` directory
- Configurable via `RUST_LOG` environment variable

```bash
# Default (INFO)
cargo run

# Debug level
RUST_LOG=debug cargo run

# Trace level (maximum verbosity)
RUST_LOG=trace cargo run
```

### 9. Debugger (`src/debugger/mod.rs`)

Stores detailed logs of all API interactions and tool executions for debugging.

**Session Structure**
```
debugger/
└── 20260114_153045/              # Session timestamp
    ├── events.jsonl              # All events in order (append-only)
    ├── 000001_api_request.json   # Full API request
    ├── 000002_api_response.json  # Full API response
    ├── 000003_tool_call.json     # Tool invocation
    └── 000004_tool_result.json   # Tool output
```

**Event Types**
| Event | Description |
|-------|-------------|
| `api_request` | Full MessageRequest sent to Anthropic |
| `api_response` | Full MessageResponse received |
| `tool_call` | Tool name and input before execution |
| `tool_result` | Tool output and error status |

**Features**
- Automatic session folders with timestamps
- Sequence numbers for event ordering
- Both individual JSON files and combined JSONL log
- Full request/response capture for debugging API issues

## Data Flow

### Request Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Input                                │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Agent::process_turn()                                           │
│  1. Get conversation history                                     │
│  2. Add user message                                             │
│  3. Build system prompt with context                             │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  AnthropicProvider::send_with_tools()                            │
│  1. Build MessageRequest                                         │
│  2. Serialize to JSON                                            │
│  3. POST to api.anthropic.com/v1/messages                        │
│  4. Parse MessageResponse                                        │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Agent::process_response()                                       │
│  For each ContentBlock:                                          │
│    - Text → print to console                                     │
│    - ToolUse → check permission → execute → create ToolResult    │
│    - Thinking → display to user + log                            │
└──────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            stop_reason:            stop_reason:
              end_turn                tool_use
                    │                       │
                    ▼                       ▼
            Save messages           Add assistant msg
            to conversation         Add tool_result msg
                    │               Loop back to LLM
                    ▼                       │
            Display final           ────────┘
              response
```

### Message Format Flow

```
User Input (string)
    │
    ▼
Message { role: "user", content: "list files" }
    │
    ▼
MessageRequest { model, max_tokens, messages, system, tools }
    │
    ▼
HTTP POST → Anthropic API
    │
    ▼
MessageResponse { content: [TextBlock, ToolUseBlock] }
    │
    ▼
Execute Tool → ToolResult
    │
    ▼
Message { role: "user", content: [ToolResultBlock] }
    │
    ▼
HTTP POST → Anthropic API (continue)
    │
    ▼
MessageResponse { content: [TextBlock], stop_reason: "end_turn" }
    │
    ▼
Save all messages to history.jsonl
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | API key for Anthropic |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5-20250929` | Model to use |
| `ANTHROPIC_MAX_TOKENS` | No | `16000` | Max tokens per response |
| `RUST_LOG` | No | `info` | Log level |

### .env.example
```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_MAX_TOKENS=8192
RUST_LOG=debug
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `reqwest` | 0.11 | HTTP client for Anthropic API |
| `tokio` | 1.0 | Async runtime |
| `async-trait` | 0.1 | Async trait support |
| `serde` / `serde_json` | 1.0 | JSON serialization |
| `anyhow` | 1.0 | Error handling |
| `tracing` | 0.1 | Logging framework |
| `tracing-subscriber` | 0.3 | Log formatting |
| `tracing-appender` | 0.2 | File logging |
| `colored` | 2.0 | Terminal colors |
| `uuid` | 1.0 | Conversation IDs |
| `chrono` | 0.4 | Timestamps |
| `glob` | 0.3 | File pattern matching |

## Extensibility

### Adding a New Tool

1. Create `src/tools/my_tool.rs`:
```rust
use async_trait::async_trait;
use super::tool::{Tool, ToolInfo, ToolResult};

pub struct MyTool { /* fields */ }

#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something" }
    fn definition(&self) -> ToolDefinition { /* ... */ }
    fn get_info(&self, input: &Value) -> ToolInfo { /* ... */ }
    async fn execute(&self, input: &Value) -> Result<ToolResult> { /* ... */ }
}
```

2. Export in `src/tools/mod.rs`:
```rust
pub mod my_tool;
pub use my_tool::MyTool;
```

3. Register in `main.rs`:
```rust
tool_registry.register(MyTool::new());
```

### Adding a Context Provider

1. Implement `ContextProvider` trait in `src/context/providers.rs`
2. Add to `ContextManager` in `main.rs`:
```rust
context_manager.add_provider(MyProvider::new());
```

## Running

```bash
# Build
cargo build

# Run with API key
ANTHROPIC_API_KEY=sk-ant-... cargo run

# Run with debug logging
RUST_LOG=debug ANTHROPIC_API_KEY=sk-ant-... cargo run
```

## Example Session

```
============================================================
  Coding Agent - Powered by Claude
============================================================

Type your message and press Enter. Type 'exit' or 'quit' to end the session.

> list the rust files in src

────────────────────────────────────────────────────────────
⚠️ Permission Required The agent wants to use tool: bash

  Execute command: find src -name "*.rs"
  Working directory: /Users/user/project

Options:
  [y] Allow this action
  [n] Deny this action
  [a] Always allow this tool
  [d] Always deny this tool
────────────────────────────────────────────────────────────
Your choice (y/n/a/d): y
✓ Allowed

src/main.rs
src/lib.rs
src/agent/mod.rs
...