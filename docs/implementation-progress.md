# Implementation Progress

This document tracks what has been implemented in the agent framework.

## Phase Summary

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: Core Types | Complete | 12 tests |
| Phase 2: Session Manager | Complete | 17 tests |
| Phase 3: Channels & Handle | Complete | 23 tests |
| Phase 4: Runtime | Complete | 8 tests |
| Phase 5: Console Renderer | Complete | - |
| Phase 6: Integration Test | Skipped | - |
| Phase 7: Permission System | Complete | 7 tests |
| Phase 8: Tool System | Complete | - |
| Phase 9: Test Agent Example | Complete | - |
| Phase 10: Helpers Module | Complete | 6 tests |
| Phase 11: Standardized Agent | Complete | 2 tests |
| Phase 12: Debugger | Complete | - |

**Total Tests:** 87 passing

---

## Phase 1: Core Types

**Location:** `src/core/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `error.rs` | Error types for the framework |
| `state.rs` | Agent state enum |
| `output.rs` | Input/output message types |
| `context.rs` | Agent context with ResourceMap |

### Key Types

#### `FrameworkError`
Error enum for framework operations:
- `SessionNotFound` - Session doesn't exist
- `AgentNotRunning` - Agent is not running
- `ChannelClosed` - Communication channel closed
- `SendError` / `ReceiveError` - Channel errors
- `Io` / `Serialization` - I/O and JSON errors
- `ToolError` / `PermissionDenied` - Tool execution errors
- `Interrupted` / `Shutdown` - Agent lifecycle
- `InvalidConfig` / `Other` - Misc errors

#### `AgentState`
Enum representing agent's current state:
- `Idle` - Waiting for input
- `Processing` - Processing input, calling LLM
- `WaitingForPermission` - Waiting for user permission
- `ExecutingTool { tool_name, tool_use_id }` - Running a tool
- `WaitingForSubAgent { session_id }` - Waiting for child agent
- `Done` - Agent completed
- `Error { message }` - Agent errored

#### `InputMessage`
Messages sent TO an agent:
- `UserInput(String)` - User text input
- `ToolResult { tool_use_id, result }` - Async tool result
- `PermissionResponse { tool_name, allowed, remember }` - Permission decision
- `SubAgentComplete { session_id, result }` - Child agent finished
- `Interrupt` - Request graceful stop
- `Shutdown` - Request termination

#### `OutputChunk`
Streaming output FROM an agent:
- Text: `TextDelta`, `TextComplete`
- Thinking: `ThinkingDelta`, `ThinkingComplete`
- Tools: `ToolStart`, `ToolProgress`, `ToolEnd`
- Permission: `PermissionRequest`
- Subagent: `SubAgentSpawned`, `SubAgentOutput`, `SubAgentComplete`
- Status: `StateChange`, `Status`, `Error`, `Done`

#### `AgentContext`
Hidden state passed to tools (not exposed to LLM):
```rust
pub struct AgentContext {
    pub session_id: String,
    pub agent_type: String,
    pub name: String,
    pub description: String,
    pub parent_session_id: Option<String>,
    pub parent_tool_use_id: Option<String>,
    pub current_turn: usize,
    pub current_tool_use_id: Option<String>,
    pub metadata: HashMap<String, Value>,  // JSON data
    pub resources: ResourceMap,             // Runtime objects
}
```

#### `ResourceMap`
Type-safe container for agent-specific runtime objects:
```rust
// Insert any Send + Sync type
ctx.insert_resource(TodoManager::new());

// Get by type (returns Arc<T>)
let todo = ctx.get_resource::<TodoManager>();
```

---

## Phase 2: Session Manager

**Location:** `src/session/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `metadata.rs` | Session metadata struct |
| `storage.rs` | File I/O helpers |
| `session.rs` | Main AgentSession struct |

### Key Types

#### `SessionMetadata`
Persisted metadata for a session:
```rust
pub struct SessionMetadata {
    pub session_id: String,
    pub agent_type: String,
    pub name: String,
    pub description: String,
    pub parent_session_id: Option<String>,
    pub parent_tool_use_id: Option<String>,
    pub child_session_ids: Vec<String>,
    pub model: String,
    pub provider: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub custom: HashMap<String, Value>,
}
```

#### `SessionStorage`
Handles file I/O for sessions:
- `save_metadata()` / `load_metadata()` - Persist/load metadata
- `append_message()` / `load_messages()` - Message history
- `list_sessions()` / `delete_session()` - Session management

Storage format:
```
sessions/
└── <session_id>/
    ├── metadata.json   # SessionMetadata
    └── history.jsonl   # Message history (one JSON per line)
```

#### `AgentSession`
Combines metadata and messages:
```rust
pub struct AgentSession {
    pub metadata: SessionMetadata,
    pub messages: Vec<Message>,
    storage: SessionStorage,
}
```

Key methods:
- `new()` / `new_subagent()` - Create sessions
- `add_message()` - Append with auto-persist
- `save()` / `load()` - Full persistence
- `delete()` - Remove session
- Subagent tracking via `child_session_ids`

---

## Phase 3: Channels & Handle

**Location:** `src/runtime/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `channels.rs` | Channel type definitions |
| `handle.rs` | AgentHandle (external interface) |
| `internals.rs` | AgentInternals (internal state) |

### Key Types

#### Channel Types
```rust
// Input: mpsc (single producer, single consumer)
pub type InputSender = mpsc::Sender<InputMessage>;
pub type InputReceiver = mpsc::Receiver<InputMessage>;

// Output: broadcast (multiple subscribers)
pub type OutputSender = broadcast::Sender<OutputChunk>;
pub type OutputReceiver = broadcast::Receiver<OutputChunk>;
```

#### `AgentHandle`
External interface for communicating with a running agent:
```rust
impl AgentHandle {
    // Input methods
    pub async fn send_input(&self, input: impl Into<String>) -> FrameworkResult<()>;
    pub async fn send_tool_result(&self, tool_use_id, result) -> FrameworkResult<()>;
    pub async fn send_permission_response(&self, tool_name, allowed, remember) -> FrameworkResult<()>;
    pub async fn interrupt(&self) -> FrameworkResult<()>;
    pub async fn shutdown(&self) -> FrameworkResult<()>;

    // Output methods
    pub fn subscribe(&self) -> OutputReceiver;  // Get streaming output
    pub fn subscriber_count(&self) -> usize;

    // State methods
    pub async fn state(&self) -> AgentState;
    pub async fn is_idle(&self) -> bool;
    pub async fn is_running(&self) -> bool;
    pub async fn wait_for_completion(&self);
}
```

#### `AgentInternals`
Internal state passed to agent functions:
```rust
impl AgentInternals {
    // Session and context
    pub session: AgentSession,
    pub context: AgentContext,

    // Input methods
    pub async fn receive(&mut self) -> Option<InputMessage>;
    pub async fn receive_or_err(&mut self) -> FrameworkResult<InputMessage>;

    // Output methods
    pub fn send(&self, chunk: OutputChunk) -> usize;
    pub fn send_text(&self, text: impl Into<String>) -> usize;
    pub fn send_thinking(&self, text: impl Into<String>) -> usize;
    pub fn send_done(&self) -> usize;
    pub fn send_tool_start(&self, id, name, input) -> usize;
    pub fn send_tool_end(&self, id, result) -> usize;

    // State methods
    pub async fn set_state(&self, state: AgentState);
    pub async fn set_idle(&self);
    pub async fn set_processing(&self);
    pub async fn set_done(&self);
    pub async fn set_error(&self, message: impl Into<String>);
    pub async fn set_executing_tool(&self, tool_name, tool_use_id);

    // Context helpers
    pub fn context_for_tool(&self, tool_use_id) -> AgentContext;
    pub fn next_turn(&mut self);
}
```

### Communication Model

```
┌─────────────────────┐         ┌─────────────────────────────────┐
│   External Code     │         │   Agent Task                    │
│   (Console, Parent) │         │   (tokio::spawn)                │
│                     │         │                                 │
│  AgentHandle        │         │  AgentInternals                 │
│  ├─ send_input() ───│──mpsc──▶│──▶ receive()                    │
│  ├─ interrupt() ────│──mpsc──▶│──▶ InputMessage::Interrupt      │
│  └─ subscribe() ◀───│◀─bcast──│◀── send()                       │
│                     │         │                                 │
└─────────────────────┘         └─────────────────────────────────┘
```

- **Input channel (mpsc):** Single-producer, single-consumer. Handle sends, internals receives.
- **Output channel (broadcast):** Multi-consumer. Internals sends, multiple handles can subscribe.
- **State:** Shared via `Arc<RwLock<AgentState>>`, accessible from both sides.

---

## Phase 4: Runtime

**Location:** `src/runtime/runtime.rs`

### Files

| File | Description |
|------|-------------|
| `runtime.rs` | AgentRuntime - spawns and manages agents |

### Key Types

#### `AgentRuntime`
Spawns and manages agent tasks:
```rust
impl AgentRuntime {
    pub fn new() -> Self;

    // Spawn an agent - returns handle immediately
    pub async fn spawn<F, Fut>(
        &self,
        session: AgentSession,
        agent_fn: F,
    ) -> AgentHandle
    where
        F: FnOnce(AgentInternals) -> Fut + Send + 'static,
        Fut: Future<Output = FrameworkResult<()>> + Send + 'static;

    // Spawn a subagent with parent linkage
    pub async fn spawn_subagent<F, Fut>(...) -> FrameworkResult<AgentHandle>;

    // Registry methods
    pub async fn get(&self, session_id: &str) -> Option<AgentHandle>;
    pub async fn is_running(&self, session_id: &str) -> bool;
    pub async fn count(&self) -> usize;
    pub async fn list_running(&self) -> Vec<String>;

    // Lifecycle methods
    pub async fn shutdown(&self, session_id: &str) -> FrameworkResult<()>;
    pub async fn interrupt(&self, session_id: &str) -> FrameworkResult<()>;
    pub async fn shutdown_all(&self) -> Vec<(String, FrameworkResult<()>)>;

    // Wait methods
    pub async fn wait_for(&self, session_id: &str) -> FrameworkResult<()>;
    pub async fn wait_all(&self);
}
```

### Usage Example

```rust
let runtime = AgentRuntime::new();
let session = AgentSession::new("my-agent", "coder", "Coder", "A coding agent")?;

// Spawn an agent
let handle = runtime.spawn(session, |mut internals| async move {
    loop {
        match internals.receive().await {
            Some(InputMessage::UserInput(text)) => {
                internals.send_text(format!("Echo: {}", text));
                internals.send_done();
            }
            Some(InputMessage::Shutdown) | None => {
                internals.set_done().await;
                break;
            }
            _ => {}
        }
    }
    Ok(())
}).await;

// Interact via handle
handle.send_input("Hello").await?;
let mut rx = handle.subscribe();
// ... receive output ...

// Shutdown
runtime.shutdown("my-agent").await?;
```

### Features

- **Auto-cleanup**: Agents are removed from registry when they exit
- **Cloneable runtime**: Multiple references can spawn/query agents
- **Error logging**: Agent errors are logged via tracing
- **Subagent support**: `spawn_subagent()` creates linked sessions

---

## Phase 5: Console Renderer

**Location:** `src/cli/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `console.rs` | Low-level terminal formatting |
| `renderer.rs` | ConsoleRenderer - subscribes to agent output |

### Key Types

#### `ConsoleRenderer`
Opt-in component that subscribes to agent output and renders to terminal:
```rust
impl ConsoleRenderer {
    /// Create a new console renderer for an agent
    pub fn new(handle: AgentHandle) -> Self;

    /// Create with a custom Console instance
    pub fn with_console(handle: AgentHandle, console: Console) -> Self;

    /// Configure whether to show thinking blocks
    pub fn show_thinking(mut self, show: bool) -> Self;

    /// Configure whether to show tool execution details
    pub fn show_tools(mut self, show: bool) -> Self;

    /// Run the interactive console loop
    pub async fn run(&self) -> io::Result<()>;

    /// Run a single turn (programmatic usage)
    pub async fn run_turn(&self, input: &str) -> io::Result<()>;

    /// Get the underlying agent handle
    pub fn handle(&self) -> &AgentHandle;

    /// Get the underlying console
    pub fn console(&self) -> &Console;
}
```

### Design Philosophy

The ConsoleRenderer is **completely decoupled** from agent logic:
- Agent runs independently in its own tokio task
- ConsoleRenderer subscribes to the agent's broadcast output channel
- Can be replaced with Tauri UI, Web UI, or any other renderer
- Programmer opts in by creating a ConsoleRenderer

### Features

- **Streaming text**: Renders text deltas as they arrive
- **Thinking blocks**: Optionally shows extended thinking (configurable)
- **Tool execution**: Shows tool start/progress/end (configurable)
- **Permission requests**: Prompts user for tool permissions via Console
- **Subagent events**: Reports spawned/completed subagents
- **State changes**: Logs agent state transitions
- **Graceful exit**: Handles "exit" and "quit" commands

### Usage Example

```rust
// Agent runs independently
let handle = runtime.spawn(session, agent_fn).await;

// Console renderer subscribes to output
let renderer = ConsoleRenderer::new(handle)
    .show_thinking(true)
    .show_tools(true);

// Run interactive loop (blocks until exit/quit)
renderer.run().await?;
```

### Render Loop

```
┌────────────────────────┐          ┌────────────────────────────────┐
│   ConsoleRenderer      │          │   Agent Task                   │
│                        │          │                                │
│  1. Read user input    │          │                                │
│  2. handle.send_input()│──mpsc───▶│  3. Process input, call LLM    │
│                        │          │  4. Send OutputChunks          │
│  5. subscribe().recv() │◀─bcast───│──── TextDelta, ToolStart, etc  │
│  6. Render to terminal │          │                                │
│                        │          │                                │
│  Loop until exit/quit  │          │  Agent runs until shutdown     │
└────────────────────────┘          └────────────────────────────────┘
```

### Examples

Two example agents demonstrate the framework:

#### `examples/test_agent.rs`
Minimal programmatic test:
- Creates session with persistent storage
- Spawns agent, sends one message
- Manually subscribes to output
- Shuts down and waits

#### `examples/console_agent.rs`
Interactive console agent:
- Creates session and spawns agent
- Uses ConsoleRenderer for interactive I/O
- Demonstrates full decoupled pattern

Run with:
```bash
cargo run --example console_agent
```

---

## Phase 7: Permission System

**Location:** `src/permissions/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports and documentation |
| `manager.rs` | Permission rules, manager, and global permissions |

### Architecture

The permission system uses a three-tier hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│  Arc<GlobalPermissions>  ← shared by ALL agents             │
│  (updates propagate immediately to all running agents)      │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ Agent 1  │         │ Agent 2  │         │ Agent 3  │
   │ ┌──────┐ │         │ ┌──────┐ │         │ ┌──────┐ │
   │ │Local │ │         │ │Local │ │         │ │Local │ │
   │ └──────┘ │         │ └──────┘ │         │ └──────┘ │
   │ ┌──────┐ │         │ ┌──────┐ │         │ ┌──────┐ │
   │ │Session│ │        │ │Session│ │        │ │Session│ │
   │ └──────┘ │         │ └──────┘ │         │ └──────┘ │
   └──────────┘         └──────────┘         └──────────┘
```

### Key Types

#### `RuleType`
```rust
pub enum RuleType {
    AllowTool,    // Allow entire tool (e.g., Read is always allowed)
    AllowPrefix,  // Allow commands starting with prefix (e.g., "cd" for Bash)
}
```

#### `PermissionRule`
```rust
pub struct PermissionRule {
    pub rule_type: RuleType,
    pub tool_name: String,           // Mandatory: "Bash", "Write", etc.
    pub prefix: Option<String>,      // For AllowPrefix: "cd", "git status", etc.
}

// Constructors
PermissionRule::allow_tool("Read")
PermissionRule::allow_prefix("Bash", "cd")
```

#### `GlobalPermissions`
Shared across all agents via `Arc`:
```rust
pub struct GlobalPermissions {
    rules: RwLock<Vec<PermissionRule>>,
}

impl GlobalPermissions {
    pub fn new() -> Self;
    pub fn with_rules(rules: Vec<PermissionRule>) -> Self;
    pub fn add_rule(&self, rule: PermissionRule);
    pub fn check(&self, tool_name: &str, input: &str) -> bool;
}
```

#### `PermissionManager`
Per-agent permission context:
```rust
pub struct PermissionManager {
    global: Arc<GlobalPermissions>,  // Shared reference
    local: Vec<PermissionRule>,      // Agent-type specific
    session: Vec<PermissionRule>,    // This session only
    interactive: bool,               // Can prompt user?
}

impl PermissionManager {
    pub fn check(&self, tool_name: &str, input: &str) -> CheckResult;
    pub fn add_rule(&mut self, rule: PermissionRule, scope: PermissionScope);
    pub fn process_decision(&mut self, tool_name, input, decision, scope) -> bool;
}
```

#### `CheckResult`
```rust
pub enum CheckResult {
    Allowed,   // Tool/action is allowed by a rule
    AskUser,   // Need to ask user for permission
    Denied,    // Denied (non-interactive mode only)
}
```

### Integration with Runtime

The `AgentRuntime` holds shared `GlobalPermissions`:
```rust
let runtime = AgentRuntime::new();
// or with initial rules:
let runtime = AgentRuntime::with_global_rules(vec![
    PermissionRule::allow_tool("Read"),
    PermissionRule::allow_tool("Glob"),
]);

// Access global permissions
runtime.global_permissions().add_rule(PermissionRule::allow_tool("Grep"));
```

Each spawned agent gets its own `PermissionManager` that references the shared global:
```rust
// Spawn with local rules
let handle = runtime.spawn_with_local_rules(
    session,
    vec![PermissionRule::allow_prefix("Bash", "git")],
    agent_fn,
).await;
```

### Integration with AgentInternals

Agents can check/request permissions:
```rust
// In agent loop
match internals.check_permission("Bash", "rm -rf /") {
    CheckResult::Allowed => { /* execute */ }
    CheckResult::AskUser => { /* prompt via renderer */ }
    CheckResult::Denied => { /* reject */ }
}

// Or use the convenience method that handles prompting:
if internals.request_permission("Bash", "Delete files", "rm -rf /tmp/*").await? {
    // Execute tool
}

// Add rules programmatically
internals.add_permission_rule(
    PermissionRule::allow_prefix("Bash", "npm"),
    PermissionScope::Session,
);
```

### Permission Flow

```
Agent wants tool → PermissionManager.check() → CheckResult
                           │
                           ├─→ Allowed (rule exists) → Execute tool
                           ├─→ Denied (non-interactive) → Return error
                           └─→ AskUser (no rule) → Send PermissionRequest
                                      │
                                      ▼
                           ConsoleRenderer receives request
                           User decides: Allow / Deny / Always Allow
                                      │
                           ┌──────────┴──────────┐
                           ▼                     ▼
                     Allow/Deny            AlwaysAllow
                     (one-time)            (add rule to session/global)
```

---

## Phase 8: Tool System

**Location:** `src/tools/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `tool.rs` | Tool trait and ToolResult |
| `registry.rs` | ToolRegistry for managing tools |
| `common/` | Built-in tool implementations |

### Common Tools (`src/tools/common/`)

| File | Description |
|------|-------------|
| `bash.rs` | BashTool - Execute shell commands |
| `read_tool.rs` | ReadTool - Read file contents |
| `write_tool.rs` | WriteTool - Write/create files |
| `edit_tool.rs` | EditTool - Edit files with string replacement |
| `glob_tool.rs` | GlobTool - Find files by pattern |
| `grep_tool.rs` | GrepTool - Search file contents |
| `todo.rs` | TodoWriteTool - Manage todo lists |

### Key Types

#### `Tool` Trait
```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn definition(&self) -> ToolDefinition;
    fn get_info(&self, input: &Value) -> ToolInfo;

    // Execute with access to agent internals
    async fn execute(&self, input: &Value, internals: &mut AgentInternals) -> Result<ToolResult>;

    fn requires_permission(&self) -> bool { true }
}
```

#### `ToolResult`
```rust
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(output: impl Into<String>) -> Self;
    pub fn error(message: impl Into<String>) -> Self;
}
```

#### `ToolRegistry`
```rust
impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: impl Tool + 'static);
    pub fn get_definitions(&self) -> Vec<ToolDefinition>;
    pub fn tool_names(&self) -> Vec<&str>;
    pub fn get_tool_info(&self, name: &str, input: &Value) -> Option<ToolInfo>;

    // Execute tool with internals for context access
    pub async fn execute(
        &self,
        name: &str,
        input: &Value,
        internals: &mut AgentInternals
    ) -> Result<ToolResult>;
}
```

---

## Phase 9: Test Agent Example

**Location:** `examples/test_agent/`

### Files

| File | Description |
|------|-------------|
| `main.rs` | Entry point, setup and configuration |
| `agent.rs` | Agent loop with permission-aware tool execution |
| `tools.rs` | Tool registry setup |

### Purpose

A complete demonstration of the agent framework:
- Runtime with shared global permissions
- Agent with permission-aware tool execution
- Console renderer for user interaction
- Read, Write, and Bash tools

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  main.rs                                                        │
│  - Creates LLM provider (AnthropicProvider)                     │
│  - Creates AgentRuntime (no pre-configured permissions)         │
│  - Creates ToolRegistry (Read, Write, Bash)                     │
│  - Creates AgentSession with persistent storage                 │
│  - Spawns agent via runtime.spawn()                             │
│  - Creates ConsoleRenderer and runs interactive loop            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  agent.rs - Agent Loop                                          │
│                                                                 │
│  loop {                                                         │
│    1. Wait for InputMessage                                     │
│    2. On UserInput → process_turn()                             │
│       a. Add user message to history                            │
│       b. LLM loop:                                              │
│          - Call LLM with tools                                  │
│          - For each tool_use → execute_tool_with_permission()   │
│          - Add results to history                               │
│          - Continue until no tool calls                         │
│    3. Send Done, persist session                                │
│    4. Loop until Interrupt/Shutdown                             │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  execute_tool_with_permission()                                 │
│                                                                 │
│  1. Check permission via internals.check_permission()           │
│     ├─ Allowed → execute immediately                            │
│     ├─ Denied → return error                                    │
│     └─ AskUser → prompt user:                                   │
│        a. Send PermissionRequest via output channel             │
│        b. Set state to WaitingForPermission                     │
│        c. Wait for PermissionResponse                           │
│        d. If "Always Allow" → add rule to session               │
│        e. Execute or deny based on response                     │
└─────────────────────────────────────────────────────────────────┘
```

### Running the Example

```bash
cargo run --example test_agent
```

This will:
1. Start the agent with no pre-configured permissions
2. Present an interactive console
3. Every tool execution will prompt for permission
4. User can choose: Allow / Deny / Always Allow

### Key Points

- **No pre-configured permissions**: Demonstrates full permission flow
- **Permission persistence**: "Always Allow" adds rule to session scope
- **Session persistence**: Conversation is saved to `./sessions/` directory
- **Streaming output**: Uses ConsoleRenderer for real-time display
- **Complete tool flow**: Read, Write, Bash tools fully functional

---

## Helpers Module

**Location:** `src/helpers/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `todo_manager.rs` | TodoListManager for task tracking |
| `context_injection.rs` | Context injection system for modifying messages before LLM calls |
| `debugger.rs` | Debugger for logging API calls and tool executions |

### Key Types

#### `TodoListManager`
Manages a todo list with turn tracking. Stored in agent's ResourceMap and accessed by TodoWriteTool.

```rust
pub struct TodoListManager { /* RwLock-protected state */ }

impl TodoListManager {
    pub fn new() -> Self;
    pub fn get_todos(&self) -> Vec<TodoItem>;
    pub fn set_todos(&self, items: Vec<TodoItem>, turn: usize);
    pub fn last_updated_turn(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn len(&self) -> usize;
    pub fn counts(&self) -> (usize, usize, usize); // (pending, in_progress, completed)
    pub fn current_task(&self) -> Option<TodoItem>;
    pub fn format(&self) -> String;
}
```

#### `TodoItem` and `TodoStatus`
```rust
pub enum TodoStatus { Pending, InProgress, Completed }

pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
    pub active_form: String,
}

impl TodoItem {
    pub fn new(content: impl Into<String>, active_form: impl Into<String>) -> Self;
    pub fn with_status(content, active_form, status) -> Self;
}
```

### Usage Pattern (TodoListManager)

```rust
// 1. In agent setup, add TodoListManager to context resources:
internals.context.insert_resource(TodoListManager::new());

// 2. Register TodoWriteTool (no constructor args needed):
registry.register(TodoWriteTool::new());

// 3. TodoWriteTool automatically finds and updates the manager
// when the LLM calls it

// 4. Console can display todos if given the manager:
let manager = Arc::new(TodoListManager::new());
let console = Console::with_todo_manager(manager.clone());
// Pass the same Arc to context.insert_resource()
```

### Context Injection System

Allows programmers to modify messages before each LLM call without writing the entire agent loop.

#### `ContextInjection` Trait
```rust
pub trait ContextInjection: Send + Sync {
    fn name(&self) -> &str;
    fn inject(&self, internals: &AgentInternals, messages: Vec<Message>) -> Vec<Message>;
}
```

#### `FnInjection`
Create injections from closures:
```rust
let injection = FnInjection::new("todo_reminder", |internals, mut messages| {
    // Check if todo needs updating, inject reminder
    messages
});
```

#### `InjectionChain`
Apply multiple injections in order:
```rust
let mut chain = InjectionChain::new();
chain.add_fn("first_msg_context", |_, msgs| { /* ... */ msgs });
chain.add_fn("todo_reminder", |_, msgs| { /* ... */ msgs });

// Before each LLM call:
let modified_messages = chain.apply(&internals, messages);
```

#### Helper Functions
```rust
// Inject <system-reminder> tags
inject_system_reminder(&mut messages, "Update your todo list");

// Add context to first user message
prepend_to_first_user_message(&mut messages, "Project context: ...\n\n");

// Append to last message
append_to_last_message(&mut messages, "\n\nBe concise.");
```

---

## Standardized Agent Module

**Location:** `src/agent/`

### Files

| File | Description |
|------|-------------|
| `mod.rs` | Module exports |
| `config.rs` | AgentConfig - configuration for the agent |
| `executor.rs` | ToolExecutor - permission-aware tool execution |
| `standard_loop.rs` | StandardAgent - the main agent implementation |

### Key Types

#### `AgentConfig`
```rust
let config = AgentConfig::new("You are a helpful assistant")
    .with_tools(tools)                     // Set tool registry
    .with_injection_chain(injections)      // Add context injections
    .with_injection_fn("name", |i, m| m)   // Add function injection
    .with_max_tool_iterations(100)         // Limit tool loops
    .with_auto_save(true);                 // Auto-save session
```

#### `StandardAgent`
```rust
let agent = StandardAgent::new(config, llm);

// Use with runtime.spawn()
let handle = runtime.spawn(session, |internals| {
    agent.run(internals)
}).await;
```

#### `ToolExecutor`
```rust
// Execute tool with permission checking
let result = ToolExecutor::execute_with_permission(
    &mut internals,
    &tools,
    "Read",
    &input
).await;

// Execute tool directly (permission already granted)
let result = ToolExecutor::execute(&mut internals, &tools, "Read", &input).await;
```

### Flow

```
User Input
    │
    ▼
┌───────────────────────────────────────────┐
│ StandardAgent.process_turn()               │
│                                           │
│  1. Add user message to history           │
│  2. Loop:                                 │
│     a. Get messages from history          │
│     b. Apply context injections ◄─────────┼── InjectionChain.apply()
│     c. Call LLM                           │
│     d. For each tool_use:                 │
│        - ToolExecutor.execute_with_permission()
│        - Collect results                  │
│     e. Add assistant message to history   │
│     f. If tool results, add & continue    │
│     g. Else check stop_reason & break     │
│  3. Auto-save session (if configured)     │
└───────────────────────────────────────────┘
    │
    ▼
Output (via internals.send_*)
```

---

## Running Tests

```bash
# All framework tests
cargo test core::
cargo test session::
cargo test runtime::
cargo test permissions::

# All tests
cargo test
```

---

## Running Examples

```bash
# Test agent with permission prompts
cargo run --example test_agent
```

---

## Phase 12: Debugger

**Location:** `src/helpers/debugger.rs`

### Purpose

Debug logging for agent development. When enabled, logs all LLM API calls and tool executions to a `debugger/` folder within the session directory.

### Key Types

#### `Debugger`
```rust
impl Debugger {
    /// Create a new debugger at session_dir/debugger/
    pub fn new(session_dir: impl AsRef<Path>) -> Result<Self>;

    /// Create a disabled debugger (no-op)
    pub fn disabled() -> Self;

    /// Check if debugging is enabled
    pub fn is_enabled(&self) -> bool;

    /// Log API request (messages, system prompt, tools)
    pub fn log_api_request(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_definitions: Option<&[Value]>,
    ) -> Result<()>;

    /// Log API response (raw JSON)
    pub fn log_api_response(&self, response: &Value) -> Result<()>;

    /// Log tool call (before execution)
    pub fn log_tool_call(&self, tool_name: &str, tool_id: &str, input: &Value) -> Result<()>;

    /// Log tool result (after execution)
    pub fn log_tool_result(&self, tool_name: &str, tool_id: &str, result: &ToolResult) -> Result<()>;

    /// Clear all debug logs
    pub fn clear(&self) -> Result<()>;
}
```

#### Event Types
```rust
pub enum EventType {
    ApiRequest,
    ApiResponse,
    ToolCall,
    ToolResult,
}
```

### Output Format

Debug events are stored as numbered JSON files:
```
sessions/<session_id>/debugger/
├── 000000_api_request.json
├── 000001_api_response.json
├── 000002_tool_call_Read.json
├── 000003_tool_result_Read.json
├── 000004_api_request.json
└── ...
```

Each file contains:
- `event_type`: Type of event
- `sequence`: Sequence number for ordering
- Event-specific data (messages, response, tool input/output, etc.)

### Integration with AgentConfig

Enable debugging via the config builder:
```rust
let config = AgentConfig::new("System prompt")
    .with_tools(tools)
    .with_debug(true);  // Enable debug logging
```

### Integration with StandardAgent

When debug is enabled, StandardAgent:
1. Creates a Debugger instance at startup
2. Stores it in `internals.context` as a resource
3. Logs API requests/responses during LLM calls
4. ToolExecutor logs tool calls/results

### Example Debug Log (api_request.json)
```json
{
  "event_type": "api_request",
  "sequence": 0,
  "system_prompt": "You are a helpful assistant...",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
  ],
  "tool_definitions": [...]
}
```

### Example Debug Log (tool_call.json)
```json
{
  "event_type": "tool_call",
  "sequence": 2,
  "tool_name": "Read",
  "tool_id": "toolu_01ABC123",
  "input": {"file_path": "/path/to/file.txt"}
}
```
