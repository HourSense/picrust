# Shadow Agent SDK

A Rust framework for building AI agents with Claude. Designed for applications that need to spawn, manage, and communicate with autonomous agents - particularly suited for Tauri apps and other frontend-backend architectures.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Key Features](#key-features)
   - [Prompt Caching](#prompt-caching)
   - [Streaming and History](#streaming-and-history)
   - [Image and PDF Support](#image-and-pdf-support)
   - [Attachment Support](#attachment-support)
   - [Ask User Questions](#ask-user-questions)
   - [Interrupt Handling](#interrupt-handling)
   - [MCP Integration](#mcp-model-context-protocol-integration)
4. [Module Reference](#module-reference)
   - [Runtime](#runtime-module)
   - [Session](#session-module)
   - [Tools](#tools-module)
   - [Permissions](#permissions-module)
   - [Hooks](#hooks-module)
   - [LLM Provider](#llm-module)
   - [Helpers](#helpers-module)
5. [Building a Frontend Integration](#building-a-frontend-integration)
6. [Examples](#examples)
7. [API Reference](#api-reference)

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
shadow-agent-sdk = { path = "path/to/shadow-agent-sdk" }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

### Minimal Example

```rust
use std::sync::Arc;
use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    llm::{AnthropicProvider, LlmProvider},
    runtime::AgentRuntime,
    session::AgentSession,
    tools::ToolRegistry,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Create LLM provider (reads ANTHROPIC_API_KEY and ANTHROPIC_MODEL from env)
    let llm: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::from_env()?);

    // 2. Create runtime (manages all agents)
    let runtime = AgentRuntime::new();

    // 3. Create session (persists conversation)
    let session = AgentSession::new(
        "my-session",      // unique ID
        "assistant",       // agent type
        "My Assistant",    // display name
        "A helpful agent", // description
    )?;

    // 4. Configure and create agent
    let config = AgentConfig::new("You are a helpful assistant.")
        .with_streaming(true);
    let agent = StandardAgent::new(config, llm);

    // 5. Spawn agent
    let handle = runtime
        .spawn(session, |internals| agent.run(internals))
        .await;

    // 6. Send input and receive output
    let mut rx = handle.subscribe();
    handle.send_input("Hello!").await?;

    // 7. Process output stream
    loop {
        match rx.recv().await {
            Ok(chunk) => {
                use shadow_agent_sdk::core::OutputChunk;
                match chunk {
                    OutputChunk::TextDelta(text) => print!("{}", text),
                    OutputChunk::Done => break,
                    OutputChunk::Error(e) => eprintln!("Error: {}", e),
                    _ => {}
                }
            }
            Err(_) => break,
        }
    }

    Ok(())
}
```

---

## Core Concepts

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Application                          │
│  (Tauri, CLI, Web Server, etc.)                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AgentRuntime                               │
│  - Spawns and manages agent tasks                               │
│  - Maintains registry of running agents                         │
│  - Shares global permissions across agents                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │ spawn()
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AgentHandle                                │
│  - External interface for communication                         │
│  - send_input(), subscribe(), state()                           │
│  - Cloneable, shareable across threads                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      StandardAgent                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Agent Loop:                                               │   │
│  │  1. Receive input → 2. Inject context                    │   │
│  │  3. Call LLM → 4. Parse response                         │   │
│  │  5. Execute tools (with permissions) → 6. Send output    │   │
│  │  7. Repeat until done                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Components:                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │   Tools    │ │ Permissions│ │   Hooks    │ │  Session   │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Types

| Type | Purpose |
|------|---------|
| `AgentRuntime` | Spawns and manages agent lifecycles |
| `AgentHandle` | Communicate with running agents |
| `AgentSession` | Conversation state and persistence |
| `AgentConfig` | Configure agent behavior |
| `StandardAgent` | Complete agent implementation |
| `ToolRegistry` | Manages available tools |
| `PermissionManager` | Three-tier permission system |
| `HookRegistry` | Intercept and modify agent behavior |
| `LlmProvider` | Trait for pluggable LLM backends |
| `AnthropicProvider` | Anthropic Claude API client |
| `GeminiProvider` | Google Gemini API client |
| `SwappableLlmProvider` | Runtime-swappable LLM provider |

### Agent States

```rust
pub enum AgentState {
    Idle,                           // Waiting for input
    Processing,                     // Calling LLM
    WaitingForPermission,          // Awaiting user approval
    ExecutingTool { tool_name, tool_use_id },
    WaitingForSubAgent { session_id },
    Done,
    Error { message },
}
```

### Communication Flow

```
Frontend                    AgentHandle                 Agent
   │                            │                         │
   │── send_input("Hello") ────►│                         │
   │                            │── UserInput ───────────►│
   │                            │                         │── Process
   │                            │◄── TextDelta ──────────│
   │◄── OutputChunk ───────────│                         │
   │                            │◄── ToolStart ──────────│
   │◄── OutputChunk ───────────│                         │
   │                            │◄── PermissionRequest ──│
   │◄── OutputChunk ───────────│                         │
   │                            │                         │
   │── send_permission ────────►│── PermissionResponse ──►│
   │                            │                         │── Execute Tool
   │                            │◄── ToolEnd ────────────│
   │◄── OutputChunk ───────────│                         │
   │                            │◄── Done ───────────────│
   │◄── OutputChunk ───────────│                         │
```

---

## Key Features

### Prompt Caching

The SDK automatically implements prompt caching to reduce API costs by up to 90% and improve latency. This feature is **enabled by default**.

#### How It Works

Prompt caching is a feature from Anthropic that allows reusing previously processed prompt content:
- Cached tokens cost only **10% of regular input tokens** (90% discount)
- Cache writes cost 25% more than regular input tokens
- Cache entries have a 5-minute lifetime by default (refreshed on each use)

The framework automatically adds cache breakpoints at three strategic locations:

1. **Tool Definitions** - Caches all tool definitions (they rarely change)
2. **System Prompt** - Caches the system prompt (static for entire conversation)
3. **Conversation History** - Caches the growing conversation history

#### Cost Example

For a 3-turn conversation:
- **Without caching**: 21,000 tokens
- **With caching**: 11,450 tokens
- **Savings**: ~46% (increases with longer conversations!)

#### Usage

```rust
// Enabled by default
let config = AgentConfig::new("You are a helpful assistant")
    .with_tools(tools);

// Optionally disable
let config = AgentConfig::new("You are a helpful assistant")
    .with_prompt_caching(false);

// Enable debug mode to see cache metrics
let config = AgentConfig::new("You are a helpful assistant")
    .with_debug(true);  // Logs cache creation/read tokens
```

#### Monitoring Cache Performance

When debug mode is enabled, cache metrics are logged:
```
Cache creation tokens: 5000
Cache read tokens: 12000
```

You can also access these programmatically via `MessageResponse.usage`:
- `usage.cache_creation_input_tokens` - Tokens written to cache
- `usage.cache_read_input_tokens` - Tokens read from cache
- `usage.input_tokens` - Uncached tokens

---

### Streaming and History

The SDK uses a **dual-channel architecture** for conversation data:

1. **Stream** - Real-time output via broadcast channels (ephemeral, fast)
2. **History** - Persistent message storage on disk (durable, reliable)

#### Critical Pattern: Subscribe Before Sending

```rust
// ✅ CORRECT: Subscribe BEFORE sending input
let mut rx = handle.subscribe();  // Subscribe first
handle.send_input("Hello").await?;  // Then send

// ❌ WRONG: Subscribe after sending
handle.send_input("Hello").await?;
let mut rx = handle.subscribe();  // Too late! Missed early output
```

#### The Foolproof Pattern for Displaying Conversations

```rust
// Step 1: Load historical messages from disk
let history = AgentSession::get_history("session-id")?;
for message in history {
    ui.display_message(message);
}

// Step 2: Subscribe to stream BEFORE sending input
let mut rx = handle.subscribe();

// Step 3: Spawn task to process stream
tokio::spawn(async move {
    while let Ok(chunk) = rx.recv().await {
        match chunk {
            OutputChunk::TextDelta(text) => ui.append_text(text),
            OutputChunk::ToolStart { name, .. } => ui.show_tool(name),
            OutputChunk::Done => {
                ui.enable_input();
                // Optionally refresh from disk for consistency
                let updated = AgentSession::get_history("session-id").ok();
                if let Some(history) = updated {
                    ui.refresh_history(history);
                }
            }
            _ => {}
        }
    }
});

// Step 4: Send user input
handle.send_input(user_text).await?;
```

#### Why This Matters

- **Stream** provides real-time responsiveness for UIs
- **History** provides ground truth for persistence and reliability
- Messages are written to disk immediately (write-through persistence)
- Stream is ephemeral - if no one is subscribed, chunks are lost
- History is durable - always available on disk

#### Multiple Agent Turns

A single user input can trigger multiple LLM calls. For example:
```
User: "Write hello.rs and run it"
  → Turn 1: LLM writes file (streams text + ToolUse)
  → Turn 2: LLM runs file (streams text + ToolUse)
  → Turn 3: LLM shows results (streams text only, Done)
```

All turns stream continuously to subscribers until `OutputChunk::Done` is received.

---

### Image and PDF Support

The Read tool supports images and PDFs for Claude's vision and document understanding capabilities.

#### Supported Formats

- **Images**: PNG, JPEG, GIF, WebP (max 5MB)
- **PDFs**: PDF documents (max 32MB)
- **Text**: All other files (no size limit, with offset/limit support)

#### Automatic Detection

File type is detected automatically based on extension:

```rust
// Read an image - automatically returns image content
handle.send_input("Read screenshot.png").await?;
// LLM receives the image and can analyze it visually

// Read a PDF - automatically returns document content
handle.send_input("Read contract.pdf").await?;
// LLM receives the PDF and can extract information

// Read text - returns formatted text with line numbers
handle.send_input("Read main.rs").await?;
```

#### Tool Result Types

The framework handles three types of content:

```rust
pub enum ToolResultData {
    Text(String),                    // Regular text files
    Image {                          // PNG, JPEG, GIF, WebP
        data: Vec<u8>,
        media_type: String,
    },
    Document {                       // PDF files
        data: Vec<u8>,
        media_type: String,
        description: String,         // "PDF file read: /path/file.pdf (240.6KB)"
    },
}
```

#### API Format

Images and PDFs are automatically base64-encoded and sent to Claude:

**For Images:**
```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "iVBORw0KGg..."
  }
}
```

**For PDFs:**
```json
[
  {
    "type": "tool_result",
    "content": "PDF file read: /path/file.pdf (240.6KB)"
  },
  {
    "type": "document",
    "source": {
      "type": "base64",
      "media_type": "application/pdf",
      "data": "JVBERi0xLj..."
    }
  }
]
```

#### Size Limits and Caching

- Images and PDFs support prompt caching just like text content
- The last content block (text, image, or PDF) gets cache control applied
- This reduces costs when analyzing the same images/documents multiple times

---

### Attachment Support

The SDK supports file attachments in user messages using special tags. Users can include files directly in their messages, and the framework automatically reads and processes them.

#### Basic Usage

Include file attachments in user messages using the `<vibe-work-attachment>` tag:

```rust
let user_input = r#"Analyze this code:
<vibe-work-attachment>/path/to/main.rs</vibe-work-attachment>"#;

handle.send_input(user_input).await?;
```

Multiple attachments can be included in a single message and will be processed in order:

```rust
let input = r#"Analyze these files:
<vibe-work-attachment>/path/to/data.csv</vibe-work-attachment>
<vibe-work-attachment>/path/to/chart.png</vibe-work-attachment>
<vibe-work-attachment>/path/to/report.pdf</vibe-work-attachment>"#;

handle.send_input(input).await?;
```

#### Supported File Types

**Text Files:**
- Extensions: Any file not matching image or PDF extensions
- Behavior: Read as UTF-8 text with line numbers
- Limits: Maximum 2000 lines displayed, lines longer than 2000 chars truncated
- Format: `cat -n` style with file path header

Example output:
```
File: /path/to/script.py

     1	import sys
     2	import os
     3
     4	def main():
     5	    print("Hello, world!")
```

**Images:**
- Supported formats: PNG, JPEG, GIF, WebP
- Maximum size: 5MB per image
- Behavior: Base64-encoded and sent to Claude's vision API
- Extensions: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`

**PDF Documents:**
- Supported format: PDF
- Maximum size: 32MB per document
- Behavior: Base64-encoded and sent to Claude's document understanding API
- Extension: `.pdf`

**Directories:**
- Behavior: Lists directory contents (files and subdirectories)
- Format: Shows type (DIR/file), size, and name
- Sorting: Directories first, then files, alphabetically within each group

Example output:
```
Directory: /path/to/project

DIR               src
DIR               tests
         1.2 KB  README.md
         4.5 KB  Cargo.toml
       125.3 KB  main.rs
```

#### How It Works

1. Framework detects `<vibe-work-attachment>` tags in user input
2. Extracts file paths and reads each file (in order)
3. **Deduplicates** files - same file referenced multiple times is only processed once
4. Creates multi-block user message:
   - Original text (with tags preserved)
   - Content block for each attachment (text/image/document/directory)
5. If a file can't be read, inserts an error message block instead

#### Deduplication

The framework automatically detects and handles duplicate file references - especially useful when using `@` symbol references where the same file might be referenced multiple times:

```rust
let input = r#"Compare @main.rs with the original:
<vibe-work-attachment>./main.rs</vibe-work-attachment>
Also check @main.rs for bugs:
<vibe-work-attachment>./main.rs</vibe-work-attachment>"#;

// Agent receives:
// - Text block: Original message
// - Text block: Content of main.rs (first occurrence)
// - Text block: "Note: File ./main.rs was already attached above" (second occurrence)
```

Deduplication is based on resolved absolute paths and helps reduce token usage and API costs.

#### Path Resolution

- **Absolute paths**: Used as-is (`/home/user/file.txt`)
- **Relative paths**: Resolved from current working directory (`./data/file.txt`)

#### Error Handling

If a file cannot be read (doesn't exist, wrong format, exceeds size limit), an error block is inserted:

```
Error: Cannot read file /path/to/file.txt - No such file or directory
```

The agent receives this error and can respond appropriately to the user.

#### Examples

**Analyzing a Code File:**
```
Review this code for bugs:
<vibe-work-attachment>./src/main.rs</vibe-work-attachment>
```

**Processing an Image:**
```
What's in this screenshot?
<vibe-work-attachment>./screenshots/error.png</vibe-work-attachment>
```

**Multiple File Analysis:**
```
Compare these designs:
<vibe-work-attachment>./design-v1.png</vibe-work-attachment>
<vibe-work-attachment>./design-v2.png</vibe-work-attachment>
```

**PDF Document Review:**
```
Summarize this report:
<vibe-work-attachment>./reports/quarterly-2024.pdf</vibe-work-attachment>
```

#### Frontend Integration

The `<vibe-work-attachment>` tags are **preserved** in the message text, allowing frontends to:
- Parse and display attachment badges/pills in the UI
- Keep tags in messages sent to the backend
- Show attachment metadata (filename, type, size)

**Message Structure:**
When attachments are detected, the framework creates a multi-block user message:
1. Text block: Original user input (with tags intact)
2. Attachment blocks: One content block per attachment, in order
   - Text files → `ContentBlock::Text` with formatted content
   - Images → `ContentBlock::Image` with base64-encoded data
   - PDFs → `ContentBlock::Document` with base64-encoded data
   - Directories → `ContentBlock::Text` with directory listing
   - Errors → `ContentBlock::Text` with error description

**Code Location:**
- Attachment processor: `src/helpers/attachments.rs`
- Integration point: `src/agent/standard_loop.rs` (process_turn method)
- Public API: `crate::helpers::process_attachments()`

#### Limitations

1. **File size limits**: Images max 5MB, PDFs max 32MB, text files display only first 2000 lines
2. **File type detection**: Based on file extension only (not magic bytes)
3. **No streaming**: Attachments are read completely before processing starts
4. **Error recovery**: Individual attachment failures don't block the message

---

### Ask User Questions

The SDK includes an `AskUserQuestion` tool that allows agents to pause execution and ask users multiple-choice questions. This enables interactive workflows where the agent needs clarification or user preferences before proceeding.

#### Overview

The agent can ask 1-4 questions at a time, each with 2-4 options. Questions support both single-select (radio buttons) and multi-select (checkboxes) modes. The framework handles the communication flow between agent and frontend.

#### Message Flow

```
Agent                          Frontend
  |                                |
  |-- OutputChunk::AskUserQuestion ->|  (display question UI)
  |                                  |
  |<- InputMessage::UserQuestionResponse (user selects answers)
  |                                  |
  | (agent continues with answers)   |
```

#### Data Structures

**Output: `OutputChunk::AskUserQuestion`**

When the agent needs user input, it emits:

```rust
OutputChunk::AskUserQuestion {
    request_id: String,           // Unique ID to match request/response
    questions: Vec<UserQuestion>,
}
```

Each `UserQuestion` contains:

```rust
pub struct UserQuestion {
    pub question: String,         // Full question text
    pub header: String,           // Short label (max 12 chars)
    pub options: Vec<QuestionOption>,
    pub multi_select: bool,       // Allow multiple selections
}

pub struct QuestionOption {
    pub label: String,            // Display text (1-5 words)
    pub description: String,      // Explanation of the option
}
```

**Input: `InputMessage::UserQuestionResponse`**

After the user makes selections, send back:

```rust
InputMessage::UserQuestionResponse {
    request_id: String,                    // Must match request
    answers: HashMap<String, String>,      // header -> selected label(s)
}
```

For multi-select questions, join multiple labels with commas: `"Option A, Option B"`.

#### Example Usage

**Agent asks:**
```json
{
  "name": "AskUserQuestion",
  "input": {
    "questions": [{
      "question": "Which authentication method should we implement?",
      "header": "Auth",
      "multiSelect": false,
      "options": [
        { "label": "JWT (Recommended)", "description": "Stateless tokens, good for APIs" },
        { "label": "Session cookies", "description": "Traditional server-side sessions" },
        { "label": "OAuth 2.0", "description": "Third-party authentication" }
      ]
    }]
  }
}
```

**User responds with:**
```rust
InputMessage::UserQuestionResponse {
    request_id: "abc-123",
    answers: {
        "Auth": "JWT (Recommended)"
    }
}
```

**Agent receives:**
```
User responded with the following answers:
{
  "Auth": "JWT (Recommended)"
}
```

#### Tauri Integration

**1. Handle the output chunk:**

```rust
match chunk {
    OutputChunk::AskUserQuestion { request_id, questions } => {
        // Emit to frontend
        app_handle.emit_all("ask-user-question", serde_json::json!({
            "requestId": request_id,
            "questions": questions.iter().map(|q| serde_json::json!({
                "question": q.question,
                "header": q.header,
                "multiSelect": q.multi_select,
                "options": q.options.iter().map(|o| serde_json::json!({
                    "label": o.label,
                    "description": o.description,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })).unwrap();
    }
    // ... other cases
}
```

**2. Create command to send response:**

```rust
#[tauri::command]
async fn send_question_response(
    request_id: String,
    answers: HashMap<String, String>,
    state: tauri::State<'_, AgentState>,
) -> Result<(), String> {
    let handle = state.handle.lock().await;
    if let Some(h) = handle.as_ref() {
        h.send(InputMessage::UserQuestionResponse {
            request_id,
            answers,
        })
        .await
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
```

**3. Frontend (React/TypeScript):**

```typescript
interface QuestionOption {
  label: string;
  description: string;
}

interface UserQuestion {
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
}

// Listen for questions
listen<AskUserQuestionEvent>("ask-user-question", (event) => {
  setQuestions(event.payload.questions);
  setRequestId(event.payload.requestId);
  setShowQuestionModal(true);
});

// Send response when user submits
async function handleSubmit(answers: Record<string, string>) {
  await invoke("send_question_response", {
    requestId: requestId,
    answers: answers,
  });
  setShowQuestionModal(false);
}
```

#### Agent State

While waiting for a response, the agent enters:

```rust
AgentState::WaitingForUserInput { request_id: String }
```

You can check this state to show a loading/waiting indicator in the UI.

#### Validation Rules

- **1-4 questions** per request
- **2-4 options** per question
- Headers should be short (max 12 characters)
- Users can always provide custom "Other" input

#### Handling Interrupts

If the user cancels:

```rust
handle.interrupt().await?;
```

The tool returns an error result to the agent, which handles it gracefully.

---

### Interrupt Handling

The SDK provides comprehensive interrupt handling to allow users to gracefully stop agent execution at any point. When an interrupt is sent via `InputMessage::Interrupt`, the agent will stop processing and add a system message to the conversation history.

#### Interrupt Scenarios

The agent can be interrupted in three different scenarios, each handled appropriately:

##### 1. During LLM Streaming

When the agent is streaming a response from Claude, users can interrupt mid-generation.

**Behavior:**
- Preserves partial text that was already streamed
- Discards incomplete thinking blocks (to avoid signature issues)
- Removes ALL tool calls (both completed and partial)
- Adds `<system>User interrupted this message</system>` to the response
- Ends the turn

**Example:**
```rust
// User sends interrupt during streaming
handle.send_interrupt().await?;
```

**History result:**
```json
{"role":"user","content":"Write an essay"}
{"role":"assistant","content":[
  {"type":"thinking","thinking":"...completed thinking..."},
  {"type":"text","text":"Here is the partial essay text..."},
  {"type":"text","text":"<system>User interrupted this message</system>"}
]}
```

##### 2. During Permission Waiting

When the agent is waiting for user permission to execute a tool, users can interrupt instead of approving/denying.

**Behavior:**
- Returns `ToolResult::error("Interrupted")` for the pending tool
- Adds interrupt result to history
- Adds `<system>User interrupted this message</system>` assistant message
- Ends the turn (does NOT retry)

**Example:**
```rust
// While agent is waiting for permission
handle.send_interrupt().await?;
```

**History result:**
```json
{"role":"user","content":"Create a file"}
{"role":"assistant","content":[{"type":"tool_use","id":"...","name":"Write","input":{...}}]}
{"role":"user","content":[{"type":"tool_result","tool_use_id":"...","content":"Interrupted","is_error":true}]}
{"role":"assistant","content":"<system>User interrupted this message</system>"}
```

##### 3. During Tool Execution

When the agent is executing tools, users can interrupt between tool executions. **Important:** The currently executing tool will complete to avoid partial side effects.

**Behavior:**
- Lets the current tool complete execution
- Returns actual results for all completed tools
- Returns `ToolResult::error("Interrupted")` for unexecuted tools
- Adds `<system>User interrupted this message</system>` to history
- Ends the turn

**Example (3 tools, interrupted after tool 2):**
```json
{"role":"user","content":"Read three files"}
{"role":"assistant","content":[
  {"type":"tool_use","id":"1","name":"Read","input":{"file_path":"file1.txt"}},
  {"type":"tool_use","id":"2","name":"Read","input":{"file_path":"file2.txt"}},
  {"type":"tool_use","id":"3","name":"Read","input":{"file_path":"file3.txt"}}
]}
{"role":"user","content":[
  {"type":"tool_result","tool_use_id":"1","content":"actual file1 contents"},
  {"type":"tool_result","tool_use_id":"2","content":"actual file2 contents"},
  {"type":"tool_result","tool_use_id":"3","content":"Interrupted","is_error":true}
]}
{"role":"assistant","content":"<system>User interrupted this message</system>"}
```

#### Key Design Decisions

1. **Let tools complete** - Don't interrupt mid-tool execution to avoid partial side effects (half-written files, incomplete API calls, etc.)

2. **Preserve completed work** - Tools that finished before interrupt return their actual results

3. **Consistent interrupt message** - All scenarios add `<system>User interrupted this message</system>` to history for context

4. **No retries after interrupt** - Agent ends the turn immediately instead of trying to handle the interruption

#### Implementation Details

Interrupts are handled using `tokio::select!` for concurrent async operations:

```rust
// During streaming
loop {
    tokio::select! {
        event_result = stream.next() => {
            // Process stream events
        }
        msg = internals.receive() => {
            if let Some(InputMessage::Interrupt) = msg {
                // Handle interrupt
                break;
            }
        }
    }
}

// During tool execution
for (index, block) in content_blocks.iter().enumerate() {
    // Execute tool
    let result = execute_tool(...).await;

    // Check for interrupt (non-blocking)
    if interrupt_detected() {
        // Add "Interrupted" error for remaining tools
        break;
    }
}
```

#### Known Limitations

1. **Non-streaming LLM calls** - When using non-streaming mode (`streaming_enabled: false`), interrupts won't be detected during the LLM call itself. The interrupt will only be processed after the full response arrives. This primarily affects extended thinking scenarios.

2. **Long-running tools** - Individual tools that take a very long time will complete fully before the interrupt is detected. Future enhancement could pass cancellation tokens to tools for graceful internal interruption.

3. **Shutdown handling** - Currently only handles `InputMessage::Interrupt`. `InputMessage::Shutdown` could be handled similarly for graceful shutdown during streaming/execution.

---

### MCP (Model Context Protocol) Integration

The SDK includes comprehensive support for MCP (Model Context Protocol), allowing your agents to access tools from external MCP servers. This enables seamless integration with a growing ecosystem of MCP tools and services.

#### What is MCP?

MCP is an open protocol that standardizes how AI applications connect to external tools and data sources. MCP servers provide tools that can be dynamically discovered and used by AI agents. Learn more at [modelcontextprotocol.io](https://modelcontextprotocol.io).

#### Key Features

- **Dynamic Tool Discovery** - Automatically fetch and expose tools from any MCP server
- **Tool Namespacing** - Avoid conflicts with `server_id__tool_name` format
- **JWT Refresh Support** - Built-in callback system for expiring tokens
- **Transparent Integration** - MCP tools appear identical to native tools
- **Multiple Servers** - Connect to multiple MCP servers simultaneously
- **Thread-Safe** - Safe to use across concurrent agents

#### Quick Start

**1. Basic Setup (No Auth)**

```rust
use shadow_agent_sdk::mcp::{MCPServerManager, MCPToolProvider};
use shadow_agent_sdk::tools::ToolRegistry;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use std::sync::Arc;

// Create transport
let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp");
let service = ().serve(transport).await?;

// Add to MCP manager
let mcp_manager = Arc::new(MCPServerManager::new());
mcp_manager.add_service("filesystem", service).await?;

// Create tool provider and add to registry
let mcp_provider = Arc::new(MCPToolProvider::new(mcp_manager));
let mut tool_registry = ToolRegistry::new();
tool_registry.add_provider(mcp_provider).await?;

// Now all MCP tools are available!
let tools = Arc::new(tool_registry);
```

**2. With Static Auth Headers**

```rust
// Create transport with auth
let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp")
    .with_header("Authorization", "Bearer static-token")
    .with_header("X-Api-Key", "your-api-key");

let service = ().serve(transport).await?;
mcp_manager.add_service("filesystem", service).await?;
```

**3. With JWT Refresh (Recommended for Proxied Servers)**

For servers requiring JWT tokens that expire, use the service refresher callback:

```rust
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// Track when we last refreshed
let last_refresh = Arc::new(RwLock::new(Instant::now()));
let jwt_provider = Arc::new(YourJwtProvider::new());

// Create refresher callback (called before EVERY MCP operation)
let refresher = {
    let last_refresh = last_refresh.clone();
    let jwt = jwt_provider.clone();

    move || {
        let last_refresh = last_refresh.clone();
        let jwt = jwt.clone();

        async move {
            // Check if we need to refresh (e.g., every 50 minutes for 1hr JWT)
            let mut last = last_refresh.write().await;
            if last.elapsed() < Duration::from_secs(50 * 60) {
                return Ok(None); // Still valid, no refresh needed
            }

            // Get fresh JWT and create new service
            let token = jwt.get_fresh_token().await?;
            let transport = StreamableHttpClientTransport::from_uri("https://backend/mcp-proxy")
                .with_header("Authorization", format!("Bearer {}", token));
            let service = ().serve(transport).await?;

            *last = Instant::now();
            Ok(Some(service)) // Replace with new service
        }
    }
};

// Create initial service
let initial_token = jwt_provider.get_fresh_token().await?;
let transport = StreamableHttpClientTransport::from_uri("https://backend/mcp-proxy")
    .with_header("Authorization", format!("Bearer {}", initial_token));
let service = ().serve(transport).await?;

// Add to manager with refresher
mcp_manager.add_service_with_refresher("remote-mcp", service, refresher).await?;
```

#### How It Works

**Service Refresher Pattern**

The refresher callback is called **before every MCP operation** (`list_tools`, `call_tool`):
- Returns `Ok(None)` if service is still valid (no refresh)
- Returns `Ok(Some(new_service))` to replace the service
- Returns `Err(...)` if refresh failed (continues with existing service)

This pattern mirrors the LLM auth provider and gives you complete control over token refresh logic.

**Automatic Reconnection**

When an MCP server crashes or restarts, the framework automatically handles reconnection:
1. Before every tool call, a health check (`list_tools`) verifies the server is alive (5-second timeout)
2. If the health check fails, the old service is dropped
3. The refresher is called to create a new service
4. The operation is retried with the new connection (up to 3 attempts)
5. All of this happens transparently - tools just work

**Thread Safety**

- Service is wrapped in `Arc<RwLock<...>>` for safe concurrent access
- Read locks allow multiple agents to use the service simultaneously
- Write lock (for refresh) waits for all reads to complete before swapping service
- In-flight tool calls complete with old service, new calls use new service

**Tool Namespacing**

MCP tools are automatically namespaced to avoid conflicts:
- **Server ID**: `filesystem`
- **Original tool name**: `read_file`
- **Exposed to agent**: `filesystem__read_file`

The double underscore (`__`) is used because Anthropic's API only allows `[a-zA-Z0-9_-]` in tool names.

#### Complete Example

```rust
use shadow_agent_sdk::{
    agent::{AgentConfig, StandardAgent},
    llm::AnthropicProvider,
    mcp::{MCPServerManager, MCPToolProvider},
    runtime::AgentRuntime,
    session::AgentSession,
    tools::ToolRegistry,
};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create LLM provider
    let llm = Arc::new(AnthropicProvider::from_env()?);

    // Create MCP manager
    let mcp_manager = Arc::new(MCPServerManager::new());

    // Connect to MCP server
    let transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp");
    let service = ().serve(transport).await?;
    mcp_manager.add_service("filesystem", service).await?;

    // Create tool registry with MCP provider
    let mut tool_registry = ToolRegistry::new();
    let mcp_provider = Arc::new(MCPToolProvider::new(mcp_manager));
    tool_registry.add_provider(mcp_provider).await?;
    let tools = Arc::new(tool_registry);

    println!("Available tools: {:?}", tools.tool_names());
    // Output: ["filesystem__read_file", "filesystem__write_file", ...]

    // Create agent
    let runtime = AgentRuntime::new();
    let session = AgentSession::new("mcp-session", "assistant", "MCP Agent", "")?;

    let config = AgentConfig::new("You are a helpful assistant with access to filesystem tools.")
        .with_tools(tools)
        .with_streaming(true);

    let agent = StandardAgent::new(config, llm);
    let handle = runtime.spawn(session, |internals| agent.run(internals)).await;

    // Use MCP tools!
    handle.send_input("Read the file ./README.md using filesystem__read_file").await?;

    // Process output
    let mut rx = handle.subscribe();
    while let Ok(chunk) = rx.recv().await {
        match chunk {
            OutputChunk::TextDelta(text) => print!("{}", text),
            OutputChunk::Done => break,
            _ => {}
        }
    }

    Ok(())
}
```

#### Tauri Integration

For Tauri apps with a backend proxy server that requires JWT authentication:

```rust
// In your Tauri backend

pub struct JwtProvider {
    auth_service: Arc<YourAuthService>,
    cached_token: Arc<RwLock<Option<CachedToken>>>,
}

struct CachedToken {
    token: String,
    expires_at: Instant,
}

impl JwtProvider {
    pub async fn get_fresh_token(&self) -> Result<String> {
        // Check cache first
        {
            let cache = self.cached_token.read().await;
            if let Some(cached) = cache.as_ref() {
                if cached.expires_at > Instant::now() + Duration::from_secs(5 * 60) {
                    return Ok(cached.token.clone());
                }
            }
        }

        // Refresh token
        let mut cache = self.cached_token.write().await;
        let new_token = self.auth_service.refresh_jwt().await?;
        let expires_at = Instant::now() + Duration::from_secs(60 * 60); // 1 hour

        *cache = Some(CachedToken {
            token: new_token.clone(),
            expires_at,
        });

        Ok(new_token)
    }
}

// Setup MCP with JWT refresh
let jwt_provider = Arc::new(JwtProvider::new(auth_service));

let refresher = {
    let jwt = jwt_provider.clone();
    let last_refresh = Arc::new(RwLock::new(Instant::now()));

    move || {
        let jwt = jwt.clone();
        let last_refresh = last_refresh.clone();

        async move {
            let mut last = last_refresh.write().await;

            // Refresh every 50 minutes (for 1hr tokens)
            if last.elapsed() < Duration::from_secs(50 * 60) {
                return Ok(None);
            }

            let token = jwt.get_fresh_token().await?;
            let transport = StreamableHttpClientTransport::from_uri("https://your-backend.com/mcp-proxy")
                .with_header("Authorization", format!("Bearer {}", token));
            let service = ().serve(transport).await?;

            *last = Instant::now();
            Ok(Some(service))
        }
    }
};

// Create initial service
let initial_token = jwt_provider.get_fresh_token().await?;
let transport = StreamableHttpClientTransport::from_uri("https://your-backend.com/mcp-proxy")
    .with_header("Authorization", format!("Bearer {}", initial_token));
let service = ().serve(transport).await?;

// Add to manager
mcp_manager.add_service_with_refresher("remote-mcp", service, refresher).await?;
```

#### Multiple MCP Servers

You can connect to multiple MCP servers simultaneously:

```rust
// Local filesystem server
let fs_transport = StreamableHttpClientTransport::from_uri("http://localhost:8005/mcp");
let fs_service = ().serve(fs_transport).await?;
mcp_manager.add_service("filesystem", fs_service).await?;

// Remote database server with auth
let db_transport = StreamableHttpClientTransport::from_uri("https://db-server.com/mcp")
    .with_header("Authorization", "Bearer db-token");
let db_service = ().serve(db_transport).await?;
mcp_manager.add_service("database", db_service).await?;

// Web search server
let search_transport = StreamableHttpClientTransport::from_uri("https://search-api.com/mcp")
    .with_header("X-Api-Key", "search-key");
let search_service = ().serve(search_transport).await?;
mcp_manager.add_service("websearch", search_service).await?;

// Tools will be namespaced:
// - filesystem__read_file, filesystem__write_file
// - database__query, database__insert
// - websearch__search, websearch__get_page
```

#### Error Handling

If an MCP server is unavailable or a tool call fails:

```rust
// Server connection failure
match mcp_manager.add_service("broken", service).await {
    Ok(_) => println!("Connected!"),
    Err(e) => eprintln!("Failed to connect: {}", e),
}

// Tool call failure (handled automatically by framework)
// The agent receives a ToolResult::error() and can respond accordingly
```

#### Best Practices

1. **Always use a service refresher** - The refresher is now mandatory for all MCP servers
2. **Implement caching in your refresher** - Check timestamps before creating new services
3. **Cache tokens in your JWT provider** - Avoid unnecessary token refresh calls
4. **Set refresh interval below expiry** - E.g., 5 minutes for 10-minute tokens, 50 minutes for 60-minute tokens
5. **Handle refresh failures gracefully** - Return `Err(...)` if refresh fails (framework continues with existing service)
6. **Test with server restarts** - The framework automatically reconnects when servers crash
7. **Use descriptive server IDs** - They appear in tool names (`server_id__tool_name`)

#### Debugging

Enable debug mode to see MCP operations:

```rust
let config = AgentConfig::new("...")
    .with_debug(true);

// Logs show:
// - [MCPServer] Created MCP server 'filesystem'
// - [MCPServer] HEALTH CHECK before calling tool 'read_file' on 'filesystem'
// - [MCPServer] ✓ HEALTH CHECK PASSED for 'filesystem' - proceeding with tool call
// - [MCPServer] Calling tool 'read_file' on server 'filesystem'
// - [MCPServer] ✓ Tool 'read_file' executed successfully on 'filesystem'

// When server crashes and restarts:
// - [MCPServer] ✗ list_tools FAILED for 'filesystem': connection error
// - [MCPServer] Will attempt FORCED RECONNECTION for 'filesystem' before retry
// - [MCPServer] Dropped old service for 'filesystem'
// - [MCPServer] Calling refresher callback for 'filesystem' to FORCE reconnect...
// - [MCPServer] ✓ Refresher provided new service for 'filesystem'
// - [MCPServer] ✓ New service installed for 'filesystem'
// - [MCPServer] ✓ list_tools SUCCESS for 'filesystem' - got 5 tools
```

#### API Reference

**MCPServerManager**

```rust
// Add server with refresher (refresher is mandatory)
mcp_manager.add_server_with_refresher(id, refresher).await?;

// Add from config (convenience - creates simple refresher with caching)
mcp_manager.add_server(MCPServerConfig::new(id, uri)).await?;

// Management
mcp_manager.get_server(id).await;
mcp_manager.server_ids().await;
mcp_manager.server_count().await;
mcp_manager.reconnect_server(id).await?; // Note: reconnection is automatic on failures
mcp_manager.health_check_all().await;
```

**MCPServer**

```rust
// Create with refresh callback (refresher is mandatory)
let server = MCPServer::new(id, refresher);

// Operations (automatic health checks and reconnection)
server.list_tools().await?;        // Includes retry logic with reconnection
server.call_tool(name, arguments).await?; // Health check + retry logic
server.health_check().await?;      // Explicit health check
```

**MCPToolProvider**

```rust
let provider = MCPToolProvider::new(mcp_manager);
tool_registry.add_provider(Arc::new(provider)).await?;

// Implements ToolProvider trait:
provider.get_tools().await?;  // Returns all MCP tools
provider.refresh().await?;     // Refreshes tool list
provider.name();               // Returns "mcp"
provider.is_dynamic();         // Returns true
```

#### See Also

- Example: `examples/mcp_agent/` - Complete MCP agent example
- rmcp documentation: [crates.io/crates/rmcp](https://crates.io/crates/rmcp)
- MCP specification: [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## Module Reference

### Runtime Module

The runtime manages agent lifecycles and provides handles for communication.

#### AgentRuntime

```rust
use shadow_agent_sdk::runtime::AgentRuntime;

// Create runtime
let runtime = AgentRuntime::new();

// Or with global permission rules
let runtime = AgentRuntime::with_global_rules(vec![
    PermissionRule::allow_tool("Read"),
]);

// Spawn an agent
let handle = runtime.spawn(session, |internals| agent.run(internals)).await;

// Spawn with local permission rules
let handle = runtime.spawn_with_local_rules(
    session,
    vec![PermissionRule::allow_tool("Read")],
    |internals| agent.run(internals),
).await;

// Get existing agent
if let Some(handle) = runtime.get("session-id").await {
    handle.send_input("Hello").await?;
}

// List running agents
let running = runtime.list_running().await; // Vec<String>

// Shutdown
runtime.shutdown("session-id").await;
runtime.shutdown_all().await;
```

#### AgentHandle

```rust
use shadow_agent_sdk::runtime::AgentHandle;
use shadow_agent_sdk::core::OutputChunk;

// Send input
handle.send_input("Write hello world").await?;

// Send permission response
handle.send_permission_response("Bash".to_string(), true, false).await?;
// (tool_name, allowed, remember)

// Subscribe to output (do this BEFORE sending input!)
let mut rx = handle.subscribe();

// Process output
while let Ok(chunk) = rx.recv().await {
    match chunk {
        OutputChunk::TextDelta(text) => print!("{}", text),
        OutputChunk::TextComplete(full_text) => {},
        OutputChunk::ThinkingDelta(text) => {},
        OutputChunk::ToolStart { id, name, input } => {},
        OutputChunk::ToolEnd { id, result } => {},
        OutputChunk::PermissionRequest { tool_name, action, input, details } => {
            // Show UI prompt, then:
            handle.send_permission_response(tool_name, true, false).await?;
        },
        OutputChunk::SubAgentSpawned { session_id, agent_type } => {},
        OutputChunk::SubAgentComplete { session_id, result } => {},
        OutputChunk::StateChange(state) => {},
        OutputChunk::Error(message) => {},
        OutputChunk::Done => break,
        _ => {}
    }
}

// Check state
let state = handle.state().await;
let is_busy = handle.is_processing().await;
let is_done = handle.is_done().await;

// Control
handle.interrupt().await;  // Cancel current operation
handle.shutdown().await;   // Stop agent entirely

// Access session metadata (even while agent is running!)
handle.set_custom_metadata("working_folder", "/path/to/folder").await?;
let folder = handle.get_custom_metadata("working_folder").await;

handle.set_conversation_name("Debug Python script").await?;
let name = handle.conversation_name().await;
```

**Session Metadata Access**: The handle now provides direct access to the running agent's session metadata. This allows you to update metadata (like working folder, user preferences, etc.) while the agent is running without race conditions.

### Session Module

Sessions persist conversation history and metadata.

#### AgentSession

```rust
use shadow_agent_sdk::session::{AgentSession, SessionStorage};

// Create new session (auto-persists to ./sessions/{id}/)
let session = AgentSession::new(
    "unique-session-id",
    "coder",              // agent_type
    "Code Assistant",     // name
    "Helps with coding",  // description
)?;

// Create subagent session (linked to parent)
let subagent_session = AgentSession::new_subagent(
    "sub-session-id",
    "researcher",
    "Research Helper",
    "Finds information",
    "parent-session-id",  // parent_session_id
    "tool_use_123",       // parent_tool_use_id
)?;

// Load existing session
let session = AgentSession::load("session-id")?;

// Access properties
let id = session.session_id();
let history = session.history();  // &[Message]
let is_sub = session.is_subagent();
let parent = session.parent_session_id();  // Option<&str>
let children = session.child_session_ids();  // &[String]

// Add message manually
session.add_message(Message::user("Hello"))?;

// Save changes
session.save()?;

// Delete session
session.delete()?;
```

#### Listing Sessions

```rust
// List all session IDs
let all = AgentSession::list_all()?;

// List only top-level sessions (not subagents)
let top_level = AgentSession::list_top_level()?;

// List with filter
let filtered = AgentSession::list_filtered(true)?;  // true = top-level only

// List with metadata (more efficient if you need metadata)
let with_meta = AgentSession::list_with_metadata(true)?;
// Returns: Vec<(String, SessionMetadata)>

for (session_id, metadata) in with_meta {
    println!("{}: {} ({})", session_id, metadata.name, metadata.agent_type);
    println!("  Created: {}", metadata.created_at);
    println!("  Is subagent: {}", metadata.is_subagent());
}
```

#### Getting History

```rust
// Get conversation history without loading full session
let history = AgentSession::get_history("session-id")?;
// Returns: Vec<Message>

// Get just metadata
let metadata = AgentSession::get_metadata("session-id")?;
// Returns: SessionMetadata

// Check existence
if AgentSession::exists("session-id") {
    // ...
}
```

#### Updating Session Metadata (Running Agents)

When an agent is running, you should update metadata through the `AgentHandle` to avoid race conditions:

```rust
// Get the handle for a running agent
if let Some(handle) = runtime.get("session-id").await {
    // Agent is running - update via handle
    handle.set_custom_metadata("working_folder", "/path/to/project").await?;
    handle.set_custom_metadata("user_preferences", serde_json::json!({
        "theme": "dark",
        "language": "en"
    })).await?;

    // Read metadata
    let folder = handle.get_custom_metadata("working_folder").await;

    // Set conversation name
    handle.set_conversation_name("Debugging Python script").await?;
} else {
    // Agent not running - update disk directly
    let mut metadata = SessionStorage::default().load_metadata("session-id")?;
    metadata.set_custom("working_folder", "/path/to/project");
    SessionStorage::default().save_metadata(&metadata)?;
}
```

**Why use the handle?** When an agent is running, it has an in-memory copy of the session. If you modify the disk directly, the agent will overwrite your changes on the next save. The handle provides thread-safe access to the shared session.

#### Conversation Naming

Sessions support an optional conversation name that describes the conversation content:

```rust
// Set conversation name (typically after first turn)
session.set_conversation_name("Help with Rust debugging")?;

// Get conversation name
if let Some(name) = session.conversation_name() {
    println!("Conversation: {}", name);
}

// Check if named
if !session.has_conversation_name() {
    // Generate name using a conversation namer helper
}
```

#### Custom Storage Location

```rust
let storage = SessionStorage::with_dir("/custom/path");
let session = AgentSession::new_with_storage(
    "session-id", "type", "name", "desc", storage
)?;
```

### Tools Module

Tools extend agent capabilities with custom actions.

#### Built-in Tools

```rust
use shadow_agent_sdk::tools::{ToolRegistry, common::*};

let mut tools = ToolRegistry::new();

// File operations
tools.register(ReadTool::new()?);       // Read text files, images (PNG/JPEG/GIF/WebP), and PDFs
tools.register(WriteTool::new()?);      // Write/create files
tools.register(EditTool::new()?);       // Edit existing files
tools.register(GlobTool::new()?);       // Find files by pattern
tools.register(GrepTool::new()?);       // Search file contents

// Shell
tools.register(BashTool::new()?);       // Execute commands

// Task management
tools.register(TodoWriteTool::new()?);  // Manage task lists

let tools = Arc::new(tools);
```

#### Creating Custom Tools

```rust
use async_trait::async_trait;
use shadow_agent_sdk::tools::{Tool, ToolResult, ToolInfo};
use shadow_agent_sdk::llm::{ToolDefinition, types::CustomTool};
use shadow_agent_sdk::runtime::AgentInternals;
use serde_json::{json, Value};

pub struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "GetWeather"
    }

    fn description(&self) -> &str {
        "Get current weather for a location"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::Custom(CustomTool {
            name: self.name().to_string(),
            description: Some(self.description().to_string()),
            input_schema: shadow_agent_sdk::llm::types::ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    }
                })),
                required: Some(vec!["location".to_string()]),
            },
            tool_type: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let location = input.get("location")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        ToolInfo {
            name: self.name().to_string(),
            action_description: format!("Get weather for {}", location),
            details: None,
        }
    }

    fn requires_permission(&self) -> bool {
        false  // Safe tool, no permission needed
    }

    async fn execute(
        &self,
        input: &Value,
        internals: &mut AgentInternals,
    ) -> anyhow::Result<ToolResult> {
        let location = input.get("location")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing location"))?;

        // Access agent context if needed
        let session_id = internals.context.session_id.clone();

        // Your implementation here...
        let weather = fetch_weather(location).await?;

        Ok(ToolResult::success(format!("Weather in {}: {}", location, weather)))
    }
}
```

#### Tools That Spawn Subagents

```rust
async fn execute(&self, input: &Value, internals: &mut AgentInternals) -> Result<ToolResult> {
    let parent_session = internals.session.session_id().to_string();
    let tool_use_id = internals.context.current_tool_use_id
        .clone()
        .unwrap_or_default();

    // Create subagent session
    let sub_session = AgentSession::new_subagent(
        format!("sub-{}", tool_use_id),
        "researcher",
        "Research Agent",
        "Researches topics",
        &parent_session,
        &tool_use_id,
    )?;

    // Notify parent of spawn
    internals.send(OutputChunk::SubAgentSpawned {
        session_id: sub_session.session_id().to_string(),
        agent_type: "researcher".to_string(),
    });

    // Get runtime from context and spawn
    if let Some(runtime) = internals.context.get_resource::<AgentRuntime>() {
        let config = AgentConfig::new("You are a researcher...");
        let agent = StandardAgent::new(config, self.llm.clone());

        let handle = runtime
            .spawn(sub_session, |internals| agent.run(internals))
            .await;

        // Send task to subagent
        handle.send_input("Research topic X").await?;

        // Wait for completion
        let mut rx = handle.subscribe();
        let mut result = String::new();

        while let Ok(chunk) = rx.recv().await {
            match chunk {
                OutputChunk::TextDelta(text) => result.push_str(&text),
                OutputChunk::Done => break,
                _ => {}
            }
        }

        // Notify completion
        internals.send(OutputChunk::SubAgentComplete {
            session_id: handle.session_id().to_string(),
            result: Some(result.clone()),
        });

        Ok(ToolResult::success(result))
    } else {
        Ok(ToolResult::error("Runtime not available"))
    }
}
```

### Permissions Module

Three-tier permission system for controlling tool access.

#### Permission Tiers

```
1. Session Rules   - Highest priority, in-memory only
2. Local Rules     - Agent-type specific, persisted
3. Global Rules    - All agents, persisted

Check order: Session → Local → Global → Ask User (if interactive)
```

#### Permission Rules

```rust
use shadow_agent_sdk::permissions::{PermissionRule, PermissionScope};

// Allow entire tool
let rule = PermissionRule::allow_tool("Read");

// Allow commands with specific prefix
let rule = PermissionRule::allow_prefix("Bash", "git ");
let rule = PermissionRule::allow_prefix("Bash", "npm ");

// Add rules at different scopes
runtime.global_permissions().add_rule(
    PermissionRule::allow_tool("Read"),
    PermissionScope::Global,
);

// Or during spawn
let handle = runtime.spawn_with_local_rules(
    session,
    vec![
        PermissionRule::allow_tool("Read"),
        PermissionRule::allow_prefix("Bash", "cd "),
    ],
    |internals| agent.run(internals),
).await;
```

#### Handling Permission Requests

```rust
// In your output handler
match chunk {
    OutputChunk::PermissionRequest { tool_name, action, input, details } => {
        // Show UI to user
        println!("Tool '{}' wants to: {}", tool_name, action);
        println!("Input: {}", input);

        // Get user decision (true = allow, false = deny)
        let allowed = show_permission_dialog(&action);
        let remember = ask_if_remember();

        // Send response
        handle.send_permission_response(tool_name, allowed, remember).await?;
    }
    _ => {}
}
```

### Hooks Module

Intercept and modify agent behavior at key points.

#### Hook Events

```rust
use shadow_agent_sdk::hooks::{HookRegistry, HookEvent, HookContext, HookResult};

let mut hooks = HookRegistry::new();

// PreToolUse - Before tool executes (can block, allow, or modify)
hooks.add(HookEvent::PreToolUse, |ctx: &mut HookContext| {
    println!("About to use tool: {}", ctx.tool_name.as_ref().unwrap());
    HookResult::none()  // Continue normal flow
})?;

// PostToolUse - After successful execution
hooks.add(HookEvent::PostToolUse, |ctx: &mut HookContext| {
    println!("Tool completed: {:?}", ctx.tool_result);
    HookResult::none()
})?;

// PostToolUseFailure - After tool fails
hooks.add(HookEvent::PostToolUseFailure, |ctx: &mut HookContext| {
    println!("Tool failed: {:?}", ctx.error);
    HookResult::none()
})?;

// UserPromptSubmit - When user sends a message
hooks.add(HookEvent::UserPromptSubmit, |ctx: &mut HookContext| {
    println!("User said: {:?}", ctx.user_prompt);
    HookResult::none()
})?;
```

#### Pattern-Based Hooks

```rust
// Only match specific tools
hooks.add_with_pattern(HookEvent::PreToolUse, "Bash", |ctx| {
    // Only runs for Bash tool
    let input = ctx.tool_input.as_ref().unwrap();
    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
        if cmd.contains("rm -rf /") {
            return HookResult::deny("Dangerous command blocked");
        }
    }
    HookResult::none()
})?;

// Regex patterns
hooks.add_with_pattern(HookEvent::PreToolUse, "Write|Edit", |ctx| {
    // Runs for Write or Edit tools
    HookResult::none()
})?;
```

#### Hook Results

```rust
// Allow - Skip permission check, execute immediately
HookResult::allow()

// Deny - Block execution with reason
HookResult::deny("Not allowed")

// Ask - Use normal permission flow
HookResult::ask()

// None - Continue normal flow (check permissions)
HookResult::none()
```

#### Modifying Tool Input

```rust
hooks.add(HookEvent::PreToolUse, |ctx: &mut HookContext| {
    if let Some(input) = ctx.tool_input.as_mut() {
        // Modify the input before execution
        if let Some(obj) = input.as_object_mut() {
            obj.insert("modified".to_string(), json!(true));
        }
    }
    HookResult::none()
})?;
```

### LLM Module

The SDK provides a pluggable LLM provider architecture, allowing you to use different LLM backends interchangeably.

**Migration Note**: If you're upgrading from an older version, the main changes are:
- `StandardAgent::new()` now takes `Arc<dyn LlmProvider>` instead of `Arc<AnthropicProvider>`
- `AnthropicProvider::from_env()` now requires the `ANTHROPIC_MODEL` environment variable
- Wrap your provider in `Arc<dyn LlmProvider>` for type compatibility
- See `docs/MIGRATION_LLM_PROVIDER.md` for detailed migration guide

#### LlmProvider Trait

All providers implement the `LlmProvider` trait:

```rust
use shadow_agent_sdk::llm::{LlmProvider, AnthropicProvider, GeminiProvider};

// Create providers
let anthropic: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::from_env()?);
let gemini: Arc<dyn LlmProvider> = Arc::new(GeminiProvider::from_env()?);

// Use with StandardAgent - both work the same way
let agent = StandardAgent::new(config, anthropic);
// or
let agent = StandardAgent::new(config, gemini);
```

#### AnthropicProvider

```rust
use shadow_agent_sdk::llm::AnthropicProvider;

// From environment variables (ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
let llm = AnthropicProvider::from_env()?;

// With explicit API key and model
let llm = AnthropicProvider::new("sk-ant-...")?
    .with_model("claude-sonnet-4-5@20250929");

// With custom max tokens
let llm = AnthropicProvider::from_env()?
    .with_max_tokens(8192);

// Wrap in Arc for sharing
let llm: Arc<dyn LlmProvider> = Arc::new(llm);
```

#### GeminiProvider

```rust
use shadow_agent_sdk::llm::GeminiProvider;

// From environment variables (GEMINI_API_KEY, GEMINI_MODEL)
let llm = GeminiProvider::from_env()?;

// With explicit API key and model
let llm = GeminiProvider::new("AIza...")?
    .with_model("gemini-3-flash-preview");

// With custom max tokens
let llm = GeminiProvider::from_env()?
    .with_max_tokens(8192);

// Wrap in Arc for sharing
let llm: Arc<dyn LlmProvider> = Arc::new(llm);
```

**Note**: GeminiProvider translates between the framework's internal message format (Anthropic-style) and Gemini's API format internally. From the agent's perspective, both providers work identically.

**Session Tracking**: The framework automatically tracks which model and provider are being used for each session in the session metadata (`model` and `provider` fields). This is especially useful when using `SwappableLlmProvider`, as the session metadata will reflect the currently active provider.

#### Creating Lightweight Variants

All providers support creating lightweight variants for specific tasks (like conversation naming):

```rust
// Create main provider
let main_llm: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::from_env()?);

// Create a Haiku variant for fast, cheap naming (shares auth config)
let naming_llm = main_llm.create_variant("claude-3-5-haiku-20241022", 1024);

// Use separate models for agent and naming
let config = AgentConfig::new("You are helpful")
    .with_naming_llm(naming_llm);

let agent = StandardAgent::new(config, main_llm);
```

#### SwappableLlmProvider

For runtime model switching (e.g., fast/pro toggle in UI):

```rust
use shadow_agent_sdk::llm::{SwappableLlmProvider, GeminiProvider, LlmProvider};

// Create initial provider
let fast = Arc::new(GeminiProvider::new("key")?.with_model("gemini-3-flash-preview"));
let swappable = SwappableLlmProvider::new(fast);

// Get a handle for external switching
let handle = swappable.handle();

// Use with agent (agent sees Arc<dyn LlmProvider>)
let llm: Arc<dyn LlmProvider> = Arc::new(swappable);
let agent = StandardAgent::new(config, llm);

// Later, switch to pro model (from UI handler, etc.)
let pro = Arc::new(GeminiProvider::new("key")?.with_model("gemini-3-pro-preview"));
handle.set_provider(pro).await;
```

#### Dynamic Authentication

For JWT tokens, rotating keys, or proxy servers:

```rust
use shadow_agent_sdk::llm::{AnthropicProvider, AuthConfig, AuthProvider};
use async_trait::async_trait;

struct MyAuthProvider {
    // Your auth state
}

#[async_trait]
impl AuthProvider for MyAuthProvider {
    async fn get_auth(&self) -> anyhow::Result<AuthConfig> {
        // Fetch/refresh credentials
        let token = refresh_jwt_token().await?;
        Ok(AuthConfig::new(token))
    }
}

let llm = AnthropicProvider::with_auth_provider_boxed(Arc::new(MyAuthProvider { ... }));
// Or use the callback-based version:
let llm = AnthropicProvider::with_auth_provider(|| async {
    let token = refresh_jwt_token().await?;
    Ok(AuthConfig::new(token))
});
```

#### Extended Thinking

Extended thinking is currently supported by AnthropicProvider using Claude's extended thinking API.

```rust
use shadow_agent_sdk::llm::ThinkingConfig;

let config = AgentConfig::new("You are a thoughtful assistant.")
    .with_thinking(16000);  // Budget in tokens

// Or with config object
let config = AgentConfig::new("...")
    .with_thinking_config(ThinkingConfig::enabled(32000));
```

#### Message Types

```rust
use shadow_agent_sdk::llm::{Message, MessageContent, ContentBlock};

// Simple text message
let msg = Message::user("Hello");
let msg = Message::assistant("Hi there!");

// With content blocks
let msg = Message::user_with_blocks(vec![
    ContentBlock::text("Analyze this"),
]);

// Access content
match &message.content {
    MessageContent::Text(text) => println!("{}", text),
    MessageContent::Blocks(blocks) => {
        for block in blocks {
            match block {
                ContentBlock::Text { text } => println!("{}", text),
                ContentBlock::ToolUse { id, name, input } => {
                    println!("Tool: {} ({})", name, id);
                }
                ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                    println!("Result for {}: {:?}", tool_use_id, content);
                }
                ContentBlock::Thinking { thinking, .. } => {
                    println!("Thinking: {}", thinking);
                }
                _ => {}
            }
        }
    }
}
```

### Helpers Module

Utilities for common agent patterns.

#### Context Injection

Modify messages before each LLM call:

```rust
use shadow_agent_sdk::helpers::{ContextInjection, InjectionChain, FnInjection};

// Function-based injection
let injection = FnInjection::new("add_timestamp", |internals, messages| {
    let timestamp = chrono::Utc::now().to_rfc3339();
    shadow_agent_sdk::helpers::inject_system_reminder(
        messages,
        &format!("Current time: {}", timestamp),
    );
});

// Chain multiple injections
let mut chain = InjectionChain::new();
chain.add(injection);
chain.add_fn("add_context", |internals, messages| {
    // Access session state
    let turn = internals.context.current_turn;
    shadow_agent_sdk::helpers::inject_system_reminder(
        messages,
        &format!("This is turn {}", turn),
    );
});

let config = AgentConfig::new("...")
    .with_injection_chain(chain);
```

#### Helper Functions

```rust
use shadow_agent_sdk::helpers::{
    prepend_to_first_user_message,
    append_to_last_message,
    inject_system_reminder,
};

// Add text to first user message
prepend_to_first_user_message(&mut messages, "IMPORTANT: ");

// Add text to last message
append_to_last_message(&mut messages, "\n\nRemember to be concise.");

// Add system reminder (creates new assistant message if needed)
inject_system_reminder(&mut messages, "The user prefers detailed explanations.");
```

#### Debugger

Log all API calls and tool executions:

```rust
let config = AgentConfig::new("...")
    .with_debug(true);

// Logs are written to: sessions/{session_id}/debugger/
// - api_request_{n}.json
// - api_response_{n}.json
// - tool_call_{n}.json
// - tool_result_{n}.json
```

#### Conversation Namer

The `StandardAgent` automatically generates descriptive names for conversations after the first turn (enabled by default). To disable:

```rust
let config = AgentConfig::new("...")
    .with_auto_name(false);  // Disable automatic naming
```

By default, the agent uses the same LLM for naming. You can configure a separate lightweight model for naming:

```rust
// Use a faster/cheaper model for naming
let naming_llm = llm.create_variant("claude-3-5-haiku-20241022", 1024);

let config = AgentConfig::new("You are a helpful assistant")
    .with_naming_llm(naming_llm);
```

You can also use the helper manually:

```rust
use shadow_agent_sdk::helpers::{ConversationNamer, generate_conversation_name};

// Using the helper struct
let namer = ConversationNamer::new(naming_llm);
let name = namer.generate_name(session.history(), Some(&session_id)).await?;
session.set_conversation_name(&name)?;

// Or using the convenience function
let name = generate_conversation_name(naming_llm, session.history(), Some(&session_id)).await?;
session.set_conversation_name(&name)?;
```

The namer:
- Can use any LlmProvider (Anthropic, Gemini, etc.)
- Generates 3-7 word descriptive names
- Analyzes the conversation content including tool usage
- Automatically integrated into StandardAgent after first turn
- Uses the main agent LLM if no separate naming LLM is configured

---

## Building a Frontend Integration

### Tauri Example Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tauri Frontend (JS/TS)                     │
│  - Chat UI                                                      │
│  - Permission dialogs                                           │
│  - Session browser                                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │ IPC Commands
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tauri Backend (Rust)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    AppState                                │  │
│  │  - runtime: AgentRuntime                                   │  │
│  │  - llm: Arc<AnthropicProvider>                            │  │
│  │  - tools: Arc<ToolRegistry>                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Commands:                                                       │
│  - create_agent(session_id, agent_type)                         │
│  - send_message(session_id, message)                            │
│  - send_permission(session_id, tool, allowed, remember)         │
│  - list_sessions(top_level_only)                                │
│  - get_history(session_id)                                      │
│  - shutdown_agent(session_id)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Backend State

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use shadow_agent_sdk::llm::{AnthropicProvider, LlmProvider};

pub struct AppState {
    pub runtime: AgentRuntime,
    pub llm: Arc<dyn LlmProvider>,
    pub tools: Arc<ToolRegistry>,
}

impl AppState {
    pub fn new() -> anyhow::Result<Self> {
        // Create LLM provider (reads from ANTHROPIC_API_KEY and ANTHROPIC_MODEL env vars)
        let llm: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::from_env()?);

        // Or use Gemini:
        // let llm: Arc<dyn LlmProvider> = Arc::new(GeminiProvider::from_env()?);

        let mut tools = ToolRegistry::new();
        tools.register(ReadTool::new()?);
        tools.register(WriteTool::new()?);
        tools.register(BashTool::new()?);
        let tools = Arc::new(tools);

        let runtime = AgentRuntime::with_global_rules(vec![
            PermissionRule::allow_tool("Read"),
        ]);

        Ok(Self { runtime, llm, tools })
    }
}
```

### Tauri Commands

```rust
#[tauri::command]
async fn create_agent(
    state: tauri::State<'_, Arc<Mutex<AppState>>>,
    session_id: String,
    agent_type: String,
    name: String,
    system_prompt: String,
) -> Result<(), String> {
    let state = state.lock().await;

    let session = AgentSession::new(&session_id, &agent_type, &name, "")
        .map_err(|e| e.to_string())?;

    let config = AgentConfig::new(&system_prompt)
        .with_tools(state.tools.clone())
        .with_streaming(true);

    let agent = StandardAgent::new(config, state.llm.clone());

    state.runtime
        .spawn(session, |internals| agent.run(internals))
        .await;

    Ok(())
}

#[tauri::command]
async fn send_message(
    state: tauri::State<'_, Arc<Mutex<AppState>>>,
    window: tauri::Window,
    session_id: String,
    message: String,
) -> Result<(), String> {
    let state = state.lock().await;

    let handle = state.runtime.get(&session_id).await
        .ok_or("Agent not found")?;

    // Subscribe BEFORE sending
    let mut rx = handle.subscribe();

    handle.send_input(&message).await
        .map_err(|e| e.to_string())?;

    // Spawn task to forward output to frontend
    let window_clone = window.clone();
    tokio::spawn(async move {
        while let Ok(chunk) = rx.recv().await {
            let event_name = match &chunk {
                OutputChunk::TextDelta(_) => "text-delta",
                OutputChunk::ToolStart { .. } => "tool-start",
                OutputChunk::ToolEnd { .. } => "tool-end",
                OutputChunk::PermissionRequest { .. } => "permission-request",
                OutputChunk::Error(_) => "error",
                OutputChunk::Done => "done",
                _ => continue,
            };

            let _ = window_clone.emit(event_name, &chunk);

            if matches!(chunk, OutputChunk::Done | OutputChunk::Error(_)) {
                break;
            }
        }
    });

    Ok(())
}

#[tauri::command]
async fn send_permission(
    state: tauri::State<'_, Arc<Mutex<AppState>>>,
    session_id: String,
    tool_name: String,
    allowed: bool,
    remember: bool,
) -> Result<(), String> {
    let state = state.lock().await;

    let handle = state.runtime.get(&session_id).await
        .ok_or("Agent not found")?;

    handle.send_permission_response(tool_name, allowed, remember).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_sessions(top_level_only: bool) -> Result<Vec<SessionInfo>, String> {
    let sessions = AgentSession::list_with_metadata(top_level_only)
        .map_err(|e| e.to_string())?;

    Ok(sessions.into_iter().map(|(id, meta)| SessionInfo {
        session_id: id,
        agent_type: meta.agent_type,
        name: meta.name,
        description: meta.description,
        created_at: meta.created_at.to_rfc3339(),
        updated_at: meta.updated_at.to_rfc3339(),
        is_subagent: meta.is_subagent(),
    }).collect())
}

#[tauri::command]
async fn get_history(session_id: String) -> Result<Vec<MessageInfo>, String> {
    let history = AgentSession::get_history(&session_id)
        .map_err(|e| e.to_string())?;

    Ok(history.into_iter().map(|msg| MessageInfo {
        role: msg.role,
        content: format_content(&msg.content),
    }).collect())
}
```

### Frontend (TypeScript)

```typescript
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

// Create agent
await invoke('create_agent', {
  sessionId: 'chat-1',
  agentType: 'assistant',
  name: 'My Assistant',
  systemPrompt: 'You are a helpful assistant.',
});

// Listen for output
await listen('text-delta', (event) => {
  appendToChat(event.payload);
});

await listen('permission-request', (event) => {
  const { tool_name, action, input } = event.payload;
  showPermissionDialog(tool_name, action, input, async (allowed, remember) => {
    await invoke('send_permission', {
      sessionId: 'chat-1',
      toolName: tool_name,
      allowed,
      remember,
    });
  });
});

await listen('done', () => {
  setLoading(false);
});

// Send message
await invoke('send_message', {
  sessionId: 'chat-1',
  message: 'Hello!',
});

// List sessions
const sessions = await invoke('list_sessions', { topLevelOnly: true });

// Get history
const history = await invoke('get_history', { sessionId: 'chat-1' });
```

---

## Examples

### Test Agent (`examples/test_agent/`)

Basic agent with common tools:

```bash
cargo run --example test_agent
```

### Gemini Test Agent (`examples/gemini_test_agent/`)

Demonstrates using GeminiProvider:

```bash
# Set environment variables
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-3-flash-preview"

cargo run --example gemini_test_agent
```

### Integration Test (`examples/integration_test/`)

Demonstrates subagent spawning:

```bash
cargo run --example integration_test
```

### Session Browser (`examples/session_browser/`)

Lists sessions and shows history:

```bash
cargo run --example session_browser
```

---

## API Reference

### OutputChunk Variants

| Variant | Description | Fields |
|---------|-------------|--------|
| `TextDelta(String)` | Streaming text token | Text content |
| `TextComplete(String)` | Full text response | Complete text |
| `ThinkingDelta(String)` | Streaming thinking token | Thinking content |
| `ThinkingComplete(String)` | Full thinking | Complete thinking |
| `ToolStart` | Tool execution starting | `id`, `name`, `input` |
| `ToolProgress` | Tool progress update | `id`, `output` |
| `ToolEnd` | Tool execution complete | `id`, `result` |
| `PermissionRequest` | Permission needed | `tool_name`, `action`, `input`, `details` |
| `SubAgentSpawned` | Subagent created | `session_id`, `agent_type` |
| `SubAgentOutput` | Subagent output | `session_id`, `chunk` |
| `SubAgentComplete` | Subagent done | `session_id`, `result` |
| `StateChange(AgentState)` | State transition | New state |
| `Status(String)` | Status message | Message |
| `Error(String)` | Error occurred | Error message |
| `Done` | Agent finished | None |

### InputMessage Variants

| Variant | Description |
|---------|-------------|
| `UserInput(String)` | User prompt |
| `ToolResult { tool_use_id, result }` | Async tool completion |
| `PermissionResponse { tool_name, allowed, remember }` | Permission decision |
| `SubAgentComplete { session_id, result }` | Subagent finished |
| `Interrupt` | Cancel current operation |
| `Shutdown` | Stop agent |

### AgentConfig Builder Methods

| Method | Description |
|--------|-------------|
| `with_tools(Arc<ToolRegistry>)` | Set available tools |
| `with_streaming(bool)` | Enable/disable streaming |
| `with_debug(bool)` | Enable debug logging |
| `with_thinking(budget)` | Enable extended thinking |
| `with_hooks(Arc<HookRegistry>)` | Set behavior hooks |
| `with_max_tool_iterations(n)` | Limit tool call loops |
| `with_auto_save(bool)` | Auto-save session |
| `with_injection_chain(chain)` | Set context injections |
| `with_auto_name(bool)` | Auto-name conversations (default: true) |
| `with_naming_llm(Arc<dyn LlmProvider>)` | Set separate LLM for naming (optional) |
| `with_prompt_caching(bool)` | Enable/disable prompt caching (default: true) |

### AgentHandle Methods

| Method | Description |
|--------|-------------|
| `handle.send_input(text)` | Send user input |
| `handle.send_permission_response(tool, allowed, remember)` | Respond to permission request |
| `handle.send_tool_result(tool_use_id, result)` | Send async tool result |
| `handle.subscribe()` | Subscribe to output stream |
| `handle.state()` | Get current state |
| `handle.is_idle()` / `is_processing()` / `is_done()` | Check specific state |
| `handle.interrupt()` | Cancel current operation |
| `handle.shutdown()` | Stop agent |
| `handle.set_custom_metadata(key, value)` | Update session metadata (works on running agent!) |
| `handle.get_custom_metadata(key)` | Get session metadata |
| `handle.set_conversation_name(name)` | Set conversation name |
| `handle.conversation_name()` | Get conversation name |

### Session Methods

| Method | Description |
|--------|-------------|
| `AgentSession::new(...)` | Create root session |
| `AgentSession::new_subagent(...)` | Create linked subagent |
| `AgentSession::load(id)` | Load from disk |
| `AgentSession::list_all()` | List all session IDs |
| `AgentSession::list_top_level()` | List non-subagents |
| `AgentSession::list_with_metadata(top_level)` | List with metadata |
| `AgentSession::get_history(id)` | Get messages only |
| `AgentSession::get_metadata(id)` | Get metadata only |
| `AgentSession::exists(id)` | Check existence |
| `session.set_conversation_name(name)` | Set conversation name |
| `session.conversation_name()` | Get conversation name |
| `session.has_conversation_name()` | Check if named |

---

## Environment Variables

### Anthropic Provider

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key for Anthropic | Required |
| `ANTHROPIC_MODEL` | Model to use (e.g., `claude-sonnet-4-5@20250929`) | Required |
| `ANTHROPIC_BASE_URL` | Custom API base URL (for proxies) | `https://api.anthropic.com/v1/messages` |
| `ANTHROPIC_MAX_TOKENS` | Maximum tokens per response | `32000` |

### Gemini Provider

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | API key for Google Gemini | Required |
| `GEMINI_MODEL` | Model to use (e.g., `gemini-3-flash-preview`) | Required |
| `GEMINI_MAX_TOKENS` | Maximum tokens per response | `8192` |

### General

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level (debug, info, warn, error) | info |

---

## License

MIT
