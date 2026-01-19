# Session Listing and History API

## Overview

This document describes the API for listing agent sessions and retrieving conversation history. These features allow applications to:
- List all sessions (optionally filtering out subagents)
- Get conversation history for any session by ID
- Get session metadata without loading full session

## API Reference

### Listing Sessions

#### List All Sessions

```rust
// List all session IDs
let all_sessions = AgentSession::list_all()?;

// With custom storage
let storage = SessionStorage::with_dir("my_sessions");
let all_sessions = AgentSession::list_all_with_storage(&storage)?;
```

#### List Top-Level Sessions Only

Top-level sessions are agents that are not subagents (i.e., they have no parent).

```rust
// List only top-level sessions (not subagents)
let top_level = AgentSession::list_top_level()?;

// With custom storage
let top_level = AgentSession::list_top_level_with_storage(&storage)?;
```

#### List with Filter Parameter

```rust
// List with explicit filter parameter
// top_level_only = true  -> only sessions without a parent
// top_level_only = false -> all sessions (same as list_all)
let sessions = AgentSession::list_filtered(true)?;

// With custom storage
let sessions = AgentSession::list_filtered_with_storage(true, &storage)?;
```

#### List with Metadata

For better performance when you need both session IDs and metadata:

```rust
// Get sessions with their metadata
// Returns Vec<(String, SessionMetadata)>
let sessions_with_meta = AgentSession::list_with_metadata(true)?;

for (session_id, metadata) in sessions_with_meta {
    println!("{}: {} ({})", session_id, metadata.name, metadata.agent_type);
    println!("  Created: {}", metadata.created_at);
    if metadata.is_subagent() {
        println!("  Parent: {:?}", metadata.parent_session_id);
    }
}
```

### Getting Conversation History

#### Get History by Session ID

This loads only the message history without the full session object:

```rust
// Get history (Vec<Message>) for a session
let history = AgentSession::get_history("my-session-id")?;

for message in &history {
    match &message.role {
        Role::User => println!("User: ..."),
        Role::Assistant => println!("Assistant: ..."),
    }
}

// With custom storage
let history = AgentSession::get_history_with_storage("my-session-id", &storage)?;
```

#### Get Metadata by Session ID

This loads only the metadata without the full session:

```rust
// Get metadata for a session
let metadata = AgentSession::get_metadata("my-session-id")?;

println!("Session: {}", metadata.session_id);
println!("Agent Type: {}", metadata.agent_type);
println!("Name: {}", metadata.name);
println!("Is Subagent: {}", metadata.is_subagent());
```

### Using SessionStorage Directly

The underlying `SessionStorage` class also provides these methods:

```rust
let storage = SessionStorage::new();

// List all sessions
let all = storage.list_sessions()?;

// List with filter
let filtered = storage.list_sessions_filtered(true)?;

// List top-level only (convenience method)
let top_level = storage.list_top_level_sessions()?;

// List with metadata
let with_meta = storage.list_sessions_with_metadata(true)?;

// Load messages directly
let messages = storage.load_messages("session-id")?;

// Load metadata directly
let metadata = storage.load_metadata("session-id")?;
```

## Use Cases

### Showing User's Active Conversations

```rust
// Get all top-level conversations (excluding subagent sessions)
let conversations = AgentSession::list_with_metadata(true)?;

println!("Your conversations:");
for (id, meta) in conversations {
    println!("  [{}] {} - {}", meta.agent_type, meta.name, meta.description);
    println!("      Last updated: {}", meta.updated_at);
}
```

### Resuming a Conversation

```rust
// Check if session exists and load it
if AgentSession::exists("my-session") {
    let session = AgentSession::load("my-session")?;

    // Access history
    for msg in session.history() {
        // Process messages...
    }
}
```

### Viewing Subagent History

```rust
// For a parent session, view its subagents' histories
let parent = AgentSession::load("parent-session")?;

for child_id in parent.child_session_ids() {
    let child_history = AgentSession::get_history(child_id)?;
    println!("Subagent {} had {} messages", child_id, child_history.len());
}
```

## Implementation Notes

1. **Performance**: `list_sessions_filtered(true)` loads metadata for each session to check if it's a subagent. For large numbers of sessions, consider caching or using `list_with_metadata` if you need the metadata anyway.

2. **Thread Safety**: `SessionStorage` operations are file-based and do not require locks. Multiple processes can safely read sessions simultaneously.

3. **Custom Storage**: All methods have `_with_storage` variants for using custom session directories.

## Example

See `examples/session_browser/main.rs` for a complete working example:

```bash
cargo run --example session_browser
```

Output:
```
=== Session Browser ===

Top-level sessions (not subagents):
------------------------------------------------------------
  Session: test-agent-session
    Type: test-agent
    Name: Test Agent
    Description: A test agent demonstrating the StandardAgent framework
    Created: 2026-01-19 12:48:35
    Updated: 2026-01-19 12:48:38

============================================================
Conversation history for: test-agent-session
------------------------------------------------------------
  Total messages: 2

  [1] USER:
      hi

  [2] ASSISTANT:
      Hello! How can I help you today?
```

## Files Modified

- `src/session/storage.rs`: Added `list_sessions_filtered`, `list_top_level_sessions`, `list_sessions_with_metadata`
- `src/session/session.rs`: Added `list_filtered`, `list_top_level`, `list_with_metadata`, `get_history`, `get_metadata` and their `_with_storage` variants
- `examples/session_browser/main.rs`: New example demonstrating these features
