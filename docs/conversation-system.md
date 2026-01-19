# Conversation System

## Overview

The agent now stores all conversations in a structured format with:
- Unique UUID for each conversation
- Metadata tracking (timestamps, model provider)
- Message history in JSONL format (Anthropic compatible)
- File-based storage for persistence

## Directory Structure

```
conversations/
├── <uuid-1>/
│   ├── metadata.json
│   └── history.jsonl
├── <uuid-2>/
│   ├── metadata.json
│   └── history.jsonl
└── ...
```

## Files

### metadata.json

Stores conversation metadata:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-01-14T10:30:45.123456Z",
  "updated_at": "2026-01-14T10:35:22.654321Z",
  "model_provider": "anthropic",
  "title": null
}
```

**Fields:**
- `id`: UUID of the conversation
- `created_at`: ISO 8601 timestamp when conversation started
- `updated_at`: ISO 8601 timestamp of last message
- `model_provider`: LLM provider used (currently "anthropic")
- `title`: Optional conversation title (can be set later)

### history.jsonl

Stores messages in JSON Lines format (one JSON object per line):

```jsonl
{"role":"user","content":"hi"}
{"role":"assistant","content":"Hello! How can I help you?"}
{"role":"user","content":"What's the weather?"}
{"role":"assistant","content":"I don't have access to real-time weather data."}
```

**Format:**
- Each line is a complete JSON object
- Compatible with Anthropic message format
- Easy to append new messages
- Efficient for large conversations

## Conversation API

### Creating a Conversation

```rust
use shadow_agent_sdk::conversation::Conversation;

// Create new conversation (generates UUID automatically)
let mut conversation = Conversation::new()?;

println!("Conversation ID: {}", conversation.id());
```

### Adding Messages

```rust
// Add user message
conversation.add_user_message("Hello!")?;

// Add assistant message
conversation.add_assistant_message("Hi there!")?;

// Add custom message
use shadow_agent_sdk::conversation::Message;
conversation.add_message(Message::user("Custom message"))?;
```

### Reading Messages

```rust
// Get all messages
let messages = conversation.get_messages()?;

for message in messages {
    println!("{}: {}", message.role, message.content);
}

// Get message count
let count = conversation.message_count()?;
println!("Total messages: {}", count);
```

### Setting Title

```rust
// Set conversation title
conversation.set_title("My First Conversation")?;
```

### Loading Existing Conversation

```rust
// Load by ID
let conversation = Conversation::load("550e8400-e29b-41d4-a716-446655440000")?;
```

### Listing All Conversations

```rust
// Get all conversation IDs
let ids = Conversation::list_all()?;

for id in ids {
    println!("Conversation: {}", id);
}
```

### Deleting a Conversation

```rust
// Delete conversation and all its files
conversation.delete()?;
```

## How It Works in the Agent

1. **Agent Initialization**: Creates a new `Conversation` automatically
   ```rust
   let agent = Agent::new(console, llm_provider)?;
   // New conversation is created with UUID
   ```

2. **User Input Received**: Agent gets the message from user

3. **Load Conversation History**: Agent retrieves all previous messages
   ```rust
   let history = conversation.get_messages()?;
   ```

4. **Send to LLM with Context**: Agent sends current message + full history
   ```rust
   llm_provider.send_message(user_message, &history, system_prompt).await?;
   ```

5. **Save User Message**: After getting LLM response, save user message
   ```rust
   conversation.add_user_message(user_message)?;
   ```

6. **Save Assistant Response**: Save the LLM's response
   ```rust
   conversation.add_assistant_message(response)?;
   ```

7. **Metadata Updates**: `updated_at` timestamp updated on every message

## Behavior

### Current (v0.1.0)
- **New conversation on every run**: Each time you run the agent, a new conversation is created
- **Automatic storage**: All messages automatically saved
- **Full context awareness**: LLM receives entire conversation history with each message
- **No loading**: Previous conversations are not loaded (coming soon)

### Future Enhancements
- **Resume conversations**: Load and continue previous conversations
- **List conversations**: CLI command to list all conversations
- **Search conversations**: Find conversations by content
- **Export conversations**: Export to markdown, PDF, etc.
- **Conversation titles**: Auto-generate from first message

## File Format Benefits

### JSONL (JSON Lines)
✅ Easy to append (just add a new line)
✅ Efficient for large files (can stream/process line by line)
✅ Human-readable
✅ Compatible with many tools (jq, grep, etc.)
✅ No need to rewrite entire file

### Metadata Separation
✅ Quick metadata access without parsing entire history
✅ Metadata changes don't require history file modification
✅ Easy to index and search

## Example Conversation Directory

After running the agent and chatting:

```
conversations/
└── a1b2c3d4-e5f6-4789-0abc-def123456789/
    ├── metadata.json
    └── history.jsonl
```

**metadata.json:**
```json
{
  "id": "a1b2c3d4-e5f6-4789-0abc-def123456789",
  "created_at": "2026-01-14T15:30:00.000000Z",
  "updated_at": "2026-01-14T15:32:15.000000Z",
  "model_provider": "anthropic",
  "title": null
}
```

**history.jsonl:**
```jsonl
{"role":"user","content":"How do I create a Rust struct?"}
{"role":"assistant","content":"To create a Rust struct, use the `struct` keyword..."}
{"role":"user","content":"Can you show an example?"}
{"role":"assistant","content":"Sure! Here's an example:\n\n```rust\nstruct Person {\n    name: String,\n    age: u32,\n}\n```"}
```

## Viewing Conversations

### Using command line tools

```bash
# List all conversations
ls conversations/

# View metadata
cat conversations/<uuid>/metadata.json

# View message history
cat conversations/<uuid>/history.jsonl

# Pretty print with jq
cat conversations/<uuid>/history.jsonl | jq

# Count messages
wc -l conversations/<uuid>/history.jsonl

# Search for text
grep "keyword" conversations/<uuid>/history.jsonl
```

### In Rust code

```rust
let conversation = Conversation::load("<uuid>")?;
let messages = conversation.get_messages()?;

for msg in messages {
    println!("[{}] {}", msg.role, msg.content);
}
```

## Error Handling

All conversation operations return `Result<T>` and can fail if:
- Filesystem permissions are insufficient
- Disk is full
- JSON parsing fails
- Directory structure is corrupted

Always handle errors:
```rust
match conversation.add_message(msg) {
    Ok(_) => println!("Message saved"),
    Err(e) => eprintln!("Failed to save: {}", e),
}
```

## Performance

- **Message appending**: O(1) - just writes one line
- **Metadata update**: O(1) - small JSON file
- **Reading all messages**: O(n) - reads entire file
- **Disk usage**: ~100-500 bytes per message (depends on content length)

## Best Practices

1. **Don't edit files manually** while the agent is running
2. **Backup important conversations** before deleting
3. **Use conversation IDs** for programmatic access
4. **Set titles** for important conversations to find them easily
5. **Check disk space** if storing very long conversations

## Logging

Conversation operations are logged:
- Conversation creation
- Message additions
- Metadata updates
- Loading/deleting operations

Check `logs/agent.log` for details.
