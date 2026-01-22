# AskUserQuestion Tool Integration Guide

This guide explains how to integrate the `AskUserQuestion` tool in a Tauri application using the Shadow Agent SDK.

## Overview

The `AskUserQuestion` tool allows the agent to pause execution and ask the user questions with multiple-choice options. The agent sends questions via `OutputChunk::AskUserQuestion`, waits for the user's response via `InputMessage::UserQuestionResponse`, and then continues execution with the answers.

## Message Flow

```
Agent                          Tauri Frontend
  |                                  |
  |-- OutputChunk::AskUserQuestion ->|  (display question UI)
  |                                  |
  |<- InputMessage::UserQuestionResponse (user selects answers)
  |                                  |
  | (agent continues with answers)   |
```

## Data Structures

### Output: `OutputChunk::AskUserQuestion`

When the agent needs user input, it emits this output chunk:

```rust
OutputChunk::AskUserQuestion {
    request_id: String,      // Unique ID to match request/response
    questions: Vec<UserQuestion>,
}
```

Each `UserQuestion` contains:

```rust
pub struct UserQuestion {
    pub question: String,    // The full question text (e.g., "Which database should we use?")
    pub header: String,      // Short label/tag (max 12 chars, e.g., "Database")
    pub options: Vec<QuestionOption>,
    pub multi_select: bool,  // If true, user can select multiple options
}

pub struct QuestionOption {
    pub label: String,       // Display text (1-5 words, e.g., "PostgreSQL")
    pub description: String, // Explanation of what this option means
}
```

### Input: `InputMessage::UserQuestionResponse`

After the user makes their selections, send this back to the agent:

```rust
InputMessage::UserQuestionResponse {
    request_id: String,                      // Must match the request
    answers: HashMap<String, String>,        // header -> selected label(s)
}
```

For `multi_select: true` questions, join multiple selected labels with commas (e.g., `"Option A, Option B"`).

## Tauri Implementation

### 1. Handle the Output Chunk

In your output stream handler, add a case for `AskUserQuestion`:

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

### 2. Create a Tauri Command to Send Response

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

### 3. Frontend (React/TypeScript Example)

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

interface AskUserQuestionEvent {
  requestId: string;
  questions: UserQuestion[];
}

// Listen for questions
listen<AskUserQuestionEvent>("ask-user-question", (event) => {
  setQuestions(event.payload);
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

### 4. Example UI Component (React)

```tsx
function QuestionModal({ questions, onSubmit }: Props) {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  return (
    <div className="modal">
      {questions.map((q) => (
        <div key={q.header} className="question">
          <h3>{q.header}</h3>
          <p>{q.question}</p>

          {q.options.map((opt) => (
            <label key={opt.label}>
              <input
                type={q.multiSelect ? "checkbox" : "radio"}
                name={q.header}
                value={opt.label}
                onChange={(e) => {
                  if (q.multiSelect) {
                    // Handle multi-select (comma-separated)
                    const current = answers[q.header]?.split(", ") || [];
                    const updated = e.target.checked
                      ? [...current, opt.label]
                      : current.filter((l) => l !== opt.label);
                    setAnswers({ ...answers, [q.header]: updated.join(", ") });
                  } else {
                    setAnswers({ ...answers, [q.header]: opt.label });
                  }
                }}
              />
              <span>{opt.label}</span>
              <small>{opt.description}</small>
            </label>
          ))}

          {/* Allow custom "Other" input */}
          <label>
            <input type={q.multiSelect ? "checkbox" : "radio"} name={q.header} value="other" />
            <span>Other</span>
            <input type="text" placeholder="Custom answer..." />
          </label>
        </div>
      ))}

      <button onClick={() => onSubmit(answers)}>Submit</button>
    </div>
  );
}
```

## Validation Rules

The agent validates input before asking:
- **1-4 questions** per request
- **2-4 options** per question
- Headers should be short (max 12 characters)

## Agent State

While waiting for a response, the agent enters the `WaitingForUserInput` state:

```rust
AgentState::WaitingForUserInput { request_id: String }
```

You can check this state to show a loading/waiting indicator in the UI.

## Handling Interrupts

If the user cancels or the app needs to interrupt:

```rust
// Send interrupt instead of response
handle.interrupt().await?;
```

The tool will return an error result to the agent, which can handle it gracefully.

## Example Agent Usage

From the agent's perspective, using the tool:

```json
{
  "name": "AskUserQuestion",
  "input": {
    "questions": [
      {
        "question": "Which authentication method should we implement?",
        "header": "Auth",
        "multiSelect": false,
        "options": [
          { "label": "JWT (Recommended)", "description": "Stateless tokens, good for APIs" },
          { "label": "Session cookies", "description": "Traditional server-side sessions" },
          { "label": "OAuth 2.0", "description": "Third-party authentication" }
        ]
      }
    ]
  }
}
```

The tool returns the user's answers as formatted JSON:

```
User responded with the following answers:
{
  "Auth": "JWT (Recommended)"
}
```
