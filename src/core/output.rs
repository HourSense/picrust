//! Input and output message types for agent communication

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::state::AgentState;
use crate::tools::ToolResult;

/// A single question option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    pub label: String,
    pub description: String,
}

/// A single question in an AskUserQuestion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestion {
    pub question: String,
    pub header: String,
    pub options: Vec<QuestionOption>,
    pub multi_select: bool,
}

/// Messages that can be sent TO an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputMessage {
    /// User input text
    UserInput(String),

    /// Result from an async tool execution
    ToolResult {
        /// ID of the tool use this result is for
        tool_use_id: String,
        /// The tool result
        result: ToolResult,
    },

    /// Permission response from user
    PermissionResponse {
        /// Tool that was requesting permission
        tool_name: String,
        /// Whether permission was granted
        allowed: bool,
        /// Whether to remember this decision
        remember: bool,
    },

    /// Subagent completed
    SubAgentComplete {
        /// Session ID of the completed subagent
        session_id: String,
        /// Final result/summary from subagent
        result: Option<String>,
    },

    /// Response to an AskUserQuestion request
    UserQuestionResponse {
        /// Unique ID matching the request
        request_id: String,
        /// Map of question header to selected answer(s)
        answers: HashMap<String, String>,
    },

    /// Request graceful interrupt
    Interrupt,

    /// Request shutdown
    Shutdown,
}

/// Output chunks streamed FROM an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputChunk {
    // --- Text Streaming ---
    /// Incremental text output
    TextDelta(String),

    /// Complete text block
    TextComplete(String),

    // --- Thinking Streaming ---
    /// Incremental thinking output
    ThinkingDelta(String),

    /// Complete thinking block
    ThinkingComplete(String),

    // --- Tool Execution ---
    /// Tool execution starting
    ToolStart {
        /// Tool use ID
        id: String,
        /// Tool name
        name: String,
        /// Tool input
        input: Value,
    },

    /// Incremental tool output (for long-running tools)
    ToolProgress {
        /// Tool use ID
        id: String,
        /// Progress output
        output: String,
    },

    /// Tool execution completed
    ToolEnd {
        /// Tool use ID
        id: String,
        /// Tool result
        result: ToolResult,
    },

    // --- Permission ---
    /// Requesting permission from user
    PermissionRequest {
        /// Tool name
        tool_name: String,
        /// Human-readable action description
        action: String,
        /// The actual input/command being executed
        input: String,
        /// Additional details
        details: Option<String>,
    },

    // --- Subagent Events ---
    /// Subagent was spawned
    SubAgentSpawned {
        /// Session ID of new subagent
        session_id: String,
        /// Type of the subagent
        agent_type: String,
    },

    /// Output from a subagent (forwarded)
    SubAgentOutput {
        /// Session ID of the subagent
        session_id: String,
        /// The output chunk from subagent
        chunk: Box<OutputChunk>,
    },

    /// Subagent completed
    SubAgentComplete {
        /// Session ID of completed subagent
        session_id: String,
        /// Final result/summary
        result: Option<String>,
    },

    // --- User Questions ---
    /// Request user to answer questions
    AskUserQuestion {
        /// Unique ID for this request
        request_id: String,
        /// The questions to ask
        questions: Vec<UserQuestion>,
    },

    // --- State & Status ---
    /// Agent state changed
    StateChange(AgentState),

    /// Status update (for progress indicators)
    Status(String),

    // --- Completion ---
    /// Error occurred
    Error(String),

    /// Agent completed this turn
    Done,
}

impl OutputChunk {
    /// Create a text delta chunk
    pub fn text(text: impl Into<String>) -> Self {
        OutputChunk::TextDelta(text.into())
    }

    /// Create a thinking delta chunk
    pub fn thinking(text: impl Into<String>) -> Self {
        OutputChunk::ThinkingDelta(text.into())
    }

    /// Create a tool start chunk
    pub fn tool_start(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        OutputChunk::ToolStart {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a tool end chunk
    pub fn tool_end(id: impl Into<String>, result: ToolResult) -> Self {
        OutputChunk::ToolEnd {
            id: id.into(),
            result,
        }
    }

    /// Create an error chunk
    pub fn error(msg: impl Into<String>) -> Self {
        OutputChunk::Error(msg.into())
    }

    /// Check if this is a terminal chunk
    pub fn is_terminal(&self) -> bool {
        matches!(self, OutputChunk::Done | OutputChunk::Error(_))
    }

    /// Check if this is a text-related chunk
    pub fn is_text(&self) -> bool {
        matches!(
            self,
            OutputChunk::TextDelta(_) | OutputChunk::TextComplete(_)
        )
    }

    /// Check if this is a thinking-related chunk
    pub fn is_thinking(&self) -> bool {
        matches!(
            self,
            OutputChunk::ThinkingDelta(_) | OutputChunk::ThinkingComplete(_)
        )
    }

    /// Check if this is a tool-related chunk
    pub fn is_tool(&self) -> bool {
        matches!(
            self,
            OutputChunk::ToolStart { .. }
                | OutputChunk::ToolProgress { .. }
                | OutputChunk::ToolEnd { .. }
        )
    }
}

impl InputMessage {
    /// Create a user input message
    pub fn user_input(text: impl Into<String>) -> Self {
        InputMessage::UserInput(text.into())
    }

    /// Create a permission response
    pub fn permission(tool_name: impl Into<String>, allowed: bool, remember: bool) -> Self {
        InputMessage::PermissionResponse {
            tool_name: tool_name.into(),
            allowed,
            remember,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_chunk_checks() {
        assert!(OutputChunk::Done.is_terminal());
        assert!(OutputChunk::error("oops").is_terminal());
        assert!(!OutputChunk::text("hello").is_terminal());

        assert!(OutputChunk::text("hello").is_text());
        assert!(OutputChunk::TextComplete("hello".into()).is_text());
        assert!(!OutputChunk::Done.is_text());

        assert!(OutputChunk::thinking("hmm").is_thinking());
        assert!(!OutputChunk::text("hello").is_thinking());
    }

    #[test]
    fn test_input_message_creation() {
        let msg = InputMessage::user_input("hello");
        assert!(matches!(msg, InputMessage::UserInput(s) if s == "hello"));

        let msg = InputMessage::permission("Bash", true, false);
        assert!(matches!(
            msg,
            InputMessage::PermissionResponse {
                allowed: true,
                remember: false,
                ..
            }
        ));
    }
}
