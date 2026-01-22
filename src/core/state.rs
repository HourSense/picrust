//! Agent state types

use serde::{Deserialize, Serialize};

/// Current state of an agent
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent is idle, waiting for input
    Idle,

    /// Agent is processing input (calling LLM, thinking)
    Processing,

    /// Agent is waiting for user permission decision
    WaitingForPermission,

    /// Agent is waiting for user to answer questions
    WaitingForUserInput {
        /// Unique ID of the question request
        request_id: String,
    },

    /// Agent is executing a tool
    ExecutingTool {
        /// Name of the tool being executed
        tool_name: String,
        /// ID of the tool use
        tool_use_id: String,
    },

    /// Agent is waiting for a subagent to complete
    WaitingForSubAgent {
        /// Session ID of the subagent
        session_id: String,
    },

    /// Agent has completed successfully
    Done,

    /// Agent encountered an error
    Error {
        /// Error message
        message: String,
    },
}

impl AgentState {
    /// Check if agent is in a terminal state (Done or Error)
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentState::Done | AgentState::Error { .. })
    }

    /// Check if agent is actively processing
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            AgentState::Processing
                | AgentState::ExecutingTool { .. }
                | AgentState::WaitingForSubAgent { .. }
        )
    }

    /// Check if agent is waiting for external input
    pub fn is_waiting(&self) -> bool {
        matches!(
            self,
            AgentState::Idle | AgentState::WaitingForPermission | AgentState::WaitingForUserInput { .. }
        )
    }

    /// Create an error state
    pub fn error(msg: impl Into<String>) -> Self {
        AgentState::Error {
            message: msg.into(),
        }
    }

    /// Create an executing tool state
    pub fn executing_tool(name: impl Into<String>, id: impl Into<String>) -> Self {
        AgentState::ExecutingTool {
            tool_name: name.into(),
            tool_use_id: id.into(),
        }
    }

    /// Create a waiting for subagent state
    pub fn waiting_for_subagent(session_id: impl Into<String>) -> Self {
        AgentState::WaitingForSubAgent {
            session_id: session_id.into(),
        }
    }

    /// Create a waiting for user input state
    pub fn waiting_for_user_input(request_id: impl Into<String>) -> Self {
        AgentState::WaitingForUserInput {
            request_id: request_id.into(),
        }
    }
}

impl Default for AgentState {
    fn default() -> Self {
        AgentState::Idle
    }
}

impl std::fmt::Display for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentState::Idle => write!(f, "Idle"),
            AgentState::Processing => write!(f, "Processing"),
            AgentState::WaitingForPermission => write!(f, "Waiting for permission"),
            AgentState::WaitingForUserInput { request_id } => {
                write!(f, "Waiting for user input: {}", request_id)
            }
            AgentState::ExecutingTool { tool_name, .. } => {
                write!(f, "Executing tool: {}", tool_name)
            }
            AgentState::WaitingForSubAgent { session_id } => {
                write!(f, "Waiting for subagent: {}", session_id)
            }
            AgentState::Done => write!(f, "Done"),
            AgentState::Error { message } => write!(f, "Error: {}", message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_checks() {
        assert!(AgentState::Done.is_terminal());
        assert!(AgentState::error("oops").is_terminal());
        assert!(!AgentState::Idle.is_terminal());

        assert!(AgentState::Processing.is_active());
        assert!(AgentState::executing_tool("Bash", "123").is_active());
        assert!(!AgentState::Idle.is_active());

        assert!(AgentState::Idle.is_waiting());
        assert!(AgentState::WaitingForPermission.is_waiting());
        assert!(!AgentState::Processing.is_waiting());
    }

    #[test]
    fn test_state_display() {
        assert_eq!(AgentState::Idle.to_string(), "Idle");
        assert_eq!(
            AgentState::executing_tool("Bash", "123").to_string(),
            "Executing tool: Bash"
        );
    }
}
