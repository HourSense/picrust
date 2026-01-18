//! Debugger for logging API calls and tool executions
//!
//! When enabled, logs all LLM interactions and tool calls to a `debugger/`
//! folder within the session directory.
//!
//! # Example
//!
//! ```ignore
//! let debugger = Debugger::new(session_dir)?;
//! debugger.log_api_request(&messages, &system_prompt)?;
//! debugger.log_api_response(&response)?;
//! debugger.log_tool_call("Read", &input)?;
//! debugger.log_tool_result("Read", &result)?;
//! ```

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::llm::Message;
use crate::tools::ToolResult;

/// Debugger for logging API calls and tool executions
pub struct Debugger {
    /// Directory where debug logs are stored
    dir: PathBuf,
    /// Sequence counter for ordering events
    sequence: AtomicU64,
    /// Whether debugging is enabled
    enabled: bool,
}

/// Types of debug events
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    ApiRequest,
    ApiResponse,
    ToolCall,
    ToolResult,
}

/// API request event
#[derive(Debug, Serialize)]
pub struct ApiRequestEvent {
    pub event_type: EventType,
    pub sequence: u64,
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    pub tool_definitions: Option<Vec<Value>>,
}

/// API response event
#[derive(Debug, Serialize)]
pub struct ApiResponseEvent {
    pub event_type: EventType,
    pub sequence: u64,
    pub response: Value,
}

/// Tool call event
#[derive(Debug, Serialize)]
pub struct ToolCallEvent {
    pub event_type: EventType,
    pub sequence: u64,
    pub tool_name: String,
    pub tool_id: String,
    pub input: Value,
}

/// Tool result event
#[derive(Debug, Serialize)]
pub struct ToolResultEvent {
    pub event_type: EventType,
    pub sequence: u64,
    pub tool_name: String,
    pub tool_id: String,
    pub output: String,
    pub is_error: bool,
}

impl Debugger {
    /// Create a new debugger that logs to a `debugger/` subdirectory
    ///
    /// # Arguments
    /// * `session_dir` - The session directory where `debugger/` will be created
    pub fn new(session_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = session_dir.as_ref().join("debugger");
        fs::create_dir_all(&dir)?;

        tracing::info!("[Debugger] Created debug directory: {:?}", dir);

        Ok(Self {
            dir,
            sequence: AtomicU64::new(0),
            enabled: true,
        })
    }

    /// Create a disabled debugger (no-op for all operations)
    pub fn disabled() -> Self {
        Self {
            dir: PathBuf::new(),
            sequence: AtomicU64::new(0),
            enabled: false,
        }
    }

    /// Check if debugging is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the debug directory path
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Get the next sequence number
    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    /// Log an API request (messages sent to LLM)
    pub fn log_api_request(
        &self,
        messages: &[Message],
        system_prompt: Option<&str>,
        tool_definitions: Option<&[Value]>,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let seq = self.next_sequence();
        let event = ApiRequestEvent {
            event_type: EventType::ApiRequest,
            sequence: seq,
            system_prompt: system_prompt.map(|s| s.to_string()),
            messages: messages.to_vec(),
            tool_definitions: tool_definitions.map(|t| t.to_vec()),
        };

        let filename = format!("{:06}_api_request.json", seq);
        let path = self.dir.join(&filename);
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &event)?;

        tracing::debug!("[Debugger] Logged API request #{}", seq);
        Ok(())
    }

    /// Log an API response (raw response from LLM)
    pub fn log_api_response(&self, response: &Value) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let seq = self.next_sequence();
        let event = ApiResponseEvent {
            event_type: EventType::ApiResponse,
            sequence: seq,
            response: response.clone(),
        };

        let filename = format!("{:06}_api_response.json", seq);
        let path = self.dir.join(&filename);
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &event)?;

        tracing::debug!("[Debugger] Logged API response #{}", seq);
        Ok(())
    }

    /// Log a tool call
    pub fn log_tool_call(&self, tool_name: &str, tool_id: &str, input: &Value) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let seq = self.next_sequence();
        let event = ToolCallEvent {
            event_type: EventType::ToolCall,
            sequence: seq,
            tool_name: tool_name.to_string(),
            tool_id: tool_id.to_string(),
            input: input.clone(),
        };

        let filename = format!("{:06}_tool_call_{}.json", seq, tool_name);
        let path = self.dir.join(&filename);
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &event)?;

        tracing::debug!("[Debugger] Logged tool call #{}: {}", seq, tool_name);
        Ok(())
    }

    /// Log a tool result
    pub fn log_tool_result(&self, tool_name: &str, tool_id: &str, result: &ToolResult) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let seq = self.next_sequence();
        let event = ToolResultEvent {
            event_type: EventType::ToolResult,
            sequence: seq,
            tool_name: tool_name.to_string(),
            tool_id: tool_id.to_string(),
            output: result.output.clone(),
            is_error: result.is_error,
        };

        let filename = format!("{:06}_tool_result_{}.json", seq, tool_name);
        let path = self.dir.join(&filename);
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &event)?;

        tracing::debug!("[Debugger] Logged tool result #{}: {}", seq, tool_name);
        Ok(())
    }

    /// Clear all debug logs in the directory
    pub fn clear(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                fs::remove_file(path)?;
            }
        }

        self.sequence.store(0, Ordering::SeqCst);
        tracing::info!("[Debugger] Cleared debug logs");
        Ok(())
    }
}
