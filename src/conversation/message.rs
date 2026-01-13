//! Message types for conversation storage
//!
//! This module re-exports the Message type from llm::types for conversation storage.
//! The Message type is Anthropic-compatible and supports content blocks.

// Re-export the Message type from llm::types
pub use crate::llm::types::{ContentBlock, Message, MessageContent};
