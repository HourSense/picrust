//! Useful helpers for agent implementations
//!
//! This module provides reusable components that agents can opt-in to:
//! - `TodoListManager` - Tracks tasks and which turn they were last updated
//! - `ContextInjection` - Modify messages before each LLM call
//! - `Debugger` - Log API calls and tool executions for debugging

mod context_injection;
mod debugger;
mod todo_manager;

pub use context_injection::{
    append_to_last_message, inject_system_reminder, prepend_to_first_user_message,
    BoxedInjection, ContextInjection, FnInjection, InjectionChain, SharedInjection,
};
pub use debugger::{
    ApiRequestEvent, ApiResponseEvent, Debugger, EventType, ToolCallEvent, ToolResultEvent,
};
pub use todo_manager::{TodoItem, TodoListManager, TodoStatus};
