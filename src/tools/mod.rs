//! Tool system for the agent
//!
//! This module provides the Tool trait and ToolRegistry for managing
//! tools that the agent can use.

pub mod bash;
pub mod file_edit;
mod registry;
pub mod todo;
mod tool;

pub use bash::BashTool;
pub use file_edit::FileEditTool;
pub use registry::ToolRegistry;
pub use todo::{new_todo_list, TodoItem, TodoList, TodoTool};
pub use tool::{Tool, ToolResult};
