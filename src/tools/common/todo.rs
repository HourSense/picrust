//! TodoWrite tool for task management
//!
//! This tool allows the agent to maintain and update a todo list
//! to track tasks it needs to perform. State is persisted.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// Status of a todo item
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single todo item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    /// The imperative form describing what needs to be done
    pub content: String,
    /// Current status of the task
    pub status: TodoStatus,
    /// The present continuous form shown during execution
    #[serde(rename = "activeForm")]
    pub active_form: String,
}

/// Shared todo list state
pub type TodoList = Arc<RwLock<Vec<TodoItem>>>;

/// Create a new shared todo list
pub fn new_todo_list() -> TodoList {
    Arc::new(RwLock::new(Vec::new()))
}

/// TodoWrite tool for managing tasks
pub struct TodoWriteTool {
    todos: TodoList,
}

/// Input for the todo tool
#[derive(Debug, Deserialize)]
struct TodoInput {
    /// The full list of todos to set
    todos: Vec<TodoItem>,
}

impl TodoWriteTool {
    /// Create a new TodoWrite tool with a shared todo list
    pub fn new(todos: TodoList) -> Self {
        Self { todos }
    }

    /// Get the current todo list
    pub fn get_todos(&self) -> Vec<TodoItem> {
        self.todos.read().unwrap().clone()
    }

    /// Format the todo list for display
    pub fn format_todos(&self) -> String {
        let todos = self.todos.read().unwrap();
        if todos.is_empty() {
            return "No tasks in the todo list.".to_string();
        }

        let mut output = String::new();
        output.push_str("Todo List:\n");

        for (i, item) in todos.iter().enumerate() {
            let status_icon = match item.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[*]",
                TodoStatus::Completed => "[x]",
            };
            output.push_str(&format!(
                "  {} {}. {}\n",
                status_icon,
                i + 1,
                item.content
            ));
        }

        // Show summary
        let pending = todos.iter().filter(|t| t.status == TodoStatus::Pending).count();
        let in_progress = todos.iter().filter(|t| t.status == TodoStatus::InProgress).count();
        let completed = todos.iter().filter(|t| t.status == TodoStatus::Completed).count();

        output.push_str(&format!(
            "\nSummary: {} pending, {} in progress, {} completed\n",
            pending, in_progress, completed
        ));

        output
    }
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &str {
        "TodoWrite"
    }

    fn description(&self) -> &str {
        "Create and manage a structured task list to track progress and organize tasks."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "TodoWrite".to_string(),
            description: Some(
                "Use this tool to create and manage a structured task list for the current session. \
                This helps track progress and organize complex tasks. \
                Each todo has content (what to do), status (pending/in_progress/completed), \
                and activeForm (present continuous description)."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "todos": {
                        "type": "array",
                        "description": "The updated todo list",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "The imperative form describing what needs to be done (e.g., 'Run tests')"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Current status of the task"
                                },
                                "activeForm": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "The present continuous form shown during execution (e.g., 'Running tests')"
                                }
                            },
                            "required": ["content", "status", "activeForm"]
                        }
                    }
                })),
                required: Some(vec!["todos".to_string()]),
            },
            tool_type: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let todo_count = input
            .get("todos")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len())
            .unwrap_or(0);

        ToolInfo {
            name: "TodoWrite".to_string(),
            action_description: format!("Update todo list ({} items)", todo_count),
            details: None,
        }
    }

    async fn execute(&self, input: &Value) -> Result<ToolResult> {
        let todo_input: TodoInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid todo input: {}", e))?;

        // Update the todo list
        {
            let mut todos = self.todos.write().unwrap();
            *todos = todo_input.todos;
        }

        // Return the formatted list
        let output = self.format_todos();
        Ok(ToolResult::success(output))
    }

    fn requires_permission(&self) -> bool {
        false // Todo updates don't need permission
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_todo_tool() {
        let todos = new_todo_list();
        let tool = TodoWriteTool::new(todos.clone());

        let input = json!({
            "todos": [
                {"content": "First task", "status": "pending", "activeForm": "Working on first task"},
                {"content": "Second task", "status": "completed", "activeForm": "Working on second task"}
            ]
        });

        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("First task"));
        assert!(result.output.contains("Second task"));

        // Check the shared state
        let list = todos.read().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].status, TodoStatus::Pending);
        assert_eq!(list[1].status, TodoStatus::Completed);
    }
}
