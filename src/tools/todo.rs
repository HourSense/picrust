//! TODO tool for task management
//!
//! This tool allows the agent to maintain and update a todo list
//! to track tasks it needs to perform.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, RwLock};

use super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};

/// A single todo item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    /// Index/ID of the todo item
    pub index: u32,
    /// Whether the task is completed
    pub completed: bool,
    /// The task description
    pub task: String,
    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_details: Option<String>,
}

/// Shared todo list state
pub type TodoList = Arc<RwLock<Vec<TodoItem>>>;

/// Create a new shared todo list
pub fn new_todo_list() -> TodoList {
    Arc::new(RwLock::new(Vec::new()))
}

/// TODO tool for managing tasks
pub struct TodoTool {
    todos: TodoList,
}

/// Input for the todo tool
#[derive(Debug, Deserialize)]
struct TodoInput {
    /// The full list of todos to set
    todos: Vec<TodoItem>,
}

impl TodoTool {
    /// Create a new TODO tool with a shared todo list
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
        for item in todos.iter() {
            let status = if item.completed { "[x]" } else { "[ ]" };
            output.push_str(&format!("  {} {}. {}\n", status, item.index, item.task));
            if let Some(ref details) = item.additional_details {
                output.push_str(&format!("      Details: {}\n", details));
            }
        }
        output
    }
}

#[async_trait]
impl Tool for TodoTool {
    fn name(&self) -> &str {
        "todo"
    }

    fn description(&self) -> &str {
        "Manage a todo list to track tasks. Update the list by providing the complete current state of all todos. Each todo has an index, completion status, task description, and optional additional details."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "todo".to_string(),
            description: Some(self.description().to_string()),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "todos": {
                        "type": "array",
                        "description": "The complete list of todo items. Each update should include ALL current todos.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "description": "Index/ID of the todo item (1-based)"
                                },
                                "completed": {
                                    "type": "boolean",
                                    "description": "Whether the task is completed"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "Description of the task"
                                },
                                "additional_details": {
                                    "type": "string",
                                    "description": "Optional additional details about the task"
                                }
                            },
                            "required": ["index", "completed", "task"]
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
            name: "todo".to_string(),
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
        let tool = TodoTool::new(todos.clone());

        let input = json!({
            "todos": [
                {"index": 1, "completed": false, "task": "First task"},
                {"index": 2, "completed": true, "task": "Second task", "additional_details": "Some details"}
            ]
        });

        let result = tool.execute(&input).await.unwrap();
        assert!(!result.is_error);
        assert!(result.output.contains("First task"));
        assert!(result.output.contains("Second task"));

        // Check the shared state
        let list = todos.read().unwrap();
        assert_eq!(list.len(), 2);
        assert!(!list[0].completed);
        assert!(list[1].completed);
    }
}
