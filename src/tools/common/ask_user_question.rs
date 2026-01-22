//! AskUserQuestion tool for interactive user queries
//!
//! This tool allows the agent to ask the user questions and wait for responses.
//! Questions are sent via `OutputChunk::AskUserQuestion`, and the tool waits
//! for the user's answers via `InputMessage::UserQuestionResponse`.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::core::output::{QuestionOption, UserQuestion};
use crate::llm::{ToolDefinition, ToolInputSchema};
use crate::runtime::AgentInternals;

/// Input for a single question option
#[derive(Debug, Deserialize)]
struct QuestionOptionInput {
    label: String,
    description: String,
}

/// Input for a single question
#[derive(Debug, Deserialize)]
struct QuestionInput {
    question: String,
    header: String,
    options: Vec<QuestionOptionInput>,
    #[serde(rename = "multiSelect", default)]
    multi_select: bool,
}

/// Input for the AskUserQuestion tool
#[derive(Debug, Deserialize)]
struct AskUserQuestionInput {
    questions: Vec<QuestionInput>,
    /// Pre-filled answers (optional, not typically used)
    #[serde(default)]
    answers: Option<std::collections::HashMap<String, String>>,
}

/// AskUserQuestion tool for interacting with users
///
/// This tool allows the agent to ask the user questions with multiple-choice
/// options and receive their responses.
pub struct AskUserQuestionTool;

impl AskUserQuestionTool {
    /// Create a new AskUserQuestion tool
    pub fn new() -> Self {
        Self
    }
}

impl Default for AskUserQuestionTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &str {
        "AskUserQuestion"
    }

    fn description(&self) -> &str {
        "Ask the user questions to gather information, clarify requirements, or get decisions."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "AskUserQuestion".to_string(),
            description: Some(
                "Use this tool to ask the user questions during execution. This allows you to:\n\
                1. Gather user preferences or requirements\n\
                2. Clarify ambiguous instructions\n\
                3. Get decisions on implementation choices as you work\n\
                4. Offer choices to the user about what direction to take.\n\n\
                Usage notes:\n\
                - Users will always be able to select \"Other\" to provide custom text input\n\
                - Use multiSelect: true to allow multiple answers to be selected for a question\n\
                - If you recommend a specific option, make that the first option in the list and add \"(Recommended)\" at the end of the label"
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "questions": {
                        "type": "array",
                        "description": "Questions to ask the user (1-4 questions)",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The complete question to ask the user. Should be clear, specific, and end with a question mark."
                                },
                                "header": {
                                    "type": "string",
                                    "description": "Very short label displayed as a chip/tag (max 12 chars). Examples: \"Auth method\", \"Library\", \"Approach\"."
                                },
                                "options": {
                                    "type": "array",
                                    "description": "The available choices for this question. Must have 2-4 options.",
                                    "minItems": 2,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "type": "string",
                                                "description": "The display text for this option (1-5 words)."
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "Explanation of what this option means or what will happen if chosen."
                                            }
                                        },
                                        "required": ["label", "description"]
                                    }
                                },
                                "multiSelect": {
                                    "type": "boolean",
                                    "default": false,
                                    "description": "Set to true to allow the user to select multiple options."
                                }
                            },
                            "required": ["question", "header", "options", "multiSelect"]
                        }
                    },
                    "answers": {
                        "type": "object",
                        "description": "Optional pre-filled answers (header -> selected label)",
                        "additionalProperties": {
                            "type": "string"
                        }
                    }
                })),
                required: Some(vec!["questions".to_string()]),
            },
            tool_type: None,
            cache_control: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let question_count = input
            .get("questions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len())
            .unwrap_or(0);

        ToolInfo {
            name: "AskUserQuestion".to_string(),
            action_description: format!("Ask user {} question(s)", question_count),
            details: None,
        }
    }

    async fn execute(&self, input: &Value, internals: &mut AgentInternals) -> Result<ToolResult> {
        // Parse the input
        let ask_input: AskUserQuestionInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid AskUserQuestion input: {}", e))?;

        // Validate: 1-4 questions
        if ask_input.questions.is_empty() {
            return Ok(ToolResult::error("At least one question is required"));
        }
        if ask_input.questions.len() > 4 {
            return Ok(ToolResult::error("Maximum of 4 questions allowed"));
        }

        // Validate each question has 2-4 options
        for (i, q) in ask_input.questions.iter().enumerate() {
            if q.options.len() < 2 {
                return Ok(ToolResult::error(format!(
                    "Question {} ('{}') must have at least 2 options",
                    i + 1,
                    q.header
                )));
            }
            if q.options.len() > 4 {
                return Ok(ToolResult::error(format!(
                    "Question {} ('{}') can have at most 4 options",
                    i + 1,
                    q.header
                )));
            }
        }

        // Convert to UserQuestion format
        let questions: Vec<UserQuestion> = ask_input
            .questions
            .into_iter()
            .map(|q| UserQuestion {
                question: q.question,
                header: q.header,
                options: q
                    .options
                    .into_iter()
                    .map(|o| QuestionOption {
                        label: o.label,
                        description: o.description,
                    })
                    .collect(),
                multi_select: q.multi_select,
            })
            .collect();

        // Generate a unique request ID
        let request_id = format!("ask_{}", uuid::Uuid::new_v4());

        // Call the helper method to ask questions and wait for response
        match internals.ask_user_question(&request_id, questions).await {
            Ok(answers) => {
                // Format answers as JSON for the tool result
                let answers_json = serde_json::to_string_pretty(&answers)
                    .unwrap_or_else(|_| format!("{:?}", answers));
                Ok(ToolResult::success(format!(
                    "User responded with the following answers:\n{}",
                    answers_json
                )))
            }
            Err(crate::core::FrameworkError::Interrupted) => {
                Ok(ToolResult::error("User interrupted the question"))
            }
            Err(crate::core::FrameworkError::Shutdown) => {
                Ok(ToolResult::error("Shutdown requested"))
            }
            Err(crate::core::FrameworkError::ChannelClosed) => {
                Ok(ToolResult::error("Connection closed before receiving response"))
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to get user response: {}", e))),
        }
    }

    fn requires_permission(&self) -> bool {
        false // Questions ARE the user interaction, no additional permission needed
    }
}
