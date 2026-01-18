//! Standardized Agent Module
//!
//! Provides a complete, configurable agent implementation that handles:
//! - The main agent loop (input → LLM → tools → output)
//! - Permission-aware tool execution
//! - Context injection before LLM calls
//! - Session persistence
//!
//! # Overview
//!
//! Instead of writing the agent loop yourself, you can use `StandardAgent`:
//!
//! ```ignore
//! // Configure the agent
//! let config = AgentConfig::new("You are a helpful assistant")
//!     .with_tools(tools)
//!     .with_injection_chain(injections);
//!
//! // Create and run
//! let agent = StandardAgent::new(config, llm);
//! let handle = runtime.spawn(session, |internals| agent.run(internals)).await;
//! ```
//!
//! # Components
//!
//! - `AgentConfig` - Configuration for the agent (system prompt, tools, injections)
//! - `StandardAgent` - The agent implementation
//! - `ToolExecutor` - Handles permission-aware tool execution

mod config;
mod executor;
mod standard_loop;

pub use config::AgentConfig;
pub use executor::ToolExecutor;
pub use standard_loop::StandardAgent;
