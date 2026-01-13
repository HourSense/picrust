use colored::*;
use std::io::{self, Write};

/// Console handles all terminal I/O with colored formatting
pub struct Console {
    user_color: Color,
    assistant_color: Color,
}

impl Console {
    /// Create a new Console with default colors
    pub fn new() -> Self {
        Self {
            user_color: Color::Cyan,
            assistant_color: Color::Green,
        }
    }

    /// Create a new Console with custom colors
    pub fn with_colors(user_color: Color, assistant_color: Color) -> Self {
        Self {
            user_color,
            assistant_color,
        }
    }

    /// Print a user message with colored formatting
    pub fn print_user(&self, message: &str) {
        println!("{} {}", "User:".color(self.user_color).bold(), message);
    }

    /// Print an assistant message prefix (without newline)
    pub fn print_assistant_prefix(&self) {
        print!("{} ", "Assistant:".color(self.assistant_color).bold());
        io::stdout().flush().unwrap();
    }

    /// Print a chunk of assistant response (for streaming)
    pub fn print_assistant_chunk(&self, chunk: &str) {
        print!("{}", chunk.color(self.assistant_color));
        io::stdout().flush().unwrap();
    }

    /// Print a complete assistant message with colored formatting
    pub fn print_assistant(&self, message: &str) {
        println!(
            "{} {}",
            "Assistant:".color(self.assistant_color).bold(),
            message.color(self.assistant_color)
        );
    }

    /// Print a newline
    pub fn println(&self) {
        println!();
    }

    /// Print a system message (errors, info, etc.)
    pub fn print_system(&self, message: &str) {
        println!("{} {}", "System:".yellow().bold(), message);
    }

    /// Print an error message
    pub fn print_error(&self, error: &str) {
        eprintln!("{} {}", "Error:".red().bold(), error);
    }

    /// Read a line of input from the user
    pub fn read_input(&self) -> io::Result<String> {
        print!("{} ", ">".color(self.user_color).bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        Ok(input.trim().to_string())
    }

    /// Print a welcome banner
    pub fn print_banner(&self) {
        println!("{}", "=".repeat(60).bright_blue());
        println!(
            "{}",
            "  Coding Agent - Powered by Claude".bright_blue().bold()
        );
        println!("{}", "=".repeat(60).bright_blue());
        println!();
        println!("Type your message and press Enter. Type 'exit' or 'quit' to end the session.");
        println!();
    }

    /// Print a separator line
    pub fn print_separator(&self) {
        println!("{}", "-".repeat(60).bright_black());
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}
