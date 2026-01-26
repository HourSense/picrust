//! Web Fetch tool using Firecrawl API
//!
//! This tool fetches and scrapes web pages into LLM-ready markdown using Firecrawl.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use super::super::tool::{Tool, ToolInfo, ToolResult};
use crate::llm::{ToolDefinition, ToolInputSchema};
use crate::runtime::AgentInternals;

/// Default timeout in milliseconds (60 seconds)
const DEFAULT_TIMEOUT_MS: u64 = 60000;
/// Maximum output length in characters
const MAX_OUTPUT_LENGTH: usize = 100000;
/// Firecrawl API endpoint for scraping
const FIRECRAWL_SCRAPE_URL: &str = "https://api.firecrawl.dev/v2/scrape";
/// Firecrawl API endpoint for searching
const FIRECRAWL_SEARCH_URL: &str = "https://api.firecrawl.dev/v2/search";

/// Web Fetch tool using Firecrawl
pub struct WebFetchTool {
    /// API key for Firecrawl
    api_key: String,
    /// HTTP client
    client: reqwest::Client,
}

/// Input for the web fetch tool
#[derive(Debug, Deserialize)]
struct WebFetchInput {
    /// The URL to scrape (required for scrape mode)
    url: Option<String>,
    /// Search query (required for search mode)
    query: Option<String>,
    /// Formats to return: "markdown", "html", "links" (default: ["markdown"])
    formats: Option<Vec<String>>,
    /// Number of search results (default 5, only for search mode)
    limit: Option<u32>,
    /// Optional description of what this fetch does
    description: Option<String>,
}

/// Firecrawl scrape request
#[derive(Debug, Serialize)]
struct FirecrawlScrapeRequest {
    url: String,
    formats: Vec<String>,
}

/// Firecrawl search request
#[derive(Debug, Serialize)]
struct FirecrawlSearchRequest {
    query: String,
    limit: u32,
}

/// Firecrawl scrape response
#[derive(Debug, Deserialize)]
struct FirecrawlScrapeResponse {
    success: bool,
    data: Option<FirecrawlScrapeData>,
    error: Option<String>,
}

/// Firecrawl scrape data
#[derive(Debug, Deserialize)]
struct FirecrawlScrapeData {
    markdown: Option<String>,
    html: Option<String>,
    metadata: Option<FirecrawlMetadata>,
}

/// Firecrawl metadata
#[derive(Debug, Deserialize)]
struct FirecrawlMetadata {
    title: Option<String>,
    description: Option<String>,
    #[serde(rename = "sourceURL")]
    source_url: Option<String>,
}

/// Firecrawl search response
#[derive(Debug, Deserialize)]
struct FirecrawlSearchResponse {
    success: bool,
    data: Option<FirecrawlSearchData>,
    error: Option<String>,
}

/// Firecrawl search data
#[derive(Debug, Deserialize)]
struct FirecrawlSearchData {
    web: Option<Vec<FirecrawlSearchResult>>,
}

/// Individual search result from Firecrawl
#[derive(Debug, Deserialize)]
struct FirecrawlSearchResult {
    title: Option<String>,
    url: String,
    description: Option<String>,
}

impl WebFetchTool {
    /// Create a new Web Fetch tool
    ///
    /// The API key is read from the FIRECRAWL_API_KEY environment variable.
    pub fn new() -> Result<Self> {
        let api_key = std::env::var("FIRECRAWL_API_KEY")
            .map_err(|_| anyhow::anyhow!("FIRECRAWL_API_KEY environment variable not set"))?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(DEFAULT_TIMEOUT_MS))
            .build()?;

        Ok(Self { api_key, client })
    }

    /// Create a new Web Fetch tool with an explicit API key
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(DEFAULT_TIMEOUT_MS))
            .build()?;

        Ok(Self {
            api_key: api_key.into(),
            client,
        })
    }

    /// Scrape a URL and return its content as markdown
    async fn scrape(&self, url: &str, formats: Vec<String>) -> Result<String> {
        tracing::info!("Scraping URL: {}", url);

        let request_body = FirecrawlScrapeRequest {
            url: url.to_string(),
            formats,
        };

        let response = self
            .client
            .post(FIRECRAWL_SCRAPE_URL)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Firecrawl API error ({}): {}", status, error_text);
        }

        let scrape_response: FirecrawlScrapeResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse response: {}", e))?;

        if !scrape_response.success {
            let error = scrape_response
                .error
                .unwrap_or_else(|| "Unknown error".to_string());
            anyhow::bail!("Firecrawl scrape failed: {}", error);
        }

        let data = scrape_response
            .data
            .ok_or_else(|| anyhow::anyhow!("No data in response"))?;

        // Format output
        let mut output = String::new();

        // Add metadata header
        if let Some(metadata) = &data.metadata {
            if let Some(title) = &metadata.title {
                output.push_str(&format!("# {}\n\n", title));
            }
            if let Some(desc) = &metadata.description {
                output.push_str(&format!("> {}\n\n", desc));
            }
            if let Some(source) = &metadata.source_url {
                output.push_str(&format!("Source: {}\n\n", source));
            }
            output.push_str("---\n\n");
        }

        // Add markdown content
        if let Some(markdown) = data.markdown {
            output.push_str(&markdown);
        } else if let Some(html) = data.html {
            output.push_str("(HTML content returned, markdown not available)\n\n");
            // Truncate HTML if too long
            let html_preview = if html.len() > 5000 {
                format!("{}...", &html[..5000])
            } else {
                html
            };
            output.push_str(&html_preview);
        }

        // Truncate if too long
        if output.len() > MAX_OUTPUT_LENGTH {
            output.truncate(MAX_OUTPUT_LENGTH);
            output.push_str("\n\n... (content truncated)");
        }

        tracing::debug!("Scrape completed, output length: {} chars", output.len());

        Ok(output)
    }

    /// Search the web and return results
    async fn search(&self, query: &str, limit: u32) -> Result<String> {
        tracing::info!("Searching web: {}", query);

        let request_body = FirecrawlSearchRequest {
            query: query.to_string(),
            limit,
        };

        let response = self
            .client
            .post(FIRECRAWL_SEARCH_URL)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Firecrawl API error ({}): {}", status, error_text);
        }

        let search_response: FirecrawlSearchResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse response: {}", e))?;

        if !search_response.success {
            let error = search_response
                .error
                .unwrap_or_else(|| "Unknown error".to_string());
            anyhow::bail!("Firecrawl search failed: {}", error);
        }

        let data = search_response
            .data
            .ok_or_else(|| anyhow::anyhow!("No data in response"))?;

        // Format results
        let mut output = String::new();
        output.push_str(&format!("# Search Results for: {}\n\n", query));

        if let Some(results) = data.web {
            output.push_str(&format!("Found {} results:\n\n", results.len()));

            for (i, result) in results.iter().enumerate() {
                let title = result.title.as_deref().unwrap_or("No title");
                output.push_str(&format!("## {}. {}\n", i + 1, title));
                output.push_str(&format!("URL: {}\n", result.url));

                if let Some(desc) = &result.description {
                    output.push_str(&format!("\n{}\n", desc));
                }
                output.push('\n');
            }
        } else {
            output.push_str("No results found.\n");
        }

        tracing::debug!("Search completed");

        Ok(output)
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "WebFetch"
    }

    fn description(&self) -> &str {
        "Fetch and scrape web pages into LLM-ready markdown, or search the web."
    }

    fn definition(&self) -> ToolDefinition {
        use crate::llm::types::CustomTool;

        ToolDefinition::Custom(CustomTool {
            name: "WebFetch".to_string(),
            description: Some(
                "Fetches web pages and converts them to clean markdown using Firecrawl. \
                Can either scrape a specific URL or search the web. \
                Use 'url' parameter to scrape a page, or 'query' parameter to search. \
                Returns LLM-ready markdown content."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(json!({
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape and convert to markdown"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query to find web pages (use instead of url for searching)"
                    },
                    "formats": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Output formats: 'markdown', 'html', 'links' (default: ['markdown'])"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of search results to return (default 5, only for search mode)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear, concise description of what this fetch does in 5-10 words, in active voice."
                    }
                })),
                required: Some(vec![]),
            },
            tool_type: None,
            cache_control: None,
        })
    }

    fn get_info(&self, input: &Value) -> ToolInfo {
        let url = input.get("url").and_then(|v| v.as_str());
        let query = input.get("query").and_then(|v| v.as_str());

        let description = input
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from);

        let action = description.unwrap_or_else(|| {
            if let Some(url) = url {
                format!("Fetch: {}", url)
            } else if let Some(query) = query {
                format!("Search: {}", query)
            } else {
                "WebFetch".to_string()
            }
        });

        let details = if let Some(url) = url {
            Some(format!("URL: {}", url))
        } else if let Some(query) = query {
            Some(format!("Query: {}", query))
        } else {
            None
        };

        ToolInfo {
            name: "WebFetch".to_string(),
            action_description: action,
            details,
        }
    }

    async fn execute(&self, input: &Value, _internals: &mut AgentInternals) -> Result<ToolResult> {
        let fetch_input: WebFetchInput = serde_json::from_value(input.clone())
            .map_err(|e| anyhow::anyhow!("Invalid web fetch input: {}", e))?;

        if let Some(ref desc) = fetch_input.description {
            tracing::info!("Fetch description: {}", desc);
        }

        // Determine mode: scrape URL or search
        if let Some(url) = fetch_input.url {
            let formats = fetch_input
                .formats
                .unwrap_or_else(|| vec!["markdown".to_string()]);

            match self.scrape(&url, formats).await {
                Ok(output) => Ok(ToolResult::success(output)),
                Err(e) => Ok(ToolResult::error(format!("Scrape failed: {}", e))),
            }
        } else if let Some(query) = fetch_input.query {
            let limit = fetch_input.limit.unwrap_or(5).min(10);

            match self.search(&query, limit).await {
                Ok(output) => Ok(ToolResult::success(output)),
                Err(e) => Ok(ToolResult::error(format!("Search failed: {}", e))),
            }
        } else {
            Ok(ToolResult::error(
                "Either 'url' or 'query' parameter is required".to_string(),
            ))
        }
    }

    fn requires_permission(&self) -> bool {
        false // Read-only web fetch doesn't need permission
    }
}
