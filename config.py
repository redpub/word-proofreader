"""
Configuration file for Word Proofreader application.
Contains model configurations and other settings.
"""

# Available LLM models
POPULAR_MODELS = [
    "x-ai/grok-4.1-fast",
    "z-ai/glm-4.5",
    "z-ai/glm-4.5-air",
    "deepseek/deepseek-v3.2"
]

# Processing configuration
DEFAULT_CHUNK_SIZE = 100   # Fallback number of paragraphs per chunk (used when model info unavailable)
MAX_CHUNK_SIZE = 500       # Upper cap on paragraphs per chunk to limit retry cost
DEFAULT_MAX_WORKERS = 5    # Number of parallel workers
DEFAULT_MAX_RETRIES = 3    # Number of retry attempts for failed chunks
DEFAULT_RETRY_DELAY = 1.0  # Initial retry delay in seconds

# Dynamic chunk sizing
TOKEN_CUSHION_FACTOR = 0.7          # Use only 70% of available tokens (30% safety margin)
TOKENS_PER_CHAR_ESTIMATE = 1.5     # Conservative estimate for CJK text tokenization
PROMPT_TEMPLATE_OVERHEAD = 500     # Estimated tokens for the user prompt template (rules, examples, etc.)
MIN_COMPLETION_RESERVE = 4000      # Minimum tokens reserved for the LLM's JSON response

# UI configuration
DEFAULT_PARAGRAPHS_PER_PAGE = 50  # For document preview pagination
DEFAULT_EDITS_PER_PAGE = 10       # For suggested edits pagination
