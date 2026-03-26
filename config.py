"""
Configuration file for Word Proofreader application.
Contains model configurations and other settings.
"""

# Prompts storage paths
PROMPTS_FILE_PATH = "prompts.json"   # Metadata (protected flag, etc.)
PROMPTS_DIR = "prompts"              # Directory containing .txt files with prompt content

# Available LLM models
POPULAR_MODELS = [
    "x-ai/grok-4.1-fast",
    "z-ai/glm-4.5",
    "z-ai/glm-4.5-air"
]

# Processing configuration
DEFAULT_CHUNK_SIZE = 100  # Number of paragraphs per chunk
DEFAULT_MAX_WORKERS = 5   # Number of parallel workers
DEFAULT_MAX_RETRIES = 3   # Number of retry attempts for failed chunks
DEFAULT_RETRY_DELAY = 1.0 # Initial retry delay in seconds

# Text matching configuration
SIMILARITY_THRESHOLD = 0.95  # Fuzzy matching threshold (0.0 to 1.0)

# UI configuration
DEFAULT_PARAGRAPHS_PER_PAGE = 50  # For document preview pagination
DEFAULT_EDITS_PER_PAGE = 10       # For suggested edits pagination
