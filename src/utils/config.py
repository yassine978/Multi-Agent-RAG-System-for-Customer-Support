"""Configuration file for the TelecomPlus multi-agent system.

This module contains all configuration parameters for models, agents, RAG, and monitoring.
"""

import os
from typing import Final

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# API Keys and Authentication
# =============================================================================

GOOGLE_API_KEY: Final[str] = os.getenv("GOOGLE_API_KEY", "")
LANGFUSE_HOST: Final[str] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY: Final[str] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: Final[str] = os.getenv("LANGFUSE_SECRET_KEY", "")

# =============================================================================
# Model Configuration
# =============================================================================

# Router Agent: Fast classification model for question routing
# Using Gemini Flash for speed and cost-efficiency
ROUTER_MODEL: Final[str] = "gemini-2.5-flash"
ROUTER_TEMPERATURE: Final[float] = 0.0  # Deterministic classification
ROUTER_MAX_TOKENS: Final[int] = 100

# RAG Agent: Answer generation from PDF documents
# Using Gemini Flash for good balance of speed and quality
RAG_MODEL: Final[str] = "gemini-2.5-flash"
RAG_TEMPERATURE: Final[float] = 0.3  # Slightly creative but focused
RAG_MAX_TOKENS: Final[int] = 500

# SQL Agent: DataFrame querying
# Using Gemini Flash for query generation
SQL_MODEL: Final[str] = "gemini-2.5-flash"
SQL_TEMPERATURE: Final[float] = 0.0  # Deterministic query generation
SQL_MAX_TOKENS: Final[int] = 300

# Synthesis Agent: Combining multiple sources
# Using Gemini Pro for better reasoning when combining complex information
SYNTHESIS_MODEL: Final[str] = "gemini-2.5-pro"
SYNTHESIS_TEMPERATURE: Final[float] = 0.4
SYNTHESIS_MAX_TOKENS: Final[int] = 600

# Evaluation Judge: LLM-as-a-judge for evaluation
JUDGE_MODEL: Final[str] = "gemini-2.5-flash"
JUDGE_TEMPERATURE: Final[float] = 0.1
JUDGE_MAX_TOKENS: Final[int] = 200

# =============================================================================
# RAG Configuration
# =============================================================================

# Vector Database
VECTOR_DB_TYPE: Final[str] = "chroma"  # Options: "chroma", "faiss"
VECTOR_DB_PATH: Final[str] = "./data/vector_db"
VECTOR_DB_COLLECTION_NAME: Final[str] = "telecomplus_faqs"

# Embeddings
EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative: "models/embedding-001" for Google embeddings

# Document Processing
# Optimized for catalog and FAQ documents
CHUNK_SIZE: Final[int] = 2000  # Characters per chunk - large enough for FULL product descriptions (colors+specs+battery)
CHUNK_OVERLAP: Final[int] = 400  # Overlap between chunks - ensures facts aren't split across boundaries
PDF_DIRECTORY: Final[str] = "./data/pdfs"

# Retrieval
# Optimized based on evaluation questions
RETRIEVAL_TOP_K: Final[int] = 15  # Number of documents to retrieve (increased for better coverage)
RETRIEVAL_SCORE_THRESHOLD: Final[float] = 0.15  # Minimum similarity score (lowered to catch more relevant docs including tables)

# =============================================================================
# Excel Data Configuration
# =============================================================================

EXCEL_DIRECTORY: Final[str] = "./data/xlsx"

EXCEL_FILES: Final[dict] = {
    "clients": "clients.xlsx",
    "forfaits": "forfaits.xlsx",
    "abonnements": "abonnements.xlsx",
    "consommation": "consommation.xlsx",
    "factures": "factures.xlsx",
    "tickets_support": "tickets_support.xlsx",
}

# =============================================================================
# Agent Configuration
# =============================================================================

# Question classification types
QUESTION_TYPES: Final[dict] = {
    "RAG": "Question answered from FAQ documents (general information)",
    "SQL": "Question requiring customer data lookup from database",
    "HYBRID": "Question requiring both FAQ documents and customer data",
    "GENERAL": "General greeting or out-of-scope question",
}

# Agent timeout (seconds)
AGENT_TIMEOUT: Final[int] = 30

# =============================================================================
# Caching Configuration
# =============================================================================

CACHE_ENABLED: Final[bool] = True
CACHE_TTL: Final[int] = 3600  # Time to live in seconds (1 hour)

# =============================================================================
# Monitoring Configuration
# =============================================================================

LANGFUSE_ENABLED: Final[bool] = True
LANGFUSE_DEBUG: Final[bool] = False

# =============================================================================
# Evaluation Configuration
# =============================================================================

EVALUATION_QUESTIONS_PATH: Final[str] = "./data/evaluation_questions.xlsx"
EVALUATION_RESULTS_PATH: Final[str] = "./evaluation_results.xlsx"

# =============================================================================
# Application Configuration
# =============================================================================

APP_TITLE: Final[str] = "TelecomPlus - Support Client"
APP_ICON: Final[str] = "📱"

# Logging
LOG_LEVEL: Final[str] = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# Model Selection Rationale (Documentation)
# =============================================================================

"""
GEMINI MODEL SELECTION RATIONALE:

1. **Gemini 1.5 Flash** (Primary model for most tasks)
   - Use Cases: Router, RAG, SQL agents
   - Rationale:
     * Excellent cost-performance ratio
     * Fast response times (critical for user experience)
     * Sufficient reasoning capability for classification and answer generation
     * Low latency for real-time chat interface
   - Cost: ~$0.000075 per 1K input tokens, ~$0.0003 per 1K output tokens

2. **Gemini 1.5 Pro** (Advanced reasoning)
   - Use Cases: Synthesis agent for complex multi-source queries
   - Rationale:
     * Superior reasoning for combining information from multiple sources
     * Better at handling nuanced requirements
     * Used sparingly to control costs (only for hybrid queries)
   - Cost: ~$0.00125 per 1K input tokens, ~$0.005 per 1K output tokens

3. **MCP Consideration**:
   - Model Context Protocol could be beneficial for:
     * Structured data access patterns
     * Tool integration standardization
     * Future extensibility
   - Decision: Will evaluate during implementation phase
   - May implement if it provides clear architectural benefits

4. **Caching Strategy**:
   - Leverage Gemini API caching for repeated queries
   - Cache vector search results
   - Cache DataFrame query results
   - Expected cost reduction: 30-50%
"""


# =============================================================================
# Validation
# =============================================================================

def validate_config() -> None:
    """Validate that all required configuration is present."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    if LANGFUSE_ENABLED and not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
        raise ValueError("Langfuse credentials not found in environment variables")

    print("[OK] Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print(f"Router Model: {ROUTER_MODEL}")
    print(f"RAG Model: {RAG_MODEL}")
    print(f"SQL Model: {SQL_MODEL}")
    print(f"Synthesis Model: {SYNTHESIS_MODEL}")
    print(f"Judge Model: {JUDGE_MODEL}")
