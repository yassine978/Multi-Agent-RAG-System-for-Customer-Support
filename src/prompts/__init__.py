"""Prompt templates for agents.

This package contains all prompt templates used by different agents.
"""

from .templates import (
    ERROR_NO_RESULTS_PROMPT,
    ERROR_SYSTEM_PROMPT,
    GENERAL_RESPONSE_PROMPT,
    RAG_PROMPT_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    ROUTER_PROMPT_TEMPLATE,
    ROUTER_SYSTEM_PROMPT,
    SQL_PROMPT_TEMPLATE,
    SQL_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    format_dataframes_info,
    format_documents_for_context,
)

__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "ROUTER_PROMPT_TEMPLATE",
    "RAG_SYSTEM_PROMPT",
    "RAG_PROMPT_TEMPLATE",
    "SQL_SYSTEM_PROMPT",
    "SQL_PROMPT_TEMPLATE",
    "SYNTHESIS_SYSTEM_PROMPT",
    "SYNTHESIS_PROMPT_TEMPLATE",
    "GENERAL_RESPONSE_PROMPT",
    "ERROR_NO_RESULTS_PROMPT",
    "ERROR_SYSTEM_PROMPT",
    "format_documents_for_context",
    "format_dataframes_info",
]
