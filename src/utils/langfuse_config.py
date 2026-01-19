"""Langfuse integration for monitoring and tracing.

This module provides utilities for integrating Langfuse monitoring into the
multi-agent system to track costs, latency, and performance.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from .config import (
    LANGFUSE_DEBUG,
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    RAG_MODEL,
    ROUTER_MODEL,
    SQL_MODEL,
    SYNTHESIS_MODEL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model Pricing (per 1K tokens)
# =============================================================================

MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input": 0.000075,  # $0.000075 per 1K input tokens
        "output": 0.0003,   # $0.0003 per 1K output tokens
    },
    "gemini-1.5-flash": {
        "input": 0.000075,
        "output": 0.0003,
    },
    "gemini-2.5-pro": {
        "input": 0.00125,   # $0.00125 per 1K input tokens
        "output": 0.005,    # $0.005 per 1K output tokens
    },
    "gemini-1.5-pro": {
        "input": 0.00125,
        "output": 0.005,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost for a model call.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    if model not in MODEL_PRICING:
        logger.warning(f"Unknown model for pricing: {model}")
        return 0.0

    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    return input_cost + output_cost


class LangfuseManager:
    """Manages Langfuse client and callback handlers for monitoring."""

    _instance: Optional["LangfuseManager"] = None
    _client: Optional[Langfuse] = None

    def __new__(cls) -> "LangfuseManager":
        """Singleton pattern to ensure single Langfuse client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize Langfuse client if enabled."""
        if not LANGFUSE_ENABLED:
            logger.info("Langfuse monitoring is disabled")
            return

        if self._client is None:
            try:
                self._client = Langfuse(
                    public_key=LANGFUSE_PUBLIC_KEY,
                    secret_key=LANGFUSE_SECRET_KEY,
                    host=LANGFUSE_HOST,
                    debug=LANGFUSE_DEBUG,
                )
                logger.info("✅ Langfuse client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self._client = None

    def get_callback_handler(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[CallbackHandler]:
        """Get a Langfuse callback handler for tracing.

        Args:
            session_id: Session identifier for grouping traces
            user_id: User identifier
            trace_name: Name of the trace (e.g., "router_agent", "rag_agent")
            metadata: Additional metadata to attach to the trace
            tags: Tags for categorizing traces

        Returns:
            CallbackHandler instance or None if Langfuse is disabled
        """
        if not LANGFUSE_ENABLED or self._client is None:
            return None

        try:
            # CallbackHandler is very simple - just create it
            # Metadata is passed via LangChain callbacks, not here
            handler = CallbackHandler()
            return handler
        except Exception as e:
            logger.error(f"Failed to create callback handler: {e}")
            return None

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")

    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        return self._client


# Global Langfuse manager instance
langfuse_manager = LangfuseManager()


def get_langfuse_handler(
    agent_name: str,
    session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[CallbackHandler]:
    """Convenience function to get a Langfuse callback handler.

    Args:
        agent_name: Name of the agent (e.g., "router", "rag", "sql")
        session_id: Session identifier
        metadata: Additional metadata

    Returns:
        CallbackHandler instance or None
    """
    tags = [f"agent:{agent_name}"]

    return langfuse_manager.get_callback_handler(
        session_id=session_id,
        trace_name=f"{agent_name}_agent",
        metadata=metadata,
        tags=tags,
    )


def flush_langfuse() -> None:
    """Flush all pending Langfuse traces."""
    langfuse_manager.flush()


# =============================================================================
# Custom Metrics Tracking
# =============================================================================

class MetricsTracker:
    """Track custom metrics for agent execution."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start tracking execution time."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop tracking and return latency in seconds.

        Returns:
            Latency in seconds
        """
        if self.start_time is None:
            return 0.0

        latency = time.time() - self.start_time
        self.metrics["latency_seconds"] = latency
        return latency

    def add_metric(self, key: str, value: Any) -> None:
        """Add a custom metric.

        Args:
            key: Metric name
            value: Metric value
        """
        self.metrics[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()


@contextmanager
def track_agent_execution(
    agent_name: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Context manager for tracking agent execution with metrics.

    Args:
        agent_name: Name of the agent
        session_id: Optional session ID
        metadata: Optional metadata

    Yields:
        MetricsTracker instance

    Example:
        ```python
        with track_agent_execution("rag", session_id="123") as tracker:
            # Your agent code here
            tracker.add_metric("documents_retrieved", 5)
        # Metrics are automatically logged
        ```
    """
    tracker = MetricsTracker()
    tracker.start()

    try:
        yield tracker
    finally:
        latency = tracker.stop()
        metrics = tracker.get_metrics()

        # Log metrics
        logger.info(
            f"[{agent_name}] Execution completed in {latency:.3f}s | Metrics: {metrics}"
        )

        # Send to Langfuse if enabled
        if LANGFUSE_ENABLED and langfuse_manager.client:
            try:
                trace_metadata = metadata or {}
                trace_metadata.update(metrics)

                # Create an event for this execution
                langfuse_manager.client.event(
                    name=f"{agent_name}_execution",
                    session_id=session_id,
                    metadata=trace_metadata,
                )
            except Exception as e:
                logger.error(f"Failed to send metrics to Langfuse: {e}")


def log_query_metrics(
    agent_name: str,
    question: str,
    answer: str,
    latency: float,
    session_id: Optional[str] = None,
    num_documents: Optional[int] = None,
    tools_used: Optional[list] = None,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Log comprehensive query metrics to Langfuse.

    Args:
        agent_name: Name of the agent
        question: Input question
        answer: Generated answer
        latency: Execution latency in seconds
        session_id: Optional session ID
        num_documents: Number of documents retrieved (for RAG)
        tools_used: List of tools used (for SQL)
        model: Model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    if not LANGFUSE_ENABLED or not langfuse_manager.client:
        return

    try:
        metadata = {
            "agent": agent_name,
            "latency_seconds": latency,
            "latency_ms": latency * 1000,
        }

        if num_documents is not None:
            metadata["documents_retrieved"] = num_documents

        if tools_used:
            metadata["tools_used"] = tools_used
            metadata["num_tools_used"] = len(tools_used)

        if model:
            metadata["model"] = model

        if input_tokens is not None and output_tokens is not None:
            metadata["input_tokens"] = input_tokens
            metadata["output_tokens"] = output_tokens
            metadata["total_tokens"] = input_tokens + output_tokens

            # Calculate cost
            if model:
                cost = calculate_cost(model, input_tokens, output_tokens)
                metadata["estimated_cost_usd"] = cost

        # Create generation event in Langfuse
        langfuse_manager.client.generation(
            name=f"{agent_name}_query",
            input=question,
            output=answer,
            session_id=session_id,
            metadata=metadata,
        )

        logger.debug(f"Logged metrics to Langfuse: {metadata}")

    except Exception as e:
        logger.error(f"Failed to log query metrics: {e}")


# Example usage
if __name__ == "__main__":
    # Test Langfuse initialization
    manager = LangfuseManager()

    if manager.client:
        print("[OK] Langfuse client initialized")

        # Test callback handler
        handler = get_langfuse_handler("test_agent")
        if handler:
            print("[OK] Callback handler created")

        # Test cost calculation
        test_cost = calculate_cost("gemini-2.5-flash", 1000, 500)
        print(f"[OK] Estimated cost for 1K input + 500 output tokens: ${test_cost:.6f}")

        # Test metrics tracker
        with track_agent_execution("test_agent", session_id="test123") as tracker:
            time.sleep(0.1)  # Simulate work
            tracker.add_metric("test_metric", 42)

        print("[OK] Metrics tracking tested")

    else:
        print("[WARN] Langfuse client not available")
