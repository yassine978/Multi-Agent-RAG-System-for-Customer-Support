"""Main module for the TelecomPlus multi-agent support system.

This module provides the main entry point for the application, integrating
all agents through the orchestrator to answer customer questions.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Handle both relative and absolute imports
if __name__ == "__main__":
    # Running as script - use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.agents.orchestrator import get_orchestrator
    from src.data.excel_loader import get_excel_loader
    from src.data.pdf_indexer import get_pdf_indexer
    from src.utils.config import validate_config
    from src.utils.langfuse_config import (
        flush_langfuse,
        log_query_metrics,
        track_agent_execution,
    )
    from src.utils.logger import setup_logger
else:
    # Running as module - use relative imports
    from .agents.orchestrator import get_orchestrator
    from .data.excel_loader import get_excel_loader
    from .data.pdf_indexer import get_pdf_indexer
    from .utils.config import validate_config
    from .utils.langfuse_config import (
        flush_langfuse,
        log_query_metrics,
        track_agent_execution,
    )
    from .utils.logger import setup_logger

logger = setup_logger(__name__)

# Global initialization flag
_initialized = False
_initialization_error: Optional[str] = None


def initialize_system(force_reload: bool = False) -> bool:
    """Initialize the multi-agent system.

    This function:
    1. Validates configuration
    2. Loads Excel data
    3. Initializes or loads vector database
    4. Prepares all agents

    Args:
        force_reload: If True, force reload of all components

    Returns:
        True if initialization successful, False otherwise
    """
    global _initialized, _initialization_error

    if _initialized and not force_reload:
        logger.info("System already initialized")
        return True

    logger.info("=" * 80)
    logger.info("INITIALIZING TELECOMPLUS MULTI-AGENT SYSTEM")
    logger.info("=" * 80)

    try:
        # Step 1: Validate configuration
        logger.info("Step 1: Validating configuration...")
        validate_config()
        logger.info("✓ Configuration validated")

        # Step 2: Load Excel data
        logger.info("Step 2: Loading Excel data...")
        excel_loader = get_excel_loader(force_reload=force_reload)
        tables = excel_loader.get_all_tables()
        logger.info(f"✓ Loaded {len(tables)} Excel tables")

        # Step 3: Initialize vector database
        logger.info("Step 3: Initializing vector database...")
        pdf_indexer = get_pdf_indexer(force_recreate=force_reload)

        # Check if vectorstore exists
        vectorstore = pdf_indexer.get_vectorstore()
        if vectorstore is None:
            logger.info("Vector database not found. Creating new index...")
            logger.info("(This may take a few minutes on first run)")
            vectorstore = pdf_indexer.index_pdfs(force_recreate=False)
        else:
            logger.info("✓ Vector database loaded from disk")

        # Step 4: Initialize orchestrator (lazy loads agents)
        logger.info("Step 4: Initializing orchestrator...")
        orchestrator = get_orchestrator()
        logger.info("✓ Orchestrator initialized")

        # Mark as initialized
        _initialized = True
        _initialization_error = None

        logger.info("=" * 80)
        logger.info("SYSTEM INITIALIZATION COMPLETE")
        logger.info("=" * 80)

        return True

    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logger.error(error_msg)
        _initialization_error = error_msg
        _initialized = False

        import traceback
        traceback.print_exc()

        return False


def answer(question: str, session_id: Optional[str] = None) -> str:
    """Answer customer questions using the multi-agent system.

    This is the main entry point called by the Streamlit app.

    Args:
        question: Customer question in French
        session_id: Optional session ID for tracking (default: generates one)

    Returns:
        Answer string
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = f"session_{int(time.time())}"

    logger.info(f"Processing question (session: {session_id}): {question[:100]}...")

    try:
        # Ensure system is initialized
        if not _initialized:
            logger.info("System not initialized. Initializing now...")
            success = initialize_system()

            if not success:
                return (
                    "Désolé, le système n'est pas encore prêt. "
                    "Veuillez réessayer dans quelques instants. "
                    f"Erreur: {_initialization_error or 'Erreur inconnue'}"
                )

        # Get orchestrator
        orchestrator = get_orchestrator()

        # Process question
        start_time = time.time()
        result = orchestrator.process_question(question, session_id=session_id)
        elapsed_time = time.time() - start_time

        # Log result
        logger.info(f"Question processed in {elapsed_time:.2f}s")
        logger.info(f"Question type: {result['question_type']}")
        logger.info(f"Used RAG: {result['used_rag']}, Used SQL: {result['used_sql']}")
        logger.info(f"Agents used: {result.get('agents_used', [])}")
        logger.info(f"Tools used: {result.get('tools_used', [])}")

        # Return answer
        answer_text = result.get("answer", "Désolé, je n'ai pas pu générer une réponse.")

        # Flush Langfuse traces
        try:
            flush_langfuse()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")

        return answer_text

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()

        return (
            "Désolé, une erreur s'est produite lors du traitement de votre question. "
            "Veuillez reformuler ou réessayer plus tard."
        )


def answer_with_metadata(
    question: str,
    session_id: Optional[str] = None
) -> dict:
    """Answer a question and return full metadata with comprehensive tracking.

    Args:
        question: User's question
        session_id: Optional session ID for tracking

    Returns:
        Dictionary with answer and metadata including agents and tools used
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = f"session_{int(time.time())}"

    logger.info(f"Processing question with metadata (session: {session_id}): {question[:100]}...")

    try:
        # Ensure system is initialized
        if not _initialized:
            logger.info("System not initialized. Initializing now...")
            success = initialize_system()

            if not success:
                return {
                    "answer": f"Désolé, le système n'est pas encore prêt. Erreur: {_initialization_error or 'Erreur inconnue'}",
                    "question": question,
                    "error": _initialization_error or "Initialization failed",
                }

        # Get orchestrator
        orchestrator = get_orchestrator()

        # Track execution with metrics
        with track_agent_execution("orchestrator", session_id=session_id) as tracker:
            # Process question
            start_time = time.time()
            result = orchestrator.process_question(question, session_id=session_id)
            elapsed_time = time.time() - start_time

            # Add metrics to tracker
            tracker.add_metric("question_type", result.get("question_type", "unknown"))
            tracker.add_metric("agents_used", result.get("agents_used", []))
            tracker.add_metric("tools_used", result.get("tools_used", []))
            tracker.add_metric("used_rag", result.get("used_rag", False))
            tracker.add_metric("used_sql", result.get("used_sql", False))

            # Add processing time to result
            result["processing_time_seconds"] = round(elapsed_time, 2)

            # Log comprehensive metrics to Langfuse
            log_query_metrics(
                agent_name="orchestrator",
                question=question,
                answer=result.get("answer", ""),
                latency=elapsed_time,
                session_id=session_id,
                tools_used=result.get("tools_used", []),
            )

        # Log result
        logger.info(f"Question processed in {elapsed_time:.2f}s")
        logger.info(f"Agents used: {result.get('agents_used', [])}")
        logger.info(f"Tools used: {result.get('tools_used', [])}")

        # Flush Langfuse traces
        try:
            flush_langfuse()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")

        return result

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()

        return {
            "answer": "Désolé, une erreur s'est produite lors du traitement de votre question.",
            "question": question,
            "error": str(e),
        }


def answer_streaming(question: str, session_id: Optional[str] = None):
    """Answer customer questions with streaming support.

    This is a generator function that yields tokens as they are generated.

    Args:
        question: Customer question in French
        session_id: Optional session ID for tracking

    Yields:
        Token strings as they are generated
    """
    # Generate session ID if not provided
    if session_id is None:
        session_id = f"session_{int(time.time())}"

    logger.info(f"Processing streaming question (session: {session_id}): {question[:100]}...")

    try:
        # Ensure system is initialized
        if not _initialized:
            logger.info("System not initialized. Initializing now...")
            success = initialize_system()

            if not success:
                yield "Désolé, le système n'est pas encore prêt. Veuillez réessayer dans quelques instants."
                return

        # Get orchestrator
        orchestrator = get_orchestrator()

        # Process question with streaming
        # Note: We'll route and retrieve first, then stream the final answer
        from .agents.router_agent import get_router_agent
        from .agents.rag_agent import get_rag_agent
        from .agents.sql_agent import get_sql_agent

        # Route question
        router = get_router_agent()
        routing_info = router.route_question(question, session_id)
        question_type = routing_info["type"]

        logger.info(f"Question type: {question_type}")

        # Gather context based on type
        rag_context = ""
        sql_context = ""

        if question_type in ["RAG", "HYBRID"]:
            rag_agent = get_rag_agent()
            rag_result = rag_agent.answer_question(question, session_id)
            rag_context = rag_result["answer"]

        if question_type in ["SQL", "HYBRID"]:
            sql_agent = get_sql_agent()
            sql_result = sql_agent.answer_question(question, session_id)
            sql_context = sql_result["answer"]

        # For GENERAL questions, use simple response
        if question_type == "GENERAL":
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            from .prompts.templates import GENERAL_RESPONSE_PROMPT
            from .utils.config import SYNTHESIS_MODEL, GOOGLE_API_KEY

            llm = ChatGoogleGenerativeAI(
                model=SYNTHESIS_MODEL,
                temperature=0.5,
                google_api_key=GOOGLE_API_KEY,
                streaming=True,
            )

            messages = [HumanMessage(content=GENERAL_RESPONSE_PROMPT.format(question=question))]

            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content

        # For RAG/SQL/HYBRID, synthesize with streaming
        elif question_type in ["RAG", "SQL", "HYBRID"]:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage
            from .prompts.templates import SYNTHESIS_PROMPT_TEMPLATE, SYNTHESIS_SYSTEM_PROMPT
            from .utils.config import SYNTHESIS_MODEL, SYNTHESIS_TEMPERATURE, GOOGLE_API_KEY

            llm = ChatGoogleGenerativeAI(
                model=SYNTHESIS_MODEL,
                temperature=SYNTHESIS_TEMPERATURE,
                google_api_key=GOOGLE_API_KEY,
                streaming=True,
            )

            if question_type == "RAG":
                answer_to_stream = rag_context
            elif question_type == "SQL":
                answer_to_stream = sql_context
            else:  # HYBRID
                # Synthesize both sources with streaming
                messages = [
                    SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                    HumanMessage(
                        content=SYNTHESIS_PROMPT_TEMPLATE.format(
                            question=question,
                            rag_results=rag_context or "Aucune information des FAQ",
                            sql_results=sql_context or "Aucune donnée client",
                        )
                    ),
                ]

                for chunk in llm.stream(messages):
                    if hasattr(chunk, 'content'):
                        yield chunk.content

                # Flush Langfuse
                try:
                    flush_langfuse()
                except Exception as e:
                    logger.warning(f"Failed to flush Langfuse: {e}")

                return

            # For RAG or SQL only, just yield the answer token by token
            for token in answer_to_stream:
                yield token
                time.sleep(0.01)  # Small delay for better UX

        # Flush Langfuse traces
        try:
            flush_langfuse()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")

    except Exception as e:
        logger.error(f"Error processing streaming question: {e}")
        import traceback
        traceback.print_exc()

        yield "Désolé, une erreur s'est produite lors du traitement de votre question."


def get_system_status() -> dict:
    """Get the current system status.

    Returns:
        Dictionary with system status information
    """
    return {
        "initialized": _initialized,
        "initialization_error": _initialization_error,
        "ready": _initialized and _initialization_error is None,
    }


# Auto-initialize on module import (for Streamlit)
# This happens in the background when the app starts
try:
    logger.info("Auto-initializing system on module import...")
    initialize_system()
except Exception as e:
    logger.warning(f"Auto-initialization failed: {e}")
    # Not critical - will retry on first question


if __name__ == "__main__":
    # Test the system
    print("\n" + "=" * 80)
    print("TESTING TELECOMPLUS MULTI-AGENT SYSTEM")
    print("=" * 80)

    # Initialize
    print("\nInitializing system...")
    success = initialize_system(force_reload=False)

    if not success:
        print("❌ Initialization failed")
        import sys
        sys.exit(1)

    print("✓ System initialized successfully\n")

    # Test questions
    test_questions = [
        "Bonjour, comment allez-vous ?",
        "Quels modes de paiement acceptez-vous ?",
        "Comment résilier mon abonnement ?",
    ]

    print("=" * 80)
    print("TESTING QUESTION ANSWERING")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}/{len(test_questions)}]")
        print(f"Q: {question}")
        print()

        answer_text = answer(question)
        print(f"A: {answer_text}")
        print("\n" + "-" * 80)

    # Show system status
    print("\n" + "=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)
    status = get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")

    print("\n✅ All tests completed successfully!")
