"""Router Agent: Question classification and routing.

This agent analyzes customer questions and determines which data source
or agent should handle the query.
"""

import logging
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..prompts.templates import ROUTER_PROMPT_TEMPLATE, ROUTER_SYSTEM_PROMPT
from ..utils.config import (
    GOOGLE_API_KEY,
    ROUTER_MAX_TOKENS,
    ROUTER_MODEL,
    ROUTER_TEMPERATURE,
)
from ..utils.langfuse_config import get_langfuse_handler
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Type definition for question types
QuestionType = Literal["RAG", "SQL", "HYBRID", "GENERAL"]


class RouterAgent:
    """Routes questions to appropriate agents based on classification."""

    def __init__(
        self,
        model_name: str = ROUTER_MODEL,
        temperature: float = ROUTER_TEMPERATURE,
        max_tokens: int = ROUTER_MAX_TOKENS,
    ):
        """Initialize Router Agent.

        Args:
            model_name: Name of the Gemini model to use
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=GOOGLE_API_KEY,
        )

        logger.info(f"RouterAgent initialized with model: {model_name}")

    def classify_question(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> QuestionType:
        """Classify a customer question.

        Args:
            question: Customer question to classify
            session_id: Optional session ID for tracking

        Returns:
            Question type: "RAG", "SQL", "HYBRID", or "GENERAL"
        """
        try:
            # Prepare messages
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(
                    content=ROUTER_PROMPT_TEMPLATE.format(question=question)
                ),
            ]

            # Get Langfuse handler for tracing
            langfuse_handler = get_langfuse_handler(
                agent_name="router",
                session_id=session_id,
                metadata={"question": question},
            )

            # Invoke LLM
            callbacks = [langfuse_handler] if langfuse_handler else []
            response = self.llm.invoke(messages, config={"callbacks": callbacks})

            # Extract and validate classification
            classification = response.content.strip().upper()

            # Map to valid types
            if "RAG" in classification:
                result = "RAG"
            elif "SQL" in classification:
                result = "SQL"
            elif "HYBRID" in classification:
                result = "HYBRID"
            elif "GENERAL" in classification:
                result = "GENERAL"
            else:
                # Default to RAG if unclear
                logger.warning(
                    f"Unclear classification '{classification}', defaulting to RAG"
                )
                result = "RAG"

            logger.info(f"Question classified as: {result}")
            logger.debug(f"Question: {question[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Error in question classification: {e}")
            # Default to RAG on error
            return "RAG"

    def route_question(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """Route a question and return routing information.

        Args:
            question: Customer question
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with routing information
        """
        question_type = self.classify_question(question, session_id)

        routing_info = {
            "question": question,
            "type": question_type,
            "use_rag": question_type in ["RAG", "HYBRID"],
            "use_sql": question_type in ["SQL", "HYBRID"],
            "is_general": question_type == "GENERAL",
        }

        logger.info(f"Routing: {routing_info}")
        return routing_info


# Singleton instance
_router_agent_instance: Optional[RouterAgent] = None


def get_router_agent() -> RouterAgent:
    """Get or create the global Router Agent instance.

    Returns:
        RouterAgent instance
    """
    global _router_agent_instance

    if _router_agent_instance is None:
        _router_agent_instance = RouterAgent()

    return _router_agent_instance


if __name__ == "__main__":
    # Test router agent
    print("Testing Router Agent...")
    print("=" * 80)

    router = RouterAgent()

    test_questions = [
        ("Quels modes de paiement acceptez-vous ?", "RAG"),
        ("Quel est mon forfait actuel ?", "SQL"),
        ("Mon forfait est-il adapté à ma consommation ?", "HYBRID"),
        ("Bonjour", "GENERAL"),
        ("Comment résilier mon abonnement ?", "RAG"),
        ("Combien ai-je consommé ce mois ?", "SQL"),
    ]

    print("\nTesting question classification:\n")

    correct = 0
    for question, expected in test_questions:
        result = router.classify_question(question)
        status = "✓" if result == expected else "✗"
        print(f"{status} Q: {question}")
        print(f"  Expected: {expected}, Got: {result}\n")

        if result == expected:
            correct += 1

    print("=" * 80)
    print(f"Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.0f}%)")
