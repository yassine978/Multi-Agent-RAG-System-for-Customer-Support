"""Orchestrator: Multi-agent coordination using LangGraph.

This module coordinates multiple agents using LangGraph's state machine
to handle complex queries that may require multiple data sources.
"""

import logging
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from .rag_agent import get_rag_agent
from .router_agent import get_router_agent
from .sql_agent import get_sql_agent
from ..prompts.templates import (
    GENERAL_RESPONSE_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
)
from ..utils.config import (
    GOOGLE_API_KEY,
    SYNTHESIS_MAX_TOKENS,
    SYNTHESIS_MODEL,
    SYNTHESIS_TEMPERATURE,
)
from ..utils.langfuse_config import get_langfuse_handler
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    """State for the multi-agent workflow."""
    question: str
    question_type: str
    rag_answer: str
    sql_answer: str
    final_answer: str
    session_id: str
    error: str
    agents_used: list  # Track which agents were executed
    tools_used: list  # Track which SQL tools were called


# =============================================================================
# Node Functions
# =============================================================================

def route_question(state: AgentState) -> AgentState:
    """Route the question to determine which agents to use.

    Args:
        state: Current state

    Returns:
        Updated state with question_type
    """
    logger.info("Node: route_question")

    question = state["question"]
    session_id = state.get("session_id", "default")

    # Initialize tracking lists if not present
    if "agents_used" not in state:
        state["agents_used"] = []
    if "tools_used" not in state:
        state["tools_used"] = []

    try:
        router = get_router_agent()
        routing_info = router.route_question(question, session_id)

        state["question_type"] = routing_info["type"]
        state["agents_used"].append("Router")
        logger.info(f"Question routed as: {routing_info['type']}")

    except Exception as e:
        logger.error(f"Error in routing: {e}")
        state["question_type"] = "RAG"  # Default fallback
        state["error"] = str(e)

    return state


def rag_node(state: AgentState) -> AgentState:
    """Process question with RAG agent.

    Args:
        state: Current state

    Returns:
        Updated state with rag_answer
    """
    logger.info("Node: rag_node")

    question = state["question"]
    session_id = state.get("session_id", "default")

    try:
        rag_agent = get_rag_agent()
        result = rag_agent.answer_question(question, session_id)

        state["rag_answer"] = result["answer"]
        state["agents_used"].append("RAG")
        logger.info("RAG answer generated")

    except Exception as e:
        logger.error(f"Error in RAG node: {e}")
        state["rag_answer"] = ""
        state["error"] = str(e)

    return state


def sql_node(state: AgentState) -> AgentState:
    """Process question with SQL agent.

    Args:
        state: Current state

    Returns:
        Updated state with sql_answer
    """
    logger.info("Node: sql_node")

    question = state["question"]
    session_id = state.get("session_id", "default")

    try:
        sql_agent = get_sql_agent()
        result = sql_agent.answer_question(question, session_id)

        state["sql_answer"] = result["answer"]
        state["agents_used"].append("SQL")

        # Track SQL tools used
        if "tools_used" in result:
            state["tools_used"].extend(result["tools_used"])

        logger.info("SQL answer generated")

    except Exception as e:
        logger.error(f"Error in SQL node: {e}")
        state["sql_answer"] = ""
        state["error"] = str(e)

    return state


def synthesis_node(state: AgentState) -> AgentState:
    """Synthesize answers from multiple sources.

    Args:
        state: Current state

    Returns:
        Updated state with final_answer
    """
    logger.info("Node: synthesis_node")

    question = state["question"]
    rag_answer = state.get("rag_answer", "")
    sql_answer = state.get("sql_answer", "")
    session_id = state.get("session_id", "default")

    try:
        # Create synthesis LLM
        llm = ChatGoogleGenerativeAI(
            model=SYNTHESIS_MODEL,
            temperature=SYNTHESIS_TEMPERATURE,
            max_output_tokens=SYNTHESIS_MAX_TOKENS,
            google_api_key=GOOGLE_API_KEY,
        )

        # Prepare prompt
        from langchain_core.messages import SystemMessage
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(
                content=SYNTHESIS_PROMPT_TEMPLATE.format(
                    question=question,
                    rag_results=rag_answer or "Aucune information des FAQ",
                    sql_results=sql_answer or "Aucune donnée client",
                )
            ),
        ]

        # Get Langfuse handler
        langfuse_handler = get_langfuse_handler(
            agent_name="synthesis",
            session_id=session_id,
            metadata={"question": question},
        )

        # Generate synthesis
        callbacks = [langfuse_handler] if langfuse_handler else []
        response = llm.invoke(messages, config={"callbacks": callbacks})

        state["final_answer"] = response.content.strip()
        state["agents_used"].append("Synthesis")
        logger.info("Synthesis completed")

    except Exception as e:
        logger.error(f"Error in synthesis: {e}")
        # Fallback: combine answers manually
        if rag_answer and sql_answer:
            state["final_answer"] = f"{rag_answer}\n\n{sql_answer}"
        elif rag_answer:
            state["final_answer"] = rag_answer
        elif sql_answer:
            state["final_answer"] = sql_answer
        else:
            state["final_answer"] = "Désolé, je n'ai pas pu générer une réponse."
        state["error"] = str(e)

    return state


def general_response_node(state: AgentState) -> AgentState:
    """Handle general/greeting questions.

    Args:
        state: Current state

    Returns:
        Updated state with final_answer
    """
    logger.info("Node: general_response_node")

    question = state["question"]
    state["agents_used"].append("General")

    try:
        llm = ChatGoogleGenerativeAI(
            model=SYNTHESIS_MODEL,
            temperature=0.5,
            max_output_tokens=200,
            google_api_key=GOOGLE_API_KEY,
        )

        response = llm.invoke([
            HumanMessage(content=GENERAL_RESPONSE_PROMPT.format(question=question))
        ])

        state["final_answer"] = response.content.strip()
        logger.info("General response generated")

    except Exception as e:
        logger.error(f"Error in general response: {e}")
        state["final_answer"] = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
        state["error"] = str(e)

    return state


# =============================================================================
# Conditional Edges
# =============================================================================

def should_use_rag(state: AgentState) -> bool:
    """Check if RAG agent should be used."""
    return state["question_type"] in ["RAG", "HYBRID"]


def should_use_sql(state: AgentState) -> bool:
    """Check if SQL agent should be used."""
    return state["question_type"] in ["SQL", "HYBRID"]


def route_after_routing(state: AgentState) -> Literal["rag", "sql", "general", "synthesis"]:
    """Determine next node after routing.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    question_type = state["question_type"]

    if question_type == "GENERAL":
        return "general"
    elif question_type == "RAG":
        return "rag"
    elif question_type == "SQL":
        return "sql"
    elif question_type == "HYBRID":
        return "rag"  # Start with RAG for hybrid
    else:
        return "rag"  # Default


def route_after_rag(state: AgentState) -> Literal["sql", "end"]:
    """Determine next node after RAG.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    if state["question_type"] == "HYBRID":
        return "sql"
    else:
        return "end"


def route_after_sql(state: AgentState) -> Literal["synthesis", "end"]:
    """Determine next node after SQL.

    Args:
        state: Current state

    Returns:
        Next node name
    """
    if state["question_type"] == "HYBRID":
        return "synthesis"
    else:
        return "end"


# =============================================================================
# Graph Construction
# =============================================================================

def create_agent_graph() -> StateGraph:
    """Create the LangGraph state machine for agent orchestration.

    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("route", route_question)
    workflow.add_node("rag", rag_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("general", general_response_node)

    # Set entry point
    workflow.set_entry_point("route")

    # Add conditional edges from routing
    workflow.add_conditional_edges(
        "route",
        route_after_routing,
        {
            "rag": "rag",
            "sql": "sql",
            "general": "general",
            "synthesis": "synthesis",
        }
    )

    # Add conditional edges from RAG
    workflow.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "sql": "sql",
            "end": END,
        }
    )

    # Add conditional edges from SQL
    workflow.add_conditional_edges(
        "sql",
        route_after_sql,
        {
            "synthesis": "synthesis",
            "end": END,
        }
    )

    # Terminal nodes
    workflow.add_edge("synthesis", END)
    workflow.add_edge("general", END)

    # Compile graph
    app = workflow.compile()

    logger.info("Agent graph created and compiled")
    return app


# =============================================================================
# Orchestrator Class
# =============================================================================

class Orchestrator:
    """Orchestrates multiple agents using LangGraph."""

    def __init__(self):
        """Initialize orchestrator."""
        self.graph = create_agent_graph()
        logger.info("Orchestrator initialized")

    def process_question(
        self,
        question: str,
        session_id: str = "default",
    ) -> dict:
        """Process a question through the multi-agent system.

        Args:
            question: Customer question
            session_id: Session ID for tracking

        Returns:
            Dictionary with final answer and metadata
        """
        logger.info(f"Processing question: {question[:100]}...")

        # Initialize state
        initial_state: AgentState = {
            "question": question,
            "question_type": "",
            "rag_answer": "",
            "sql_answer": "",
            "final_answer": "",
            "session_id": session_id,
            "error": "",
            "agents_used": [],
            "tools_used": [],
        }

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)

            # Determine final answer
            if final_state.get("final_answer"):
                answer = final_state["final_answer"]
            elif final_state.get("rag_answer"):
                answer = final_state["rag_answer"]
            elif final_state.get("sql_answer"):
                answer = final_state["sql_answer"]
            else:
                answer = "Désolé, je n'ai pas pu générer une réponse à votre question."

            # Prepare result
            result = {
                "question": question,
                "answer": answer,
                "question_type": final_state.get("question_type", "unknown"),
                "used_rag": bool(final_state.get("rag_answer")),
                "used_sql": bool(final_state.get("sql_answer")),
                "agents_used": final_state.get("agents_used", []),
                "tools_used": final_state.get("tools_used", []),
                "error": final_state.get("error", ""),
            }

            logger.info(f"Question processed. Type: {result['question_type']}")
            return result

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": "Désolé, une erreur s'est produite lors du traitement de votre question.",
                "question_type": "error",
                "used_rag": False,
                "used_sql": False,
                "error": str(e),
            }


# Singleton instance
_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the global Orchestrator instance.

    Returns:
        Orchestrator instance
    """
    global _orchestrator_instance

    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()

    return _orchestrator_instance


if __name__ == "__main__":
    # Test orchestrator
    print("Testing Orchestrator...")
    print("=" * 80)

    orchestrator = Orchestrator()

    test_questions = [
        "Quels modes de paiement acceptez-vous ?",  # RAG
        "Bonjour",  # GENERAL
        # Add more test questions as needed
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        result = orchestrator.process_question(question)
        print(f"Type: {result['question_type']}")
        print(f"A: {result['answer']}")
        print("-" * 80)
