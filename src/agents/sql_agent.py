"""SQL Agent: Customer data queries using LangChain tools with ReAct pattern.

This agent queries customer data from Excel tables to answer
client-specific questions using a custom ReAct (Reasoning + Acting) implementation.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..data.excel_loader import get_excel_loader
from ..prompts.templates import SQL_PROMPT_TEMPLATE, SQL_SYSTEM_PROMPT
from ..utils.cache import cache_dataframe_query
from ..utils.config import (
    GOOGLE_API_KEY,
    SQL_MAX_TOKENS,
    SQL_MODEL,
    SQL_TEMPERATURE,
)
from ..utils.langfuse_config import get_langfuse_handler
from ..utils.logger import setup_logger
from .sql_tools import TOOLS_DESCRIPTION, get_sql_tools

logger = setup_logger(__name__)


class SQLAgent:
    """Queries customer data using structured LangChain tools with ReAct pattern."""

    def __init__(
        self,
        model_name: str = SQL_MODEL,
        temperature: float = SQL_TEMPERATURE,
        max_tokens: int = SQL_MAX_TOKENS,
    ):
        """Initialize SQL Agent with LangChain tools and ReAct logic.

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

        # Get Excel loader
        self.excel_loader = get_excel_loader()
        self.dataframes = self.excel_loader.get_all_tables()

        # Get SQL tools and create tool map
        self.tools = get_sql_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        logger.info(f"SQLAgent initialized with model: {model_name}")
        logger.info(f"Loaded {len(self.tools)} tools for querying data")

    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        tools_desc = []
        for tool in self.tools:
            tools_desc.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_desc)

    def _parse_tool_call(self, text: str) -> Optional[tuple]:
        """Parse tool call from LLM response.

        Expected format: "Action: tool_name(param1='value1', param2='value2')"

        Returns:
            Tuple of (tool_name, kwargs) or None if not found
        """
        # Try to find action pattern
        action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", text, re.IGNORECASE)
        if not action_match:
            return None

        tool_name = action_match.group(1)
        args_str = action_match.group(2)

        # Parse arguments
        kwargs = {}
        if args_str.strip():
            # Simple parsing for key=value pairs
            for arg in args_str.split(','):
                arg = arg.strip()
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    kwargs[key] = value

        return (tool_name, kwargs)

    def _execute_react_loop(self, question: str, max_iterations: int = 5) -> tuple:
        """Execute ReAct loop: Reasoning + Acting.

        Args:
            question: User question
            max_iterations: Maximum number of reasoning-action iterations

        Returns:
            Tuple of (answer, tools_used)
        """
        system_prompt = f"""{SQL_SYSTEM_PROMPT}

{TOOLS_DESCRIPTION}

OUTILS DISPONIBLES:
{self._get_tools_description()}

INSTRUCTIONS ReAct:
1. Pense (Thought) à ce que tu dois faire
2. Agis (Action) en appelant un outil avec: Action: tool_name(param1='value1', param2='value2')
3. Observe le résultat
4. Répète si nécessaire
5. Quand tu as toutes les informations, donne ta réponse finale avec: Final Answer: [ta réponse]

Commence TOUJOURS par get_client_by_name si le client donne son nom.
Réponds en français de manière claire."""

        conversation = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}")
        ]

        observations = []
        tools_used = []  # Track which tools were called

        for iteration in range(max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}/{max_iterations}")

            # Get LLM response
            try:
                response = self.llm.invoke(conversation)
                response_text = response.content if hasattr(response, 'content') else str(response)

                logger.info(f"LLM Response: {response_text[:200]}...")

                # Check for final answer
                if "final answer:" in response_text.lower():
                    # Extract final answer
                    final_match = re.search(r"final answer:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL)
                    if final_match:
                        return (final_match.group(1).strip(), tools_used)

                # Parse tool call
                tool_call = self._parse_tool_call(response_text)

                if tool_call:
                    tool_name, kwargs = tool_call

                    if tool_name in self.tool_map:
                        try:
                            # Execute tool
                            logger.info(f"Executing tool: {tool_name} with {kwargs}")
                            tool = self.tool_map[tool_name]
                            result = tool.invoke(kwargs)

                            # Track tool usage
                            tools_used.append(tool_name)

                            observation = f"Observation: {json.dumps(result, ensure_ascii=False)}"
                            observations.append(observation)

                            # Add to conversation
                            conversation.append(AIMessage(content=response_text))
                            conversation.append(HumanMessage(content=observation))

                            logger.info(f"Tool result: {str(result)[:200]}...")
                        except Exception as e:
                            error_msg = f"Erreur lors de l'exécution de {tool_name}: {str(e)}"
                            logger.error(error_msg)
                            conversation.append(AIMessage(content=response_text))
                            conversation.append(HumanMessage(content=f"Observation: {error_msg}"))
                    else:
                        logger.warning(f"Tool {tool_name} not found")
                        conversation.append(AIMessage(content=response_text))
                        conversation.append(HumanMessage(content=f"Observation: Outil {tool_name} non trouvé. Outils disponibles: {list(self.tool_map.keys())}"))
                else:
                    # No tool call found, might be thinking or final answer
                    conversation.append(AIMessage(content=response_text))
                    conversation.append(HumanMessage(content="Continue ton raisonnement ou donne ta réponse finale avec 'Final Answer:'"))

            except Exception as e:
                logger.error(f"Error in ReAct loop: {e}")
                return (f"Désolé, une erreur s'est produite: {str(e)}", tools_used)

        # Max iterations reached
        return ("Désolé, je n'ai pas pu trouver une réponse complète dans le temps imparti.", tools_used)

    def get_table_info(self) -> str:
        """Get information about available tables.

        Returns:
            String describing available tables
        """
        info_parts = []
        for table_name, df in self.dataframes.items():
            columns = ", ".join(df.columns.tolist())
            info_parts.append(
                f"{table_name} ({len(df)} rows): {columns}"
            )
        return "\n".join(info_parts)

    @cache_dataframe_query
    def query_data(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> tuple:
        """Query customer data to answer a question using ReAct pattern.

        Args:
            question: Customer question requiring data lookup
            session_id: Optional session ID for tracking

        Returns:
            Tuple of (answer, tools_used)
        """
        try:
            logger.info(f"Querying data for: {question[:100]}...")

            # Get Langfuse handler for tracing (optional)
            langfuse_handler = get_langfuse_handler(
                agent_name="sql",
                session_id=session_id,
                metadata={"question": question},
            )

            # Execute ReAct loop
            answer, tools_used = self._execute_react_loop(question, max_iterations=5)

            logger.info("Data query completed successfully using ReAct pattern")
            return (answer.strip(), tools_used)

        except Exception as e:
            logger.error(f"Error querying data: {e}")
            return (f"Désolé, une erreur s'est produite lors de la requête de données. Veuillez reformuler votre question.", [])

    def answer_question(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """Answer a question using customer data.

        Args:
            question: Customer question
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing SQL question: {question[:100]}...")

        # Query data
        answer, tools_used = self.query_data(question, session_id)

        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "tables_used": list(self.dataframes.keys()),
            "tools_used": tools_used,
        }

        return result

    def get_client_data(self, client_identifier: str) -> Optional[Dict]:
        """Get client data by ID, email, phone, or name.

        Args:
            client_identifier: Client ID, email, phone, or name

        Returns:
            Dictionary with client data or None
        """
        try:
            client = self.excel_loader.search_client(client_identifier)
            if client is not None:
                return client.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting client data: {e}")
            return None


# Singleton instance
_sql_agent_instance: Optional[SQLAgent] = None


def get_sql_agent() -> SQLAgent:
    """Get or create the global SQL Agent instance.

    Returns:
        SQLAgent instance
    """
    global _sql_agent_instance

    if _sql_agent_instance is None:
        _sql_agent_instance = SQLAgent()

    return _sql_agent_instance


if __name__ == "__main__":
    # Test SQL agent
    print("Testing SQL Agent...")
    print("=" * 80)

    sql_agent = SQLAgent()

    # Show available tables
    print("\nAvailable tables:")
    print(sql_agent.get_table_info())

    # Test questions
    test_questions = [
        "Combien de clients avons-nous au total ?",
        "Quels sont les forfaits disponibles avec leurs prix ?",
        "Combien de tickets de support sont ouverts ?",
    ]

    print("\n" + "=" * 80)
    print("Testing SQL question answering:")
    print("=" * 80)

    for question in test_questions:
        print(f"\nQ: {question}")
        result = sql_agent.answer_question(question)
        print(f"A: {result['answer']}\n")
        print("-" * 80)
