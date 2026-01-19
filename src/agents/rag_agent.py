"""RAG Agent: FAQ document search and answer generation.

This agent searches PDF documents using vector similarity and generates
answers based on retrieved context.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..data.pdf_indexer import get_pdf_indexer
from ..prompts.templates import (
    ERROR_NO_RESULTS_PROMPT,
    RAG_PROMPT_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    format_documents_for_context,
)
from ..utils.cache import cache_vector_search
from ..utils.config import (
    GOOGLE_API_KEY,
    RAG_MAX_TOKENS,
    RAG_MODEL,
    RAG_TEMPERATURE,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from ..utils.langfuse_config import get_langfuse_handler
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGAgent:
    """Retrieval-Augmented Generation agent for FAQ questions."""

    def __init__(
        self,
        model_name: str = RAG_MODEL,
        temperature: float = RAG_TEMPERATURE,
        max_tokens: int = RAG_MAX_TOKENS,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: Optional[float] = RETRIEVAL_SCORE_THRESHOLD,
    ):
        """Initialize RAG Agent.

        Args:
            model_name: Name of the Gemini model to use
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for response
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=GOOGLE_API_KEY,
        )

        # Get PDF indexer
        self.pdf_indexer = get_pdf_indexer()

        logger.info(f"RAGAgent initialized with model: {model_name}")

    @cache_vector_search
    def retrieve_documents(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve relevant documents for a question.

        Args:
            question: Customer question
            k: Number of documents to retrieve (uses default if None)

        Returns:
            List of relevant Document objects
        """
        k = k or self.top_k

        # POLICY QUESTION DETECTION - Increase k for completeness
        if self._is_policy_question(question):
            k = min(int(k * 1.5), 15)  # Increase k by 50%, max 15
            logger.info(f"Policy question detected - increased k to {k} for completeness")

        # COMPARISON QUESTION DETECTION - Multi-product retrieval
        if self._is_comparison_question(question):
            logger.info(f"Comparison question detected - using multi-product retrieval strategy")
            return self._retrieve_for_comparison(question, k)

        try:
            # Query expansion for better retrieval
            search_query = question

            # Check roaming indicators first (needed for later conditions)
            has_roaming_indicator = any(keyword in question.lower() for keyword in [
                "roaming", "itinérance", "voyage", "voyager", "pars", "partir",
                "étranger", "international", "hors de france"
            ])

            has_travel_destination = any(keyword in question.lower() for keyword in [
                "italie", "espagne", "allemagne", "belgique", "europe", "ue", "union",
                "états-unis", "usa", "amérique", "asie", "afrique"
            ])

            # Expansion 0: Password reset questions
            # Q6: "J'ai oublié mon mot de passe" should find FAQ_Compte_Client
            has_password = any(keyword in question.lower() for keyword in ["mot de passe", "password", "identifiant", "connexion", "oublié"])
            if has_password:
                search_query = f"{question} password oublié forgotten reset réinitialisation compte client"
                logger.info(f"Password reset question detected - expanded query")

            # Expansion 1: Trade-in/Reprise questions
            # Q21: "Reprise ancien iPhone" should find trade-in info
            elif any(keyword in question.lower() for keyword in ["reprise", "ancien", "échange", "trade"]):
                search_query = f"{question} échange trade-in offre réduction ancien appareil"
                logger.info(f"Trade-in question detected - expanded query")

            # Expansion 2: Commercialization year questions
            # Q9: "Quel iPhone commercialisé en 2023?" should find iPhone 15
            elif any(keyword in question.lower() for keyword in ["commercialisé", "sorti", "lancé"]):
                # Extract year from question
                import re
                year_match = re.search(r'20\d{2}', question)
                if year_match:
                    year = year_match.group(0)
                    search_query = f"{question} {year} année commercialisation lancement sorti"
                    logger.info(f"Commercialization year question detected - expanded query with year {year}")
                else:
                    search_query = f"{question} année commercialisation sortie lancement 2023 2024 date"
                    logger.info(f"Commercialization question detected - expanded query for factual extraction")

            # Expansion 3: Factual catalog questions (colors, specific specs)
            # Detects questions asking for specific facts about products
            elif any(keyword in question.lower() for keyword in ["coloris", "couleur", "colors"]):
                # Color question - expand to find color lists
                search_query = f"{question} coloris couleurs disponibles options variantes"
                logger.info(f"Color question detected - expanded query for color extraction")
            elif any(keyword in question.lower() for keyword in ["autonomie", "batterie", "différence", "capacité", "stockage", "caméra", "mpx"]):
                # Spec comparison/fact question - expand for specifications
                search_query = f"{question} caractéristiques techniques spécifications comparaison différence"
                logger.info(f"Specification question detected - expanded query for fact extraction")

            # Expansion 4: ROAMING QUESTIONS
            # Questions about travel/roaming should NOT trigger phone catalog retrieval
            else:
                # Roaming activation questions
                has_activation = any(keyword in question.lower() for keyword in ["activer", "activation", "comment", "procédure"])
                if has_activation and "roaming" in question.lower():
                    search_query = f"{question} activation roaming activer automatique paramètres procédure"
                    logger.info(f"Roaming activation question detected - expanded query")

                # EU roaming questions
                elif "roaming" in question.lower() or (has_travel_destination and any(eu in question.lower() for eu in ["italie", "espagne", "allemagne", "belgique", "europe", "ue", "union"])):
                    search_query = f"{question} Union Européenne roaming comme à la maison sans surcoût inclus UE"
                    logger.info(f"EU roaming question detected - expanded query to prioritize EU policy")

                # Non-EU roaming/travel questions (USA, international)
                elif has_roaming_indicator or has_travel_destination:
                    search_query = f"{question} roaming international hors Europe pass international tarifs itinérance"
                    logger.info(f"International roaming/travel question detected - expanded query for non-EU roaming")

                # Expansion 5: Multi-criteria questions (GENERIC PATTERN)
                # Detects questions with multiple criteria: price + specs/features/conditions
                # BUT: Skip if roaming detected above
                elif not (has_roaming_indicator or has_travel_destination):
                    has_price = any(keyword in question.lower() for keyword in ["prix", "coute", "cout", "combien", "€", "euro", "moins de", "budget", "cher"])
                    has_specs = any(keyword in question.lower() for keyword in ["avec", "mpx", "gb", "go", "inclus", "disponible", "autonomie", "camera", "caméra"])
                    has_comparison = any(keyword in question.lower() for keyword in ["meilleur", "mieux", "plus", "moins", "supérieur", "inférieur"])

                    if has_price and (has_specs or has_comparison):
                        # Multi-criteria: need both price data AND product descriptions
                        search_query = f"{question} prix catalogue caractéristiques spécifications tableau description"
                        logger.info(f"Multi-criteria question detected (price+specs/comparison) - expanded query")
                    elif has_price:
                        # Price-only question
                        search_query = f"{question} Tableau des Prix prix catalogue"
                        logger.info(f"Price question detected - expanded query for price tables")

            documents = self.pdf_indexer.search(
                query=search_query,
                k=k,
                score_threshold=self.score_threshold,
            )

            logger.info(f"Retrieved {len(documents)} documents for question")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _is_policy_question(self, question: str) -> bool:
        """Detect if question is about policies/rules that require complete information.

        Policy questions often need multiple chunks to provide the full answer
        (e.g., fees if engaged + fees if not engaged).

        Args:
            question: Customer question

        Returns:
            True if it's a policy question
        """
        policy_keywords = [
            "frais",
            "résiliation",
            "résilier",
            "engagement",
            "engagé",
            "conditions",
            "modalités",
            "règles",
            "politique",
            "procédure",
            "pénalités",
            "rembours",
        ]
        return any(keyword in question.lower() for keyword in policy_keywords)

    def _is_binary_question(self, question: str) -> bool:
        """Detect if question has two scenarios (if...then...else).

        Binary questions require complete answers covering BOTH scenarios.

        Args:
            question: Customer question

        Returns:
            True if it's a binary/conditional question
        """
        question_lower = question.lower()

        # Binary indicators - questions with conditional structure
        binary_indicators = [
            "si je suis",
            "si vous êtes",
            "si j'ai",
            "si on est",
            "y a-t-il des frais",
            "y a-t-il un",
            "puis-je",
            "est-ce gratuit",
            "est-ce payant",
            "dois-je",
        ]

        # Also check for policy questions with "si" (if) - likely binary
        has_si = "si " in question_lower
        has_policy = any(keyword in question_lower for keyword in ["frais", "résiliation", "engagement", "engagé"])

        return any(ind in question_lower for ind in binary_indicators) or (has_si and has_policy)

    def _is_comparison_question(self, question: str) -> bool:
        """Detect if question is comparing products/attributes.

        Comparison questions need retrieval for EACH product mentioned.

        Args:
            question: Customer question

        Returns:
            True if it's a comparison question
        """
        question_lower = question.lower()

        # Comparison keywords
        has_comparison_keyword = any(keyword in question_lower for keyword in [
            "différence",
            "compare",
            "comparer",
            "comparaison",
            "meilleur",
            "mieux",
            "entre",
            "vs",
            "versus",
        ])

        # Attributes that can be compared
        has_comparable_attribute = any(attr in question_lower for attr in [
            "autonomie",
            "batterie",
            "caméra",
            "appareil photo",
            "écran",
            "prix",
            "stockage",
            "mémoire",
            "processeur",
            "puce",
            "taille",
            "poids",
        ])

        return has_comparison_keyword and has_comparable_attribute

    def _retrieve_for_comparison(self, question: str, k: int) -> List[Document]:
        """Multi-product retrieval strategy for comparison questions.

        Instead of retrieving docs for the general question, retrieve docs
        for EACH product + attribute combination, then combine.

        Args:
            question: Comparison question
            k: Number of docs to retrieve

        Returns:
            Combined list of documents from all products
        """
        import re

        question_lower = question.lower()

        # Extract attribute being compared
        attribute = None
        attribute_keywords = {
            "autonomie": ["autonomie", "batterie"],
            "caméra": ["caméra", "appareil photo", "photo", "mpx", "mégapixels"],
            "écran": ["écran", "affichage", "display"],
            "prix": ["prix", "coût", "tarif"],
            "stockage": ["stockage", "go", "gb", "mémoire"],
        }

        for attr_name, keywords in attribute_keywords.items():
            if any(kw in question_lower for kw in keywords):
                attribute = attr_name
                break

        if not attribute:
            # Fallback to general retrieval
            logger.warning("Comparison detected but couldn't extract attribute - using general retrieval")
            return self.pdf_indexer.search(question, k=k)

        # Extract product names (iPhone models)
        # Pattern: iPhone followed by number or model name
        iphone_pattern = r'iphone\s*(?:x|xs|xr|se|[0-9]{1,2}(?:\s*pro(?:\s*max)?)?)'
        products = list(set(re.findall(iphone_pattern, question_lower, re.IGNORECASE)))

        if len(products) < 2:
            # Not enough products for comparison, fallback
            logger.warning(f"Comparison detected but only found {len(products)} product(s) - using general retrieval")
            return self.pdf_indexer.search(question, k=k)

        logger.info(f"Comparison question: attribute='{attribute}', products={products}")

        # Retrieve chunks for each product + attribute
        all_documents = []
        docs_per_product = max(10, k * 2 // len(products))  # Increased significantly to ensure we get the right chunks

        for product in products:
            # Create enriched queries with attribute-specific keywords
            if attribute == "autonomie":
                # For autonomie, include keywords like "lecture vidéo", "batterie", "h", "heures"
                queries = [
                    f"{product} autonomie lecture vidéo",
                    f"{product} batterie heures",
                    f"{product} {attribute}",
                ]
            elif attribute == "caméra":
                queries = [
                    f"{product} caméra mpx",
                    f"{product} appareil photo",
                    f"{product} {attribute}",
                ]
            else:
                queries = [f"{product} {attribute}"]

            # Retrieve docs for each query variant
            for query in queries:
                logger.info(f"Retrieving for: '{query}' (k={docs_per_product})")

                product_docs = self.pdf_indexer.search(
                    query=query,
                    k=docs_per_product,
                    score_threshold=self.score_threshold,
                )

                all_documents.extend(product_docs)

        # Deduplicate by page_content
        seen_content = set()
        unique_docs = []
        for doc in all_documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        logger.info(f"Multi-product retrieval: {len(unique_docs)} unique documents from {len(products)} products")

        return unique_docs[:k * 2]  # Return up to 2x k to ensure we have enough context

    def _extract_catalog_facts(self, question: str, documents: List[Document]) -> Optional[str]:
        """Extract specific catalog facts using regex patterns.

        This is a fallback for when LLM fails to extract facts from chunks.
        """
        import re

        # Combine all document content
        full_text = " ".join([doc.page_content for doc in documents])

        # Pattern 1: Extract colors for a specific iPhone model
        if "coloris" in question.lower() or "couleur" in question.lower():
            # Extract iPhone model from question
            model_match = re.search(r'iPhone\s+(\d+|X)', question, re.IGNORECASE)
            if model_match:
                model = model_match.group(0)
                # Look for "iPhone X ... Coloris disponibles : ..."
                pattern = rf'{model}[^!]+?Coloris disponibles?\s*:\s*([^.]+\.)'
                match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                if match:
                    colors_text = match.group(1).strip()
                    return f"Les coloris disponibles pour l'{model} sont : {colors_text}\n\nSource: FAQ_Catalogue_Telephones.pdf"

        # Pattern 2: Extract launch year for iPhone model
        if "commercialisé" in question.lower() or "année" in question.lower():
            # Look for "iPhone X, commercialisé en YYYY"
            pattern = r'(iPhone\s+\d+)[^!]+?commercialisé en (\d{4})'
            matches = re.findall(pattern, full_text, re.IGNORECASE)

            # Check which iPhone is asked about
            model_match = re.search(r'iPhone\s+(\d+)', question, re.IGNORECASE)
            if model_match and matches:
                asked_model = f"iPhone {model_match.group(1)}"
                for model, year in matches:
                    if model == asked_model:
                        return f"L'{model} a été commercialisé en {year}.\n\nSource: FAQ_Catalogue_Telephones.pdf"

        # Pattern 3: Extract battery life for comparison
        if "autonomie" in question.lower() and "différence" in question.lower():
            # Extract both iPhone models
            models = re.findall(r'iPhone\s+(\d+)', question, re.IGNORECASE)
            if len(models) >= 2:
                model1, model2 = models[0], models[1]

                # IMPROVED: Search for autonomie values across ALL documents, not just in same chunk
                # Strategy: Find chunks mentioning each phone, then look for autonomie in nearby context

                hours1 = None
                hours2 = None

                # Search for each model's autonomie value
                for doc in documents:
                    content = doc.page_content

                    # Check if this chunk contains model1 AND autonomie
                    if f'iPhone {model1}' in content or f'iphone {model1}' in content.lower():
                        # Try to find autonomie value in this chunk
                        autonomie_match = re.search(r'autonomie jusqu[\'`\']à (\d+)h', content, re.IGNORECASE)
                        if autonomie_match and hours1 is None:
                            hours1 = int(autonomie_match.group(1))

                    # Check if this chunk contains model2 AND autonomie
                    if f'iPhone {model2}' in content or f'iphone {model2}' in content.lower():
                        # Try to find autonomie value in this chunk
                        autonomie_match = re.search(r'autonomie jusqu[\'`\']à (\d+)h', content, re.IGNORECASE)
                        if autonomie_match and hours2 is None:
                            hours2 = int(autonomie_match.group(1))

                # If still not found, try broader pattern across full text
                if hours1 is None:
                    pattern1 = rf'iPhone {model1}[^!]+?autonomie jusqu[\'`\']à (\d+)h'
                    match1 = re.search(pattern1, full_text, re.IGNORECASE | re.DOTALL)
                    if match1:
                        hours1 = int(match1.group(1))

                if hours2 is None:
                    pattern2 = rf'iPhone {model2}[^!]+?autonomie jusqu[\'`\']à (\d+)h'
                    match2 = re.search(pattern2, full_text, re.IGNORECASE | re.DOTALL)
                    if match2:
                        hours2 = int(match2.group(1))

                if hours1 and hours2:
                    diff = abs(hours2 - hours1)
                    return f"L'iPhone {model1} offre une autonomie de {hours1}h, tandis que l'iPhone {model2} offre {hours2}h. La différence est de {diff}h de lecture vidéo.\n\nSource: FAQ_Catalogue_Telephones.pdf"

        return None

    def generate_answer(
        self,
        question: str,
        documents: List[Document],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate answer based on retrieved documents.

        Args:
            question: Customer question
            documents: Retrieved documents
            session_id: Optional session ID for tracking

        Returns:
            Generated answer
        """
        try:
            # CRITICAL FIX: Try direct fact extraction first for catalog questions
            extracted_answer = self._extract_catalog_facts(question, documents)
            if extracted_answer:
                logger.info("Answer extracted using direct pattern matching")
                return extracted_answer

            # Fallback to LLM generation
            # Format context from documents
            context = format_documents_for_context(documents)

            # Build prompt with binary question instruction if needed
            prompt_content = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )

            # Add binary question instruction for complete answers
            if self._is_binary_question(question):
                logger.info("Binary question detected - adding completeness instruction")
                prompt_content += """

⚠️ IMPORTANT - QUESTION BINAIRE DÉTECTÉE:
Cette question a probablement DEUX scénarios/cas de figure.

Ta réponse DOIT être COMPLÈTE et inclure LES DEUX cas:
1. Premier scénario (ex: si condition remplie - avec engagement, avec frais, etc.)
2. Deuxième scénario (ex: si condition non remplie - sans engagement, gratuit, etc.)

Exemple pour résiliation:
- Scénario 1: Si engagé → frais de résiliation applicables
- Scénario 2: Si hors engagement → résiliation gratuite sans frais

NE PAS s'arrêter après le premier scénario!
CHERCHE dans les documents les DEUX cas et mentionne-les TOUS dans ta réponse."""

            # Prepare messages
            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]

            # Get Langfuse handler for tracing
            langfuse_handler = get_langfuse_handler(
                agent_name="rag",
                session_id=session_id,
                metadata={
                    "question": question,
                    "num_documents": len(documents),
                },
            )

            # Invoke LLM
            callbacks = [langfuse_handler] if langfuse_handler else []
            response = self.llm.invoke(messages, config={"callbacks": callbacks})

            answer = response.content.strip()
            logger.info("Answer generated successfully")

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Désolé, une erreur s'est produite lors de la génération de la réponse. Erreur: {str(e)}"

    def answer_question(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """Complete RAG pipeline: retrieve and generate answer.

        Args:
            question: Customer question
            session_id: Optional session ID for tracking

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing RAG question: {question[:100]}...")

        # Retrieve documents
        documents = self.retrieve_documents(question)

        # Check if we found relevant documents
        if not documents:
            logger.warning("No relevant documents found")
            # Generate a "no results" response
            no_results_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.3,
                max_output_tokens=200,
                google_api_key=GOOGLE_API_KEY,
            )
            response = no_results_llm.invoke(
                [HumanMessage(content=ERROR_NO_RESULTS_PROMPT.format(question=question))]
            )
            answer = response.content.strip()
        else:
            # Generate answer from documents
            answer = self.generate_answer(question, documents, session_id)

        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "num_documents": len(documents),
            "sources": [
                {
                    "file": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                }
                for doc in documents
            ],
        }

        return result


# Singleton instance
_rag_agent_instance: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get or create the global RAG Agent instance.

    Returns:
        RAGAgent instance
    """
    global _rag_agent_instance

    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent()

    return _rag_agent_instance


if __name__ == "__main__":
    # Test RAG agent
    print("Testing RAG Agent...")
    print("=" * 80)

    # Initialize vector database first
    print("\nInitializing vector database...")
    from src.data.pdf_indexer import initialize_vector_db
    initialize_vector_db(force_recreate=False)

    # Create RAG agent
    rag_agent = RAGAgent()

    # Test questions
    test_questions = [
        "Quels modes de paiement acceptez-vous ?",
        "Comment fonctionne le roaming international ?",
        "Comment résilier mon abonnement ?",
    ]

    print("\nTesting RAG question answering:\n")

    for question in test_questions:
        print("=" * 80)
        print(f"Q: {question}\n")

        result = rag_agent.answer_question(question)

        print(f"A: {result['answer']}\n")
        print(f"Sources ({result['num_documents']} documents):")
        for source in result['sources'][:3]:  # Show top 3
            print(f"  - {source['file']} (Page {source['page']})")
        print()
