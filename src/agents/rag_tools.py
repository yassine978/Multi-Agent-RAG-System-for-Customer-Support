"""RAG Tools for structured fact extraction from documents.

This module provides LangChain tools for extracting specific facts
from the RAG system, similar to how SQL tools work.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain.tools import tool

from ..data.pdf_indexer import get_pdf_indexer
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Product Catalog Tools
# =============================================================================

@tool
def get_product_colors(product_name: str) -> Dict[str, Any]:
    """Obtient les coloris disponibles pour un produit.

    Args:
        product_name: Nom du produit (ex: "iPhone 16", "iPhone 15")

    Returns:
        Dictionary avec les coloris disponibles

    Example:
        get_product_colors("iPhone 16")
    """
    try:
        indexer = get_pdf_indexer()

        # Search for product description - be very specific
        query = f"{product_name} coloris disponibles"
        docs = indexer.search(query, k=15)

        colors = []
        for doc in docs:
            content = doc.page_content

            # APPROACH: Look for the specific pattern where the product name HEADER
            # is followed by its description including "coloris disponibles"
            # Pattern: "iPhone XX\n" or "iPhone XX\nCe modèle" followed by description

            # Find product header pattern (product name at start of line or after header marker)
            # The PDF has "iPhone 16" as a header followed by description
            header_pattern = rf'(?:^|\n)\s*{re.escape(product_name)}\s*(?:\n|Ce mod)'

            header_match = re.search(header_pattern, content, re.IGNORECASE | re.MULTILINE)

            if not header_match:
                # Try alternative: product name followed by "Ce modèle, commercialisé"
                alt_pattern = rf'{re.escape(product_name)}\s*\n?\s*Ce mod[eè]le,?\s*commercialis'
                header_match = re.search(alt_pattern, content, re.IGNORECASE)

            if not header_match:
                continue

            # Found the header - now extract the section
            section_start = header_match.start()

            # Find end: next iPhone header or end of content
            next_iphone = re.search(r'\niPhone\s+(\d+|X)\s*\n', content[section_start + 10:], re.IGNORECASE)
            if next_iphone:
                section_end = section_start + 10 + next_iphone.start()
            else:
                section_end = len(content)

            product_section = content[section_start:section_end]

            # Now extract colors from this section
            color_match = re.search(r'[Cc]oloris\s+disponibles?\s*:\s*([^.]+\.)', product_section)
            if color_match:
                color_text = color_match.group(1)
                # Split by comma and clean up
                raw_colors = re.split(r',\s*', color_text)
                found_colors = []
                for c in raw_colors:
                    # Clean the color name
                    c = c.strip().rstrip('.')
                    # Remove "Une gamme..." suffix
                    if 'Une gamme' in c or len(c) > 30:
                        continue
                    if c:
                        found_colors.append(c)
                if found_colors:
                    colors = found_colors
                    break

        if colors:
            return {
                "product": product_name,
                "colors": colors,
                "found": True
            }
        else:
            return {
                "product": product_name,
                "colors": [],
                "found": False,
                "message": f"Coloris non trouvés pour {product_name}"
            }

    except Exception as e:
        logger.error(f"Error in get_product_colors: {e}")
        return {"error": str(e)}


@tool
def get_product_launch_year(product_name: str) -> Dict[str, Any]:
    """Obtient l'année de commercialisation d'un produit.

    Args:
        product_name: Nom du produit (ex: "iPhone 15", "iPhone 14")

    Returns:
        Dictionary avec l'année de commercialisation

    Example:
        get_product_launch_year("iPhone 15")
    """
    try:
        indexer = get_pdf_indexer()

        # Search for product description with year
        query = f"{product_name} commercialisé année sortie"
        docs = indexer.search(query, k=15)

        year = None
        for doc in docs:
            content = doc.page_content

            # APPROACH: Look for the specific pattern where the product name HEADER
            # is followed by its description including "commercialisé"

            # Find product header pattern
            header_pattern = rf'(?:^|\n)\s*{re.escape(product_name)}\s*(?:\n|Ce mod)'
            header_match = re.search(header_pattern, content, re.IGNORECASE | re.MULTILINE)

            if not header_match:
                # Try alternative: product name followed by "Ce modèle, commercialisé"
                alt_pattern = rf'{re.escape(product_name)}\s*\n?\s*Ce mod[eè]le,?\s*commercialis'
                header_match = re.search(alt_pattern, content, re.IGNORECASE)

            if not header_match:
                continue

            # Found the header - now extract the section
            section_start = header_match.start()

            # Find end: next iPhone header or end of content
            next_iphone = re.search(r'\niPhone\s+(\d+|X)\s*\n', content[section_start + 10:], re.IGNORECASE)
            if next_iphone:
                section_end = section_start + 10 + next_iphone.start()
            else:
                section_end = len(content)

            product_section = content[section_start:section_end]

            # Now extract year from this section
            year_match = re.search(r'commercialis[eé]\s+en\s+(\d{4})', product_section, re.IGNORECASE)
            if year_match:
                year = int(year_match.group(1))
                break

        if year:
            return {
                "product": product_name,
                "year": year,
                "found": True
            }
        else:
            return {
                "product": product_name,
                "year": None,
                "found": False,
                "message": f"Année de commercialisation non trouvée pour {product_name}"
            }

    except Exception as e:
        logger.error(f"Error in get_product_launch_year: {e}")
        return {"error": str(e)}


@tool
def get_product_battery_life(product_name: str) -> Dict[str, Any]:
    """Obtient l'autonomie de la batterie d'un produit.

    Args:
        product_name: Nom du produit (ex: "iPhone 16", "iPhone 13")

    Returns:
        Dictionary avec l'autonomie en heures

    Example:
        get_product_battery_life("iPhone 16")
    """
    try:
        indexer = get_pdf_indexer()

        # Search for product battery specs
        query = f"{product_name} autonomie batterie heures lecture video"
        docs = indexer.search(query, k=15)

        battery_hours = None
        for doc in docs:
            content = doc.page_content

            # APPROACH: Look for the specific pattern where the product name HEADER
            # is followed by its description including "autonomie"

            # Find product header pattern
            header_pattern = rf'(?:^|\n)\s*{re.escape(product_name)}\s*(?:\n|Ce mod)'
            header_match = re.search(header_pattern, content, re.IGNORECASE | re.MULTILINE)

            if not header_match:
                # Try alternative: product name followed by "Ce modèle, commercialisé"
                alt_pattern = rf'{re.escape(product_name)}\s*\n?\s*Ce mod[eè]le,?\s*commercialis'
                header_match = re.search(alt_pattern, content, re.IGNORECASE)

            if not header_match:
                continue

            # Found the header - now extract the section
            section_start = header_match.start()

            # Find end: next iPhone header or end of content
            next_iphone = re.search(r'\niPhone\s+(\d+|X)\s*\n', content[section_start + 10:], re.IGNORECASE)
            if next_iphone:
                section_end = section_start + 10 + next_iphone.start()
            else:
                section_end = len(content)

            product_section = content[section_start:section_end]

            # Now extract battery from this section
            # Pattern: "autonomie jusqu'à XXh" or "autonomie jusqu'a XXh"
            battery_match = re.search(r"autonomie[^0-9]*?(\d+)\s*h", product_section, re.IGNORECASE)
            if battery_match:
                battery_hours = int(battery_match.group(1))
                break

        if battery_hours:
            return {
                "product": product_name,
                "battery_hours": battery_hours,
                "found": True
            }
        else:
            return {
                "product": product_name,
                "battery_hours": None,
                "found": False,
                "message": f"Autonomie non trouvée pour {product_name}"
            }

    except Exception as e:
        logger.error(f"Error in get_product_battery_life: {e}")
        return {"error": str(e)}


@tool
def search_faq(question: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Recherche une réponse dans les FAQ.

    Args:
        question: Question à rechercher
        category: Catégorie optionnelle (ex: "roaming", "résiliation", "compte")

    Returns:
        Dictionary avec les documents pertinents

    Example:
        search_faq("Comment activer le roaming ?", category="roaming")
    """
    try:
        indexer = get_pdf_indexer()

        # Build search query
        if category:
            query = f"{question} {category}"
        else:
            query = question

        # Search with higher k for FAQ
        docs = indexer.search(query, k=3)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:500],  # First 500 chars
                "source": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })

        return {
            "question": question,
            "results": results,
            "found": len(results) > 0
        }

    except Exception as e:
        logger.error(f"Error in search_faq: {e}")
        return {"error": str(e)}


# =============================================================================
# Get all tools for the agent
# =============================================================================

def get_rag_tools() -> list:
    """Get all RAG tools for the agent.

    Returns:
        List of LangChain tools
    """
    return [
        get_product_colors,
        get_product_launch_year,
        get_product_battery_life,
        search_faq,
    ]


# =============================================================================
# Tool descriptions for LLM
# =============================================================================

TOOLS_DESCRIPTION = """
Tu as accès aux outils suivants pour interroger les documents:

1. **get_product_colors(product_name)**: Obtient les coloris d'un produit
   - Exemple: get_product_colors("iPhone 16")
   - Retourne: liste des coloris disponibles

2. **get_product_launch_year(product_name)**: Obtient l'année de commercialisation
   - Exemple: get_product_launch_year("iPhone 15")
   - Retourne: année (ex: 2023)

3. **get_product_battery_life(product_name)**: Obtient l'autonomie batterie
   - Exemple: get_product_battery_life("iPhone 16")
   - Retourne: heures d'autonomie

4. **search_faq(question, category)**: Recherche dans les FAQ
   - Exemple: search_faq("Comment activer le roaming?", category="roaming")
   - Retourne: documents pertinents

MÉTHODOLOGIE:
- Utilise ces outils pour extraire des faits précis des documents
- Combine les résultats pour construire ta réponse
- Cite toujours la source des informations
"""
