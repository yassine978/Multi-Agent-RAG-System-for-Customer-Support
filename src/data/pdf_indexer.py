"""PDF document indexing for RAG system.

This module handles loading, chunking, and indexing FAQ PDF documents
into a vector database for semantic search.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ..utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PDF_DIRECTORY,
    RETRIEVAL_TOP_K,
    VECTOR_DB_COLLECTION_NAME,
    VECTOR_DB_PATH,
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFIndexer:
    """Handles PDF document loading and vector database indexing."""

    def __init__(
        self,
        pdf_directory: str = PDF_DIRECTORY,
        vector_db_path: str = VECTOR_DB_PATH,
        collection_name: str = VECTOR_DB_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        """Initialize PDF indexer.

        Args:
            pdf_directory: Directory containing PDF files
            vector_db_path: Path to store vector database
            collection_name: Name of the vector database collection
            embedding_model: Name of the embedding model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.pdf_directory = Path(pdf_directory)
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore: Optional[Chroma] = None

    def load_pdfs(self) -> List[Document]:
        """Load all PDF files from the directory.

        Returns:
            List of Document objects with content and metadata
        """
        documents = []

        if not self.pdf_directory.exists():
            logger.error(f"PDF directory not found: {self.pdf_directory}")
            return documents

        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_directory}")

        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()

                # Add source metadata
                for doc in pdf_docs:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["source_path"] = str(pdf_file)

                documents.extend(pdf_docs)
                logger.info(f"  ✓ Loaded {len(pdf_docs)} pages from {pdf_file.name}")

            except Exception as e:
                logger.error(f"  ✗ Error loading {pdf_file.name}: {e}")

        logger.info(f"Total pages loaded: {len(documents)}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with product-aware splitting for catalog documents.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        import re as regex_module

        logger.info(f"Chunking {len(documents)} documents...")

        all_chunks = []

        # Group documents by source file for catalog processing
        docs_by_source = {}
        for doc in documents:
            source_file = doc.metadata.get("source_file", "")
            if source_file not in docs_by_source:
                docs_by_source[source_file] = []
            docs_by_source[source_file].append(doc)

        for source_file, source_docs in docs_by_source.items():
            # Special handling for the telephone catalog - merge all pages first
            if "Catalogue_Telephones" in source_file:
                # Sort by page number and merge all content
                sorted_docs = sorted(source_docs, key=lambda d: d.metadata.get("page", 0))
                merged_content = "\n\n".join(doc.page_content for doc in sorted_docs)

                # Find all iPhone section headers (iPhone X, iPhone 11, ..., iPhone 17)
                # Pattern matches "iPhone XX" followed by description start "Ce modèle"
                # Note: Sometimes there's a page header/footer between iPhone name and "Ce modèle"
                # so we need to match across newlines ([\s\S]) up to "Ce mod"
                product_pattern = r'(iPhone\s+(?:\d+|X))[\s\S]{0,200}?Ce mod[èe]le'
                matches = list(regex_module.finditer(product_pattern, merged_content, regex_module.IGNORECASE))

                if matches:
                    logger.info(f"Found {len(matches)} product sections in {source_file}")

                    # Split content into product sections
                    for i, match in enumerate(matches):
                        section_start = match.start()
                        # Section ends at next product or end of content
                        if i + 1 < len(matches):
                            section_end = matches[i + 1].start()
                        else:
                            section_end = len(merged_content)

                        product_section = merged_content[section_start:section_end].strip()
                        product_name = match.group(1)

                        # Create a document for this product section
                        if len(product_section) > 100:  # Minimum viable section
                            chunk_doc = Document(
                                page_content=product_section,
                                metadata={
                                    "source_file": source_file,
                                    "source_path": sorted_docs[0].metadata.get("source_path", ""),
                                    "product": product_name,
                                    "chunk_type": "product_section"
                                }
                            )
                            all_chunks.append(chunk_doc)
                            logger.debug(f"Created product chunk for {product_name} ({len(product_section)} chars)")

                    # Also add any content before the first product (intro, price table)
                    if matches[0].start() > 100:
                        intro_content = merged_content[:matches[0].start()].strip()
                        if intro_content:
                            intro_doc = Document(
                                page_content=intro_content,
                                metadata={
                                    "source_file": source_file,
                                    "source_path": sorted_docs[0].metadata.get("source_path", ""),
                                    "chunk_type": "catalog_intro"
                                }
                            )
                            all_chunks.append(intro_doc)
                else:
                    # No product sections found, use default splitting
                    for doc in source_docs:
                        chunks = self.text_splitter.split_documents([doc])
                        all_chunks.extend(chunks)
            else:
                # For other documents, use default text splitter
                for doc in source_docs:
                    chunks = self.text_splitter.split_documents([doc])
                    all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks (with product-aware splitting)")
        return all_chunks

    def create_vectorstore(
        self,
        documents: List[Document],
        force_recreate: bool = False
    ) -> Chroma:
        """Create or load vector database.

        Args:
            documents: List of Document chunks to index
            force_recreate: If True, recreate the index even if it exists

        Returns:
            Chroma vectorstore instance
        """
        persist_directory = self.vector_db_path

        # Check if vectorstore already exists
        if os.path.exists(persist_directory) and not force_recreate:
            logger.info(f"Loading existing vectorstore from {persist_directory}")
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory,
                )
                logger.info(f"✓ Vectorstore loaded successfully")
                return self.vectorstore
            except Exception as e:
                logger.warning(f"Failed to load existing vectorstore: {e}")
                logger.info("Creating new vectorstore...")

        # Create new vectorstore
        logger.info(f"Creating vectorstore with {len(documents)} documents...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=persist_directory,
        )

        logger.info(f"✓ Vectorstore created and persisted to {persist_directory}")
        return self.vectorstore

    def index_pdfs(self, force_recreate: bool = False) -> Chroma:
        """Complete indexing pipeline: load, chunk, and index PDFs.

        Args:
            force_recreate: If True, recreate the index even if it exists

        Returns:
            Chroma vectorstore instance
        """
        logger.info("=" * 80)
        logger.info("Starting PDF indexing pipeline")
        logger.info("=" * 80)

        # Step 1: Load PDFs
        documents = self.load_pdfs()
        if not documents:
            raise ValueError("No documents loaded. Check PDF directory.")

        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)

        # Step 3: Create vectorstore
        vectorstore = self.create_vectorstore(chunks, force_recreate=force_recreate)

        logger.info("=" * 80)
        logger.info("PDF indexing pipeline completed successfully")
        logger.info("=" * 80)

        return vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the vectorstore instance.

        Returns:
            Chroma vectorstore or None if not initialized
        """
        if self.vectorstore is None:
            logger.info("Vectorstore not initialized. Loading from disk...")
            try:
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.vector_db_path,
                )
                logger.info("✓ Vectorstore loaded")
            except Exception as e:
                logger.error(f"Failed to load vectorstore: {e}")
                return None

        return self.vectorstore

    def search(
        self,
        query: str,
        k: int = RETRIEVAL_TOP_K,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of documents to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of relevant Document objects
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            logger.error("Vectorstore not available for search")
            return []

        try:
            # CRITICAL FIX: For product-specific queries (iPhone X, iPhone 15, etc.)
            # Do keyword filtering FIRST to get exact model, then semantic search
            import re as regex_module
            product_match = regex_module.search(r'iPhone\s+(\d+|X)', query, regex_module.IGNORECASE)

            if product_match:
                # Extract exact product name
                product_name = product_match.group(0)  # e.g., "iPhone 16"
                logger.info(f"Product-specific query detected: {product_name}")

                # Get MORE documents to increase chance of finding descriptive chunks
                # Increase k multiplier from 3 to 5 to get more candidates
                docs_and_scores = vectorstore.similarity_search_with_score(query, k=k * 5)

                # CRITICAL FIX: Re-rank to prioritize descriptive chunks over price tables
                # Detect factual questions (colors, battery, year, etc.)
                catalog_keywords = ["coloris", "couleur", "colors", "autonomie", "batterie",
                                    "commercialisé", "année", "sortie", "caractéristiques",
                                    "différence", "spécifications"]
                has_catalog_query = any(keyword in query.lower() for keyword in catalog_keywords)

                # Filter and score documents
                scored_docs = []
                for doc, base_score in docs_and_scores:
                    if product_name in doc.page_content:
                        content_lower = doc.page_content.lower()

                        # Calculate content richness score
                        richness_score = 0

                        # CRITICAL: Product name position matters - prefer chunks that START with the product
                        # This avoids chunks that have multiple products (iPhone 15 + iPhone 16)
                        product_index = content_lower.find(product_name.lower())
                        if product_index >= 0 and product_index < 100:  # Product name in first 100 chars
                            richness_score += 50
                        elif product_index >= 0 and product_index < 500:  # Product name in first 500 chars
                            richness_score += 25

                        # Boost: Descriptive chunks (longer, more detailed)
                        if len(doc.page_content) > 1000:
                            richness_score += 10

                        # Boost: Contains catalog facts AFTER the product name
                        # Check that coloris appears AFTER the product mention to avoid wrong phone
                        coloris_index = content_lower.find("coloris disponibles")
                        if coloris_index >= 0 and product_index >= 0 and coloris_index > product_index:
                            richness_score += 30
                        if "autonomie" in content_lower and "h" in content_lower:
                            richness_score += 15
                        if "commercialisé en" in content_lower:
                            richness_score += 15

                        # Penalty: Price table chunks (short, repetitive, page 0)
                        if doc.metadata.get("page") == 0:
                            richness_score -= 30
                        if "tableau des prix" in content_lower or "price" in content_lower:
                            richness_score -= 20
                        if len(doc.page_content) < 800:
                            richness_score -= 10

                        # For catalog queries, prioritize content richness over semantic similarity
                        if has_catalog_query:
                            final_score = richness_score * 2 + base_score
                        else:
                            final_score = base_score + richness_score * 0.5

                        scored_docs.append((doc, final_score))

                # Sort by final score (higher is better for our richness scoring)
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Take top k documents
                filtered_docs = [doc for doc, score in scored_docs[:k]]

                if filtered_docs:
                    logger.info(f"Found {len(filtered_docs)} documents containing '{product_name}' (re-ranked by content richness)")

                    # FIX #2: Fallback semantic search if <k documents found
                    if len(filtered_docs) < k:
                        logger.info(f"Only {len(filtered_docs)} product-specific docs found, completing with semantic search...")

                        # Get additional documents via semantic search
                        semantic_docs = vectorstore.similarity_search(query, k=k * 2)

                        # Add docs that aren't already in filtered_docs
                        existing_ids = {id(doc) for doc in filtered_docs}
                        for doc in semantic_docs:
                            if id(doc) not in existing_ids and len(filtered_docs) < k:
                                filtered_docs.append(doc)

                        logger.info(f"Completed to {len(filtered_docs)} documents total")

                    return filtered_docs
                else:
                    logger.warning(f"No documents found containing '{product_name}', falling back to semantic search")

            # Standard semantic search
            if score_threshold is not None:
                # Search with score threshold
                docs_and_scores = vectorstore.similarity_search_with_score(
                    query, k=k
                )
                # Filter by threshold (lower score = more similar in some embeddings)
                docs = [
                    doc for doc, score in docs_and_scores
                    if score >= score_threshold
                ]
            else:
                # Regular similarity search
                docs = vectorstore.similarity_search(query, k=k)

            logger.info(f"Found {len(docs)} relevant documents for query")
            return docs

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


# Singleton instance for easy access
_pdf_indexer_instance: Optional[PDFIndexer] = None


def get_pdf_indexer(force_recreate: bool = False) -> PDFIndexer:
    """Get or create the global PDF indexer instance.

    Args:
        force_recreate: If True, recreate the singleton instance

    Returns:
        PDFIndexer instance
    """
    global _pdf_indexer_instance

    if _pdf_indexer_instance is None or force_recreate:
        _pdf_indexer_instance = PDFIndexer()

    return _pdf_indexer_instance


def initialize_vector_db(force_recreate: bool = False) -> Chroma:
    """Initialize the vector database (convenience function).

    Args:
        force_recreate: If True, recreate the index even if it exists

    Returns:
        Chroma vectorstore instance
    """
    indexer = get_pdf_indexer()
    return indexer.index_pdfs(force_recreate=force_recreate)


if __name__ == "__main__":
    # Test PDF indexing
    print("Testing PDF Indexer...")
    print("=" * 80)

    indexer = PDFIndexer()

    # Index PDFs
    vectorstore = indexer.index_pdfs(force_recreate=False)

    # Test search
    print("\nTesting search...")
    test_query = "Quels modes de paiement acceptez-vous ?"
    results = indexer.search(test_query, k=3)

    print(f"\nQuery: {test_query}")
    print(f"Found {len(results)} results:\n")

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"  Content: {doc.page_content[:200]}...")
        print()
