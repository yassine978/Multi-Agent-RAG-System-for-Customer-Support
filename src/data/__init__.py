"""Data loading and indexing modules.

This package contains:
- PDF Indexer: Loads and indexes FAQ documents into vector database
- Excel Loader: Loads customer data from Excel files into DataFrames
"""

from .excel_loader import ExcelDataLoader, get_excel_loader
from .pdf_indexer import PDFIndexer, get_pdf_indexer, initialize_vector_db

__all__ = [
    "PDFIndexer",
    "get_pdf_indexer",
    "initialize_vector_db",
    "ExcelDataLoader",
    "get_excel_loader",
]
