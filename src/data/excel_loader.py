"""Excel data loader for customer database.

This module loads customer data from Excel files and provides
utility functions for querying the data.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils.config import EXCEL_DIRECTORY, EXCEL_FILES
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ExcelDataLoader:
    """Handles loading and accessing customer data from Excel files."""

    def __init__(
        self,
        excel_directory: str = EXCEL_DIRECTORY,
        excel_files: Dict[str, str] = EXCEL_FILES,
    ):
        """Initialize Excel data loader.

        Args:
            excel_directory: Directory containing Excel files
            excel_files: Dictionary mapping table names to file names
        """
        self.excel_directory = Path(excel_directory)
        self.excel_files = excel_files
        self.dataframes: Dict[str, pd.DataFrame] = {}

    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all Excel tables into DataFrames.

        Returns:
            Dictionary mapping table names to DataFrames
        """
        logger.info("=" * 80)
        logger.info("Loading Excel data tables")
        logger.info("=" * 80)

        if not self.excel_directory.exists():
            logger.error(f"Excel directory not found: {self.excel_directory}")
            return {}

        for table_name, filename in self.excel_files.items():
            file_path = self.excel_directory / filename

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            try:
                logger.info(f"Loading {table_name} from {filename}...")
                df = pd.read_excel(file_path)
                self.dataframes[table_name] = df
                logger.info(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"    Columns: {', '.join(df.columns.tolist())}")

            except Exception as e:
                logger.error(f"  ✗ Error loading {filename}: {e}")

        logger.info("=" * 80)
        logger.info(f"Successfully loaded {len(self.dataframes)} tables")
        logger.info("=" * 80)

        return self.dataframes

    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a specific table DataFrame.

        Args:
            table_name: Name of the table (e.g., 'clients', 'forfaits')

        Returns:
            DataFrame or None if not found
        """
        if table_name not in self.dataframes:
            logger.warning(f"Table '{table_name}' not loaded")
            return None

        return self.dataframes[table_name]

    def get_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded DataFrames.

        Returns:
            Dictionary of all DataFrames
        """
        return self.dataframes

    def get_client_by_id(self, client_id: int) -> Optional[pd.Series]:
        """Get client information by ID.

        Args:
            client_id: Client ID

        Returns:
            Series with client data or None if not found
        """
        clients_df = self.get_table("clients")
        if clients_df is None:
            return None

        result = clients_df[clients_df["client_id"] == client_id]
        if result.empty:
            return None

        return result.iloc[0]

    def get_client_by_email(self, email: str) -> Optional[pd.Series]:
        """Get client information by email.

        Args:
            email: Client email address

        Returns:
            Series with client data or None if not found
        """
        clients_df = self.get_table("clients")
        if clients_df is None:
            return None

        result = clients_df[clients_df["email"].str.lower() == email.lower()]
        if result.empty:
            return None

        return result.iloc[0]

    def get_client_by_phone(self, phone: str) -> Optional[pd.Series]:
        """Get client information by phone number.

        Args:
            phone: Client phone number

        Returns:
            Series with client data or None if not found
        """
        clients_df = self.get_table("clients")
        if clients_df is None:
            return None

        # Normalize phone number (remove spaces, dashes)
        phone_normalized = phone.replace(" ", "").replace("-", "")

        result = clients_df[
            clients_df["telephone"].astype(str).str.replace(" ", "").str.replace("-", "") == phone_normalized
        ]

        if result.empty:
            return None

        return result.iloc[0]

    def get_client_subscription(self, client_id: int) -> Optional[pd.DataFrame]:
        """Get client subscription information.

        Args:
            client_id: Client ID

        Returns:
            DataFrame with subscription data or None if not found
        """
        abonnements_df = self.get_table("abonnements")
        if abonnements_df is None:
            return None

        result = abonnements_df[abonnements_df["client_id"] == client_id]
        return result if not result.empty else None

    def get_client_invoices(self, client_id: int) -> Optional[pd.DataFrame]:
        """Get client invoices.

        Args:
            client_id: Client ID

        Returns:
            DataFrame with invoice data or None if not found
        """
        factures_df = self.get_table("factures")
        if factures_df is None:
            return None

        result = factures_df[factures_df["client_id"] == client_id]
        return result if not result.empty else None

    def get_client_consumption(self, client_id: int) -> Optional[pd.DataFrame]:
        """Get client consumption data.

        Args:
            client_id: Client ID

        Returns:
            DataFrame with consumption data or None if not found
        """
        consommation_df = self.get_table("consommation")
        if consommation_df is None:
            return None

        result = consommation_df[consommation_df["client_id"] == client_id]
        return result if not result.empty else None

    def get_client_support_tickets(self, client_id: int) -> Optional[pd.DataFrame]:
        """Get client support tickets.

        Args:
            client_id: Client ID

        Returns:
            DataFrame with support ticket data or None if not found
        """
        tickets_df = self.get_table("tickets_support")
        if tickets_df is None:
            return None

        result = tickets_df[tickets_df["client_id"] == client_id]
        return result if not result.empty else None

    def get_table_schema(self, table_name: str) -> Optional[str]:
        """Get a human-readable schema description for a table.

        Args:
            table_name: Name of the table

        Returns:
            String description of the table schema
        """
        df = self.get_table(table_name)
        if df is None:
            return None

        schema = f"\n{table_name.upper()} Table Schema:\n"
        schema += "-" * 50 + "\n"
        schema += f"Total rows: {len(df)}\n"
        schema += f"Columns ({len(df.columns)}):\n"

        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()

            schema += f"  - {col}: {dtype}"
            if null_count > 0:
                schema += f" ({null_count} nulls)"
            schema += f" ({unique_count} unique values)\n"

        # Show a sample row
        schema += f"\nSample row:\n{df.iloc[0].to_dict()}\n"

        return schema

    def get_all_schemas(self) -> str:
        """Get schemas for all loaded tables.

        Returns:
            String with all table schemas
        """
        schemas = []
        for table_name in self.dataframes.keys():
            schema = self.get_table_schema(table_name)
            if schema:
                schemas.append(schema)

        return "\n".join(schemas)

    def search_client(self, query: str) -> Optional[pd.Series]:
        """Search for a client by ID, email, phone, or name.

        Args:
            query: Search query (can be ID, email, phone, or name)

        Returns:
            Series with client data or None if not found
        """
        # Try as client ID (if numeric)
        if query.isdigit():
            result = self.get_client_by_id(int(query))
            if result is not None:
                return result

        # Try as email
        if "@" in query:
            result = self.get_client_by_email(query)
            if result is not None:
                return result

        # Try as phone
        if any(char.isdigit() for char in query):
            result = self.get_client_by_phone(query)
            if result is not None:
                return result

        # Try as name
        clients_df = self.get_table("clients")
        if clients_df is not None:
            result = clients_df[
                clients_df["nom"].str.contains(query, case=False, na=False) |
                clients_df["prenom"].str.contains(query, case=False, na=False)
            ]
            if not result.empty:
                return result.iloc[0]

        return None


# Singleton instance for easy access
_excel_loader_instance: Optional[ExcelDataLoader] = None


def get_excel_loader(force_reload: bool = False) -> ExcelDataLoader:
    """Get or create the global Excel loader instance.

    Args:
        force_reload: If True, recreate the singleton instance

    Returns:
        ExcelDataLoader instance
    """
    global _excel_loader_instance

    if _excel_loader_instance is None or force_reload:
        _excel_loader_instance = ExcelDataLoader()
        _excel_loader_instance.load_all_tables()

    return _excel_loader_instance


if __name__ == "__main__":
    # Test Excel loader
    print("Testing Excel Data Loader...")
    print("=" * 80)

    loader = ExcelDataLoader()
    dataframes = loader.load_all_tables()

    print("\nLoaded tables:", list(dataframes.keys()))

    # Test client lookup
    print("\n" + "=" * 80)
    print("Testing client lookup...")

    # Get first client
    clients_df = loader.get_table("clients")
    if clients_df is not None and len(clients_df) > 0:
        first_client_id = clients_df.iloc[0]["client_id"]
        print(f"\nClient ID {first_client_id}:")

        client = loader.get_client_by_id(first_client_id)
        if client is not None:
            print(f"  Name: {client['prenom']} {client['nom']}")
            print(f"  Email: {client['email']}")
            print(f"  Phone: {client['telephone']}")

        # Get subscription
        subscription = loader.get_client_subscription(first_client_id)
        if subscription is not None:
            print(f"\n  Subscriptions: {len(subscription)}")

        # Get invoices
        invoices = loader.get_client_invoices(first_client_id)
        if invoices is not None:
            print(f"  Invoices: {len(invoices)}")

    # Show table schemas
    print("\n" + "=" * 80)
    print("Table Schemas:")
    print(loader.get_all_schemas())
