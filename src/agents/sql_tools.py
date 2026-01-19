"""SQL Tools for querying customer data.

This module provides LangChain tools with proper decorators for
querying Excel data as if it were a SQL database.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from langchain.tools import tool

from ..data.excel_loader import get_excel_loader
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Client Lookup Tools
# =============================================================================

@tool
def get_client_by_name(nom: str, prenom: str) -> Dict[str, Any]:
    """Recherche un client par son nom et prénom.

    Args:
        nom: Nom de famille du client (ex: "Bertrand")
        prenom: Prénom du client (ex: "Jean")

    Returns:
        Informations du client incluant client_id, email, téléphone, adresse

    Example:
        get_client_by_name("Bertrand", "Jean")
    """
    try:
        loader = get_excel_loader()
        clients_df = loader.get_table("clients")

        if clients_df is None:
            return {"error": "Table clients non disponible"}

        # Search case-insensitive
        result = clients_df[
            (clients_df["nom"].str.lower() == nom.lower()) &
            (clients_df["prenom"].str.lower() == prenom.lower())
        ]

        if result.empty:
            return {"error": f"Aucun client trouvé avec le nom {prenom} {nom}"}

        client = result.iloc[0].to_dict()
        logger.info(f"Client trouvé: {client['client_id']}")
        return client

    except Exception as e:
        logger.error(f"Erreur dans get_client_by_name: {e}")
        return {"error": str(e)}


@tool
def get_client_consumption(client_id: int) -> Dict[str, Any]:
    """Obtient la consommation data du client pour le mois en cours.

    Args:
        client_id: ID du client

    Returns:
        Informations de consommation (data utilisée, minutes, SMS)

    Example:
        get_client_consumption(1)
    """
    try:
        loader = get_excel_loader()
        consommation_df = loader.get_table("consommation")

        if consommation_df is None:
            return {"error": "Table consommation non disponible"}

        # Get latest consumption for this client
        result = consommation_df[consommation_df["client_id"] == client_id]

        if result.empty:
            return {"error": f"Aucune consommation trouvée pour le client {client_id}"}

        # Get most recent month
        latest = result.sort_values("mois", ascending=False).iloc[0]

        return {
            "client_id": int(latest["client_id"]),
            "mois": str(latest["mois"]),
            "data_utilise_gb": float(latest["data_utilise_gb"]),
            "minutes_utilisees": int(latest["minutes_utilisees"]),
            "sms_utilises": int(latest["sms_utilises"]),
        }

    except Exception as e:
        logger.error(f"Erreur dans get_client_consumption: {e}")
        return {"error": str(e)}


@tool
def get_client_forfait(client_id: int) -> Dict[str, Any]:
    """Obtient le forfait actuel du client avec tous les détails.

    Args:
        client_id: ID du client

    Returns:
        Détails du forfait (nom, data, prix, engagement)

    Example:
        get_client_forfait(1)
    """
    try:
        loader = get_excel_loader()
        abonnements_df = loader.get_table("abonnements")
        forfaits_df = loader.get_table("forfaits")

        if abonnements_df is None or forfaits_df is None:
            return {"error": "Tables non disponibles"}

        # Get active subscription (status starts with "Actif")
        result = abonnements_df[
            (abonnements_df["client_id"] == client_id) &
            (abonnements_df["statut"].str.startswith("Actif"))
        ]

        if result.empty:
            return {"error": f"Aucun abonnement actif pour le client {client_id}"}

        abonnement = result.iloc[0]
        forfait_id = abonnement["forfait_id"]

        # Get forfait details
        forfait = forfaits_df[forfaits_df["forfait_id"] == forfait_id].iloc[0]

        # Handle "Illimité/Illimitée/Illimités/Illimitées" values for minutes and SMS
        minutes_val = forfait["minutes_incluses"]
        sms_val = forfait["sms_inclus"]

        # Check if value starts with "illimit" (covers all variations)
        def is_unlimited(val):
            return str(val).lower().startswith("illimit")

        minutes_incluses = str(minutes_val) if is_unlimited(minutes_val) else int(minutes_val)
        sms_inclus = str(sms_val) if is_unlimited(sms_val) else int(sms_val)

        return {
            "forfait_nom": str(forfait["nom_forfait"]),
            "data_mensuel_gb": float(forfait["data_mensuel_gb"]),
            "prix_mensuel": float(forfait["prix_mensuel"]),
            "engagement_mois": int(forfait["engagement_mois"]),
            "minutes_incluses": minutes_incluses,
            "sms_inclus": sms_inclus,
            "date_debut": str(abonnement["date_debut"]),
            "statut": str(abonnement["statut"]),
        }

    except Exception as e:
        logger.error(f"Erreur dans get_client_forfait: {e}")
        return {"error": str(e)}


@tool
def get_client_factures(client_id: int) -> Dict[str, Any]:
    """Obtient les factures du client.

    Args:
        client_id: ID du client

    Returns:
        Liste des factures avec montants et statuts

    Example:
        get_client_factures(1)
    """
    try:
        loader = get_excel_loader()
        factures_df = loader.get_table("factures")

        if factures_df is None:
            return {"error": "Table factures non disponible"}

        result = factures_df[factures_df["client_id"] == client_id]

        if result.empty:
            return {"error": f"Aucune facture pour le client {client_id}"}

        # Get latest invoice
        latest = result.sort_values("mois", ascending=False).iloc[0]

        # Count unpaid
        unpaid = result[result["statut_paiement"] == "Impayé"]

        return {
            "derniere_facture_montant": float(latest["montant"]),
            "derniere_facture_mois": str(latest["mois"]),
            "derniere_facture_statut": str(latest["statut_paiement"]),
            "date_echeance": str(latest["date_echeance"]),
            "factures_impayees": len(unpaid),
            "total_impaye": float(unpaid["montant"].sum()) if len(unpaid) > 0 else 0.0,
        }

    except Exception as e:
        logger.error(f"Erreur dans get_client_factures: {e}")
        return {"error": str(e)}


@tool
def get_client_tickets(client_id: int) -> Dict[str, Any]:
    """Obtient les tickets de support du client.

    Args:
        client_id: ID du client

    Returns:
        Liste des tickets de support

    Example:
        get_client_tickets(1)
    """
    try:
        loader = get_excel_loader()
        tickets_df = loader.get_table("tickets_support")

        if tickets_df is None:
            return {"error": "Table tickets non disponible"}

        result = tickets_df[tickets_df["client_id"] == client_id]

        if result.empty:
            return {"error": f"Aucun ticket pour le client {client_id}"}

        # Get open tickets
        open_tickets = result[result["statut"].isin(["Ouvert", "En cours"])]

        tickets_info = []
        for _, ticket in open_tickets.iterrows():
            tickets_info.append({
                "ticket_id": int(ticket["ticket_id"]),
                "categorie": str(ticket["categorie"]),
                "sujet": str(ticket["sujet"]),
                "statut": str(ticket["statut"]),
                "priorite": str(ticket["priorite"]),
            })

        return {
            "total_tickets": len(result),
            "tickets_ouverts": len(open_tickets),
            "tickets_details": tickets_info[:5],  # Limit to 5 most recent
        }

    except Exception as e:
        logger.error(f"Erreur dans get_client_tickets: {e}")
        return {"error": str(e)}


@tool
def list_all_forfaits() -> Dict[str, Any]:
    """Liste tous les forfaits disponibles avec leurs caractéristiques.

    Returns:
        Liste de tous les forfaits avec prix et data

    Example:
        list_all_forfaits()
    """
    try:
        loader = get_excel_loader()
        forfaits_df = loader.get_table("forfaits")

        if forfaits_df is None:
            return {"error": "Table forfaits non disponible"}

        # Helper to check if value is unlimited (any variation)
        def is_unlimited(val):
            return str(val).lower().startswith("illimit")

        forfaits = []
        for _, forfait in forfaits_df.iterrows():
            # Handle "Illimité/Illimitée/Illimités/Illimitées" values
            minutes_val = forfait["minutes_incluses"]
            sms_val = forfait["sms_inclus"]

            minutes_incluses = str(minutes_val) if is_unlimited(minutes_val) else int(minutes_val)
            sms_inclus = str(sms_val) if is_unlimited(sms_val) else int(sms_val)

            forfaits.append({
                "nom": str(forfait["nom_forfait"]),
                "data_gb": float(forfait["data_mensuel_gb"]),
                "prix": float(forfait["prix_mensuel"]),
                "engagement_mois": int(forfait["engagement_mois"]),
                "minutes_incluses": minutes_incluses,
                "sms_inclus": sms_inclus,
            })

        return {"forfaits": forfaits}

    except Exception as e:
        logger.error(f"Erreur dans list_all_forfaits: {e}")
        return {"error": str(e)}


# =============================================================================
# Get all tools for the agent
# =============================================================================

def get_sql_tools() -> list:
    """Get all SQL tools for the agent.

    Returns:
        List of LangChain tools
    """
    return [
        get_client_by_name,
        get_client_consumption,
        get_client_forfait,
        get_client_factures,
        get_client_tickets,
        list_all_forfaits,
    ]


# =============================================================================
# Tool descriptions for LLM
# =============================================================================

TOOLS_DESCRIPTION = """
Tu as accès aux outils suivants pour interroger les données clients:

1. **get_client_by_name(nom, prenom)**: Trouve un client par son nom complet
   - Utilise TOUJOURS cet outil en premier pour obtenir le client_id
   - Exemple: get_client_by_name("Bertrand", "Jean")

2. **get_client_consumption(client_id)**: Obtient la consommation data du mois
   - Retourne data utilisée, minutes, SMS

3. **get_client_forfait(client_id)**: Obtient le forfait actuel du client
   - Retourne nom du forfait, data incluse, prix, engagement

4. **get_client_factures(client_id)**: Obtient les factures du client
   - Retourne dernière facture et factures impayées

5. **get_client_tickets(client_id)**: Obtient les tickets de support
   - Retourne tickets ouverts et en cours

6. **list_all_forfaits()**: Liste tous les forfaits disponibles
   - Utile pour répondre aux questions générales sur les forfaits

MÉTHODOLOGIE OBLIGATOIRE:
1. Extraire le prénom et nom du client de la question
2. Appeler get_client_by_name(nom, prenom) pour obtenir le client_id
3. Utiliser le client_id pour appeler les autres outils selon la question
4. Formater la réponse en français de manière claire et naturelle
"""
