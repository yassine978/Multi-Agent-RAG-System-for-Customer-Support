# TelecomPlus - Système Multi-Agent de Support Client

## Description du Projet

Système multi-agent intelligent pour le support client de **TelecomPlus**, un opérateur téléphonique fictif. Le système répond aux questions clients en combinant :
- **RAG (Retrieval-Augmented Generation)** : Recherche sémantique dans les documents PDF (FAQ)
- **SQL Agent** : Requêtes sur les données clients (Excel/DataFrame)
- **Orchestration LangGraph** : Coordination intelligente des agents selon le type de question

---

## Architecture du Système

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│                           (LangGraph StateGraph)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐                                                          │
│    │    ROUTER    │  Classifie la question en 4 types:                       │
│    │    AGENT     │  RAG | SQL | HYBRID | GENERAL                            │
│    └──────┬───────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│    ┌──────────────────────────────────────────────────────────┐              │
│    │                    ROUTING CONDITIONNEL                   │              │
│    └──────────────────────────────────────────────────────────┘              │
│           │                    │                    │                        │
│           ▼                    ▼                    ▼                        │
│    ┌────────────┐       ┌────────────┐       ┌────────────┐                  │
│    │    RAG     │       │    SQL     │       │  GENERAL   │                  │
│    │   AGENT    │       │   AGENT    │       │  RESPONSE  │                  │
│    └─────┬──────┘       └─────┬──────┘       └────────────┘                  │
│          │                    │                                              │
│          │    (HYBRID)        │                                              │
│          └────────┬───────────┘                                              │
│                   ▼                                                          │
│            ┌────────────┐                                                    │
│            │ SYNTHESIS  │  Combine RAG + SQL pour questions hybrides         │
│            │   AGENT    │                                                    │
│            └────────────┘                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Composants Principaux

#### 1. Router Agent (`src/agents/router_agent.py`)
- **Rôle** : Classification des questions entrantes
- **Modèle** : Gemini 2.5 Flash (rapide, déterministe)
- **Types de classification** :
  - `RAG` : Questions générales (FAQ, procédures, catalogue)
  - `SQL` : Questions spécifiques client identifié par nom
  - `HYBRID` : Combinaison FAQ + données client
  - `GENERAL` : Salutations, remerciements

#### 2. RAG Agent (`src/agents/rag_agent.py`)
- **Rôle** : Recherche et génération de réponses à partir des PDF
- **Composants** :
  - Vector Store (ChromaDB) pour la recherche sémantique
  - Embeddings (sentence-transformers/all-MiniLM-L6-v2)
  - Query expansion intelligente selon le type de question
- **Optimisations** :
  - Détection de questions de comparaison (multi-product retrieval)
  - Détection de questions binaires (couverture complète des scénarios)
  - Extraction directe de faits par regex (fallback)
  - Cache des résultats de recherche vectorielle

#### 3. SQL Agent (`src/agents/sql_agent.py`)
- **Rôle** : Requêtes sur les données clients via pattern ReAct
- **Outils disponibles** (`src/agents/sql_tools.py`) :
  - `get_client_by_name(nom, prenom)` : Identification client
  - `get_client_consumption(client_id)` : Consommation data/minutes/SMS
  - `get_client_forfait(client_id)` : Détails du forfait actuel
  - `get_client_factures(client_id)` : Factures et impayés
  - `get_client_tickets(client_id)` : Tickets de support
  - `list_all_forfaits()` : Liste tous les forfaits
- **Pattern ReAct** : Reasoning + Acting en boucle (max 5 itérations)

#### 4. Synthesis Agent (intégré dans `src/agents/orchestrator.py`)
- **Rôle** : Fusion intelligente des résultats RAG et SQL
- **Modèle** : Gemini 2.5 Pro (meilleur raisonnement)
- **Règles** :
  - Priorité RAG pour les politiques (roaming, frais, conditions)
  - Priorité SQL pour les données client (forfait, consommation)
  - Personnalisation avec les détails du client

#### 5. Orchestrator (`src/agents/orchestrator.py`)
- **Framework** : LangGraph StateGraph
- **Flux de travail** :
  ```
  route_question → [rag_node | sql_node | general_response]
                        ↓           ↓
                   (si HYBRID) → synthesis_node → END
  ```

### Structure des Fichiers

```
src/
├── __init__.py
├── main.py                 # Point d'entrée principal, fonction answer()
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py     # Orchestration LangGraph
│   ├── router_agent.py     # Classification des questions
│   ├── rag_agent.py        # Agent RAG avec recherche PDF
│   ├── sql_agent.py        # Agent SQL avec pattern ReAct
│   └── sql_tools.py        # Outils LangChain pour requêtes données
├── data/
│   ├── __init__.py
│   ├── excel_loader.py     # Chargement des tables Excel
│   └── pdf_indexer.py      # Indexation PDF → ChromaDB
├── prompts/
│   ├── __init__.py
│   └── templates.py        # Tous les prompts optimisés
└── utils/
    ├── __init__.py
    ├── config.py           # Configuration centralisée
    ├── cache.py            # Système de cache
    ├── langfuse_config.py  # Intégration Langfuse
    └── logger.py           # Logging configuré
```

---

## Choix Techniques et Justifications

### 1. Modèles LLM (Google Gemini)

| Agent | Modèle | Justification |
|-------|--------|---------------|
| Router | Gemini 2.5 Flash | Rapidité, classification simple, coût minimal |
| RAG | Gemini 2.5 Flash | Bon équilibre vitesse/qualité pour génération |
| SQL | Gemini 2.5 Flash | Génération déterministe de tool calls |
| Synthesis | Gemini 2.5 Pro | Raisonnement avancé pour fusion multi-sources |
| Évaluation | Gemini 2.0 Flash | LLM-as-a-judge efficace et économique |

### 2. Vector Database : ChromaDB
- **Choix** : Stockage local persistant
- **Avantages** :
  - Pas de service externe requis
  - Persistence sur disque (`data/vector_db/`)
  - API simple avec LangChain

### 3. Embeddings : sentence-transformers/all-MiniLM-L6-v2
- **Avantages** :
  - Modèle léger et rapide (CPU)
  - Bonne qualité pour le français
  - Gratuit et local

### 4. Framework d'Orchestration : LangGraph
- **Avantages** :
  - Gestion d'état typée (TypedDict)
  - Graphe de flux conditionnel
  - Traçabilité native
  - Support streaming

### 5. Pattern ReAct pour SQL Agent
- **Implémentation custom** plutôt que LangChain Agent standard
- **Avantages** :
  - Contrôle total sur le parsing des tool calls
  - Gestion d'erreur robuste
  - Logging détaillé des étapes

### 6. Optimisations RAG

| Optimisation | Description |
|--------------|-------------|
| Query Expansion | Enrichissement automatique des requêtes (roaming, prix, etc.) |
| Multi-Product Retrieval | Stratégie spéciale pour questions de comparaison |
| Binary Question Detection | Couverture complète des scénarios (si engagé/si non engagé) |
| Regex Fallback | Extraction directe de faits (coloris, années, autonomie) |
| Chunking optimisé | 2000 chars + 400 overlap pour préserver contexte |

### 7. Monitoring : Langfuse
- Traçabilité complète des appels LLM
- Métriques de latence et coûts
- Session tracking

---

## Installation et Exécution

### Prérequis
- Python 3.10+
- Clé API Google Gemini
- (Optionnel) Credentials Langfuse

### 1. Installation

```bash
# Cloner le repository
git clone <repository-url>
cd Multi-Agent-RAG-System-for-Customer-Support

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration

Créer un fichier `.env` à la racine :

```env
# Obligatoire
GOOGLE_API_KEY=votre_clé_api_google

# Optionnel - Monitoring Langfuse
LANGFUSE_PUBLIC_KEY=votre_clé_publique
LANGFUSE_SECRET_KEY=votre_clé_secrète
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Lancer l'Application

```bash
# Interface Streamlit
streamlit run app.py

# Ou test en ligne de commande
python -m src.main
```

### 4. Lancer l'Évaluation

```bash
python evaluate.py
```

Les résultats sont sauvegardés dans `evaluation_results.xlsx`.

---

## Résultats d'Évaluation

### Score Global : 100% (25/25 questions)

L'évaluation utilise un **LLM-as-a-judge** (Gemini 2.0 Flash) avec évaluation par batch pour optimiser les appels API.

### Détail par Difficulté

| Difficulté | Questions | Score | Taux |
|------------|-----------|-------|------|
| Facile | 7 | 7/7 | 100% |
| Moyen | 11 | 11/11 | 100% |
| Difficile | 3 | 3/3 | 100% |
| Très Difficile | 4 | 4/4 | 100% |

### Critères d'Évaluation LLM-as-Judge

- **Score 1** : Réponse contient l'information essentielle attendue
- **Score 0.5** : Réponse partiellement correcte (info manquante mineure)
- **Score 0** : Réponse incorrecte ou information critique manquante

### Optimisations API
- **Batching** : 5 questions par appel API d'évaluation
- **Total appels** : 5 au lieu de 25 (réduction de 80%)

---

## Données

### Tables Excel (6 fichiers dans `data/xlsx/`)

| Table | Colonnes | Description |
|-------|----------|-------------|
| clients.xlsx | client_id, nom, prenom, email, telephone, date_inscription, adresse, ville, code_postal | 20 clients |
| forfaits.xlsx | forfait_id, nom_forfait, data_mensuel_gb, prix_mensuel, engagement_mois, minutes_incluses, sms_inclus | 5 forfaits |
| abonnements.xlsx | abonnement_id, client_id, forfait_id, date_debut, date_fin, statut | Abonnements actifs |
| consommation.xlsx | client_id, mois, data_utilise_gb, minutes_utilisees, sms_utilises | Consommation mensuelle |
| factures.xlsx | facture_id, client_id, mois, montant, statut_paiement, date_echeance | Historique factures |
| tickets_support.xlsx | ticket_id, client_id, categorie, sujet, description, statut, date_creation, priorite | Tickets support |

### Documents PDF (7 fichiers dans `data/pdfs/`)

- FAQ_Facturation_et_Paiements.pdf
- FAQ_Forfaits_et_Abonnements.pdf
- FAQ_Support_Technique.pdf
- FAQ_Roaming_International.pdf
- FAQ_Compte_Client.pdf
- FAQ_Resiliation_et_Modifications.pdf
- FAQ_Catalogue_Telephones.pdf

---

## Exemples de Questions

### Type RAG (FAQ)
```
Q: Quels modes de paiement acceptez-vous ?
R: Nous acceptons les paiements par carte bancaire, prélèvement automatique,
   virement bancaire et PayPal. Le prélèvement automatique garantit de ne
   jamais manquer une échéance. Source: FAQ_Facturation_et_Paiements.pdf
```

### Type SQL (Données client)
```
Q: Je m'appelle Jean Bertrand. Quelle est ma consommation data ce mois-ci ?
R: Bonjour Monsieur Bertrand, ce mois-ci vous avez consommé 3.20GB de data
   sur votre forfait Essentiel 5GB.
```

### Type HYBRID (FAQ + Données client)
```
Q: Je m'appelle Jean Bertrand. Je pars en Italie 10 jours, dois-je changer de forfait ?
R: Bonjour Monsieur Bertrand, vous avez actuellement le forfait Essentiel 5GB.
   Pour votre voyage en Italie, bonne nouvelle : vous bénéficiez du roaming
   inclus dans l'Union Européenne selon la politique "roaming comme à la maison".
   Vous pouvez utiliser votre forfait normalement sans frais supplémentaires.
```

---

## Auteur

Projet réalisé dans le cadre du cours **Generative AI** - Université Paris Dauphine - IASD 2025-2026
