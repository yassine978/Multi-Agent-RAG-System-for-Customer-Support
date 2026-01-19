"""Streamlit chat interface for TelecomPlus customer support.

Enhanced interface with:
- Loading indicators
- System status display
- Example questions
- Session tracking
- Error handling
"""

import time

import streamlit as st

from src.main import answer, answer_with_metadata, answer_streaming, get_system_status

# Page configuration
st.set_page_config(
    page_title="TelecomPlus Support",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .example-question {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        cursor: pointer;
    }
    .status-ready {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    **TelecomPlus Support Client**

    Assistant intelligent alimenté par:
    - 🤖 Multi-Agent AI System
    - 📚 RAG (Retrieval-Augmented Generation)
    - 🗄️ Accès aux données clients
    - 🔄 LangGraph Orchestration
    """)

    st.divider()

    # System status
    st.subheader("📊 Statut du Système")
    status = get_system_status()

    if status["ready"]:
        st.markdown('<p class="status-ready">✅ Système opérationnel</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-error">⚠️ Système en cours d\'initialisation</p>', unsafe_allow_html=True)
        if status["initialization_error"]:
            with st.expander("Détails de l'erreur"):
                st.error(status["initialization_error"])

    st.divider()

    # Example questions
    st.subheader("💡 Questions Exemples")

    example_questions = [
        "Quels modes de paiement acceptez-vous ?",
        "Comment fonctionne le roaming international ?",
        "Quels sont vos forfaits disponibles ?",
        "Comment résilier mon abonnement ?",
        "Y a-t-il des frais de résiliation ?",
    ]

    st.markdown("Cliquez sur une question pour l'essayer:")

    for i, question in enumerate(example_questions):
        if st.button(f"❓ {question}", key=f"example_{i}", use_container_width=True):
            # Add to session state to be processed in main area
            st.session_state.pending_question = question
            st.rerun()

    st.divider()

    # Clear chat button
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Info about the system
    with st.expander("ℹ️ Comment ça marche ?"):
        st.markdown("""
        Le système utilise plusieurs agents IA spécialisés:

        1. **Router Agent**: Classe votre question
        2. **RAG Agent**: Cherche dans les FAQs
        3. **SQL Agent**: Accède aux données clients
        4. **Orchestrator**: Coordonne les agents

        Alimenté par Google Gemini et LangGraph.
        """)

# Main area
st.title("📱 TelecomPlus - Support Client")
st.markdown("*Assistant intelligent pour répondre à vos questions*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{int(time.time())}"

# Welcome message
if len(st.session_state.messages) == 0:
    st.info("👋 Bonjour ! Je suis votre assistant virtuel TelecomPlus. Comment puis-je vous aider aujourd'hui ?")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle pending question from sidebar
if hasattr(st.session_state, 'pending_question'):
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Stream the response token by token
            for token in answer_streaming(prompt, session_id=st.session_state.session_id):
                full_response += token
                message_placeholder.markdown(full_response + "▌")

            # Remove cursor and show final response
            message_placeholder.markdown(full_response)
            response = full_response

        except Exception as e:
            error_msg = f"❌ Une erreur s'est produite: {str(e)}"
            st.error(error_msg)
            response = error_msg

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Posez votre question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Stream the response token by token
            for token in answer_streaming(prompt, session_id=st.session_state.session_id):
                full_response += token
                message_placeholder.markdown(full_response + "▌")

            # Remove cursor and show final response
            message_placeholder.markdown(full_response)
            response = full_response

        except Exception as e:
            error_msg = f"❌ Une erreur s'est produite: {str(e)}"
            st.error(error_msg)
            response = error_msg

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.caption("🤖 Propulsé par Multi-Agent RAG System | Google Gemini | LangGraph | Langfuse")
