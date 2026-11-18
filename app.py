
import traceback
import asyncio
from pathlib import Path

import streamlit as st

from research_assistant.config import (
    DOCUMENTS_DIR,
    ensure_directories
)

from research_assistant.db import (
    initialize_db,
    list_documents,
    create_conversation,
    list_conversations,
    get_conversation_messages
)

from research_assistant.ingest import ingest_pdf
from research_assistant.agent import ask_research_question

def get_async_loop():
    """Get or create a persistent asyncio event loop for this Streamlit session."""

    if "async_loop" not in st.session_state:
        st.session_state.async_loop = asyncio.new_event_loop()

    return st.session_state.async_loop


def run_async(coroutine):
    """Execute an asynchronous function using the persistent event loop."""

    loop = get_async_loop()

    return loop.run_until_complete(coroutine)

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Research Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

ensure_directories()
initialize_db()


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = create_conversation(
        "New Research Conversation"
    )


# --------------------------------------------------
# 3. HELPER FUNCTION
# --------------------------------------------------

def start_new_conversation():
    """Create and activate a new conversation."""

    conversation_id = create_conversation(
        "New Research Conversation"
    )

    st.session_state.active_conversation_id = conversation_id


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.title("📚 AI Research & Knowledge Assistant")

st.caption(
    "Research across your PDFs using hybrid retrieval, "
    "reranking, MCP and conversational memory."
)


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("💬 Research Conversations")

    if st.button(
        "➕ New Conversation",
        use_container_width=True
    ):
        start_new_conversation()
        st.rerun()

    st.divider()

    conversations = list_conversations(30)

    active_id = st.session_state.active_conversation_id

    if conversations:

        for conversation in conversations:

            conversation_id = conversation["conversation_id"]
            title = conversation["title"]

            button_label = (
                f"🟢 {title}"
                if conversation_id == active_id
                else f"💬 {title}"
            )

            if st.button(
                button_label,
                key=f"conversation_{conversation_id}",
                use_container_width=True
            ):
                st.session_state.active_conversation_id = conversation_id
                st.rerun()

    st.divider()

    # ----------------------------------------------
    # DOCUMENT UPLOAD
    # ----------------------------------------------

    st.header("📄 Add Documents")

    uploads = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Index Uploaded PDFs",
        disabled=not uploads,
        use_container_width=True
    ):

        progress = st.empty()

        for file in uploads:

            target = DOCUMENTS_DIR / Path(file.name).name

            target.write_bytes(file.getvalue())

            try:
                result = ingest_pdf(target)

                progress.success(
                    f"Indexed {result['filename']}: "
                    f"{result['pages']} pages, "
                    f"{result['chunks']} chunks"
                )

            except Exception as exc:
                progress.error(
                    f"{file.name}: {type(exc).__name__}: {exc}"
                )

        st.rerun()

    # ----------------------------------------------
    # INDEXED DOCUMENTS
    # ----------------------------------------------

    st.divider()

    st.header("📚 Indexed Documents")

    docs = list_documents()

    if docs:

        for doc in docs:

            st.write(f"📄 **{doc['filename']}**")

            st.caption(
                f"{doc['pages']} pages · {doc['chunks']} chunks"
            )

    else:
        st.info("No documents indexed yet.")


# --------------------------------------------------
# 6. LOAD ACTIVE CONVERSATION
# --------------------------------------------------

active_id = st.session_state.active_conversation_id

history = get_conversation_messages(
    active_id,
    limit=100
)


# --------------------------------------------------
# 7. DISPLAY CHAT HISTORY
# --------------------------------------------------

st.subheader("Research Chat")

if not history:

    st.info(
        "Start by asking a question about your indexed "
        "research papers."
    )

else:

    for message in history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# --------------------------------------------------
# 8. CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your research papers..."
)


# --------------------------------------------------
# 9. PROCESS USER QUESTION
# --------------------------------------------------

if question:

    if not docs:

        st.warning(
            "Please upload and index at least one PDF first."
        )

    else:

        # Display user question immediately
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching papers and preparing your answer..."
            ):

                try:
                    answer = run_async(
                    ask_research_question(question, active_id)
                    )

                    st.markdown(answer)

                except Exception as exc:
                    st.error("Research failed. Full error details:")

                    st.exception(exc)

                    error_details = traceback.format_exc()
                    st.code(error_details, language="python")

                    print("\n========== FULL ERROR TRACEBACK ==========")
                    print(error_details)
                    print("==========================================\n")

                    st.stop()

        # Refresh UI from SQLite
        st.rerun()
