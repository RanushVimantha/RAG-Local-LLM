"""Main Streamlit application — RAG Local LLM."""

import json
from pathlib import Path

import streamlit as st

from src.config import settings
from src.conversation_manager import ConversationManager
from src.document_processor import DocumentProcessor
from src.document_registry import DocumentRegistry
from src.embedding_manager import EmbeddingManager
from src.llm_manager import LLMManager
from src.rag_engine import RAGEngine
from src.text_chunker import TextChunker
from src.utils import configure_logging, format_file_size
from src.vector_store import VectorStore

configure_logging()

# --- Page Config ---
st.set_page_config(
    page_title="RAG Local LLM",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Initialize Components ---
@st.cache_resource
def get_processor() -> DocumentProcessor:
    return DocumentProcessor()


@st.cache_resource
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@st.cache_resource
def get_vector_store() -> VectorStore:
    settings.ensure_dirs()
    return VectorStore()


@st.cache_resource
def get_registry() -> DocumentRegistry:
    settings.ensure_dirs()
    return DocumentRegistry()


@st.cache_resource
def get_llm() -> LLMManager:
    return LLMManager()


@st.cache_resource
def get_conversation_manager() -> ConversationManager:
    settings.ensure_dirs()
    return ConversationManager()


@st.cache_resource
def get_rag_engine() -> RAGEngine:
    return RAGEngine(
        vector_store=get_vector_store(),
        embedding_manager=get_embedding_manager(),
        llm_manager=get_llm(),
    )


processor = get_processor()
embedder = get_embedding_manager()
store = get_vector_store()
registry = get_registry()
llm = get_llm()
conv_mgr = get_conversation_manager()
rag = get_rag_engine()


# --- Session State Initialization ---
def init_session():
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session()


# --- Sidebar ---
def render_sidebar():
    with st.sidebar:
        st.header("📄 RAG Local LLM")

        # --- Conversation History ---
        st.subheader("💬 Conversations")

        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.current_conv_id = None
            st.session_state.messages = []
            st.rerun()

        conversations = conv_mgr.get_conversations(limit=20)
        for conv in conversations:
            col_title, col_del = st.columns([5, 1])
            is_active = st.session_state.current_conv_id == conv["id"]
            label = f"{'▶ ' if is_active else ''}{conv['title']}"
            with col_title:
                if st.button(
                    label,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    load_conversation(conv["id"])
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"cdel_{conv['id']}"):
                    conv_mgr.delete_conversation(conv["id"])
                    if st.session_state.current_conv_id == conv["id"]:
                        st.session_state.current_conv_id = None
                        st.session_state.messages = []
                    st.rerun()

        st.divider()

        # --- Document List ---
        st.subheader("📁 Documents")
        docs = registry.get_all()
        if docs:
            stats = registry.get_stats()
            c1, c2 = st.columns(2)
            c1.metric("Docs", stats["total_documents"])
            c2.metric("Chunks", stats["total_chunks"])

            for doc in docs:
                status_icon = {"ready": "✅", "processing": "⏳", "error": "❌"}.get(
                    doc["status"], "❓"
                )
                with st.expander(f"{status_icon} {doc['filename']}"):
                    st.caption(f"Type: {doc['file_type'].upper()}")
                    st.caption(f"Size: {format_file_size(doc['file_size_bytes'])}")
                    st.caption(f"Chunks: {doc['chunk_count']} | Pages: {doc['page_count']}")
                    st.caption(f"Uploaded: {doc['created_at']}")
                    if doc["status"] == "error":
                        st.error(doc["error_message"])
                    if st.button("🗑️ Delete", key=f"del_{doc['id']}"):
                        delete_document(doc)
                        st.rerun()
        else:
            st.info("No documents yet.")

        st.divider()

        # --- Chunking Settings ---
        st.subheader("⚙️ Settings")
        chunk_size = st.slider("Chunk Size", 200, 2000, settings.chunk_size, step=50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, settings.chunk_overlap, step=10)
        strategy = st.selectbox("Strategy", list(TextChunker.STRATEGIES.keys()), index=0)

        return chunk_size, chunk_overlap, strategy


def load_conversation(conv_id: str):
    """Load a conversation from the database into session state."""
    st.session_state.current_conv_id = conv_id
    db_messages = conv_mgr.get_messages(conv_id)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "sources": m.get("sources")}
        for m in db_messages
    ]


def delete_document(doc: dict):
    """Delete a document from registry and vector store."""
    store.delete_by_source(doc["filename"])
    registry.delete(doc["id"])
    file_path = Path(doc["file_path"])
    if file_path.exists():
        file_path.unlink()


# --- Main Tabs ---
def render_main(chunk_size: int, chunk_overlap: int, strategy: str):
    tab_chat, tab_upload, tab_docs, tab_settings = st.tabs(
        ["💬 Ask Questions", "📤 Upload Documents", "📊 Document Dashboard", "⚙️ Settings"]
    )

    with tab_chat:
        render_chat_tab()

    with tab_upload:
        render_upload_tab(chunk_size, chunk_overlap, strategy)

    with tab_docs:
        render_document_dashboard()

    with tab_settings:
        render_settings_tab(chunk_size, chunk_overlap, strategy)


# --- Chat Tab ---
def render_chat_tab():
    health = llm.health_check()
    if not health["ollama_running"]:
        st.error(
            "⚠️ Ollama is not running. Start it with `ollama serve` "
            "and pull a model: `ollama pull mistral`"
        )
        return

    if store.get_stats()["total_chunks"] == 0:
        st.info("Upload some documents first, then come back to ask questions!")
        return

    # Source filter
    sources = store.get_all_sources()
    source_options = ["All Documents"] + sources
    col_filter, col_export = st.columns([3, 1])
    with col_filter:
        selected_source = st.selectbox("Search in:", source_options, key="source_filter")
    with col_export:
        if st.session_state.current_conv_id and st.session_state.messages:
            export_format = st.selectbox("Export", ["—", "Markdown", "JSON"], key="export_fmt")
            if export_format == "Markdown":
                md = conv_mgr.export_markdown(st.session_state.current_conv_id)
                st.download_button("📥 Download", md, "conversation.md", "text/markdown")
            elif export_format == "JSON":
                data = conv_mgr.export_json(st.session_state.current_conv_id)
                st.download_button(
                    "📥 Download",
                    json.dumps(data, indent=2),
                    "conversation.json",
                    "application/json",
                )

    source_filter = None if selected_source == "All Documents" else selected_source

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_citations(msg["sources"])

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Create conversation if needed
        if st.session_state.current_conv_id is None:
            conv_id = conv_mgr.create_conversation(model_name=llm.model)
            st.session_state.current_conv_id = conv_id

        conv_id = st.session_state.current_conv_id

        # Show user message
        st.session_state.messages.append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user"):
            st.markdown(question)

        # Persist user message
        conv_mgr.add_message(conv_id, "user", question)

        # Build chat history for context
        chat_history = conv_mgr.get_chat_history_text(conv_id, window=settings.conversation_window)

        # Generate response
        with st.chat_message("assistant"):
            try:
                token_stream, search_results = rag.query_stream(
                    question=question,
                    chat_history=chat_history,
                    source_filter=source_filter,
                )
                response = st.write_stream(token_stream)
                render_citations(search_results)

                # Persist assistant message
                conv_mgr.add_message(
                    conv_id, "assistant", response, sources=search_results
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": search_results,
                })

                # Auto-title after first exchange
                if len(st.session_state.messages) == 2:
                    conv_mgr.auto_title(conv_id)

            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Clear conversation
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation"):
            if st.session_state.current_conv_id:
                conv_mgr.delete_conversation(st.session_state.current_conv_id)
            st.session_state.current_conv_id = None
            st.session_state.messages = []
            st.rerun()


def render_citations(sources: list[dict]):
    if not sources:
        return
    st.caption("📚 **Sources:**")
    for src in sources:
        score_label = f"Score: {src['score']:.2f}"
        if "rerank_score" in src:
            score_label += f" | Re-rank: {src['rerank_score']:.2f}"
        with st.expander(f"📄 {src['source']} — Page {src['page']} ({score_label})"):
            st.text(src["text"][:500])


# --- Upload Tab ---
def render_upload_tab(chunk_size: int, chunk_overlap: int, strategy: str):
    st.subheader("📤 Upload Documents")
    st.caption(f"Supported: PDF, TXT, Markdown, DOCX. Max {settings.max_file_size_mb}MB per file.")

    uploaded_files = st.file_uploader(
        "Drop your documents here",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size > settings.max_file_size_mb * 1024 * 1024:
                st.error(f"❌ {uploaded_file.name} exceeds {settings.max_file_size_mb}MB limit.")
                continue

            existing = registry.get_by_filename(uploaded_file.name)
            if existing and existing["status"] == "ready":
                st.warning(f"⚠️ {uploaded_file.name} already uploaded. Delete it first to re-upload.")
                continue

            if st.button(f"Process {uploaded_file.name}", key=f"proc_{uploaded_file.name}"):
                process_document(uploaded_file, chunk_size, chunk_overlap, strategy)


def process_document(uploaded_file, chunk_size: int, chunk_overlap: int, strategy: str):
    settings.ensure_dirs()

    save_path = settings.uploads_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc_id = registry.register(
        filename=uploaded_file.name,
        file_path=str(save_path),
        file_size_bytes=uploaded_file.size,
        file_type=processor.get_file_type(uploaded_file.name),
    )

    progress = st.progress(0, text="Starting...")

    try:
        progress.progress(10, text="📖 Extracting text...")
        pages = processor.extract_text(save_path)

        progress.progress(30, text="✂️ Chunking text...")
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
        chunks = chunker.chunk_pages(pages, uploaded_file.name)

        progress.progress(50, text="🧠 Generating embeddings (local)...")
        embeddings = embedder.embed_texts([c.text for c in chunks])

        progress.progress(80, text="💾 Storing in vector database...")
        store.add_chunks(chunks, embeddings)

        total_chars = sum(len(p["text"]) for p in pages)
        registry.update_status(
            doc_id=doc_id,
            status="ready",
            page_count=len(pages),
            chunk_count=len(chunks),
            character_count=total_chars,
            chunk_size_used=chunk_size,
            chunk_overlap_used=chunk_overlap,
            embedding_model=embedder.model_name,
        )

        progress.progress(100, text="✅ Complete!")
        st.success(f"✅ **{uploaded_file.name}** processed successfully!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", len(pages))
        col2.metric("Chunks", len(chunks))
        col3.metric("Characters", f"{total_chars:,}")

    except Exception as e:
        registry.update_status(doc_id=doc_id, status="error", error_message=str(e))
        progress.progress(100, text="❌ Error")
        st.error(f"Failed to process {uploaded_file.name}: {e}")


# --- Document Dashboard Tab ---
def render_document_dashboard():
    st.subheader("📊 Document Dashboard")

    docs = registry.get_all()
    if not docs:
        st.info("No documents uploaded yet.")
        return

    # Summary stats
    stats = registry.get_stats()
    store_stats = store.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Documents", stats["total_documents"])
    c2.metric("Total Chunks", stats["total_chunks"])
    c3.metric("Total Characters", f"{stats['total_characters']:,}")
    c4.metric("Total Size", format_file_size(stats["total_size_bytes"]))

    st.divider()

    # Document table
    for doc in docs:
        status_icon = {"ready": "✅", "processing": "⏳", "error": "❌"}.get(doc["status"], "❓")
        with st.expander(f"{status_icon} **{doc['filename']}**", expanded=False):
            col_info, col_meta = st.columns(2)

            with col_info:
                st.markdown("**File Info**")
                st.write(f"- **Type:** {doc['file_type'].upper()}")
                st.write(f"- **Size:** {format_file_size(doc['file_size_bytes'])}")
                st.write(f"- **Pages:** {doc['page_count']}")
                st.write(f"- **Characters:** {doc['character_count']:,}")

            with col_meta:
                st.markdown("**Processing Info**")
                st.write(f"- **Chunks:** {doc['chunk_count']}")
                st.write(f"- **Chunk Size:** {doc['chunk_size_used']}")
                st.write(f"- **Overlap:** {doc['chunk_overlap_used']}")
                st.write(f"- **Embedding Model:** {doc['embedding_model']}")
                st.write(f"- **Uploaded:** {doc['created_at']}")

            if doc["status"] == "error":
                st.error(f"Error: {doc['error_message']}")

            col_search, col_delete = st.columns([3, 1])
            with col_search:
                search_q = st.text_input(
                    "Search within this document",
                    key=f"search_{doc['id']}",
                    placeholder="Ask a question about this document only...",
                )
                if search_q:
                    with st.spinner("Searching..."):
                        query_emb = embedder.embed_query(search_q)
                        results = store.search(query_emb, top_k=3, source_filter=doc["filename"])
                        if results:
                            for r in results:
                                st.info(f"**Page {r['page']}** (Score: {r['score']:.2f})\n\n{r['text'][:300]}")
                        else:
                            st.warning("No relevant results found.")
            with col_delete:
                st.write("")  # spacing
                if st.button("🗑️ Delete Document", key=f"dash_del_{doc['id']}"):
                    delete_document(doc)
                    st.rerun()


# --- Settings Tab ---
def render_settings_tab(chunk_size: int, chunk_overlap: int, strategy: str):
    st.subheader("⚙️ Settings & System Health")

    col_left, col_right = st.columns(2)

    with col_left:
        # --- System Health ---
        st.markdown("### System Health")
        health = llm.health_check()

        if health["ollama_running"]:
            st.success("Ollama: Running")
            if health["model_available"]:
                st.success(f"Model `{health['configured_model']}`: Available")
            else:
                st.warning(
                    f"Model `{health['configured_model']}` not found. "
                    f"Run: `ollama pull {health['configured_model']}`"
                )
            if health["available_models"]:
                st.caption(f"Available models: {', '.join(health['available_models'])}")
        else:
            st.error("Ollama: Not running. Start it with `ollama serve`")

        store_stats = store.get_stats()
        st.info(f"Vector store: {store_stats['total_chunks']} chunks across {store_stats['total_documents']} documents")

        st.markdown("### Embedding Model")
        st.write(f"- **Current:** {embedder.model_name}")
        st.write(f"- **Dimensions:** {settings.embedding_dimensions}")

        # --- LLM Settings ---
        st.markdown("### LLM Model Selection")
        available_models = llm.list_models()
        if available_models:
            current_idx = 0
            for i, m in enumerate(available_models):
                if settings.llm_model in m:
                    current_idx = i
                    break
            selected_model = st.selectbox(
                "LLM Model", available_models, index=current_idx, key="llm_model_select"
            )
            st.caption(f"Selected: {selected_model}")
        else:
            st.warning("No Ollama models available. Install one: `ollama pull mistral`")

    with col_right:
        # --- Retrieval Settings ---
        st.markdown("### Retrieval Settings")
        top_k = st.slider("Top-K Results", 1, 20, settings.top_k, key="setting_top_k")
        similarity_threshold = st.slider(
            "Similarity Threshold", 0.0, 1.0, settings.similarity_threshold,
            step=0.05, key="setting_sim_thresh"
        )
        st.caption(
            f"Retrieve top {top_k} chunks with similarity >= {similarity_threshold:.2f}"
        )

        st.markdown("### LLM Generation Settings")
        temperature = st.slider(
            "Temperature", 0.0, 2.0, settings.llm_temperature,
            step=0.1, key="setting_temp",
            help="Lower = more factual (better for RAG). Higher = more creative."
        )
        max_tokens = st.slider(
            "Max Tokens", 256, 4096, settings.llm_max_tokens,
            step=128, key="setting_max_tokens"
        )

        st.markdown("### Chunking Preview")
        st.caption(f"Current: {chunk_size} chars, {chunk_overlap} overlap, {strategy}")

        preview_text = st.text_area(
            "Paste text to preview chunking:",
            height=100,
            key="chunk_preview_input",
            placeholder="Paste some text here to see how it would be chunked...",
        )
        if preview_text:
            chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
            preview = chunker.preview_chunks(
                [{"page": 1, "text": preview_text}], source_file="preview", max_preview=10
            )
            for p in preview:
                st.code(f"Chunk {p['index']} (Page {p['page']}, {p['length']} chars):\n{p['preview']}")

    st.divider()

    # --- Danger Zone ---
    st.markdown("### Danger Zone")
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🗑️ Reset Vector Database", type="secondary"):
            store.reset()
            st.warning("Vector database has been reset.")
            st.rerun()
    with col_reset2:
        if st.button("🗑️ Clear All Conversations", type="secondary"):
            for conv in conv_mgr.get_conversations():
                conv_mgr.delete_conversation(conv["id"])
            st.session_state.current_conv_id = None
            st.session_state.messages = []
            st.warning("All conversations cleared.")
            st.rerun()


# --- Run ---
chunk_size, chunk_overlap, strategy = render_sidebar()
render_main(chunk_size, chunk_overlap, strategy)
