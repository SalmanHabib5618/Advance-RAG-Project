"""
Advanced RAG (Retrieval-Augmented Generation) App
----------------------------------------------------
Builds on the Basic RAG project with techniques that make retrieval more
accurate and the chat more natural:

1. Hybrid Search   — combines keyword search (BM25) with semantic search
                      (FAISS) via an EnsembleRetriever, so both exact terms
                      and conceptual matches are found.
2. Query Expansion — an LLM rewrites the user's question into several
                      phrasings (MultiQueryRetriever) before retrieving, so
                      relevant chunks aren't missed due to wording alone.
3. Re-ranking      — a cross-encoder model re-scores every retrieved chunk
                      against the question and keeps only the most relevant
                      ones before they reach the LLM.
4. Conversational memory — follow-up questions ("what about the second
                      one?") are understood using chat history via
                      ConversationalRetrievalChain's question condensing.

Each stage can be toggled in the sidebar so the effect of each technique is
visible on its own.

LLM: Groq (fixed model, key read only from Streamlit secrets — never shown
     or requested in the UI)
Embeddings: local HuggingFace sentence-transformers (free, no extra API key)
Re-ranker: local HuggingFace cross-encoder (free, no extra API key)

Stack: Streamlit + LangChain (0.3.x) + FAISS + BM25 + Groq
"""

import ast
import io
import logging
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.chains import ConversationalRetrievalChain


# ----------------------------- Page setup -----------------------------
st.set_page_config(page_title="Advanced RAG App", page_icon="🧠", layout="wide")
st.title("🧠 Advanced RAG — Hybrid Search + Re-ranking + Query Expansion")
st.caption(
    "Upload documents from different sources, then chat. Retrieval uses "
    "hybrid search, query expansion, and cross-encoder re-ranking for "
    "more accurate, grounded answers."
)

# ----------------------------- Session state -----------------------------
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("qa_chain", None)
st.session_state.setdefault("chat_history", [])  # list of dicts: q, a, sources, expanded_queries

# ----------------------------- API key (hidden — not shown in UI) -----------------------------
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Fixed model — change here, not exposed in the UI.
MODEL_NAME = "openai/gpt-oss-20b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("📥 Add Sources")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )
    web_url = st.text_input("Or add a web page URL (optional)")

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)

    st.divider()
    st.header("🧪 Retrieval Techniques")
    use_hybrid = st.checkbox("Hybrid search (BM25 + FAISS)", value=True)
    use_multi_query = st.checkbox("Query expansion (multi-query)", value=True)
    use_reranking = st.checkbox("Cross-encoder re-ranking", value=True)
    top_k = st.slider("Chunks retrieved per query", 4, 20, 10)
    top_n_final = st.slider("Chunks kept after re-ranking", 2, 8, 4)

    build_clicked = st.button("🔨 Build Knowledge Base", use_container_width=True)
    clear_clicked = st.button("🗑️ Clear Conversation", use_container_width=True)


if clear_clicked:
    st.session_state.chat_history = []


# ----------------------------- Helpers -----------------------------
def load_document(uploaded_file):
    """Save an uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".csv":
            loader = CSVLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_cross_encoder():
    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)


def build_retriever(chunks, llm):
    """Assemble the retrieval pipeline according to the sidebar toggles."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    if use_hybrid:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6],
        )
    else:
        retriever = faiss_retriever

    if use_multi_query:
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    if use_reranking:
        reranker = CrossEncoderReranker(model=get_cross_encoder(), top_n=top_n_final)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=retriever
        )

    return retriever


def ask_with_query_log(chain, inputs):
    """Invoke the chain while capturing MultiQueryRetriever's generated
    query variations from its logger, for display in the UI. Falls back
    to an empty list if the log can't be parsed (never breaks the app)."""
    logger = logging.getLogger("langchain.retrievers.multi_query")
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        result = chain.invoke(inputs)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    expanded_queries = []
    try:
        for line in log_stream.getvalue().splitlines():
            if "Generated queries:" in line:
                part = line.split("Generated queries:", 1)[1].strip()
                expanded_queries = ast.literal_eval(part)
    except Exception:
        expanded_queries = []
    return result, expanded_queries


# ----------------------------- Build knowledge base -----------------------------
if build_clicked:
    if not GROQ_API_KEY:
        st.sidebar.error(
            "No Groq API key is configured for this app. Set GROQ_API_KEY "
            "in Streamlit secrets (Settings → Secrets) and reboot the app."
        )
    elif not uploaded_files and not web_url:
        st.sidebar.error("Upload at least one file or provide a URL.")
    else:
        with st.spinner("Reading sources and building the retrieval pipeline..."):
            all_docs = []

            for f in uploaded_files or []:
                try:
                    all_docs.extend(load_document(f))
                except Exception as e:
                    st.sidebar.error(f"Failed to load {f.name}: {e}")

            if web_url:
                try:
                    web_docs = WebBaseLoader(web_url).load()
                    for d in web_docs:
                        d.metadata["source"] = web_url
                    all_docs.extend(web_docs)
                except Exception as e:
                    st.sidebar.error(f"Failed to load URL: {e}")

            if all_docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                chunks = splitter.split_documents(all_docs)

                llm = ChatGroq(model=MODEL_NAME, temperature=0)
                retriever = build_retriever(chunks, llm)
                st.session_state.retriever = retriever

                st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                )
                st.session_state.chat_history = []

                st.sidebar.success(
                    f"Knowledge base built from {len(all_docs)} document(s), "
                    f"{len(chunks)} chunks."
                )
            else:
                st.sidebar.error("No content could be loaded from the given sources.")


# ----------------------------- Main: chat interface -----------------------------
st.subheader("💬 Chat")

if st.session_state.qa_chain is None:
    st.info("Upload documents and click **Build Knowledge Base** in the sidebar to get started.")
else:
    question = st.text_input("Your question", placeholder="Ask a follow-up — I remember context.")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        with st.spinner("Retrieving and thinking..."):
            try:
                history_pairs = [(h["q"], h["a"]) for h in st.session_state.chat_history]
                result, expanded_queries = ask_with_query_log(
                    st.session_state.qa_chain,
                    {"question": question, "chat_history": history_pairs},
                )
                answer = result["answer"]
                sources = sorted({
                    doc.metadata.get("source", "unknown")
                    for doc in result.get("source_documents", [])
                })
                st.session_state.chat_history.insert(0, {
                    "q": question,
                    "a": answer,
                    "sources": sources,
                    "expanded_queries": expanded_queries,
                })
            except Exception as e:
                st.error(f"Something went wrong while answering: {e}")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["a"])
            if turn["sources"]:
                st.caption("Sources: " + ", ".join(turn["sources"]))
            if turn["expanded_queries"]:
                with st.expander("🔍 Query variations used for retrieval"):
                    for q in turn["expanded_queries"]:
                        st.write(f"- {q}")
