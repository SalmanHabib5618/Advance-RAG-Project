# Advanced RAG — Hybrid Search + Re-ranking + Query Expansion

Builds on the Basic RAG project with retrieval techniques that noticeably
improve answer quality and relevance:

- **Hybrid search** — BM25 (keyword) + FAISS (semantic), combined via an
  EnsembleRetriever, so exact terms and conceptual matches both surface.
- **Query expansion** — a MultiQueryRetriever asks the LLM to rewrite the
  question into several phrasings before retrieving, catching relevant
  chunks that a single phrasing would miss.
- **Cross-encoder re-ranking** — every retrieved chunk is re-scored against
  the question with a cross-encoder and only the top matches reach the LLM.
- **Conversational memory** — follow-up questions are understood in context
  via ConversationalRetrievalChain's question condensing.

Each technique can be toggled on/off in the sidebar to demonstrate its
individual effect — useful for a portfolio walkthrough.

## Stack
- Streamlit (UI)
- LangChain 0.3.x (orchestration)
- FAISS + BM25 (hybrid retrieval)
- Local HuggingFace embeddings (`all-MiniLM-L6-v2`) and cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — both free, no API key
- Groq (`openai/gpt-oss-20b`) for generation — key kept server-side only,
  never shown or requested in the UI

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your real GROQ_API_KEY
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to GitHub — keep `requirements.txt` and `runtime.txt`
   **lowercase** exactly, Streamlit Cloud only auto-detects those names.
2. Create the app, main file path `advanced-rag/app.py` (or wherever you
   place it in your repo).
3. In **Advanced settings**, set the Python version dropdown to **3.11**
   explicitly — don't rely on `runtime.txt` alone.
4. Add your key under **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "your-real-key"
   ```
5. Deploy, upload a document, build the knowledge base, and chat.
