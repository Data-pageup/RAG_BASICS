# 🧠 DocMind

**Multi-model PDF assistant — ask anything across your documents.**

Live at → [multi-docs.streamlit.app](https://multi-docs.streamlit.app/)

---

## What it does

Upload one or more PDFs, pick your LLM provider, and chat with your documents. DocMind chunks and embeds your files locally, builds a vector store, and routes your questions through a RAG chain backed by either Groq or Gemini.

---

## Stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| Embeddings | HuggingFace (`langchain-huggingface`) |
| Vector store | Chroma (`langchain-chroma`) |
| LLM providers | Groq · Gemini |
| PDF loading | PyPDF (`pypdf`) |
| Orchestration | LangChain |

---

## Getting started

### 1. Clone

```bash
git clone https://github.com/your-username/docmind.git
cd docmind
```

### 2. Install dependencies

Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### 3. Run

```bash
uv run python -m streamlit run app.py
```

### 4. Use

1. Choose a provider — **Groq** or **Gemini**
2. Paste your API key
3. Upload your PDFs
4. Click **Build pipeline**
5. Start asking questions

---

## Project structure

```
docmind/
├── app.py                  # Streamlit UI
├── src/
│   ├── loader.py           # PDF loading
│   ├── splitter.py         # Text chunking
│   ├── embeddings.py       # HuggingFace embeddings
│   ├── vector_store.py     # Chroma vector store + retriever
│   └── rag_chain.py        # RAG chain (Groq / Gemini)
├── pyproject.toml
└── README.md
```

---

## API keys

- **Groq** → [console.groq.com](https://console.groq.com)
- **Gemini** → [aistudio.google.com](https://aistudio.google.com)

Keys are never stored — entered per session in the UI.

---

Built by **Amirtha Ganesh R.**
