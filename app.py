import streamlit as st
import tempfile
import os
from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, get_retriever
from src.rag_chain import create_rag_chain

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0d0d0d; }

    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #1f1f1f;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * { color: #c9c9c9 !important; }

    /* Sidebar inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #e0e0e0 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #5b5fc7 !important;
        box-shadow: none !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background-color: #1a1a1a !important;
        border: 1px dashed #2a2a2a !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #5b5fc7 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 0.45rem 1rem !important;
        width: 100%;
    }
    .stButton > button:hover { background-color: #4a4eb5 !important; }

    .stButton > button:disabled {
        background-color: #2a2a2a !important;
        color: #555 !important;
    }

    hr { border-color: #1f1f1f !important; margin: 0.75rem 0 !important; }

    /* Chat messages */
    .msg-wrap-user { display: flex; justify-content: flex-end; margin: 10px 0; }
    .msg-wrap-ai   { display: flex; justify-content: flex-start; margin: 10px 0; }

    .msg-user {
        background: #1e1e2e;
        border: 1px solid #2e2e4e;
        border-radius: 14px 14px 3px 14px;
        padding: 10px 14px;
        color: #dcdcf0;
        max-width: 72%;
        font-size: 14px;
        line-height: 1.55;
    }
    .msg-ai {
        background: #131313;
        border: 1px solid #222;
        border-radius: 14px 14px 14px 3px;
        padding: 10px 14px;
        color: #c8c8c8;
        max-width: 72%;
        font-size: 14px;
        line-height: 1.55;
    }

    .tag {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #444 !important;
        margin-bottom: 3px;
    }
    .tag-right { text-align: right; }

    /* Empty state */
    .empty-state {
        text-align: center;
        color: #333;
        padding: 5rem 0;
        font-size: 14px;
    }

    /* Status dot */
    .dot-green { color: #4ade80; font-size: 11px; }
    .dot-yellow { color: #facc15; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [("messages", []), ("chain", None), ("pipeline_ready", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DocMind")
    st.markdown("<p style='font-size:12px;color:#444;margin-top:-6px;'>Multi-model PDF assistant</p>", unsafe_allow_html=True)
    st.markdown("---")

    provider = st.selectbox(
        "Model provider",
        options=["groq", "gemini"],
        format_func=lambda x: "Groq" if x == "groq" else "Gemini",
    )

    api_key = st.text_input(
        "API key",
        type="password",
        placeholder=f"{provider.capitalize()} API key…",
    )

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f"<p style='font-size:11px;color:#555;margin:2px 0'>📄 {f.name}</p>", unsafe_allow_html=True)

    st.markdown("---")

    build_btn = st.button("Build pipeline", use_container_width=True)

    if build_btn:
        if not api_key.strip():
            st.error("API key required.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uf in uploaded_files:
                    with open(os.path.join(tmp_dir, uf.name), "wb") as f:
                        f.write(uf.read())

                with st.spinner("Loading…"):
                    docs = load_documents(tmp_dir)
                with st.spinner("Splitting…"):
                    chunks = split_documents(docs)
                with st.spinner("Embedding…"):
                    embeddings = get_embeddings()
                with st.spinner("Indexing…"):
                    vs = create_vector_store(chunks=chunks, embeddings=embeddings)
                    retriever = get_retriever(vs)
                with st.spinner("Chaining…"):
                    st.session_state.chain = create_rag_chain(
                        retriever=retriever,
                        provider=provider,
                        api_key=api_key,
                    )

            st.session_state.pipeline_ready = True
            st.session_state.messages = []
            st.success(f"{len(chunks)} chunks ready.")

    st.markdown("---")

    if st.session_state.pipeline_ready:
        st.markdown('<span class="dot-green">● Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot-yellow">● Not built</span>', unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## DocMind")
st.markdown("<p style='color:#444;margin-top:-12px;font-size:13px;'>Ask anything about your uploaded documents.</p>", unsafe_allow_html=True)
st.markdown("---")

# Chat
with st.container():
    if not st.session_state.messages:
        st.markdown('<div class="empty-state">Upload PDFs → Build pipeline → Start asking</div>', unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="tag tag-right">You</div>'
                    f'<div class="msg-wrap-user"><div class="msg-user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="tag">DocMind</div>'
                    f'<div class="msg-wrap-ai"><div class="msg-ai">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )

st.markdown("---")

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "q",
        label_visibility="collapsed",
        placeholder="Ask a question…",
        disabled=not st.session_state.pipeline_ready,
        key="question_input",
    )
with col2:
    send = st.button("Send", disabled=not st.session_state.pipeline_ready)

if send and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking…"):
        answer = st.session_state.chain.invoke(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()