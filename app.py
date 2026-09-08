"""
PDF Question/Answer Generator
A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF
and ask questions about it, answered by a Groq-hosted LLM.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo (app.py + requirements.txt) to GitHub.
    2. On https://share.streamlit.io create a new app pointing at this repo.
    3. In "Advanced settings -> Secrets", add:
           GROQ_API_KEY = "your-groq-api-key-here"
       This keeps the key out of the code and out of the UI entirely.
"""

import os
import time
import textwrap

import numpy as np
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss


# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Question/Answer Generator",
    page_icon="Q&A",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# STYLING — animated, gradient, glassy UI
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Fade-in-up animation for content blocks */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fadeIn { animation: fadeInUp 0.6s ease-out both; }

/* Glass card */
.glassCard {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    margin-bottom: 1rem;
}

/* Animated gradient title */
.heroTitle {
    font-size: 2.6rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #43cea2, #185a9d, #43cea2);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
    margin-bottom: 0.2rem;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.heroSubtitle {
    text-align: center;
    color: #cfd8dc;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 1.6rem;
}

/* Chat bubbles */
.chatBubbleUser {
    background: linear-gradient(135deg, #185a9d, #43cea2);
    color: white;
    padding: 0.85rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0;
    max-width: 85%;
    margin-left: auto;
    animation: fadeInUp 0.4s ease-out both;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.chatBubbleAi {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: #f1f1f1;
    padding: 0.85rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 0;
    max-width: 85%;
    margin-right: auto;
    animation: fadeInUp 0.4s ease-out both;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

/* Source chip */
.sourceChip {
    display: inline-block;
    background: rgba(67, 206, 162, 0.15);
    border: 1px solid rgba(67, 206, 162, 0.4);
    color: #7fffd4;
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    margin: 0.2rem 0.3rem 0 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.85);
    backdrop-filter: blur(10px);
}

/* Status pill */
.statusPill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
}
.statusReady { background: rgba(67, 206, 162, 0.18); color: #43cea2; border: 1px solid #43cea2; }
.statusWait  { background: rgba(255, 193, 7, 0.15); color: #ffc107; border: 1px solid #ffc107; }

.pulseDot {
    width: 9px; height: 9px; border-radius: 50%;
    background: currentColor;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
    70%  { box-shadow: 0 0 0 8px transparent; opacity: 0.6; }
    100% { box-shadow: 0 0 0 0 transparent; opacity: 1; }
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #43cea2, #185a9d);
    color: white;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 6px 18px rgba(67, 206, 162, 0.35);
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SECRETS / API KEY HANDLING (kept hidden from the UI at all times)
# --------------------------------------------------------------------------
def get_groq_api_key() -> str:
    """
    Resolve the Groq API key without ever displaying it.
    Priority:
      1. Streamlit secrets (st.secrets["GROQ_API_KEY"])           <- recommended for deployment
      2. Environment variable GROQ_API_KEY                        <- recommended for local dev
      3. A masked password field in the sidebar (session-only)    <- fallback for quick testing
    """
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("GROQ_API_KEY", "")

    if not key:
        key = st.session_state.get("_manual_groq_key", "")

    return key


def render_api_key_sidebar_fallback():
    """Only shown if no key was found in secrets/env. Input is masked and never echoed back."""
    if get_groq_api_key():
        return
    with st.sidebar:
        st.markdown("##### Groq API Key")
        entered = st.text_input(
            "Not found in secrets/env — enter it here for this session",
            type="password",
            key="_manual_groq_key_input",
            placeholder="gsk_************************",
            help="This is stored only in memory for your current session and is never displayed or logged.",
        )
        if entered:
            st.session_state["_manual_groq_key"] = entered
            st.rerun()


# --------------------------------------------------------------------------
# CACHED RESOURCES
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client():
    api_key = get_groq_api_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------------------------------
def extract_text_from_pdf(file) -> list:
    """Returns a list of (page_number, text) tuples."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_pages(pages, chunk_size=900, overlap=150):
    """
    Splits page text into overlapping word-based chunks.
    Each chunk keeps track of which page(s) it came from.
    """
    chunks = []
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append({"page": page_num, "text": chunk_text})
            if end >= len(words):
                break
            start = end - overlap
    return chunks


def build_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query, embedder, index, chunks, top_k=4):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


# --------------------------------------------------------------------------
# LLM ANSWER GENERATION
# --------------------------------------------------------------------------
GROQ_MODEL_OPTIONS = {
    "Llama 3.3 70B Versatile (best quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B Instant (fastest)": "llama-3.1-8b-instant",
    "Gemma2 9B IT": "gemma2-9b-it",
}


def ask_groq(client, model, question, context_chunks, history):
    context_text = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = textwrap.dedent(
        """
        You are a precise, helpful assistant that answers questions strictly using
        the provided PDF excerpts. Rules:
        - Only use information found in the provided context.
        - If the answer is not contained in the context, say so honestly instead of guessing.
        - Cite the page number(s) you used, like (Page 3), where relevant.
        - Be concise and well-structured.
        """
    ).strip()

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = f"Context from the PDF:\n\n{context_text}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    return stream


# --------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------
defaults = {
    "chat_history": [],
    "chunks": None,
    "index": None,
    "pdf_name": None,
    "pdf_pages": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
render_api_key_sidebar_fallback()

with st.sidebar:
    st.markdown("## Settings")

    model_label = st.selectbox("Model (Groq — free tier)", list(GROQ_MODEL_OPTIONS.keys()))
    selected_model = GROQ_MODEL_OPTIONS[model_label]

    top_k = st.slider("Chunks retrieved per question", 2, 8, 4)

    st.markdown("---")
    st.markdown("## Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    process_clicked = st.button("Process PDF", use_container_width=True)

    st.markdown("---")
    if st.session_state.pdf_name:
        st.markdown(
            f"<div class='statusPill statusReady'><span class='pulseDot'></span>"
            f"{st.session_state.pdf_name} · {st.session_state.pdf_pages} pages indexed</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='statusPill statusWait'><span class='pulseDot'></span>No PDF processed yet</div>",
            unsafe_allow_html=True,
        )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown('<div class="heroTitle">PDF Question/Answer Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="heroSubtitle">Upload a PDF, then ask it anything — powered by Retrieval-Augmented '
    'Generation and Groq\'s lightning-fast LLMs.</div>',
    unsafe_allow_html=True,
)

api_key_present = bool(get_groq_api_key())
if not api_key_present:
    st.warning(
        "No Groq API key found. Add one in Streamlit secrets as `GROQ_API_KEY`, "
        "set it as an environment variable, or enter it in the sidebar to continue.",
    )

# --------------------------------------------------------------------------
# PROCESS PDF
# --------------------------------------------------------------------------
if process_clicked:
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Reading and indexing your PDF... this only takes a moment"):
            embedder = load_embedder()
            pages = extract_text_from_pdf(uploaded_file)
            if not pages:
                st.error("Couldn't extract any text from this PDF. It may be a scanned/image-only PDF.")
            else:
                chunks = chunk_pages(pages)
                index, _ = build_index(chunks, embedder)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_pages = len(pages)
                st.session_state.chat_history = []
        st.success(f"Indexed '{uploaded_file.name}' ({len(pages)} pages, {len(chunks)} chunks). Ask away!")
        st.balloons()


# --------------------------------------------------------------------------
# MAIN CHAT AREA
# --------------------------------------------------------------------------
st.markdown('<div class="glassCard fadeIn">', unsafe_allow_html=True)

if st.session_state.index is None:
    st.info("Upload a PDF and click **Process PDF** in the sidebar to get started.")
else:
    # render history
    for turn in st.session_state.chat_history:
        css_class = "chatBubbleUser" if turn["role"] == "user" else "chatBubbleAi"
        st.markdown(f'<div class="{css_class}">{turn["content"]}</div>', unsafe_allow_html=True)
        if turn.get("sources"):
            chips = "".join(
                f'<span class="sourceChip">Page {s}</span>' for s in turn["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="chatBubbleUser">{question}</div>', unsafe_allow_html=True)

        client = get_groq_client()
        if client is None:
            st.error("Groq API key missing — cannot generate an answer.")
        else:
            embedder = load_embedder()
            results = retrieve(question, embedder, st.session_state.index, st.session_state.chunks, top_k=top_k)
            pages_used = sorted({r["page"] for r in results})

            placeholder = st.empty()
            answer_text = ""
            try:
                stream = ask_groq(client, selected_model, question, results, st.session_state.chat_history)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    answer_text += delta
                    placeholder.markdown(
                        f'<div class="chatBubbleAi">{answer_text}▌</div>', unsafe_allow_html=True
                    )
                placeholder.markdown(f'<div class="chatBubbleAi">{answer_text}</div>', unsafe_allow_html=True)
            except Exception as e:
                answer_text = f"Error calling Groq: {e}"
                placeholder.markdown(f'<div class="chatBubbleAi">{answer_text}</div>', unsafe_allow_html=True)

            chips = "".join(f'<span class="sourceChip">Page {p}</span>' for p in pages_used)
            st.markdown(chips, unsafe_allow_html=True)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_text, "sources": pages_used}
            )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#8892a0; font-size:0.8rem; margin-top:2rem;'>"
    "Built with Streamlit · Groq · Sentence-Transformers · FAISS</p>",
    unsafe_allow_html=True,
)
