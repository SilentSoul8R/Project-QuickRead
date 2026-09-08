# 📄 PDF Question/Answer Generator

A RAG (Retrieval-Augmented Generation) app: upload a PDF, ask questions, get
answers grounded in the document — powered by free Groq-hosted LLMs.

## How it works
1. **Extract** — text is pulled from every page of your PDF (`pypdf`).
2. **Chunk** — text is split into overlapping chunks.
3. **Embed & Index** — chunks are embedded with `all-MiniLM-L6-v2`
   (`sentence-transformers`) and indexed with `faiss` for fast similarity search.
4. **Retrieve** — your question is embedded and matched against the most
   relevant chunks.
5. **Generate** — the question + retrieved context is sent to a Groq model
   (Llama 3.3 70B, Llama 3.1 8B, or Gemma2 9B — all free tier) to produce a
   grounded, cited answer.

## Run locally
```bash
pip install -r requirements.txt

# Option A: environment variable
export GROQ_API_KEY="your-groq-api-key-here"

# Option B: local secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your real key

streamlit run app.py
```

Get a free Groq API key at https://console.groq.com/keys.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and this README to a GitHub repo
   (do **not** commit a real `secrets.toml`).
2. Go to https://share.streamlit.io and create a new app pointing at the repo.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. Deploy. The key stays server-side in Streamlit's secrets store and is
   never rendered in the UI, logs, or source code.

## Notes
- If no key is found in secrets or the environment, the app shows a masked
  password field in the sidebar as a session-only fallback — nothing is ever
  echoed back to the screen or stored on disk.
- Scanned/image-only PDFs won't extract text since there's no OCR step;
  use a text-based PDF for best results.
