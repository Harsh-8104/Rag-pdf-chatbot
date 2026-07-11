# Ghost — RAG Document Chatbot

*Ask the document, not the model.*

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly
from an uploaded PDF — no hallucinated answers, no outside knowledge leaking
in. Built from scratch to understand the full RAG pipeline: chunking
strategy, embedding generation, vector similarity search, and prompt
grounding.

**Live demo:** [rag-pdf-chatbot-eu6h.onrender.com](https://rag-pdf-chatbot-eu6h.onrender.com)
> Hosted on Render's free tier — the first request after a period of
> inactivity can take 30–60 seconds to wake up. After that, it's fast.

---

## What it does

1. **Upload a PDF** — dropped into the folio, or picked via file dialog.
2. **The document is indexed** — text extracted, split into overlapping
   chunks, embedded, and stored in a Pinecone vector index.
3. **Ask a question** — the question is embedded, the most relevant chunks
   are retrieved by similarity search, and an LLM answers using *only* that
   retrieved context.
4. **If the answer isn't in the document**, the bot says so explicitly
   instead of guessing:
   > *"I couldn't find that information in the uploaded document."*

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Orchestration | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Chat model | Google Gemini (`gemini-2.5-flash`) |
| Vector database | Pinecone (serverless, cosine similarity) |
| PDF parsing | PyPDF |
| Frontend | Hand-built HTML/CSS/JS, served directly by FastAPI |
| Hosting | Render (free tier) |

Entirely free to run — Gemini's free tier and Pinecone's free tier both
require no credit card.

## Features

- Upload any text-based PDF via a REST endpoint or the web UI
- Automatic chunking with configurable size/overlap (`RecursiveCharacterTextSplitter`)
- Semantic search over document chunks (top-3 retrieval)
- Answers strictly grounded in retrieved context — explicitly refuses to
  answer from the model's general knowledge if the document doesn't contain
  the answer
- A distinctive interface: answers render as *marginalia* — annotations
  alongside the conversation — rather than generic chat bubbles
- Fully documented, modular codebase (config / ingestion / retrieval / API
  layers are cleanly separated)

## Architecture

```
                    ┌─────────────────┐
                    │   PDF Upload     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Extract Text     │  PyPDF
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Split into       │  LangChain
                    │  Chunks           │  RecursiveCharacterTextSplitter
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Embed Chunks     │  Gemini Embeddings
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Store Vectors    │  Pinecone
                    └───────────────────┘

                    ┌─────────────────┐
                    │  User Question   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Embed Question   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Retrieve Top-3   │  Pinecone similarity search
                    │  Chunks           │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Build Prompt +   │  Gemini 2.5 Flash
                    │  Generate Answer  │
                    └───────────────────┘
```

## Folder structure

```
rag-chatbot/
├── app.py              # FastAPI app: routes, serves the UI
├── rag.py               # Chat pipeline: embed question -> retrieve -> prompt -> LLM
├── ingest.py             # Upload pipeline: extract -> chunk -> embed -> Pinecone
├── config.py              # Loads and validates all environment variables
├── static/
│   └── index.html          # The chat UI
├── requirements.txt
├── .env.example
├── .python-version
├── .gitignore
├── uploads/               # Uploaded PDFs are saved here (gitignored)
└── README.md
```

---

## Running it locally

### 1. Prerequisites

- Python 3.13
- A free [Google AI Studio API key](https://aistudio.google.com/apikey) (no credit card required)
- A free [Pinecone account](https://app.pinecone.io) and API key

### 2. Set up a virtual environment

```bash
cd rag-chatbot
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

```
GOOGLE_API_KEY=your-gemini-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=rag-chatbot
PINECONE_ENVIRONMENT=us-east-1
```

The Pinecone index is created automatically on first upload — no manual
setup needed in the Pinecone console.

### 5. Run the app

```bash
uvicorn app:app --reload
```

Open **`http://127.0.0.1:8000/`** — that's the chat UI. (`/docs` still works
too, if you want the raw Swagger API view.)

---

## Deploying to Render (free)

1. Push the repo to GitHub.
2. On [Render](https://render.com), create a **New → Web Service**, connect
   the repo.
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Add the same four environment variables from `.env` in Render's
   dashboard.
5. Deploy. Render gives you a live HTTPS URL.

`.python-version` pins the build to Python 3.13 — without it, Render may
default to a newer Python that some dependencies don't support yet.

## API reference

### `POST /upload`

Upload and index a PDF.

```bash
curl -X POST https://rag-pdf-chatbot-eu6h.onrender.com/upload \
  -F "file=@/path/to/document.pdf"
```

```json
{
  "filename": "document.pdf",
  "chunks_uploaded": 7,
  "message": "File uploaded and indexed successfully."
}
```

### `POST /chat`

Ask a question about the uploaded document.

```bash
curl -X POST https://rag-pdf-chatbot-eu6h.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

```json
{
  "answer": "This document is about..."
}
```

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Missing required environment variable` | `.env` missing or incomplete | Copy `.env.example` to `.env` and fill in all values |
| `400: Only PDF files are supported` | Uploaded a non-PDF file | Only `.pdf` files are accepted |
| `400: No extractable text found in this PDF` | Scanned/image-only PDF | Use a PDF with a real text layer (OCR is out of scope) |
| `401` / `PermissionDenied` from Gemini | Invalid or missing `GOOGLE_API_KEY` | Regenerate at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `429 RESOURCE_EXHAUSTED` from Gemini | Hit the free-tier rate limit | Wait a bit and retry |
| `PineconeApiException` on startup | Invalid key or region mismatch | Verify `PINECONE_API_KEY` and that `PINECONE_ENVIRONMENT` matches your plan's supported region |
| First request very slow on the live demo | Render free tier cold start | Normal — subsequent requests are fast |

## Scope note

This is intentionally a **minimal, learning-focused** RAG pipeline. It does
not include auth, a database, streaming responses, conversation memory,
multi-PDF support, hybrid search/reranking, or container orchestration —
left out deliberately to keep the core RAG concepts clear and the codebase
easy to read end-to-end.

---

Built by [Harsh Chauhan](https://github.com/Harsh-8104) — a.k.a. Ghost.
