# RAG Chatbot

A minimal, production-quality Retrieval-Augmented Generation (RAG) chatbot.
Upload a PDF, ask questions about it, and get answers grounded **only** in
that document's content — no hallucinated answers outside the source.

Built from scratch to understand the full RAG pipeline: chunking strategy,
embedding generation, vector similarity search, and prompt grounding —
not a wrapper around a pre-built framework template.

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Orchestration | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Chat model | Google Gemini (`gemini-2.5-flash`) |
| Vector database | Pinecone (serverless, cosine similarity) |
| PDF parsing | PyPDF |

## Features

- Upload any text-based PDF via a REST endpoint
- Automatic chunking with configurable size/overlap (`RecursiveCharacterTextSplitter`)
- Semantic search over document chunks (top-3 retrieval)
- Answers strictly grounded in retrieved context — explicitly refuses to
  answer from the model's general knowledge if the document doesn't contain
  the answer
- Fully documented, modular codebase (config / ingestion / retrieval / API
  layers are cleanly separated)

## How it works

1. You upload a PDF → text is extracted, split into chunks, embedded, and
   stored in a Pinecone vector index.
2. You ask a question → the question is embedded, the top-3 most similar
   chunks are retrieved from Pinecone, and those chunks are passed to an
   LLM as context to generate a grounded answer.
3. If the answer isn't in the document, the bot says so explicitly instead
   of guessing.

## Folder structure

```
rag-chatbot/
├── app.py           # FastAPI app: routes, request/response models
├── rag.py           # Chat pipeline: embed question -> retrieve -> prompt -> LLM
├── ingest.py        # Upload pipeline: extract -> chunk -> embed -> Pinecone
├── config.py        # Loads and validates all environment variables
├── requirements.txt
├── .env.example
├── .gitignore
├── uploads/         # Uploaded PDFs are saved here (gitignored)
└── README.md
```


## 1. Prerequisites

- Python 3.12+
- A free [Google AI Studio API key](https://aistudio.google.com/apikey) (no credit card required)
- A [Pinecone account](https://app.pinecone.io) and API key (free tier is fine)

## 2. Set up a virtual environment

```bash
cd rag-chatbot
python -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure your `.env` file

Copy the example file:

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```
GOOGLE_API_KEY=AIza...your-key...
PINECONE_API_KEY=pcsk-...your-key...
PINECONE_INDEX_NAME=rag-chatbot
PINECONE_ENVIRONMENT=us-east-1
```

- Get `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) — click "Create API key". No credit card needed; this is Google's free tier (Gemini 2.5 Flash + Gemini Embeddings).
- `PINECONE_ENVIRONMENT` should be the **region** you want your serverless
  index created in (e.g. `us-east-1`). Check the "Create Index" screen in
  the Pinecone console if you're unsure which regions are available on
  your plan.

> **Note:** You do NOT need to manually create the Pinecone index yourself.
> The app automatically creates an index named `rag-chatbot` (cosine
> similarity, dimension 768) the first time you upload a PDF, if it
> doesn't already exist.

## 5. Run the app

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Interactive docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

## 6. Test the endpoints

### Using Swagger UI (easiest)

Go to `http://127.0.0.1:8000/docs` and try the endpoints directly in the browser.

### Using Postman

**Upload a PDF**

- Method: `POST`
- URL: `http://127.0.0.1:8000/upload`
- Body type: `form-data`
- Key: `file` → type `File` → select a `.pdf` from your computer

Example response:

```json
{
  "filename": "resume.pdf",
  "chunks_uploaded": 7,
  "message": "File uploaded and indexed successfully."
}
```

**Ask a question**

- Method: `POST`
- URL: `http://127.0.0.1:8000/chat`
- Body type: `raw` → `JSON`
- Body:

```json
{
  "question": "What is this document about?"
}
```

Example response:

```json
{
  "answer": "This document is a resume describing the candidate's experience in..."
}
```

If you ask something unrelated to the document:

```json
{
  "answer": "I couldn't find that information in the uploaded document."
}
```

### Using curl

```bash
# Upload
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/your/document.pdf"

# Chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Missing required environment variable` | `.env` file missing or incomplete | Copy `.env.example` to `.env` and fill in all values |
| `400: Only PDF files are supported` | Uploaded a non-PDF file | Only `.pdf` files are accepted |
| `400: No extractable text found in this PDF` | PDF is scanned/image-only (no text layer) | Use a PDF with real text, or OCR it first (OCR is out of scope for this project) |
| `401 Unauthorized` / `PermissionDenied` from Gemini | Invalid or missing `GOOGLE_API_KEY` | Double-check the key in `.env`, generated from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `429 RESOURCE_EXHAUSTED` from Gemini | Hit the free-tier rate limit (requests per minute/day) | Wait a bit and retry — free tier has generous but finite daily quotas |
| `PineconeApiException` on startup | Invalid `PINECONE_API_KEY` or unavailable `PINECONE_ENVIRONMENT` region | Verify your key and check which regions your Pinecone plan supports |
| `"I couldn't find that information..."` for everything | No PDF uploaded yet, or wrong index | Upload a PDF via `/upload` first |
| `ModuleNotFoundError` | Dependencies not installed / venv not activated | Run `pip install -r requirements.txt` inside your activated venv |
| Slow first request after upload | Pinecone index just created / cold start | This is normal — subsequent requests are faster |

## Scope note

This is intentionally a **minimal, learning-focused** RAG pipeline. It does
not include auth, a database, streaming responses, conversation memory,
multi-PDF support, hybrid search/reranking, or deployment tooling
(Docker/CI/CD/Kubernetes) — those are deliberately left out to keep the
core RAG concepts clear and easy to follow.
