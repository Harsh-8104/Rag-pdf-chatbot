"""
app.py

FastAPI application entry point.

This file only handles HTTP concerns: routes, request/response models,
file validation, and turning exceptions into proper HTTP status codes.
All actual RAG logic lives in ingest.py (upload pipeline) and rag.py
(chat pipeline).

Run with:
    uvicorn app:app --reload
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from ingest import ingest_pdf
from rag import answer_question

app = FastAPI(
    title="RAG Chatbot",
    description="A minimal Retrieval-Augmented Generation chatbot over a single uploaded PDF.",
    version="1.0.0",
)

# Make sure the uploads directory exists at startup.
os.makedirs(config.UPLOAD_DIR, exist_ok=True)


# --- Request/response models ---

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    question: str


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    answer: str


class UploadResponse(BaseModel):
    """Response body for POST /upload."""
    filename: str
    chunks_uploaded: int
    message: str


# --- Endpoints ---

@app.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF document, extract its text, chunk it, embed it, and
    store the embeddings in Pinecone.

    Args:
        file: The uploaded PDF file (multipart/form-data).

    Returns:
        UploadResponse with the filename and number of chunks stored.

    Raises:
        HTTPException 400: If the file isn't a PDF, or if no text could
            be extracted from it.
        HTTPException 500: For unexpected errors during ingestion.
    """
    # Validate: only accept .pdf files.
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Save the uploaded file to disk.
    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )
    finally:
        file.file.close()

    # Run the ingestion pipeline (extract -> chunk -> embed -> upsert).
    try:
        chunks_uploaded = ingest_pdf(file_path=file_path, filename=file.filename)
    except ValueError as exc:
        # Expected failure case: e.g. scanned PDF with no text layer.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Unexpected failure: embedding API error, Pinecone error, etc.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {exc}",
        )

    return UploadResponse(
        filename=file.filename,
        chunks_uploaded=chunks_uploaded,
        message="File uploaded and indexed successfully.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question using retrieval-augmented generation over the
    previously uploaded document(s).

    Args:
        request: ChatRequest containing the user's question.

    Returns:
        ChatResponse containing the generated answer.

    Raises:
        HTTPException 400: If the question is empty.
        HTTPException 500: For unexpected errors during retrieval/generation.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answer = answer_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate an answer: {exc}",
        )

    return ChatResponse(answer=answer)


@app.get("/")
async def serve_ui() -> FileResponse:
    """Serve the chat UI (static/index.html)."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> dict:
    """Simple health-check endpoint."""
    return {"status": "ok", "message": "RAG Chatbot API is running."}
