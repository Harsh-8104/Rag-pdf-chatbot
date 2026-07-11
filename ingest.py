"""
ingest.py

Handles the "upload" side of the RAG pipeline:
  1. Extract raw text from a PDF file.
  2. Split that text into overlapping chunks.
  3. Embed each chunk.
  4. Upsert the embeddings (+ metadata) into Pinecone.

app.py calls `ingest_pdf()` -- everything else here is a private helper.
"""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

import config


def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file, page by page.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        The full extracted text, with pages joined by newlines.

    Raises:
        ValueError: If no extractable text is found (e.g. a scanned/
            image-only PDF with no text layer).
    """
    reader = PdfReader(file_path)

    pages_text = []
    for page in reader.pages:
        # extract_text() can return None for pages with no text layer.
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only document."
        )

    return full_text


def _split_text_into_chunks(text: str) -> list[str]:
    """
    Split a long text into overlapping chunks suitable for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split
    on paragraph/sentence boundaries first, and only falls back to hard
    character cuts if a chunk would otherwise be too large.

    Args:
        text: The full document text.

    Returns:
        A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def _get_pinecone_index():
    """
    Get a handle to the Pinecone index, creating it first if it doesn't exist.

    Returns:
        A Pinecone Index object ready for upsert/query operations.
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing_indexes:
        # Create a serverless index with cosine similarity, sized to match
        # the embedding model's output dimension.
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_ENVIRONMENT,
            ),
        )

    return pc.Index(config.PINECONE_INDEX_NAME)


def ingest_pdf(file_path: str, filename: str) -> int:
    """
    Run the full ingestion pipeline for a single PDF: extract, chunk,
    embed, and upload to Pinecone.

    Args:
        file_path: Path to the saved PDF file on disk.
        filename: Original filename, stored as metadata so answers can be
            traced back to the source document.

    Returns:
        The number of chunks that were embedded and uploaded.

    Raises:
        ValueError: If the PDF has no extractable text.
    """
    # Step 1: extract raw text
    text = _extract_text_from_pdf(file_path)

    # Step 2: split into chunks
    chunks = _split_text_into_chunks(text)

    # Step 3: embed all chunks in a single batched call.
    # task_type="RETRIEVAL_DOCUMENT" tells the model these are documents to
    # be searched over (as opposed to queries) -- Gemini uses this hint to
    # produce better-optimized embeddings for retrieval.
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        output_dimensionality=config.EMBEDDING_DIMENSION,
        task_type="RETRIEVAL_DOCUMENT",
    )
    vectors = embeddings_model.embed_documents(chunks)

    # Step 4: build Pinecone upsert payload.
    # Each vector needs: a unique id, the embedding values, and metadata
    # so we can display/reference the original chunk later.
    index = _get_pinecone_index()

    upsert_payload = []
    for chunk_number, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
        vector_id = f"{filename}-chunk-{chunk_number}"
        upsert_payload.append({
            "id": vector_id,
            "values": vector,
            "metadata": {
                "filename": filename,
                "chunk_number": chunk_number,
                "text": chunk_text,
            },
        })

    # Upsert in one batch. For very large PDFs you'd chunk this into
    # smaller batches, but that's beyond the "minimal" scope here.
    index.upsert(vectors=upsert_payload)

    return len(chunks)
