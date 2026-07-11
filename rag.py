"""
rag.py

Handles the "chat" side of the RAG pipeline:
  1. Embed the user's question.
  2. Query Pinecone for the top-K most similar chunks.
  3. Build a prompt that includes only that retrieved context.
  4. Send the prompt to the chat LLM and return its answer.

app.py calls `answer_question()` -- everything else here is a private helper.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone

import config

# The chatbot must answer ONLY from retrieved context, and must use this
# exact fallback line when the answer isn't in the context.
PROMPT_TEMPLATE = """You are a helpful AI assistant.
Only answer using the provided context.
If the answer is missing from the context, say:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:"""


def _embed_question(question: str) -> list[float]:
    """
    Generate an embedding vector for the user's question, using the same
    embedding model that was used to embed the document chunks. This is
    required -- query and document embeddings must come from the same
    model to be comparable via cosine similarity.

    Args:
        question: The user's natural-language question.

    Returns:
        The embedding vector for the question.
    """
    # task_type="RETRIEVAL_QUERY" is the query-side counterpart to
    # RETRIEVAL_DOCUMENT used when embedding chunks in ingest.py -- Gemini
    # optimizes each side differently for better retrieval matching.
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        output_dimensionality=config.EMBEDDING_DIMENSION,
        task_type="RETRIEVAL_QUERY",
    )
    return embeddings_model.embed_query(question)


def _retrieve_relevant_chunks(question_vector: list[float]) -> list[str]:
    """
    Query Pinecone for the top-K chunks most similar to the question.

    Args:
        question_vector: Embedding vector of the user's question.

    Returns:
        A list of chunk text strings, ordered from most to least relevant.
        Empty list if the index has no vectors yet (e.g. nothing uploaded).
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    results = index.query(
        vector=question_vector,
        top_k=config.TOP_K,
        include_metadata=True,
    )

    # Pull the original chunk text back out of each match's metadata.
    return [match["metadata"]["text"] for match in results["matches"]]


def _build_prompt(context: str, question: str) -> str:
    """
    Fill in the fixed prompt template with retrieved context and the
    user's question.

    Args:
        context: Retrieved chunk text, concatenated together.
        question: The user's question.

    Returns:
        The fully-formed prompt string to send to the LLM.
    """
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer_question(question: str) -> str:
    """
    Run the full retrieval-augmented answer pipeline for a single question.

    Args:
        question: The user's natural-language question.

    Returns:
        The LLM's answer, grounded only in retrieved document context.
    """
    # Step 1: embed the question
    question_vector = _embed_question(question)

    # Step 2: retrieve the most relevant chunks from Pinecone
    chunks = _retrieve_relevant_chunks(question_vector)

    if not chunks:
        # Nothing in the index at all -- no document uploaded yet.
        return "I couldn't find that information in the uploaded document."

    # Step 3: build the prompt from retrieved context
    context = "\n\n".join(chunks)
    prompt = _build_prompt(context=context, question=question)

    # Step 4: send the prompt to the chat LLM
    chat_model = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0,  # deterministic, factual answers -- no creativity needed
    )
    response = chat_model.invoke(prompt)

    return response.content
