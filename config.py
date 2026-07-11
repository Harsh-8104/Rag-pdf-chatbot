"""
config.py

Single source of truth for all configuration/environment variables.

No other file in this project should call os.getenv() directly -- they
should import the values they need from here. This makes it obvious,
in one place, exactly what secrets/config this app depends on.
"""

import os

from dotenv import load_dotenv

# Load variables from a local .env file into the process environment.
# In production you'd typically set these as real environment variables
# instead of relying on a .env file, but this works for both cases.
load_dotenv()


def _get_required_env(var_name: str) -> str:
    """
    Fetch a required environment variable, or raise a clear error if missing.

    Args:
        var_name: Name of the environment variable to fetch.

    Returns:
        The value of the environment variable.

    Raises:
        RuntimeError: If the variable is not set (or is empty), so the app
            fails fast at startup instead of failing later with a confusing
            error deep inside a request handler.
    """
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: '{var_name}'. "
            f"Did you copy .env.example to .env and fill it in?"
        )
    return value


# --- Google Gemini (free tier, no billing required) ---
GOOGLE_API_KEY: str = _get_required_env("GOOGLE_API_KEY")

# --- Pinecone ---
PINECONE_API_KEY: str = _get_required_env("PINECONE_API_KEY")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")

# Used as the Pinecone serverless "region" (e.g. "us-east-1").
# Pinecone's newer serverless indexes use cloud + region instead of the
# older pod-based "environment" concept, but we keep the env var name
# PINECONE_ENVIRONMENT since that's the more commonly recognized term.
PINECONE_ENVIRONMENT: str = _get_required_env("PINECONE_ENVIRONMENT")

# Cloud provider for the Pinecone serverless index.
# Hardcoded to "aws" since that's the most widely available option on
# Pinecone's free tier. Change here if you need gcp/azure.
PINECONE_CLOUD: str = "aws"

# --- Embedding + chat models ---
# Centralized here so swapping models later is a one-line change.
# Both are on Google's free tier (aistudio.google.com) -- no credit card needed.
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
EMBEDDING_DIMENSION: int = 768  # reduced via MRL from the 3072 default -- smaller/cheaper to store, still strong retrieval quality
CHAT_MODEL: str = "gemini-2.5-flash"

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# --- Retrieval ---
TOP_K: int = 3

# --- File storage ---
UPLOAD_DIR: str = "uploads"
