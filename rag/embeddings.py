"""
CareerCraft AI - Embedding Model Configuration
Demonstrates: LLM Basics + Embeddings (Syllabus Topic #1)

Uses sentence-transformers for local, free embedding generation.
Embeddings convert text into dense vectors for semantic similarity search.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings


def get_embedding_model():
    """
    Initialize the embedding model for converting text to vectors.
    Uses the singleton from settings to prevent expensive re-loads.
    """
    from config.settings import get_embeddings
    return get_embeddings()


def embed_text(text: str) -> list[float]:
    """
    Convert a single text string into an embedding vector.
    
    Args:
        text: The text to embed
        
    Returns:
        list[float]: 384-dimensional embedding vector
    """
    model = get_embedding_model()
    return model.embed_query(text)


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Convert a list of documents into embedding vectors.
    
    Args:
        documents: List of text strings to embed
        
    Returns:
        list[list[float]]: List of embedding vectors
    """
    model = get_embedding_model()
    return model.embed_documents(documents)
