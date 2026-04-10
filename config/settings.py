"""
CareerCraft AI - Application Configuration
Uses pydantic-settings for type-safe configuration management.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    tavily_api_key: str = Field(default="", env="TAVILY_API_KEY")

    # --- LLM Configuration ---
    llm_model: str = Field(default="llama-3.3-70b-versatile", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")

    # --- Embedding Model ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(
        default="./chroma_db", env="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="careercraft_knowledge", env="CHROMA_COLLECTION_NAME"
    )

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./careercraft.db", env="DATABASE_URL"
    )

    # --- App Metadata ---
    app_name: str = Field(default="CareerCraft AI", env="APP_NAME")
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton settings instance
settings = Settings()


# --- Singletons ---
_llm_instance = None
_embeddings_instance = None


def get_llm():
    """Initialize and return a singleton Groq LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key,
        )
    return _llm_instance


def get_embeddings():
    """Initialize and return a singleton embedding model."""
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance
