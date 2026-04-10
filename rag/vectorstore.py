"""
CareerCraft AI - ChromaDB Vector Store
Demonstrates: RAG (Syllabus Topic #4)

ChromaDB stores document embeddings for semantic retrieval.
This is the core of our RAG pipeline - enabling the agents
to retrieve relevant knowledge before generating responses.
"""

import chromadb
from langchain_chroma import Chroma
from rag.embeddings import get_embedding_model
from config.settings import settings


def get_vectorstore() -> Chroma:
    """
    Initialize or load the ChromaDB vector store.
    
    The vector store persists to disk so knowledge survives restarts.
    Uses the embedding model to convert queries into vectors for
    similarity search against stored documents.
    
    Returns:
        Chroma: LangChain-wrapped ChromaDB vector store
    """
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    
    return vectorstore


def get_retriever(search_k: int = 5):
    """
    Create a retriever from the vector store for RAG queries.
    
    The retriever performs similarity search to find the most
    relevant documents for a given query.
    
    Args:
        search_k: Number of documents to retrieve (default: 5)
        
    Returns:
        VectorStoreRetriever: Configured retriever for RAG
    """
    vectorstore = get_vectorstore()
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k},
    )
    
    return retriever


def similarity_search(query: str, k: int = 5):
    """
    Perform direct similarity search against the knowledge base.
    
    Args:
        query: Search query text
        k: Number of results to return
        
    Returns:
        list[Document]: Most relevant documents
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)


def similarity_search_with_score(query: str, k: int = 5):
    """
    Perform similarity search with relevance scores.
    
    Args:
        query: Search query text
        k: Number of results to return
        
    Returns:
        list[tuple[Document, float]]: Documents with similarity scores
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)
