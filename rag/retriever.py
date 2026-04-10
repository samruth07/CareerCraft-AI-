"""
CareerCraft AI - RAG Retriever
Demonstrates: RAG + LangChain (Syllabus Topics #4, #6)

Combines vector retrieval with LLM generation using LangChain.
This is the full RAG chain that retrieves context and generates
grounded responses.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.vectorstore import get_retriever
from config.settings import get_llm


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'knowledge_base')}\n{doc.page_content}"
        for doc in docs
    )


def get_rag_chain(system_prompt: str, human_prompt: str):
    """
    Create a RAG chain that retrieves context and generates responses.
    
    This demonstrates the full RAG pipeline:
    1. User query → Embedding → Vector search
    2. Retrieved documents → Context formatting
    3. Context + Query → LLM → Response
    
    Args:
        system_prompt: System message for the LLM
        human_prompt: Human message template (must contain {context} and {question})
        
    Returns:
        Runnable: Executable RAG chain
    """
    llm = get_llm()
    retriever = get_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def retrieve_relevant_context(query: str, k: int = 5) -> str:
    """
    Retrieve relevant context from the knowledge base for a query.
    
    This is used by agents to get RAG context before calling the LLM.
    
    Args:
        query: The search query
        k: Number of documents to retrieve
        
    Returns:
        str: Formatted context string from retrieved documents
    """
    retriever = get_retriever(search_k=k)
    docs = retriever.invoke(query)
    return format_docs(docs)
