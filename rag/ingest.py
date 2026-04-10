"""
CareerCraft AI - Knowledge Base Ingestion Script
Demonstrates: Embeddings + RAG (Syllabus Topics #1, #4)

This script loads all knowledge base data (skill taxonomies,
job descriptions, interview questions) into ChromaDB as embeddings.
Run this once before starting the application.

Usage:
    python -m rag.ingest
"""

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag.vectorstore import get_vectorstore
from rag.embeddings import get_embedding_model
from config.settings import settings

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")


def load_skills_taxonomy() -> list[Document]:
    """Load skills taxonomy into documents for embedding."""
    filepath = os.path.join(KNOWLEDGE_DIR, "skills_taxonomy.json")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    
    # Process skill categories
    for cat_key, cat_data in data.get("categories", {}).items():
        content = f"Skill Category: {cat_data['name']}\nSkills: {', '.join(cat_data['skills'])}"
        doc = Document(
            page_content=content,
            metadata={
                "source": "skills_taxonomy",
                "type": "skill_category",
                "category": cat_key,
            },
        )
        documents.append(doc)
    
    # Process role skill profiles
    for role_key, role_data in data.get("role_skill_profiles", {}).items():
        content = (
            f"Role: {role_data['title']}\n"
            f"Critical Skills: {', '.join(role_data['critical_skills'])}\n"
            f"Important Skills: {', '.join(role_data['important_skills'])}\n"
            f"Nice to Have: {', '.join(role_data['nice_to_have'])}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "source": "skills_taxonomy",
                "type": "role_profile",
                "role": role_key,
            },
        )
        documents.append(doc)
    
    return documents


def load_job_descriptions() -> list[Document]:
    """Load job descriptions into documents for embedding."""
    jd_dir = os.path.join(KNOWLEDGE_DIR, "job_descriptions")
    documents = []
    
    if not os.path.exists(jd_dir):
        return documents
    
    for filename in os.listdir(jd_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(jd_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "job_description",
                    "type": "job_description",
                    "filename": filename,
                    "role": filename.replace(".txt", "").replace("_", " ").title(),
                },
            )
            documents.append(doc)
    
    return documents


def load_interview_questions() -> list[Document]:
    """Load interview questions into documents for embedding."""
    iq_dir = os.path.join(KNOWLEDGE_DIR, "interview_questions")
    documents = []
    
    if not os.path.exists(iq_dir):
        return documents
    
    for filename in os.listdir(iq_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(iq_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Process each question category in the JSON
            for category_key, questions in data.items():
                if isinstance(questions, list):
                    for q in questions:
                        if isinstance(q, dict) and "question" in q:
                            content = f"Interview Question ({category_key}):\n"
                            content += f"Q: {q['question']}\n"
                            if "difficulty" in q:
                                content += f"Difficulty: {q['difficulty']}\n"
                            if "expected_points" in q:
                                content += f"Key Points: {', '.join(q['expected_points'])}\n"
                            if "what_they_assess" in q:
                                content += f"Assesses: {q['what_they_assess']}\n"
                            if "framework" in q:
                                content += f"Framework: {q['framework']}\n"
                            if "tips" in q:
                                content += f"Tips: {q['tips']}\n"
                            if "follow_ups" in q:
                                content += f"Follow-ups: {', '.join(q['follow_ups'])}"
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": "interview_questions",
                                    "type": "interview",
                                    "category": category_key,
                                    "filename": filename,
                                },
                            )
                            documents.append(doc)
    
    return documents


def ingest_knowledge_base():
    """
    Main ingestion function: loads all knowledge base data into ChromaDB.
    
    Steps:
    1. Load documents from all sources
    2. Split into chunks for better retrieval
    3. Embed and store in ChromaDB
    """
    print("=" * 60)
    print("🧠 CareerCraft AI - Knowledge Base Ingestion")
    print("=" * 60)
    
    # Load all documents
    print("\n📚 Loading documents...")
    
    all_documents = []
    
    skills_docs = load_skills_taxonomy()
    print(f"  ✅ Skills taxonomy: {len(skills_docs)} documents")
    all_documents.extend(skills_docs)
    
    jd_docs = load_job_descriptions()
    print(f"  ✅ Job descriptions: {len(jd_docs)} documents")
    all_documents.extend(jd_docs)
    
    iq_docs = load_interview_questions()
    print(f"  ✅ Interview questions: {len(iq_docs)} documents")
    all_documents.extend(iq_docs)
    
    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    
    # Split documents into chunks
    print("\n✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"  ✅ Created {len(chunks)} chunks")
    
    # Store in ChromaDB
    print("\n💾 Embedding and storing in ChromaDB...")
    vectorstore = get_vectorstore()
    
    # Add in batches to avoid overwhelming the embedding model
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  📦 Batch {i // batch_size + 1}: Added {len(batch)} chunks")
    
    print(f"\n✅ Knowledge base ingestion complete!")
    print(f"   Total chunks stored: {len(chunks)}")
    print(f"   Persist directory: {os.path.abspath(settings.chroma_persist_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    ingest_knowledge_base()
