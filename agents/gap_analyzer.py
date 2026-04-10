"""
CareerCraft AI - Gap Analyzer Agent
Demonstrates: Agents + RAG + Embeddings (Syllabus Topics #2, #4, #1)

This agent compares a parsed resume against a target job description,
using RAG to retrieve relevant skill profiles and industry context.
It identifies exact skill gaps, strengths, and provides a match score.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm
from config.prompts import GAP_ANALYZER_SYSTEM, GAP_ANALYZER_HUMAN
from rag.retriever import retrieve_relevant_context
from tools.web_search import perform_web_search
from tools.github_analyzer import analyze_github_profile


def analyze_gaps(
    parsed_resume: dict,
    job_description: str,
    target_role: str = "",
) -> dict:
    """
    Analyze skill gaps between a resume and target job.
    
    This agent demonstrates:
    - RAG-powered analysis (retrieves relevant skill profiles)
    - Embedding-based context retrieval
    - Multi-step reasoning with LLM
    
    Args:
        parsed_resume: Structured resume data from parser agent
        job_description: Target job description text
        target_role: Target role title (optional, improves RAG retrieval)
        
    Returns:
        dict: Gap analysis with match %, missing skills, strengths, recommendations
    """
    llm = get_llm()
    
    # Build RAG query from the target role and key job requirements
    rag_query = f"Skills required for {target_role}: {job_description[:500]}"
    
    # Retrieve relevant context from knowledge base (reduced k to save tokens)
    rag_context = retrieve_relevant_context(rag_query, k=2)
    
    # Enrich with live web search for modern market trends
    web_query = f"Latest tech skills and top requirements for {target_role} in 2026"
    web_results = perform_web_search(web_query, max_results=2)
    
    # Check for GitHub link to analyze real codebase
    github_url = parsed_resume.get("github_url", "")
    github_context = ""
    if github_url and "http" in github_url:
        github_context = analyze_github_profile(github_url)
    
    rag_context = f"--- INTERNAL KNOWLEDGE BASE ---\n{rag_context}\n\n--- LIVE WEB DATA ---\n{web_results}\n\n--- GITHUB PROFILE DATA ---\n{github_context}"
    
    # Aggressively trim context to stay under 6,000 TPM limit
    rag_context = rag_context[:2000]
    
    # Compress inputs to stay under TPM limits
    # Only send the most critical parts of the parsed resume
    resume_summary = {
        "skills": parsed_resume.get("skills", {}),
        "experience": [
            {"title": exp.get("title"), "company": exp.get("company"), "tech": exp.get("technologies", [])} 
            for exp in parsed_resume.get("experience", [])[:5]
        ],
        "projects": parsed_resume.get("projects", [])[:3],
        "deep_understanding": parsed_resume.get("deep_understanding", {})
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", GAP_ANALYZER_SYSTEM),
        ("human", GAP_ANALYZER_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "parsed_resume": json.dumps(resume_summary, indent=1),
        "job_description": job_description[:1000], # Reduced from 1500 to stay under 6000 TPM
        "rag_context": rag_context,
    })
    
    # Parse JSON response
    analysis = _extract_json(response)
    
    return analysis


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    import re
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    
    return {
        "match_percentage": 0,
        "matching_skills": [],
        "missing_skills": {"critical": [], "important": [], "nice_to_have": []},
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
        "overall_assessment": text[:500],
        "_parse_error": "Could not parse structured gap analysis",
    }
