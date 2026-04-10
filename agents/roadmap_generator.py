"""
CareerCraft AI - Roadmap Generator Agent
Demonstrates: Agents + RAG (Syllabus Topics #2, #4)

This agent creates a personalized, week-by-week learning roadmap
based on the skill gaps identified by the Gap Analyzer.
Uses RAG to retrieve relevant learning resources and skill taxonomies.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm
from config.prompts import ROADMAP_GENERATOR_SYSTEM, ROADMAP_GENERATOR_HUMAN
from rag.retriever import retrieve_relevant_context
from tools.web_search import perform_web_search


def generate_roadmap(
    gap_analysis: dict,
    target_role: str,
    parsed_resume: dict = {},
) -> dict:
    """
    Generate a personalized learning roadmap from gap analysis.
    
    This agent demonstrates:
    - Actionable output generation
    - RAG for resource recommendations
    - Structured learning path creation
    
    Args:
        gap_analysis: Gap analysis results from the gap analyzer agent
        target_role: Target job role
        
    Returns:
        dict: Week-by-week learning roadmap with free resources
    """
    llm = get_llm()
    
    # Build RAG query for learning resources
    missing_skills = []
    if isinstance(gap_analysis.get("missing_skills"), dict):
        for category in ["critical", "important", "nice_to_have"]:
            skills = gap_analysis["missing_skills"].get(category, [])
            if isinstance(skills, list):
                missing_skills.extend(skills)
    
    rag_query = f"Learning roadmap for {target_role}: {', '.join(missing_skills[:10])}"
    
    rag_context = retrieve_relevant_context(rag_query, k=3)
    
    # Enrich with live web search for actual usable course links
    if missing_skills:
        web_query = f"Best free courses and tutorials to learn {', '.join(missing_skills[:3])} in 2026"
        web_results = perform_web_search(web_query, max_results=2)
    else:
        web_results = "No critical missing skills identified for specific search."

    rag_context = f"--- INTERNAL KNOWLEDGE BASE ---\n{rag_context}\n\n--- LIVE WEB COURSE LINKS ---\n{web_results}"
    
    # Compress inputs to stay under TPM limits
    resume_summary = {
        "skills": parsed_resume.get("skills", {}),
        "experience": [{"title": exp.get("title")} for exp in parsed_resume.get("experience", [])[:5]],
        "projects": parsed_resume.get("projects", [])[:3]
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROADMAP_GENERATOR_SYSTEM),
        ("human", ROADMAP_GENERATOR_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "gap_analysis": json.dumps(gap_analysis, indent=1),
        "target_role": target_role,
        "rag_context": rag_context[:2000], # Cap the context further
    })
    
    roadmap = _extract_json(response)
    
    return roadmap


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
        "roadmap_title": f"Learning Roadmap",
        "total_duration_weeks": 0,
        "phases": [],
        "free_resources_summary": [],
        "_parse_error": "Could not parse structured roadmap",
        "_raw_response": text[:500],
    }
