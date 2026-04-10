"""
CareerCraft AI - AI Matchmaker Agent
Demonstrates: Proactive Web Searching + LLM Agents
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm
from config.prompts import MATCHMAKER_SYSTEM, MATCHMAKER_HUMAN
from tools.web_search import perform_web_search


def match_jobs(parsed_resume: dict) -> dict:
    """Find live matching jobs using web search and LLM."""
    llm = get_llm()
    
    # Extract top skills
    skills_data = parsed_resume.get("skills", {})
    tech_skills = skills_data.get("technical", []) if isinstance(skills_data, dict) else []
    soft_skills = skills_data.get("soft", []) if isinstance(skills_data, dict) else []
    
    target_skills = ", ".join(tech_skills[:5])
    if not target_skills:
        target_skills = ", ".join(soft_skills[:5])
        
    if not target_skills:
        target_skills = "Entry level"
    
    # Search the web for live jobs
    web_query = f"Latest open job postings hiring for: {target_skills} 2026 apply URL"
    web_results = perform_web_search(web_query, max_results=2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", MATCHMAKER_SYSTEM),
        ("human", MATCHMAKER_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "parsed_skills": json.dumps(skills_data, indent=2),
        "web_results": web_results,
    })
    
    return _extract_json(response)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    import re
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    if matches:
        try:
            return json.loads(matches[0].strip())
        except:
            pass
    
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except:
            pass
    
    return {"matched_jobs": [], "overall_market_outlook": "Failed to parse."}
