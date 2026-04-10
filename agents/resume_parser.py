"""
CareerCraft AI - Resume Parser Agent
Demonstrates: Agents (Single Agent) + LangChain + LLM Basics (Syllabus Topics #2, #6, #1)

This agent takes raw resume text and extracts structured information
using LLM-powered parsing. It converts unstructured resume text into
a clean JSON format for downstream agents.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm
from config.prompts import RESUME_PARSER_SYSTEM, RESUME_PARSER_HUMAN


def parse_resume(resume_text: str) -> dict:
    """
    Parse raw resume text into structured data using LLM.
    
    This agent demonstrates:
    - Single agent pattern
    - LLM chain with structured output
    - Prompt engineering for reliable JSON extraction
    
    Args:
        resume_text: Raw text extracted from the resume PDF/DOCX
        
    Returns:
        dict: Structured resume data with sections like skills, experience, etc.
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESUME_PARSER_SYSTEM),
        ("human", RESUME_PARSER_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"resume_text": resume_text})
    
    # Parse JSON from response
    parsed = _extract_json(response)
    
    return parsed


def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        text: LLM response that may contain JSON
        
    Returns:
        dict: Parsed JSON data
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from code blocks
    import re
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try finding JSON object in text
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    
    # Return a minimal structure if all parsing fails
    return {
        "name": "",
        "email": "",
        "phone": "",
        "summary": text[:500],
        "skills": {"technical": [], "soft": [], "tools": [], "languages": []},
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "total_experience_years": 0,
        "_parse_error": "Could not parse structured data from resume",
    }
