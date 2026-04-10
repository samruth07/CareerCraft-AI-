"""
Reasoning Agent for CareerCraft AI
Generates a human-like explanation for the career analysis results.
Inspired by reasoning_agent.py in the Loan-Agentic-System.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm

def generate_career_reasoning(analysis_results: dict) -> str:
    """
    Summarizes the technical analysis into a cohesive reasoning narrative.
    """
    llm = get_llm()
    
    # Extract key data for the prompt
    target_role = analysis_results.get("target_role", "Target Role")
    match_pct = analysis_results.get("gap_analysis", {}).get("match_percentage", 0)
    strengths = analysis_results.get("gap_analysis", {}).get("strengths", [])
    missing = analysis_results.get("gap_analysis", {}).get("missing_skills", {})
    critical_missing = missing.get("critical", []) if isinstance(missing, dict) else []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the 'Career Reasoning Agent'. Your job is to explain a career match analysis in a professional, encouraging, and highly detailed way.
        
        Focus on:
        1. WHY the candidate is a good/moderate/poor match for the {target_role}.
        2. HOW their existing strengths align with the role.
        3. WHAT the single most important next step is to close the gap.
        
        Use a professional "Advisor" tone. Avoid bullet points in the main reasoning section; use narrative paragraphs.
        Match Percentage: {match_pct}%"""),
        ("human", """Here is the technical data:
        Strengths: {strengths}
        Critical Missing Skills: {critical_missing}
        
        Please provide a 'Reasoning Profile' (approx 150-200 words).""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    reasoning = chain.invoke({
        "target_role": target_role,
        "match_pct": match_pct,
        "strengths": ", ".join(strengths[:5]),
        "critical_missing": ", ".join(critical_missing[:5])
    })
    
    return reasoning
