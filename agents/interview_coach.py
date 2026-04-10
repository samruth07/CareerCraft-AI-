"""
CareerCraft AI - Interview Coach Agent
Demonstrates: Agents + RAG + Memory (Syllabus Topics #2, #4, #5)

This agent generates role-specific interview questions, provides
preparation tips, and can evaluate candidate answers in mock
interview mode. Uses RAG to retrieve relevant interview Q&A.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm
from config.prompts import (
    INTERVIEW_COACH_SYSTEM,
    INTERVIEW_COACH_HUMAN,
    ANSWER_EVALUATOR_SYSTEM,
    ANSWER_EVALUATOR_HUMAN,
)
from rag.retriever import retrieve_relevant_context


def prepare_interview(
    parsed_resume: dict,
    target_role: str,
    gap_analysis: dict,
) -> dict:
    """
    Generate comprehensive interview preparation material.
    
    This agent demonstrates:
    - RAG-powered question generation
    - Personalized content based on candidate profile
    - Structured output for UI rendering
    
    Args:
        parsed_resume: Structured resume data
        target_role: Target job role
        gap_analysis: Gap analysis results
        
    Returns:
        dict: Interview questions, tips, and preparation guidance
    """
    llm = get_llm()
    
    # Retrieve relevant interview questions from knowledge base (reduced k to save tokens)
    rag_query = f"Interview questions for {target_role} role"
    rag_context = retrieve_relevant_context(rag_query, k=3)
    
    # Compress inputs to stay under TPM limits
    resume_summary = {
        "skills": parsed_resume.get("skills", {}),
        "experience": [{"title": exp.get("title")} for exp in parsed_resume.get("experience", [])[:5]]
    }
    
    gap_summary = {
        "matching_skills": gap_analysis.get("matching_skills", []),
        "missing_skills": gap_analysis.get("missing_skills", {})
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", INTERVIEW_COACH_SYSTEM),
        ("human", INTERVIEW_COACH_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "parsed_resume": json.dumps(resume_summary, indent=1),
        "target_role": target_role,
        "gap_analysis": json.dumps(gap_summary, indent=1),
        "rag_context": rag_context[:2000],
    })
    
    interview_prep = _extract_json(response)
    
    return interview_prep


def evaluate_answer(
    question: str,
    answer: str,
    target_role: str,
) -> dict:
    """
    Evaluate a candidate's answer to an interview question.
    
    This creates a feedback loop:
    Question → Answer → Evaluation → Improvement
    This IS agentic behavior — not just static AI.
    
    Args:
        question: The interview question
        answer: The candidate's answer
        target_role: Target role for context
        
    Returns:
        dict: Evaluation with score, feedback, and model answer
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_EVALUATOR_SYSTEM),
        ("human", ANSWER_EVALUATOR_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": question,
        "answer": answer,
        "target_role": target_role,
    })
    
    evaluation = _extract_json(response)
    
    return evaluation


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
        "role": "",
        "preparation_summary": text[:500],
        "questions": {"behavioral": [], "technical": [], "situational": []},
        "preparation_tips": {},
        "_parse_error": "Could not parse structured interview prep",
    }
