"""
CareerCraft AI - Resume Tailor Agent
Rewrites resume sections for maximum ATS compatibility.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm
from config.prompts import RESUME_TAILOR_SYSTEM, RESUME_TAILOR_HUMAN


def tailor_resume(parsed_resume: dict, gap_analysis: dict, target_role: str) -> dict:
    """Rewrite portions of the resume to boost ATS score."""
    # Compress inputs to stay under TPM limits
    resume_summary = {
        "skills": parsed_resume.get("skills", {}),
        "experience": parsed_resume.get("experience", [])[:5]
    }

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESUME_TAILOR_SYSTEM),
        ("human", RESUME_TAILOR_HUMAN),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "parsed_resume": json.dumps(resume_summary, indent=1),
        "gap_analysis": json.dumps(gap_analysis, indent=1),
        "target_role": target_role,
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
            
    return {"tailored_summary": "", "tailored_experience": [], "added_keywords": []}
