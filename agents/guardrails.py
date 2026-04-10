"""
CareerCraft AI - Guardrails
Demonstrates: Safety & Guardrails (Syllabus Topic #2)

Logic to validate inputs and ensure agents operate within professional bounds.
"""

import re
from typing import Tuple

def validate_resume_text(text: str) -> Tuple[bool, str]:
    """
    Validates if the provided text is likely a resume.
    
    Checks for structural keywords and minimum length.
    """
    if not text or len(text) < 200:
        return False, "Input text is too short to be a valid resume."
        
    # Structural keywords typically found in resumes
    resume_keywords = [
        r"(?i)experience", 
        r"(?i)education", 
        r"(?i)skills", 
        r"(?i)projects", 
        r"(?i)objective", 
        r"(?i)summary",
        r"(?i)university",
        r"(?i)college",
        r"(?i)employment"
    ]
    
    match_count = 0
    for pattern in resume_keywords:
        if re.search(pattern, text):
            match_count += 1
            
    # Require at least 3 matching sections
    if match_count < 3:
        return False, "You have not uploaded a resume. So please upload only resume"
        
    return True, "Validation successful."

def validate_agent_output(output_text: str) -> bool:
    """
    Basic output guardrail to ensure professional tone and no obvious hallucinations.
    """
    forbidden_phrases = [
        "as an AI language model",
        "I don't know",
        "hallucination",
        "toxic",
        "offensive"
    ]
    
    for phrase in forbidden_phrases:
        if phrase.lower() in output_text.lower():
            return False
            
    return True
