"""
CareerCraft AI - Skill Matcher Tool
Demonstrates: Embeddings (Syllabus Topic #1)

Uses embedding similarity to match skills between resume and job requirements.
This provides a more intelligent matching than simple keyword matching,
as it understands semantic relationships between skills.
"""

from rag.embeddings import get_embedding_model
import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector
        
    Returns:
        float: Similarity score between 0 and 1
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def match_skills(
    candidate_skills: list[str],
    required_skills: list[str],
    threshold: float = 0.65
) -> dict:
    """
    Match candidate skills against required skills using semantic similarity.
    
    Instead of exact string matching, this uses embeddings to find
    semantic matches (e.g., "JS" matches "JavaScript", "ML" matches
    "Machine Learning").
    
    Args:
        candidate_skills: Skills from the candidate's resume
        required_skills: Skills required by the job
        threshold: Minimum similarity score for a match (0-1)
        
    Returns:
        dict: Matching results with scores
    """
    if not candidate_skills or not required_skills:
        return {
            "matched": [],
            "missing": required_skills or [],
            "match_percentage": 0,
        }
    
    model = get_embedding_model()
    
    # Embed all skills
    candidate_embeddings = model.embed_documents(candidate_skills)
    required_embeddings = model.embed_documents(required_skills)
    
    matched = []
    missing = []
    
    for i, req_skill in enumerate(required_skills):
        best_match = None
        best_score = 0.0
        
        for j, cand_skill in enumerate(candidate_skills):
            score = cosine_similarity(
                required_embeddings[i], 
                candidate_embeddings[j]
            )
            
            if score > best_score:
                best_score = score
                best_match = cand_skill
        
        if best_score >= threshold:
            matched.append({
                "required": req_skill,
                "matched_with": best_match,
                "similarity": round(best_score, 3),
            })
        else:
            missing.append({
                "skill": req_skill,
                "best_candidate_match": best_match,
                "best_similarity": round(best_score, 3),
            })
    
    match_percentage = (
        len(matched) / len(required_skills) * 100
        if required_skills
        else 0
    )
    
    return {
        "matched": matched,
        "missing": missing,
        "match_percentage": round(match_percentage, 1),
        "total_required": len(required_skills),
        "total_matched": len(matched),
        "total_missing": len(missing),
    }
