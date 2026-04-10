"""
CareerCraft AI - Tools Tests
Tests for PDF parser and skill matcher tools.
"""

import pytest


class TestPdfParser:
    """Tests for PDF/DOCX text extraction."""
    
    def test_clean_resume_text(self):
        """Test text cleaning function."""
        from tools.pdf_parser import clean_resume_text
        
        dirty = "  Hello   World  \n\n\n\nTest  \n  123  \n  Line  "
        clean = clean_resume_text(dirty)
        
        assert "Hello World" in clean
        assert "\n\n\n" not in clean  # No excessive newlines
    
    def test_clean_resume_text_empty(self):
        """Test cleaning empty text."""
        from tools.pdf_parser import clean_resume_text
        
        result = clean_resume_text("")
        assert result == ""
    
    def test_parse_resume_file_no_input(self):
        """Test error when no input provided."""
        from tools.pdf_parser import parse_resume_file
        
        with pytest.raises(ValueError):
            parse_resume_file()


class TestSkillMatcher:
    """Tests for embedding-based skill matcher."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        from tools.skill_matcher import cosine_similarity
        
        vec = [1.0, 0.0, 0.0]
        score = cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        from tools.skill_matcher import cosine_similarity
        
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        score = cosine_similarity(vec_a, vec_b)
        assert abs(score) < 0.001
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        from tools.skill_matcher import cosine_similarity
        
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 0.0]
        score = cosine_similarity(vec_a, vec_b)
        assert score == 0.0
    
    def test_match_skills_empty(self):
        """Test skill matching with empty inputs."""
        from tools.skill_matcher import match_skills
        
        result = match_skills([], ["Python"])
        assert result["match_percentage"] == 0
        assert len(result["missing"]) == 1
