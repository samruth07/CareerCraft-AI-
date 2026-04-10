"""
CareerCraft AI - Agent Tests
Tests for individual agents and the multi-agent supervisor.
"""

import pytest
import json


class TestResumeParser:
    """Tests for the resume parser agent."""
    
    def test_extract_json_valid(self):
        """Test JSON extraction from clean response."""
        from agents.resume_parser import _extract_json
        
        json_str = '{"name": "John Doe", "skills": {"technical": ["Python"]}}'
        result = _extract_json(json_str)
        assert result["name"] == "John Doe"
        assert "Python" in result["skills"]["technical"]
    
    def test_extract_json_with_code_block(self):
        """Test JSON extraction from markdown code block."""
        from agents.resume_parser import _extract_json
        
        text = 'Here is the result:\n```json\n{"name": "Jane"}\n```'
        result = _extract_json(text)
        assert result["name"] == "Jane"
    
    def test_extract_json_embedded(self):
        """Test JSON extraction from text with embedded JSON."""
        from agents.resume_parser import _extract_json
        
        text = 'The parsed resume is: {"name": "Test"} and that is it.'
        result = _extract_json(text)
        assert result["name"] == "Test"
    
    def test_extract_json_fallback(self):
        """Test fallback when JSON extraction fails."""
        from agents.resume_parser import _extract_json
        
        text = "This is not JSON at all"
        result = _extract_json(text)
        assert "_parse_error" in result
        assert isinstance(result["skills"], dict)


class TestGapAnalyzer:
    """Tests for the gap analyzer agent."""
    
    def test_extract_json(self):
        """Test gap analysis JSON extraction."""
        from agents.gap_analyzer import _extract_json
        
        data = {
            "match_percentage": 75,
            "matching_skills": ["Python", "SQL"],
            "missing_skills": {"critical": ["Docker"]},
        }
        result = _extract_json(json.dumps(data))
        assert result["match_percentage"] == 75
        assert "Python" in result["matching_skills"]


class TestRoadmapGenerator:
    """Tests for the roadmap generator agent."""
    
    def test_extract_json(self):
        """Test roadmap JSON extraction."""
        from agents.roadmap_generator import _extract_json
        
        data = {
            "roadmap_title": "Test Roadmap",
            "total_duration_weeks": 8,
            "phases": [{"phase_number": 1, "title": "Foundations"}],
        }
        result = _extract_json(json.dumps(data))
        assert result["total_duration_weeks"] == 8
        assert len(result["phases"]) == 1


class TestInterviewCoach:
    """Tests for the interview coach agent."""
    
    def test_extract_json(self):
        """Test interview prep JSON extraction."""
        from agents.interview_coach import _extract_json
        
        data = {
            "role": "Software Engineer",
            "questions": {
                "behavioral": [{"question": "Tell me about yourself"}],
                "technical": [{"question": "What is REST?"}],
            },
        }
        result = _extract_json(json.dumps(data))
        assert result["role"] == "Software Engineer"


class TestSupervisor:
    """Tests for the LangGraph supervisor."""
    
    def test_should_continue_after_parse(self):
        """Test routing after resume parsing."""
        from agents.supervisor import should_continue
        
        state = {"current_step": "resume_parsed", "errors": []}
        assert should_continue(state) == "analyze_gaps"
    
    def test_should_continue_after_gaps(self):
        """Test routing after gap analysis."""
        from agents.supervisor import should_continue
        
        state = {"current_step": "gaps_analyzed", "errors": []}
        assert should_continue(state) == "generate_roadmap"
    
    def test_should_continue_after_roadmap(self):
        """Test routing after roadmap generation."""
        from agents.supervisor import should_continue
        
        state = {"current_step": "roadmap_generated", "errors": []}
        assert should_continue(state) == "prepare_interview"
    
    def test_should_stop_after_interview(self):
        """Test pipeline ends after interview prep."""
        from agents.supervisor import should_continue
        
        state = {"current_step": "interview_prepared", "errors": []}
        assert should_continue(state) == "end"
    
    def test_should_stop_on_too_many_errors(self):
        """Test pipeline stops after 3 errors."""
        from agents.supervisor import should_continue
        
        state = {
            "current_step": "resume_parsed",
            "errors": ["err1", "err2", "err3"],
        }
        assert should_continue(state) == "end"
    
    def test_build_career_graph(self):
        """Test that the LangGraph compiles successfully."""
        from agents.supervisor import build_career_graph
        
        app = build_career_graph()
        assert app is not None
