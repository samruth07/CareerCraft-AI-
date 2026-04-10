"""
CareerCraft AI - RAG Tests
Tests for the RAG pipeline components.
"""

import pytest
import os
import json


class TestKnowledgeBaseData:
    """Tests to ensure knowledge base data is valid."""
    
    def test_skills_taxonomy_exists(self):
        """Test that skills taxonomy file exists and is valid JSON."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "knowledge_base", "skills_taxonomy.json"
        )
        assert os.path.exists(path), f"Skills taxonomy not found at {path}"
        
        with open(path, "r") as f:
            data = json.load(f)
        
        assert "categories" in data
        assert "role_skill_profiles" in data
        assert len(data["categories"]) > 0
        assert len(data["role_skill_profiles"]) > 0
    
    def test_job_descriptions_exist(self):
        """Test that job description files exist."""
        jd_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "knowledge_base", "job_descriptions"
        )
        assert os.path.exists(jd_dir)
        
        txt_files = [f for f in os.listdir(jd_dir) if f.endswith(".txt")]
        assert len(txt_files) >= 3, f"Expected at least 3 JD files, found {len(txt_files)}"
    
    def test_interview_questions_exist(self):
        """Test that interview question files exist and are valid JSON."""
        iq_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "knowledge_base", "interview_questions"
        )
        assert os.path.exists(iq_dir)
        
        json_files = [f for f in os.listdir(iq_dir) if f.endswith(".json")]
        assert len(json_files) >= 1
        
        for jf in json_files:
            path = os.path.join(iq_dir, jf)
            with open(path, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict)
    
    def test_skills_taxonomy_has_profiles(self):
        """Test role profiles have required fields."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "knowledge_base", "skills_taxonomy.json"
        )
        with open(path, "r") as f:
            data = json.load(f)
        
        for role_key, profile in data["role_skill_profiles"].items():
            assert "title" in profile, f"Missing title in {role_key}"
            assert "critical_skills" in profile, f"Missing critical_skills in {role_key}"
            assert "important_skills" in profile, f"Missing important_skills in {role_key}"
            assert len(profile["critical_skills"]) > 0


class TestRAGIngest:
    """Tests for the ingestion pipeline."""
    
    def test_load_skills_taxonomy(self):
        """Test skills taxonomy loading."""
        from rag.ingest import load_skills_taxonomy
        
        docs = load_skills_taxonomy()
        assert len(docs) > 0
        assert all(doc.page_content for doc in docs)
        assert all(doc.metadata.get("source") == "skills_taxonomy" for doc in docs)
    
    def test_load_job_descriptions(self):
        """Test job description loading."""
        from rag.ingest import load_job_descriptions
        
        docs = load_job_descriptions()
        assert len(docs) >= 3
        assert all(doc.metadata.get("source") == "job_description" for doc in docs)
    
    def test_load_interview_questions(self):
        """Test interview question loading."""
        from rag.ingest import load_interview_questions
        
        docs = load_interview_questions()
        assert len(docs) > 0
        assert all(doc.metadata.get("source") == "interview_questions" for doc in docs)
