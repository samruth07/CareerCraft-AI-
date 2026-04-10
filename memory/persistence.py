"""
CareerCraft AI - Persistence Layer
Demonstrates: Memory + Persistence (Syllabus Topic #5)

CRUD operations for saving and retrieving analysis history
and conversation logs from the SQLite database.
"""

import json
from typing import Optional
from memory.models import (
    AnalysisHistory,
    ConversationLog,
    get_session,
    init_database,
)


class PersistenceManager:
    """Manages all database operations for CareerCraft AI."""

    def __init__(self):
        """Initialize the persistence manager and ensure tables exist."""
        init_database()

    def save_analysis(
        self,
        session_id: str,
        resume_filename: str,
        target_role: str,
        job_description: str,
        parsed_resume: dict,
        gap_analysis: dict,
        match_percentage: float,
        roadmap: dict = None,
        interview_prep: dict = None,
    ) -> int:
        """
        Save a complete analysis result to the database.
        
        Args:
            session_id: Unique session identifier
            resume_filename: Name of uploaded resume file
            target_role: Target job role
            job_description: Job description text
            parsed_resume: Parsed resume data (dict)
            gap_analysis: Gap analysis results (dict)
            match_percentage: Overall match percentage
            roadmap: Learning roadmap (dict, optional)
            interview_prep: Interview prep data (dict, optional)
            
        Returns:
            int: ID of the saved record
        """
        session = get_session()
        try:
            record = AnalysisHistory(
                session_id=session_id,
                resume_filename=resume_filename,
                target_role=target_role,
                job_description=job_description,
                parsed_resume=parsed_resume,
                gap_analysis=gap_analysis,
                match_percentage=match_percentage,
                roadmap=roadmap,
                interview_prep=interview_prep,
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_analysis_history(
        self, session_id: str = None, limit: int = 10
    ) -> list[dict]:
        """
        Retrieve analysis history, optionally filtered by session.
        
        Args:
            session_id: Filter by session ID (optional)
            limit: Maximum number of records to return
            
        Returns:
            list[dict]: List of analysis records
        """
        session = get_session()
        try:
            query = session.query(AnalysisHistory)
            if session_id:
                query = query.filter(AnalysisHistory.session_id == session_id)
            
            records = (
                query.order_by(AnalysisHistory.timestamp.desc())
                .limit(limit)
                .all()
            )
            
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "resume_filename": r.resume_filename,
                    "target_role": r.target_role,
                    "match_percentage": r.match_percentage,
                    "parsed_resume": r.parsed_resume,
                    "gap_analysis": r.gap_analysis,
                    "roadmap": r.roadmap,
                    "interview_prep": r.interview_prep,
                }
                for r in records
            ]
        finally:
            session.close()

    def save_conversation(
        self, session_id: str, role: str, content: str, agent_name: str = ""
    ):
        """
        Save a conversation message to the database.
        
        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            agent_name: Name of the agent that generated the message
        """
        session = get_session()
        try:
            log = ConversationLog(
                session_id=session_id,
                role=role,
                content=content,
                agent_name=agent_name,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> list[dict]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            
        Returns:
            list[dict]: Conversation messages in chronological order
        """
        session = get_session()
        try:
            records = (
                session.query(ConversationLog)
                .filter(ConversationLog.session_id == session_id)
                .order_by(ConversationLog.timestamp.asc())
                .limit(limit)
                .all()
            )
            
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "agent_name": r.agent_name,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in records
            ]
        finally:
            session.close()

    def get_latest_analysis(self, session_id: str) -> Optional[dict]:
        """Get the most recent analysis for a session."""
        history = self.get_analysis_history(session_id=session_id, limit=1)
        return history[0] if history else None
