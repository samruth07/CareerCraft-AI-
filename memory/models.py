"""
CareerCraft AI - Database Models
Demonstrates: Memory + Persistence (Syllabus Topic #5)

SQLAlchemy models for persisting user data, analysis results,
and conversation history across sessions.
"""

import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from config.settings import settings

Base = declarative_base()


class AnalysisHistory(Base):
    """Stores the results of each career analysis session."""
    
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Input data
    resume_filename = Column(String(255))
    target_role = Column(String(255))
    job_description = Column(Text)
    
    # Parsed resume (JSON)
    parsed_resume = Column(JSON)
    
    # Analysis results (JSON)
    gap_analysis = Column(JSON)
    match_percentage = Column(Float)
    
    # Roadmap (JSON)
    roadmap = Column(JSON)
    
    # Interview prep (JSON)
    interview_prep = Column(JSON)
    
    def __repr__(self):
        return (
            f"<AnalysisHistory(id={self.id}, role='{self.target_role}', "
            f"match={self.match_percentage}%)>"
        )


class ConversationLog(Base):
    """Stores conversation messages for memory persistence."""
    
    __tablename__ = "conversation_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    role = Column(String(50))  # 'user', 'assistant', 'system'
    content = Column(Text)
    agent_name = Column(String(100))  # Which agent generated this
    
    def __repr__(self):
        return (
            f"<ConversationLog(id={self.id}, role='{self.role}', "
            f"agent='{self.agent_name}')>"
        )


# Database initialization
def get_engine():
    """Create database engine."""
    return create_engine(settings.database_url, echo=False)


def get_session():
    """Create a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ Database initialized successfully")
