"""
CareerCraft AI - Conversation Memory
Demonstrates: Memory + LangChain (Syllabus Topics #5, #6)

Implements LangChain conversation memory with SQLite backend.
This allows the AI to remember context across multiple interactions
within a session, creating a more natural conversation flow.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memory.persistence import PersistenceManager


class ConversationMemory:
    """
    Manages conversation memory for CareerCraft AI sessions.
    
    Combines in-memory buffer for fast access with SQLite persistence
    for durability. Supports LangChain message format integration.
    """

    def __init__(self, session_id: str, max_messages: int = 20):
        """
        Initialize conversation memory for a session.
        
        Args:
            session_id: Unique session identifier
            max_messages: Maximum messages to keep in buffer (sliding window)
        """
        self.session_id = session_id
        self.max_messages = max_messages
        self.persistence = PersistenceManager()
        self.messages: list[dict] = []
        
        # Load existing conversation from database
        self._load_from_db()

    def _load_from_db(self):
        """Load conversation history from database into memory buffer."""
        history = self.persistence.get_conversation_history(
            self.session_id, limit=self.max_messages
        )
        self.messages = history

    def add_user_message(self, content: str):
        """Add a user message to memory and persist."""
        msg = {"role": "user", "content": content}
        self.messages.append(msg)
        self._trim_buffer()
        self.persistence.save_conversation(
            self.session_id, "user", content
        )

    def add_ai_message(self, content: str, agent_name: str = ""):
        """Add an AI response to memory and persist."""
        msg = {"role": "assistant", "content": content, "agent_name": agent_name}
        self.messages.append(msg)
        self._trim_buffer()
        self.persistence.save_conversation(
            self.session_id, "assistant", content, agent_name
        )

    def add_system_message(self, content: str):
        """Add a system message to memory."""
        msg = {"role": "system", "content": content}
        self.messages.append(msg)
        self._trim_buffer()

    def _trim_buffer(self):
        """Keep only the most recent messages within the window."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_langchain_messages(self) -> list:
        """
        Convert memory buffer to LangChain message format.
        
        Returns:
            list: LangChain message objects for use in chains
        """
        lc_messages = []
        for msg in self.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
        return lc_messages

    def get_context_summary(self) -> str:
        """
        Get a brief summary of conversation context.
        
        Returns:
            str: Formatted conversation context for agent prompts
        """
        if not self.messages:
            return "No previous conversation."
        
        recent = self.messages[-5:]  # Last 5 messages
        summary_parts = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:200]  # Truncate long messages
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)

    def clear(self):
        """Clear the memory buffer (does not delete from DB)."""
        self.messages = []
