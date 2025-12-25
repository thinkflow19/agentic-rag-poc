"""Service for RAG queries."""

import logging
from app.core.agent.agent import create_agent_executor

logger = logging.getLogger(__name__)


class QueryService:
    """Handles RAG queries through the agent."""
    
    def __init__(self):
        """Initialize query service."""
        self._agent_executor = None
    
    def _get_agent(self):
        """
        Get or create agent executor (singleton pattern).
        
        Returns:
            Agent executor function
            
        Raises:
            Exception: If agent initialization fails
        """
        if self._agent_executor is None:
            try:
                self._agent_executor = create_agent_executor()
                logger.info("Agent executor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize agent: {e}", exc_info=True)
                raise
        return self._agent_executor
    
    def process_query(self, query: str, chat_history: list = None) -> dict:
        """
        Process a query through the agent.
        
        Args:
            query: User query string
            chat_history: Optional list of previous messages with 'role' and 'content' keys
            
        Returns:
            Dictionary with:
            - response: Agent's response string
            - status: 'success'
        """
        if chat_history is None:
            chat_history = []
        
        agent = self._get_agent()
        result = agent({
            'input': query,
            'chat_history': chat_history
        })
        
        return {
            'response': result.get('output', ''),
            'status': 'success'
        }
    
    def is_ready(self) -> bool:
        """
        Check if agent is ready to process queries.
        
        Returns:
            True if agent can be initialized, False otherwise
        """
        try:
            self._get_agent()
            return True
        except Exception:
            return False

