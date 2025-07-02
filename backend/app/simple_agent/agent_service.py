from app.simple_agent.agent import agent_pipeline
import logging

logger = logging.getLogger(__name__)

class AgentService:
    """Service class for handling agent operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def search(self, user_id: str, prompt: str):
        """
        Search using the agent pipeline
        
        Args:
            user_id: The user identifier
            prompt: The search prompt
            
        Yields:
            Events from the agent pipeline
        """
        try:
            self.logger.info(f"Starting search for user_id={user_id}")
            async for event in agent_pipeline(user_id, prompt):
                yield event
        except Exception as e:
            self.logger.error(f"Error in agent search: {str(e)}")
            yield {
                "type": "error",
                "message": "An error occurred during search"
            } 