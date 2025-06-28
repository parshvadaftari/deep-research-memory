from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.models import SearchRequest
from app.services.agent_service import AgentService
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search")
async def search(request: Request):
    """Search endpoint that streams results"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        prompt = data.get("prompt")
        
        if not user_id or not prompt:
            raise HTTPException(status_code=400, detail="user_id and prompt are required")
        
        logger.info(f"Search request for user_id={user_id} with prompt={prompt}")
        
        agent_service = AgentService()
        
        async def event_stream():
            async for event in agent_service.search(user_id, prompt):
                yield f"data: {json.dumps(event)}\n\n"
        
        return StreamingResponse(
            event_stream(), 
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 