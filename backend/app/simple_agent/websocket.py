from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.simple_agent.agent_service import AgentService
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time search results"""
    await websocket.accept()
    
    try:
        # Receive initial data
        data = await websocket.receive_text()
        data = json.loads(data)
        user_id = data.get("user_id")
        prompt = data.get("prompt")
        
        if not user_id or not prompt:
            await websocket.send_json({
                "type": "error",
                "message": "user_id and prompt are required"
            })
            await websocket.close()
            return
        
        logger.info(f"WebSocket connection for user_id={user_id} with prompt={prompt}")
        
        # Send thinking event
        await websocket.send_json({"type": "thinking"})
        
        # Stream results
        agent_service = AgentService()
        async for event in agent_service.search(user_id, prompt):
            await websocket.send_json(event)
        
        await websocket.close()
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid JSON format"
        })
        await websocket.close()
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": "Internal server error"
        })
        await websocket.close() 