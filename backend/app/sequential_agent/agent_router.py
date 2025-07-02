from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.sequential_agent.agentic_graph import graph
from app.sequential_agent.agentic_state import ResearchState
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/agent/answer")
async def agent_answer(user_id: str, prompt: str):
    state = ResearchState(user_id=user_id, prompt=prompt)
    result = await graph.ainvoke(state)
    return result

@router.websocket("/ws/agent")
async def agentic_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time agentic research results"""
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
        logger.info(f"Agentic WebSocket connection for user_id={user_id} with prompt={prompt}")
        await websocket.send_json({"type": "thinking"})
        # Run the agentic workflow
        state = ResearchState(user_id=user_id, prompt=prompt)
        result = await graph.ainvoke(state)
        logger.info(f"Agentic result: {result}")
        # Use dict-style access for result
        # Send rationale (plain text)
        rationale = result.get("rationale") if isinstance(result, dict) else getattr(result, "rationale", None)
        if rationale:
            await websocket.send_json({"type": "rationale_complete", "rationale": rationale})
        # Send rationale as annotated HTML if available
        rationale_html = result.get("rationale_html") if isinstance(result, dict) else getattr(result, "rationale_html", None)
        if rationale_html:
            await websocket.send_json({"type": "rationale_annotated_html", "rationale_html": rationale_html})
        # Send answer (plain text)
        answer = result.get("answer") if isinstance(result, dict) else getattr(result, "answer", None)
        if answer:
            await websocket.send_json({"type": "answer_complete", "answer": answer})
        # Send answer as annotated HTML if available
        answer_html = result.get("answer_html") if isinstance(result, dict) else getattr(result, "answer_html", None)
        if answer_html:
            await websocket.send_json({"type": "answer_annotated_html", "answer_html": answer_html})
        # Send citations
        citations = result.get("citations") if isinstance(result, dict) else getattr(result, "citations", None)
        if citations:
            await websocket.send_json({"type": "citations", "citations": citations})
        # Optionally send history (not used by frontend, but kept for debugging)
        # history = result.get("history") if isinstance(result, dict) else getattr(result, "history", None)
        # if history:
        #     await websocket.send_json({"type": "history", "history": history})
        # Signal completion
        await websocket.send_json({"type": "done"})
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("Agentic WebSocket client disconnected")
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid JSON format"
        })
        await websocket.close()
    except Exception as e:
        logger.error(f"Error in Agentic WebSocket endpoint: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": "Internal server error"
        })
        await websocket.close() 