from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.simple_agent.websocket import router as simple_ws_router
from app.simple_agent.search_router import router as simple_search_router
from app.simple_agent.agent_service import AgentService

from app.sequential_agent.agent_router import router as sequential_agent_router

from app.multiagent.router import router as multiagent_router

from app.core.config import settings

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Deep Research Memory API",
        description="A FastAPI application for deep research memory management",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers with clear prefixes
    app.include_router(simple_ws_router, prefix="/api/v1/simple")
    app.include_router(simple_search_router, prefix="/api/v1/simple")
    app.include_router(sequential_agent_router, prefix="/api/v1/sequential")
    app.include_router(multiagent_router, prefix="/api/v1/multiagent")
    
    @app.get("/")
    async def root():
        return {"message": "Deep Research Memory API is running"}
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
