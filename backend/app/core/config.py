from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Deep Research Memory API"
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./research_agent_conversations.db"
    
    # Memory Settings
    MEMORY_DB_PATH: str = "./db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # This will ignore extra fields instead of raising errors
    )

settings = Settings() 