import pytest
import asyncio
import tempfile
import os
import sqlite3
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from typing import Generator, AsyncGenerator

# Mock mem0 client before importing app modules
with patch('mem0.Memory') as mock_memory:
    mock_memory.from_config.return_value = Mock()
    
    # Import your app and components
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from main import app
    from app.utils.database import fetch_conversation_history, store_conversation
    from app.utils.memory import write_memory, fetch_cited_memories
    from app.utils.search import bm25_hybrid_search
    from app.utils.llm import ground_context, llm_annotate_with_citations
    from app.utils.context import format_context
    from app.agent import agent_pipeline
    from app.services.agent_service import AgentService
    from app.core.config import Settings


@pytest.fixture
def test_client() -> Generator:
    """Create a test client for the FastAPI app"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    return Settings(
        API_V1_STR="/api/v1",
        PROJECT_NAME="Test Deep Research Memory API",
        OPENAI_API_KEY="test-key",
        ALLOWED_ORIGINS=["http://localhost:3000"],
        DATABASE_URL="sqlite:///./test_conversations.db",
        MEMORY_DB_PATH="./test_db",
        LOG_LEVEL="DEBUG"
    )


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Create test database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            user_id TEXT, role TEXT, content TEXT, timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def sample_memories():
    """Sample memory data for testing"""
    return [
        {
            'id': 'mem_001',
            'memory': 'Machine learning is a subset of artificial intelligence.',
            'created_at': '2024-01-01T10:00:00Z',
            'updated_at': '2024-01-01T10:00:00Z'
        },
        {
            'id': 'mem_002',
            'memory': 'Deep learning uses neural networks with multiple layers.',
            'created_at': '2024-01-02T10:00:00Z',
            'updated_at': '2024-01-02T10:00:00Z'
        },
        {
            'id': 'mem_003',
            'memory': 'Natural language processing helps computers understand human language.',
            'created_at': '2024-01-03T10:00:00Z',
            'updated_at': '2024-01-03T10:00:00Z'
        }
    ]


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing"""
    return [
        ('user', 'What is machine learning?', '1640995200.0'),
        ('agent', 'Machine learning is a subset of AI that enables computers to learn from data.', '1640995200.0'),
        ('user', 'How does deep learning work?', '1641081600.0'),
        ('agent', 'Deep learning uses neural networks with multiple layers to process data.', '1641081600.0')
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="Mocked LLM response"))
    
    # Create a proper async iterator for astream
    async def mock_astream(messages):
        # If the mock has a side effect set, raise it
        if hasattr(mock.astream, 'side_effect') and mock.astream.side_effect:
            raise mock.astream.side_effect
        # Yield a 'thinking' event for integration tests
        yield Mock(content="thinking")
        yield Mock(content="Mocked")
        yield Mock(content=" LLM")
        yield Mock(content=" response")
    
    mock.astream = mock_astream
    return mock


@pytest.fixture
def mock_mem0_client():
    """Mock mem0 client for testing"""
    mock = Mock()
    mock.add = Mock(return_value={'id': 'test_memory_id'})
    mock.get = Mock(return_value={
        'id': 'test_memory_id',
        'memory': 'Test memory content',
        'created_at': '2024-01-01T10:00:00Z'
    })
    mock.get_all = Mock(return_value={'results': []})
    return mock


@pytest.fixture
def agent_service():
    """Create an AgentService instance for testing"""
    return AgentService()


# Async test utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client():
    """Async test client for testing async endpoints"""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client 