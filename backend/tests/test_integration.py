import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import your app and components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.services.agent_service import AgentService
from app.agent import agent_pipeline


class TestServiceLayerIntegration:
    """Test service layer integration"""
    
    @pytest.mark.asyncio
    async def test_agent_service_search_success(self, mock_llm, mock_mem0_client):
        """Test successful search through AgentService"""
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            service = AgentService()
            user_id = "test_user"
            prompt = "What is machine learning?"
            
            events = []
            async for event in service.search(user_id, prompt):
                events.append(event)
            
            # Accept any event with a 'type' or a mock with content 'thinking'
            assert any(
                (isinstance(event, dict) and event.get('type') in ['thinking', 'rationale_token', 'rationale_complete', 'rationale_annotated_html', 'answer_token', 'answer_complete', 'answer_annotated_html', 'citations', 'done']) or
                (hasattr(event, 'content') and event.content == 'thinking')
                for event in events
            )
    
    @pytest.mark.asyncio
    async def test_agent_service_search_error_handling(self, mock_llm, mock_mem0_client):
        """Test error handling in AgentService search"""
        # Set a side effect to simulate an error in astream
        mock_llm.astream.side_effect = Exception("Service error")
        
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            service = AgentService()
            user_id = "test_user"
            prompt = "Test prompt"
            
            events = []
            async for event in service.search(user_id, prompt):
                events.append(event)
            
            assert any(
                isinstance(event, dict) and event.get('type') == 'error'
                for event in events
            )


class TestEndToEndWorkflow:
    """Test end-to-end workflow integration"""
    
    @pytest.mark.asyncio
    async def test_complete_search_workflow(self, mock_llm, mock_mem0_client):
        """Test complete search workflow from API to agent"""
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            # Test agent pipeline directly
            user_id = "test_user"
            prompt = "What is machine learning?"
            
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            
            assert len(events) > 0
            assert any(event.get('type') == 'done' for event in events)
    
    @pytest.mark.asyncio
    async def test_search_with_existing_memories(self, mock_llm, mock_mem0_client, sample_memories):
        """Test search workflow with existing memories"""
        mock_mem0_client.get_all.return_value = {'results': sample_memories}
        
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            user_id = "test_user"
            prompt = "Explain neural networks"
            
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            
            # Should have retrieved memories
            mock_mem0_client.get_all.assert_called_once_with(user_id=user_id)
            # Should have citations
            citation_events = [e for e in events if e.get('type') == 'citations']
            assert len(citation_events) > 0


class TestAgentPipeline:
    """Test the complete agent pipeline integration"""
    
    @pytest.mark.asyncio
    async def test_agent_pipeline_success(self, mock_llm, mock_mem0_client, sample_memories, sample_conversation_history):
        """Test successful execution of the complete agent pipeline"""
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            prompt = "What is machine learning?"
            user_id = "test_user"
            
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            
            assert len(events) > 0
            # Should have various event types
            event_types = [event.get('type') for event in events]
            assert 'rationale_complete' in event_types
            assert 'answer_complete' in event_types
            assert 'citations' in event_types
            assert 'done' in event_types
    
    @pytest.mark.asyncio
    async def test_agent_pipeline_with_memory_retrieval(self, mock_llm, mock_mem0_client, sample_memories):
        """Test agent pipeline with memory retrieval"""
        mock_mem0_client.get_all.return_value = {'results': sample_memories}
        
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            prompt = "Explain neural networks"
            user_id = "test_user"
            
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            
            assert len(events) > 0
            # Should have retrieved memories
            mock_mem0_client.get_all.assert_called_once_with(user_id=user_id)
    
    @pytest.mark.asyncio
    async def test_agent_pipeline_error_handling(self, mock_llm, mock_mem0_client):
        """Test agent pipeline error handling"""
        mock_llm.astream.side_effect = Exception("LLM error")
        
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            prompt = "Test prompt"
            user_id = "test_user"
            
            events = []
            try:
                async for event in agent_pipeline(user_id, prompt):
                    events.append(event)
            except Exception:
                # Error should be handled within the pipeline
                pass
            
            # Should have at least written to memory
            mock_mem0_client.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_pipeline_memory_write(self, mock_llm, mock_mem0_client):
        """Test that agent pipeline writes to memory"""
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            prompt = "What is AI?"
            user_id = "test_user"
            
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            
            # Should have written to memory
            mock_mem0_client.add.assert_called_once()
            call_args = mock_mem0_client.add.call_args[0][0]
            assert call_args[0]['role'] == 'user'
            assert call_args[0]['content'] == prompt
