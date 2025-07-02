import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.utils.llm import ground_context, llm_annotate_with_citations
from app.simple_agent.agent import agent_pipeline
from app.simple_agent.agent_service import AgentService

class TestContextGrounding:
    def test_ground_context_success(self, mock_llm):
        prompt = "What is machine learning?"
        context = "Machine learning is a subset of AI."
        result = ground_context(context, prompt, mock_llm)
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    def test_ground_context_error_handling(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("LLM error")
        with pytest.raises(Exception, match="LLM error"):
            ground_context("test context", "test prompt", mock_llm)
    def test_ground_context_empty_context(self, mock_llm):
        result = ground_context("", "test prompt", mock_llm)
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()

class TestCitationAnnotation:
    def test_llm_annotate_with_citations_success(self, mock_llm):
        answer = "Machine learning is a subset of artificial intelligence."
        memories = [
            {'id': 'mem_001', 'content': 'Machine learning enables computers to learn from data.'},
            {'id': 'mem_002', 'content': 'AI encompasses various technologies including ML.'}
        ]
        result = llm_annotate_with_citations(answer, memories, mock_llm)
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    def test_llm_annotate_with_citations_no_memories(self, mock_llm):
        answer = "This is a standalone answer."
        memories = []
        result = llm_annotate_with_citations(answer, memories, mock_llm)
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    def test_llm_annotate_with_citations_error_handling(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("LLM error")
        answer = "Test answer"
        memories = [{'id': 'mem_001', 'content': 'Test memory'}]
        with pytest.raises(Exception, match="LLM error"):
            llm_annotate_with_citations(answer, memories, mock_llm)
    def test_llm_annotate_with_citations_complex_answer(self, mock_llm):
        answer = """
        Machine learning has revolutionized many fields. Deep learning, a subset of ML, 
        uses neural networks with multiple layers. Natural language processing enables 
        computers to understand human language. These technologies work together to 
        create intelligent systems.
        """
        memories = [
            {'id': 'mem_001', 'content': 'Machine learning algorithms learn patterns from data.'},
            {'id': 'mem_002', 'content': 'Deep learning uses artificial neural networks.'},
            {'id': 'mem_003', 'content': 'NLP focuses on language understanding and generation.'}
        ]
        result = llm_annotate_with_citations(answer, memories, mock_llm)
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    def test_llm_annotate_with_citations_memory_formatting(self, mock_llm):
        answer = "Test answer"
        memories = [
            {
                'id': 'mem_001',
                'content': 'Memory content with special characters: < > & " ''',
                'title': 'Memory Title'
            }
        ]
        result = llm_annotate_with_citations(answer, memories, mock_llm)
        assert result == "Mocked LLM response"
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_str = call_args[0]['content']
        assert 'Memory content with special characters' in prompt_str

class TestAsyncGeneration:
    @pytest.mark.asyncio
    async def test_agent_pipeline_basic_flow(self, mock_llm, mock_mem0_client):
        with patch('app.utils.memory.mem0_client', mock_mem0_client), \
             patch('app.simple_agent.agent.ChatOpenAI', return_value=mock_llm):
            user_id = "test_user"
            prompt = "What is machine learning?"
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            assert len(events) > 0
            event_types = [event.get('type') for event in events]
            assert 'rationale_complete' in event_types
            assert 'answer_complete' in event_types
            assert 'citations' in event_types
            assert 'done' in event_types
    @pytest.mark.asyncio
    async def test_agent_pipeline_with_memories(self, mock_llm, mock_mem0_client, sample_memories):
        mock_mem0_client.get_all.return_value = {'results': sample_memories}
        with patch('app.utils.memory.mem0_client', mock_mem0_client), \
             patch('app.simple_agent.agent.ChatOpenAI', return_value=mock_llm):
            user_id = "test_user"
            prompt = "Explain neural networks"
            events = []
            async for event in agent_pipeline(user_id, prompt):
                events.append(event)
            mock_mem0_client.get_all.assert_called_once_with(user_id=user_id)
            mock_mem0_client.add.assert_called_once()
    @pytest.mark.asyncio
    async def test_agent_pipeline_error_handling(self, mock_llm, mock_mem0_client):
        mock_llm.astream.side_effect = Exception("LLM error")
        with patch('app.utils.memory.mem0_client', mock_mem0_client), \
             patch('app.simple_agent.agent.ChatOpenAI', return_value=mock_llm):
            user_id = "test_user"
            prompt = "Test prompt"
            events = []
            try:
                async for event in agent_pipeline(user_id, prompt):
                    events.append(event)
            except Exception:
                pass
            mock_mem0_client.add.assert_called_once()

class TestPromptTemplates:
    def test_prompt_templates_import(self):
        from app.prompts import ANSWER_GENERATOR_PROMPT, GROUND_CONTEXT_PROMPT, REASONING_PROMPT
        assert '{context}' in GROUND_CONTEXT_PROMPT.template
        assert '{prompt}' in GROUND_CONTEXT_PROMPT.template
        assert '{grounded_context}' in REASONING_PROMPT.template
        assert '{context}' in ANSWER_GENERATOR_PROMPT.template
        assert '{rationale}' in ANSWER_GENERATOR_PROMPT.template
    def test_prompt_formatting(self):
        from app.prompts import GROUND_CONTEXT_PROMPT
        context = "Test context"
        prompt = "Test prompt"
        formatted = GROUND_CONTEXT_PROMPT.format(context=context, prompt=prompt)
        assert "Test context" in formatted
        assert "Test prompt" in formatted
        assert "{context}" not in formatted
        assert "{prompt}" not in formatted 