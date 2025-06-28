import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.agent import (
    ground_context,
    generate_rationale,
    generate_answer,
    llm_annotate_with_citations,
    agent_pipeline
)


class TestContextGrounding:
    """Test context grounding functionality"""
    
    def test_ground_context_success(self, mock_llm):
        """Test successful context grounding"""
        prompt = "What is machine learning?"
        context = "Machine learning is a subset of AI."
        
        result = ground_context(context, prompt, mock_llm)
        
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    
    def test_ground_context_error_handling(self, mock_llm):
        """Test error handling in context grounding"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            ground_context("test context", "test prompt", mock_llm)
    
    def test_ground_context_empty_context(self, mock_llm):
        """Test context grounding with empty context"""
        result = ground_context("", "test prompt", mock_llm)
        
        assert result == "Mocked LLM response"
        # Should still call LLM even with empty context
        mock_llm.invoke.assert_called_once()


class TestRationaleGeneration:
    """Test rationale generation functionality"""
    
    def test_generate_rationale_success(self, mock_llm):
        """Test successful rationale generation"""
        grounded_context = "AI has both positive and negative impacts."
        prompt = "What is the impact of AI on society?"
        
        result = generate_rationale(grounded_context, prompt, mock_llm)
        
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    
    def test_generate_rationale_error_handling(self, mock_llm):
        """Test error handling in rationale generation"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            generate_rationale("test context", "test prompt", mock_llm)
    
    def test_generate_rationale_complex_prompt(self, mock_llm):
        """Test rationale generation with complex prompts"""
        grounded_context = "Autonomous vehicles raise questions about safety, liability, and job displacement."
        prompt = "Analyze the ethical implications of autonomous vehicles in urban environments"
        
        result = generate_rationale(grounded_context, prompt, mock_llm)
        
        assert result == "Mocked LLM response"
        # Should handle complex prompts without issues
        mock_llm.invoke.assert_called_once()


class TestAnswerGeneration:
    """Test answer generation functionality"""
    
    def test_generate_answer_success(self, mock_llm):
        """Test successful answer generation"""
        grounded_context = "France is a country in Europe."
        rationale = "Paris is the capital city of France."
        prompt = "What is the capital of France?"
        
        result = generate_answer(grounded_context, rationale, prompt, mock_llm)
        
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    
    def test_generate_answer_error_handling(self, mock_llm):
        """Test error handling in answer generation"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            generate_answer("test context", "test rationale", "test prompt", mock_llm)
    
    def test_generate_answer_without_rationale(self, mock_llm):
        """Test answer generation without rationale"""
        result = generate_answer("test context", "", "test prompt", mock_llm)
        
        assert result == "Mocked LLM response"
        # Should still work without rationale
        mock_llm.invoke.assert_called_once()


class TestCitationAnnotation:
    """Test citation annotation functionality"""
    
    def test_llm_annotate_with_citations_success(self, mock_llm):
        """Test successful citation annotation"""
        answer = "Machine learning is a subset of artificial intelligence."
        memories = [
            {'id': 'mem_001', 'content': 'Machine learning enables computers to learn from data.'},
            {'id': 'mem_002', 'content': 'AI encompasses various technologies including ML.'}
        ]
        
        result = llm_annotate_with_citations(answer, memories, mock_llm)
        
        assert result == "Mocked LLM response"
        mock_llm.invoke.assert_called_once()
    
    def test_llm_annotate_with_citations_no_memories(self, mock_llm):
        """Test citation annotation with no memories"""
        answer = "This is a standalone answer."
        memories = []
        
        result = llm_annotate_with_citations(answer, memories, mock_llm)
        
        assert result == "Mocked LLM response"
        # Should still call LLM even with no memories
        mock_llm.invoke.assert_called_once()
    
    def test_llm_annotate_with_citations_error_handling(self, mock_llm):
        """Test error handling in citation annotation"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        answer = "Test answer"
        memories = [{'id': 'mem_001', 'content': 'Test memory'}]
        
        with pytest.raises(Exception, match="LLM error"):
            llm_annotate_with_citations(answer, memories, mock_llm)
    
    def test_llm_annotate_with_citations_complex_answer(self, mock_llm):
        """Test citation annotation with complex answers"""
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
        """Test that memory formatting is correct in citation annotation"""
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
        # Check that memory content is properly formatted in the prompt
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_str = call_args[0]['content']
        assert 'Memory content with special characters' in prompt_str


class TestAsyncGeneration:
    """Test async generation functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_pipeline_basic_flow(self, mock_llm, mock_mem0_client):
        """Test basic agent pipeline flow"""
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            user_id = "test_user"
            prompt = "What is machine learning?"
            
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
    async def test_agent_pipeline_with_memories(self, mock_llm, mock_mem0_client, sample_memories):
        """Test agent pipeline with memory retrieval"""
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
            # Should have written to memory
            mock_mem0_client.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_pipeline_error_handling(self, mock_llm, mock_mem0_client):
        """Test agent pipeline error handling"""
        mock_llm.astream.side_effect = Exception("LLM error")
        
        with patch('app.agent.mem0_client', mock_mem0_client), \
             patch('app.agent.ChatOpenAI', return_value=mock_llm):
            
            user_id = "test_user"
            prompt = "Test prompt"
            
            # Should handle errors gracefully
            events = []
            try:
                async for event in agent_pipeline(user_id, prompt):
                    events.append(event)
            except Exception:
                # Error should be handled within the pipeline
                pass
            
            # Should have at least written to memory
            mock_mem0_client.add.assert_called_once()


class TestPromptTemplates:
    """Test prompt template functionality"""
    
    def test_prompt_templates_import(self):
        """Test that prompt templates can be imported"""
        from app.prompts import ANSWER_GENERATOR_PROMPT, GROUND_CONTEXT_PROMPT, REASONING_PROMPT
        # Check that prompts contain expected placeholders in their template string
        assert '{context}' in GROUND_CONTEXT_PROMPT.template
        assert '{prompt}' in GROUND_CONTEXT_PROMPT.template
        assert '{grounded_context}' in REASONING_PROMPT.template
        assert '{context}' in ANSWER_GENERATOR_PROMPT.template
        assert '{rationale}' in ANSWER_GENERATOR_PROMPT.template
    
    def test_prompt_formatting(self):
        """Test prompt template formatting"""
        from app.prompts import GROUND_CONTEXT_PROMPT
        
        context = "Test context"
        prompt = "Test prompt"
        
        formatted = GROUND_CONTEXT_PROMPT.format(context=context, prompt=prompt)
        
        assert "Test context" in formatted
        assert "Test prompt" in formatted
        assert "{context}" not in formatted  # Should be replaced
        assert "{prompt}" not in formatted   # Should be replaced 