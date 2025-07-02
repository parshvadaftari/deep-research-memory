import pytest
from unittest.mock import patch, Mock, AsyncMock
from app.multiagent.agents import supervisor_agent, memory_agent, conversation_agent, context_agent, reasoning_agent, answer_agent, citation_agent
from app.multiagent.state import MultiAgentState

class TestMultiAgentPipeline:
    @pytest.mark.asyncio
    async def test_supervisor_agent_direct_answer(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?", memories=[{'id': 'mem_1', 'memory': 'AI is intelligence.'}])
        new_state = await supervisor_agent(state)
        assert hasattr(new_state, 'answer')
        assert new_state.answer == 'AI is intelligence.' or 'clarification' in new_state.answer.lower() or 'need more information' in new_state.answer.lower()
    @pytest.mark.asyncio
    async def test_supervisor_agent_correction(self):
        state = MultiAgentState(user_id="test_user", prompt="This is a rumor.", memories=[{'id': 'mem_1', 'memory': 'AI is intelligence.'}])
        new_state = await supervisor_agent(state)
        assert hasattr(new_state, 'answer')
        assert 'Correction:' in new_state.answer
    @pytest.mark.asyncio
    async def test_supervisor_agent_clarification(self):
        state = MultiAgentState(user_id="test_user", prompt="Unknown question.", memories=[])
        new_state = await supervisor_agent(state)
        assert hasattr(new_state, 'answer')
        assert 'clarification' in new_state.answer.lower() or 'need more information' in new_state.answer.lower()
    @pytest.mark.asyncio
    async def test_memory_agent(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?")
        with patch('app.multiagent.agents.get_all_memories', return_value=[{'id': 'mem_1', 'memory': 'AI is intelligence.'}]), \
             patch('app.multiagent.agents.write_memory') as mock_write:
            new_state = await memory_agent(state)
            assert hasattr(new_state, 'memories')
            assert new_state.memories[0]['memory'] == 'AI is intelligence.'
            mock_write.assert_called_once()
    @pytest.mark.asyncio
    async def test_conversation_agent(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?")
        with patch('app.multiagent.agents.fetch_conversation_history', return_value=[('user', 'Hi', '123')]):
            new_state = await conversation_agent(state)
            assert hasattr(new_state, 'conversations')
            assert any('Hi' in c for c in [x[1] for x in new_state.conversations])
    @pytest.mark.asyncio
    async def test_context_agent(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?", memories=[{'id': 'mem_1', 'memory': 'AI is intelligence.'}], conversations=[('user', 'Hi', '123')])
        with patch('app.utils.context.format_context', return_value="context string"):
            new_state = await context_agent(state)
            assert hasattr(new_state, 'context')
            assert 'context' in new_state.context or '*Past Conversations:*' in new_state.context
    @pytest.mark.asyncio
    async def test_reasoning_agent(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?", context="context string")
        mock_llm = AsyncMock()
        mock_llm.astream = AsyncMock(return_value=iter([Mock(content="rationale")]))
        with patch('app.utils.llm.get_llm', return_value=mock_llm):
            new_state = await reasoning_agent(state)
            assert hasattr(new_state, 'rationale')
            assert isinstance(new_state.rationale, str) and len(new_state.rationale) > 0
    @pytest.mark.asyncio
    async def test_answer_agent(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?", context="context string", rationale="rationale string")
        mock_llm = AsyncMock()
        mock_llm.astream = AsyncMock(return_value=iter([Mock(content="answer")]))
        with patch('app.utils.llm.get_llm', return_value=mock_llm):
            new_state = await answer_agent(state)
            assert hasattr(new_state, 'answer')
            assert isinstance(new_state.answer, str) and len(new_state.answer) > 0
    @pytest.mark.asyncio
    async def test_citation_agent(self):
        state = MultiAgentState(user_id="test_user", prompt="What is AI?", memories=[{'id': 'mem_1', 'memory': 'AI is intelligence.'}], answer="AI is intelligence.")
        with patch('app.utils.memory.fetch_cited_memories', return_value=[{'id': 'mem_1', 'memory': 'AI is intelligence.'}]), \
             patch('app.utils.llm.llm_annotate_with_citations', return_value="AI is intelligence. [1]"):
            new_state = await citation_agent(state)
            assert hasattr(new_state, 'citations')
            assert new_state.citations[0]['id'] == 'mem_1'
            assert 'AI is intelligence.' in new_state.answer or 'AI is intelligence.' in getattr(new_state, 'answer_html', '') 