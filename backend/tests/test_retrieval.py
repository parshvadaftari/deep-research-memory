import pytest
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from app.utils.database import fetch_conversation_history
from app.utils.search import bm25_hybrid_search
from app.utils.memory import fetch_cited_memories, write_memory
from app.utils.context import format_context


class TestConversationHistoryRetrieval:
    """Test conversation history retrieval functionality"""
    
    def test_fetch_conversation_history_empty(self, temp_db):
        """Test fetching conversation history when no conversations exist"""
        with patch('app.utils.database.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            
            result = fetch_conversation_history("test_user", limit=10)
            
            assert result == []
            # Should call execute twice (CREATE TABLE and SELECT)
            assert mock_cursor.execute.call_count == 2
    
    def test_fetch_conversation_history_with_data(self, temp_db):
        """Test fetching conversation history with existing data"""
        # Create test data - SQL query returns in DESC order by timestamp, then gets reversed
        # The function should return data in chronological order (oldest first)
        test_data_from_db = [
            ('agent', 'I am fine, thank you!', '1641081600.0'),
            ('user', 'How are you?', '1641081600.0'),
            ('agent', 'Hi there!', '1640995200.0'),
            ('user', 'Hello', '1640995200.0')
        ]
        
        # Expected result after reversal (chronological order)
        expected_result = [
            ('user', 'Hello', '1640995200.0'),
            ('agent', 'Hi there!', '1640995200.0'),
            ('user', 'How are you?', '1641081600.0'),
            ('agent', 'I am fine, thank you!', '1641081600.0')
        ]
        
        with patch('app.utils.database.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = test_data_from_db
            
            result = fetch_conversation_history("test_user", limit=5)
            
            assert len(result) == 4
            # Should return in chronological order (the function reverses DESC order from DB)
            assert result == expected_result
    
    def test_fetch_conversation_history_limit(self, temp_db):
        """Test that conversation history respects the limit parameter"""
        test_data = [
            ('user', f'Message {i}', f'1640995200.{i}') 
            for i in range(15)
        ]
        
        with patch('app.utils.database.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = test_data[:5]  # Mock DB returns limited results
            
            result = fetch_conversation_history("test_user", limit=5)
            
            assert len(result) == 5
            # Should call execute twice (CREATE TABLE and SELECT)
            assert mock_cursor.execute.call_count == 2


class TestBM25HybridSearch:
    """Test BM25 hybrid search functionality"""
    
    def test_bm25_search_empty_corpus(self):
        """Test BM25 search with empty document corpus"""
        memories = []
        conversation_history = []
        prompt = "test query"
        
        result = bm25_hybrid_search(prompt, memories, conversation_history)
        
        assert result == []
    
    def test_bm25_search_with_memories_only(self, sample_memories):
        """Test BM25 search with only memory documents"""
        conversation_history = []
        prompt = "machine learning"
        
        result = bm25_hybrid_search(prompt, sample_memories, conversation_history, top_n=2)
        
        assert len(result) == 2
        assert all(r['type'] == 'memory' for r in result)
        # Should prioritize memory about machine learning
        assert any('machine learning' in r['meta']['memory'].lower() for r in result)
    
    def test_bm25_search_with_conversations_only(self, sample_conversation_history):
        """Test BM25 search with only conversation documents"""
        memories = []
        prompt = "deep learning"
        
        result = bm25_hybrid_search(prompt, memories, sample_conversation_history, top_n=2)
        
        assert len(result) == 2
        assert all(r['type'] == 'conversation' for r in result)
        # Should find conversation about deep learning
        assert any('deep learning' in r['content'].lower() for r in result)
    
    def test_bm25_search_hybrid(self, sample_memories, sample_conversation_history):
        """Test BM25 search with both memories and conversations"""
        prompt = "neural networks"
        
        result = bm25_hybrid_search(prompt, sample_memories, sample_conversation_history, top_n=3)
        
        assert len(result) == 3
        # Should find both memory and conversation about neural networks
        memory_results = [r for r in result if r['type'] == 'memory']
        conversation_results = [r for r in result if r['type'] == 'conversation']
        
        assert len(memory_results) > 0
        assert len(conversation_results) > 0
    
    def test_bm25_search_relevance_scoring(self, sample_memories):
        """Test that BM25 search returns results in relevance order"""
        conversation_history = []
        prompt = "artificial intelligence"
        
        result = bm25_hybrid_search(prompt, sample_memories, conversation_history, top_n=3)
        
        assert len(result) == 3
        # First result should be most relevant to AI
        first_result = result[0]['meta']['memory']
        assert 'artificial intelligence' in first_result.lower()
    
    def test_bm25_search_case_insensitive(self, sample_memories):
        """Test that BM25 search is case insensitive"""
        conversation_history = []
        prompt = "MACHINE LEARNING"
        
        result = bm25_hybrid_search(prompt, sample_memories, conversation_history, top_n=1)
        
        assert len(result) == 1
        assert 'machine learning' in result[0]['meta']['memory'].lower()


class TestMemoryRetrieval:
    """Test memory retrieval and citation functionality"""
    
    def test_fetch_cited_memories_success(self, mock_mem0_client):
        """Test successful fetching of cited memories"""
        with patch('app.utils.memory.mem0_client', mock_mem0_client):
            citations = [('mem_001', '2024-01-01T10:00:00Z'), ('mem_002', '2024-01-02T10:00:00Z')]
            
            result = fetch_cited_memories(citations)
            
            assert len(result) == 2
            assert all('id' in mem for mem in result)
            assert all('content' in mem for mem in result)
            assert all('title' in mem for mem in result)
    
    def test_fetch_cited_memories_duplicate_handling(self, mock_mem0_client):
        """Test that duplicate memory IDs are handled correctly"""
        with patch('app.utils.memory.mem0_client', mock_mem0_client):
            citations = [('mem_001', '2024-01-01T10:00:00Z'), ('mem_001', '2024-01-01T10:00:00Z')]
            
            result = fetch_cited_memories(citations)
            
            assert len(result) == 1  # Should deduplicate
            assert result[0]['id'] == 'mem_001'
    
    def test_fetch_cited_memories_not_found(self, mock_mem0_client):
        """Test handling of memories that are not found"""
        mock_mem0_client.get.return_value = None
        
        with patch('app.utils.memory.mem0_client', mock_mem0_client):
            citations = [('mem_001', '2024-01-01T10:00:00Z')]
            
            result = fetch_cited_memories(citations)
            
            assert len(result) == 1
            assert result[0]['title'] == '[Memory not found]'
            assert result[0]['content'] == '[Memory not found]'
    
    def test_fetch_cited_memories_error_handling(self, mock_mem0_client):
        """Test error handling when fetching memories fails"""
        mock_mem0_client.get.side_effect = Exception("Database error")
        
        with patch('app.utils.memory.mem0_client', mock_mem0_client):
            citations = [('mem_001', '2024-01-01T10:00:00Z')]
            
            result = fetch_cited_memories(citations)
            
            assert len(result) == 1
            assert result[0]['title'] == '[Error fetching memory]'
            assert 'Database error' in result[0]['content']


class TestContextFormatting:
    """Test context formatting functionality"""
    
    def test_format_context_with_memories_and_conversations(self, sample_memories, sample_conversation_history):
        """Test formatting context with both memories and conversations"""
        result = format_context(sample_memories, sample_conversation_history)
        
        # Should contain both sections
        assert '*Past Conversations:*' in result
        assert '*Relevant Memories (Hybrid Search):*' in result
        
        # Should contain memory content
        for memory in sample_memories:
            assert memory['memory'] in result
            assert memory['id'] in result
        
        # Should contain conversation content
        for role, content, timestamp in sample_conversation_history:
            assert content in result
            assert timestamp in result
    
    def test_format_context_empty_memories(self, sample_conversation_history):
        """Test formatting context with empty memories"""
        result = format_context([], sample_conversation_history)
        
        assert '*Past Conversations:*' in result
        assert '*Relevant Memories (Hybrid Search):*' in result
        assert 'Memory:' not in result  # No memory formatting
    
    def test_format_context_empty_conversations(self, sample_memories):
        """Test formatting context with empty conversations"""
        result = format_context(sample_memories, [])
        
        assert '*Past Conversations:*' in result
        assert '*Relevant Memories (Hybrid Search):*' in result
        assert 'Message Index:' not in result  # No conversation formatting
    
    def test_format_context_both_empty(self):
        """Test formatting context with both empty memories and conversations"""
        result = format_context([], [])
        assert '*Past Conversations:*' in result
        assert '*Relevant Memories (Hybrid Search):*' in result
        # The function adds two newlines at the end
        expected = '*Past Conversations:*\n\n\n*Relevant Memories (Hybrid Search):*\n\n'
        assert result == expected


class TestMemoryWriting:
    """Test memory writing functionality"""
    
    def test_write_memory_success(self, mock_mem0_client):
        """Test successful memory writing"""
        with patch('app.utils.memory.mem0_client', mock_mem0_client):
            result = write_memory("Test memory content", "test_user")
            
            assert result == {'id': 'test_memory_id'}
            mock_mem0_client.add.assert_called_once()
    
    def test_write_memory_error_handling(self, mock_mem0_client):
        """Test error handling in memory writing"""
        mock_mem0_client.add.side_effect = Exception("Memory write error")
        
        with patch('app.utils.memory.mem0_client', mock_mem0_client):
            result = write_memory("Test memory content", "test_user")
            
            assert result is None