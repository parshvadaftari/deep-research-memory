"""
Search utilities for hybrid search functionality.
"""

import nltk
from rank_bm25 import BM25Okapi

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def bm25_hybrid_search(prompt: str, memories: list, conversation_history: list, top_n: int = 5):
    """
    Perform BM25 hybrid search across memories and conversation history.
    
    Args:
        prompt: The search query
        memories: List of memory dictionaries
        conversation_history: List of conversation tuples (role, content, timestamp)
        top_n: Number of top results to return
        
    Returns:
        List of search results with metadata
    """
    docs = []
    doc_meta = []
    
    # Add memories to search corpus
    for m in memories:
        docs.append(m['memory'])
        doc_meta.append({'type': 'memory', 'id': m['id'], 'meta': m})
    
    # Add conversation history to search corpus
    for i, (role, content, timestamp) in enumerate(conversation_history):
        docs.append(content)
        doc_meta.append({
            'type': 'conversation', 
            'index': i, 
            'role': role, 
            'timestamp': timestamp, 
            'content': content
        })
    
    # Handle empty corpus case
    if not docs:
        return []
    
    # Perform BM25 search
    tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = nltk.word_tokenize(prompt.lower())
    scores = bm25.get_scores(tokenized_query)
    
    # Get top results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = [doc_meta[i] for i in top_indices]
    
    return results 