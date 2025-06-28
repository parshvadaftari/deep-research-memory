"""
Context utilities for formatting and organizing context information.
"""

def format_context(memories: list, conversation_history: list):
    """
    Format memories and conversation history into a structured context string.
    
    Args:
        memories: List of memory dictionaries
        conversation_history: List of conversation tuples (role, content, timestamp)
        
    Returns:
        Formatted context string
    """
    formatted_memories = "\n".join(
        [
            f"Memory: {m['memory']}\n[ref: {m['id']}, timestamp: {m.get('updated_at') or m.get('created_at', 'N/A')}]"
            for m in memories
        ]
    )
    
    past_conversations_str = "\n".join(
        [
            f"Message Index: {i}\nTimestamp: {timestamp}\n{content}"
            for i, (role, content, timestamp) in enumerate(conversation_history)
        ]
    )
    
    return f"*Past Conversations:*\n{past_conversations_str}\n\n*Relevant Memories (Hybrid Search):*\n{formatted_memories}\n" 