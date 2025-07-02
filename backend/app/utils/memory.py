import logging
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("memory")

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0",
            "path": "db",
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    }
}

mem0_client = Memory.from_config(config)

def write_memory(prompt: str, user_id: str):
    """
    Write a memory to the vector store.
    
    Args:
        prompt: The content to store as memory
        user_id: The user identifier
        
    Returns:
        Memory write result or None if failed
    """
    try:
        logger.info(f"Writing memory for user {user_id}: {prompt}")
        result = mem0_client.add([{"role": "user", "content": prompt}], user_id=user_id)
        logger.info(f"Memory write result: {result}")
        return result
    except Exception as e:
        logger.error(f"Could not write memory: {e}")
        return None

def fetch_cited_memories(citations):
    """
    Fetch memory details for cited memory IDs.
    
    Args:
        citations: List of tuples (memory_id, timestamp)
        
    Returns:
        List of memory dictionaries with citation details
    """
    cited_memories = []
    seen_ids = set()
    for mem_id, timestamp in set(citations):
        if mem_id not in seen_ids:
            try:
                mem_data = mem0_client.get(mem_id)
                if mem_data and 'memory' in mem_data:
                    mem_ts = mem_data.get('updated_at') or mem_data.get('created_at', 'N/A')
                    content = mem_data['memory']
                    title = content[:50]
                    cited_memories.append({
                        "id": mem_id,
                        "title": title,
                        "memory_id": mem_id,
                        "timestamp": mem_ts,
                        "content": content
                    })
                else:
                    cited_memories.append({
                        "id": mem_id,
                        "title": "[Memory not found]",
                        "memory_id": mem_id,
                        "timestamp": timestamp,
                        "content": "[Memory not found]"
                    })
            except Exception as e:
                cited_memories.append({
                    "id": mem_id,
                    "title": "[Error fetching memory]",
                    "memory_id": mem_id,
                    "timestamp": timestamp,
                    "content": f"[Error fetching memory: {e}]"
                })
            seen_ids.add(mem_id)
    logger.info(f"Cited memories returned: {cited_memories}")
    return cited_memories

def get_all_memories(user_id: str):
    """
    Get all memories for a user.
    
    Args:
        user_id: The user identifier
        
    Returns:
        List of all memories for the user
    """
    return mem0_client.get_all(user_id=user_id).get('results', []) 