import time
import sqlite3
import logging

logger = logging.getLogger("database")

def fetch_conversation_history(user_id: str, limit: int = 10):
    """
    Fetch conversation history for a user from the database.
    
    Args:
        user_id: The user identifier
        limit: Maximum number of conversations to fetch
        
    Returns:
        List of conversation tuples (role, content, timestamp)
    """
    conn = sqlite3.connect("research_agent_conversations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            user_id TEXT, role TEXT, content TEXT, timestamp TEXT
        )
    """)
    c.execute("SELECT role, content, timestamp FROM conversation_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

def store_conversation(user_id: str, prompt: str, answer: str):
    """
    Store a conversation exchange in the database.
    
    Args:
        user_id: The user identifier
        prompt: The user's prompt/message
        answer: The agent's response
    """
    try:
        conn = sqlite3.connect("research_agent_conversations.db")
        c = conn.cursor()
        now = time.time()
        c.execute("INSERT INTO conversation_history (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (user_id, "user", prompt, now))
        if answer:
            c.execute("INSERT INTO conversation_history (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (user_id, "agent", answer, now))
        conn.commit()
        conn.close()
        logger.info(f"Stored conversation for user {user_id}.")
    except Exception as e:
        logger.error(f"Could not store conversation: {e}") 