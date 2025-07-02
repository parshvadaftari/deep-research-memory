import logging
from langchain_openai import ChatOpenAI
from app.prompts import ANSWER_GENERATOR_PROMPT, REASONING_PROMPT
from app.utils.memory import get_all_memories, fetch_cited_memories, write_memory
from app.utils.database import fetch_conversation_history, store_conversation
from app.utils.search import bm25_hybrid_search
from app.utils.llm import llm_annotate_with_citations, ground_context
from app.utils.context import format_context

# Set up logger
logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

async def agent_pipeline(user_id, prompt):
    """
    Main agent pipeline that processes user prompts and generates responses.
    
    Args:
        user_id: The user identifier
        prompt: The user's prompt/message
        
    Yields:
        Streaming response tokens and metadata
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", streaming=True)
    write_memory(prompt, user_id)
    conversation_history = fetch_conversation_history(user_id, limit=10)
    all_memories = get_all_memories(user_id)
    hybrid_results = bm25_hybrid_search(prompt, all_memories, conversation_history, top_n=5)
    
    # Handle empty hybrid results
    if not hybrid_results:
        hybrid_memories = []
        hybrid_conversations = []
    else:
        hybrid_memories = [r['meta'] for r in hybrid_results if r['type'] == 'memory']
        hybrid_conversations = [
            (r['role'], r['content'], r['timestamp'])
            for r in hybrid_results if r['type'] == 'conversation'
        ]
    
    try:
        citations = [(m['id'], m.get('updated_at') or m.get('created_at', 'N/A')) for m in hybrid_memories]
        cited_memories = fetch_cited_memories(citations)
    except Exception as e:
        logger.error(f"Error fetching citations: {e}")
        cited_memories = []
    
    context = format_context(hybrid_memories, hybrid_conversations)
    grounded_context = ground_context(context, prompt, llm)

    # Stream rationale tokens
    rationale_prompt = REASONING_PROMPT.format(grounded_context=grounded_context, prompt=prompt)
    rationale = ""
    async for chunk in llm.astream([{"role": "user", "content": rationale_prompt}]):
        token = chunk.content if hasattr(chunk, "content") else chunk
        rationale += token
        yield {"type": "rationale_token", "token": token}
    yield {"type": "rationale_complete", "rationale": rationale}
    annotated_rationale_html = llm_annotate_with_citations(rationale, cited_memories, llm)
    yield {"type": "rationale_annotated_html", "rationale_html": annotated_rationale_html}

    # Stream answer tokens
    answer_prompt = ANSWER_GENERATOR_PROMPT.format(context=grounded_context, rationale=rationale, prompt=prompt)
    answer = ""
    async for chunk in llm.astream([{"role": "user", "content": answer_prompt}]):
        token = chunk.content if hasattr(chunk, "content") else chunk
        answer += token
        yield {"type": "answer_token", "token": token}
    yield {"type": "answer_complete", "answer": answer}
    annotated_answer_html = llm_annotate_with_citations(answer, cited_memories, llm)
    yield {"type": "answer_annotated_html", "answer_html": annotated_answer_html}

    yield {"type": "citations", "citations": cited_memories}
    
    # Store the conversation
    store_conversation(user_id, prompt, answer)
    
    # Signal completion
    yield {"type": "done"}
