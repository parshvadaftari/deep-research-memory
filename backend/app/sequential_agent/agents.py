from .agentic_state import ResearchState
from app.utils.memory import get_all_memories, fetch_cited_memories, write_memory
from app.utils.database import fetch_conversation_history
from app.utils.search import bm25_hybrid_search
from app.utils.context import format_context
from app.utils.llm import get_llm, cot_reasoning_prompt, answer_prompt, annotate_with_citations

# Model selection for each agent
MEMORY_MODEL = "gpt-4.1-mini"  # fast, cheap, sufficient context
CONVERSATION_MODEL = "gpt-4.1-mini"  # fast, cheap
CONTEXT_MODEL = "gpt-4o"  # large context window for synthesis
REASONING_MODEL = "gpt-4o"  # large context window for CoT
ANSWER_MODEL = "gpt-4o"  # turbo for answer generation
CITATION_MODEL = "gpt-4.1-mini"  # fast, cheap, for annotation

async def memory_agent(state: ResearchState):
    # Store the new prompt as a memory for the user in Mem0
    write_memory(state.prompt, state.user_id)
    all_memories = get_all_memories(state.user_id)
    # Use hybrid search to rank memories
    hybrid_results = bm25_hybrid_search(state.prompt, all_memories, [], top_n=10)
    top_memories = [r['meta'] for r in hybrid_results if r['type'] == 'memory']
    state.memories = top_memories
    state.history.append(f"MemoryAgent({MEMORY_MODEL}): stored new memory and retrieved memories")
    return state

async def conversation_agent(state: ResearchState):
    conversations = fetch_conversation_history(state.user_id, limit=10)
    # Use hybrid search to rank conversations
    hybrid_results = bm25_hybrid_search(state.prompt, [], conversations, top_n=10)
    top_conversations = [
        (r['role'], r['content'], r['timestamp'])
        for r in hybrid_results if r['type'] == 'conversation']
    state.conversations = top_conversations
    state.history.append(f"ConversationAgent({CONVERSATION_MODEL}): retrieved conversations")
    return state

async def context_agent(state: ResearchState):
    # If context synthesis ever needs LLM, use CONTEXT_MODEL
    state.context = format_context(state.memories, state.conversations)
    state.history.append(f"ContextAgent({CONTEXT_MODEL}): formatted context")
    return state

async def reasoning_agent(state: ResearchState):
    llm = get_llm(REASONING_MODEL)
    rationale_prompt = cot_reasoning_prompt(state.context, state.prompt)
    rationale = ""
    async for chunk in llm.astream([{"role": "user", "content": rationale_prompt}]):
        token = chunk.content if hasattr(chunk, "content") else chunk
        rationale += token
    state.rationale = rationale
    state.history.append(f"ReasoningAgent({REASONING_MODEL}): generated rationale")
    return state

async def answer_agent(state: ResearchState):
    llm = get_llm(ANSWER_MODEL)
    answer_prompt_str = answer_prompt(state.context, state.rationale, state.prompt)
    answer = ""
    async for chunk in llm.astream([{"role": "user", "content": answer_prompt_str}]):
        token = chunk.content if hasattr(chunk, "content") else chunk
        answer += token
    state.answer = answer
    state.history.append(f"AnswerAgent({ANSWER_MODEL}): generated answer")
    return state

async def citation_agent(state: ResearchState):
    citations = [(m['id'], m.get('updated_at') or m.get('created_at', 'N/A')) for m in state.memories]
    cited_memories = fetch_cited_memories(citations)
    state.citations = cited_memories
    state.answer = annotate_with_citations(state.answer, cited_memories)
    state.history.append(f"CitationAgent({CITATION_MODEL}): annotated answer with citations")
    return state 