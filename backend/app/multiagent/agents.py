from .state import MultiAgentState
from ..utils.memory import get_all_memories, fetch_cited_memories, write_memory
from ..utils.database import fetch_conversation_history
from ..utils.search import bm25_hybrid_search
from ..utils.context import format_context
from ..utils.llm import get_llm, cot_reasoning_prompt, answer_prompt, annotate_with_citations, llm_annotate_with_citations

SUPERVISOR_MODEL = "chatgpt-4.1"
SUPERVISOR_PROMPT = """
You are a research supervisor agent. Your job is to:
- Answer the user's question directly if you have enough information from memories or context.
- If the user makes a speculative, hypothetical, or factually incorrect statement, gently correct them and provide the most accurate information you have.
- Only ask for clarification if you truly cannot answer with the available information.
- Always cite relevant memories or context in your answer.
- If you do not know, say so clearly and suggest how the user can help you answer.
Be concise, factual, and helpful. Do not over-plan or over-clarify.
"""
MEMORY_MODEL = "gpt-4.1-mini"
CONVERSATION_MODEL = "gpt-4.1-mini"
CONTEXT_MODEL = "gpt-4o"
REASONING_MODEL = "gpt-4o"
ANSWER_MODEL = "gpt-4-turbo"
CITATION_MODEL = "gpt-4.1-mini"

async def supervisor_agent(state: MultiAgentState):
    llm = get_llm(model=SUPERVISOR_MODEL)
    # Compose the system prompt
    system_prompt = SUPERVISOR_PROMPT
    # Gather context and memories
    memories = state.memories or []
    conversations = state.conversations or []
    context = format_context(memories, conversations)
    prompt = state.prompt

    # Try to find a direct answer in memories/context
    direct_answer = None
    for mem in memories:
        if prompt.lower() in (mem.get('memory', '').lower() or ''):
            direct_answer = mem.get('memory')
            break
    # If direct answer found, answer and cite
    if direct_answer:
        state.answer = direct_answer
        state.citations = [memories[0]] if memories else []
        state.history.append("Supervisor: answered directly from context.")
        return state
    # If user statement is hallucinated/speculative, correct it
    # (Simple heuristic: if prompt contains future/past tense or known falsehoods)
    hallucination_keywords = ["will be", "would be", "is going to", "was", "were", "supposed to", "rumor", "fake", "incorrect", "not true", "speculative", "hypothetical"]
    if any(kw in prompt.lower() for kw in hallucination_keywords):
        state.answer = "Correction: Your statement appears to be speculative or not supported by the information I have. Here is what I know: "
        if memories:
            state.answer += " " + memories[0].get('memory', '')
            state.citations = [memories[0]]
        else:
            state.answer += " I do not have any supporting information in your memories."
        state.history.append("Supervisor: corrected user hallucination.")
        return state
    # If not enough info, ask for clarification
    state.answer = "I need more information or clarification to answer your question. Could you please provide more details?"
    state.history.append("Supervisor: asked for clarification.")
    return state

async def memory_agent(state: MultiAgentState):
    write_memory(state.prompt, state.user_id)
    all_memories = get_all_memories(state.user_id)
    hybrid_results = bm25_hybrid_search(state.prompt, all_memories, [], top_n=10)
    top_memories = [r['meta'] for r in hybrid_results if r['type'] == 'memory']
    state.memories = top_memories
    state.history.append(f"MemoryAgent({MEMORY_MODEL}): stored new memory and retrieved memories")
    return state

async def conversation_agent(state: MultiAgentState):
    conversations = fetch_conversation_history(state.user_id, limit=10)
    hybrid_results = bm25_hybrid_search(state.prompt, [], conversations, top_n=10)
    top_conversations = [
        (r['role'], r['content'], r['timestamp'])
        for r in hybrid_results if r['type'] == 'conversation']
    state.conversations = top_conversations
    state.history.append(f"ConversationAgent({CONVERSATION_MODEL}): retrieved conversations")
    return state

async def context_agent(state: MultiAgentState):
    state.context = format_context(state.memories, state.conversations)
    state.history.append(f"ContextAgent({CONTEXT_MODEL}): formatted context")
    return state

async def reasoning_agent(state: MultiAgentState):
    
    # Decide model based on state or task
    model = getattr(state, 'reasoning_model', 'gpt-4o')
    llm = get_llm(model=model)
    rationale_prompt = cot_reasoning_prompt(state.context, state.prompt)
    rationale = ""
    async for chunk in llm.astream([{"role": "user", "content": rationale_prompt}]):
        token = chunk.content if hasattr(chunk, "content") else chunk
        rationale += token
    state.rationale = rationale
    state.history.append(f"ReasoningAgent({model}): generated rationale")
    return state

async def answer_agent(state: MultiAgentState):
    
    # Decide model based on state or task
    model = getattr(state, 'answer_model', 'gpt-4-turbo')
    llm = get_llm(model=model)
    answer_prompt_str = answer_prompt(state.context, state.rationale, state.prompt)
    answer = ""
    async for chunk in llm.astream([{"role": "user", "content": answer_prompt_str}]):
        token = chunk.content if hasattr(chunk, "content") else chunk
        answer += token
    state.answer = answer
    state.history.append(f"AnswerAgent({model}): generated answer")
    return state

async def citation_agent(state: MultiAgentState):
    # Always cite at least the top memory if any
    if state.memories:
        citations = [(m['id'], m.get('updated_at') or m.get('created_at', 'N/A')) for m in state.memories]
        cited_memories = fetch_cited_memories(citations)
        state.citations = cited_memories if cited_memories else [state.memories[0]]
        llm = get_llm(model='gpt-4.1-mini')
        state.answer_html = llm_annotate_with_citations(state.answer, cited_memories, llm)
    else:
        state.citations = []
        state.answer_html = state.answer
    state.history.append(f"CitationAgent({CITATION_MODEL}): annotated answer with citations (HTML)")
    return state

async def memory_retrieval_agent(state: MultiAgentState):
    state.memories = get_all_memories(state.user_id)
    state.history.append("Memory retrieval complete.")
    return state

async def conversation_retrieval_agent(state: MultiAgentState):
    state.conversations = fetch_conversation_history(state.user_id, limit=20)
    state.history.append("Conversation retrieval complete.")
    return state 