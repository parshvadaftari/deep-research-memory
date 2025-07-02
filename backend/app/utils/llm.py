from ..prompts import GROUND_CONTEXT_PROMPT
from langchain_openai import ChatOpenAI
from typing import List, Dict

def ground_context(context: str, prompt: str, llm):
    """
    Ground the context using the LLM.
    
    Args:
        context: The context to ground
        prompt: The user prompt
        llm: The LLM instance
        
    Returns:
        Grounded context from LLM
    """
    grounded = llm.invoke([
        {"role": "user", "content": GROUND_CONTEXT_PROMPT.format(context=context, prompt=prompt)}
    ])
    return grounded.content if hasattr(grounded, "content") else grounded

def llm_annotate_with_citations(text: str, cited_memories: list, llm):
    """
    Annotate text with inline citation tags using LLM.
    
    Args:
        text: The text to annotate
        cited_memories: List of memory dictionaries for citations
        llm: The LLM instance
        
    Returns:
        HTML-annotated text with citation tags
    """
    citation_list = "\n".join([
        f"[{i+1}] {mem['content']}" for i, mem in enumerate(cited_memories)
    ])
    
    annotation_prompt = f'''
You are an expert research assistant. Your job is to annotate the following text with inline citation tags, using the provided list of memory citations.

IMPORTANT: Do NOT use Ellipsis, [N], [1], [2], etc. citation markers anywhere in the output. Only use <cite data-citation="N">...</cite> tags for citations, or refer to evidence in natural language. If you see [N] in the text, replace it with the appropriate <cite> tag. Do not output any [N] style citations.

For every phrase or sentence in the text that is directly supported by a memory, wrap it in a <cite data-citation="N">...</cite> tag, where N is the number of the memory in the list below. Do this for every citation that applies. Do not add, remove, or change any text. Only add <cite> tags. Return valid HTML only.

{citation_list}

{text}
'''
    annotated = llm.invoke([{"role": "user", "content": annotation_prompt}])
    return annotated.content if hasattr(annotated, "content") else annotated 

def get_llm(model: str = "gpt-4.1-mini"):
    # Returns a streaming LLM instance (can be customized/configured)
    return ChatOpenAI(model=model, streaming=True)

def cot_reasoning_prompt(context: str, prompt: str) -> str:
    # Chain-of-thought prompt for rationale generation
    return (
        "You are an expert research assistant.\n"
        "Given the following context from the user's memories and past conversations, reason step by step to answer the user's question.\n"
        "If you need more information, specify what to retrieve next.\n"
        "\nContext:\n" + context + "\n\nQuestion: " + prompt + "\n\nLet's think step by step."
    )

def answer_prompt(context: str, rationale: str, prompt: str) -> str:
    # Prompt for answer generation, grounded in rationale and context
    return (
        "You are an expert research assistant.\n"
        "Given the following context and rationale, synthesize a clear, well-structured answer to the user's question.\n"
        "\nContext:\n" + context + "\n\nRationale:\n" + rationale + "\n\nQuestion: " + prompt + "\n\nAnswer:"
    )

def annotate_with_citations(answer: str, cited_memories: List[Dict]) -> str:
    # Simple inline citation annotation (can be improved for production)
    for i, mem in enumerate(cited_memories):
        ref = f"[ref: {mem['id']}]"
        if mem['title'][:20] in answer:
            answer = answer.replace(mem['title'][:20], mem['title'][:20] + " " + ref)
    return answer 