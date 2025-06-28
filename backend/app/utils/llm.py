"""
LLM utilities for interaction and annotation.
"""

from ..prompts import GROUND_CONTEXT_PROMPT

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

# Memory Citations
{citation_list}

# Text to Annotate
{text}
'''
    annotated = llm.invoke([{"role": "user", "content": annotation_prompt}])
    return annotated.content if hasattr(annotated, "content") else annotated 