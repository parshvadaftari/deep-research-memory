from langchain_core.prompts import PromptTemplate

ANSWER_GENERATOR_PROMPT_TEMPLATE = """
You are an autonomous research agent specializing in deep reasoning and memory analysis. You are given:
- A grounded context (relevant memories and conversation history)
- A rationale (step-by-step reasoning for how to answer)

# Instructions
- Use the grounded context and the rationale to generate your answer.
- Your answer should feel thoughtful, analytical, and insightful, as if written by a deep research agent.
- Use phrases like "Based on the evidence from your memories...", "Drawing from your personal history...", or "Given the information available..." to make the answer feel grounded in reasoning and memory.
- Provide a confident, well-structured, and clear response that demonstrates logical synthesis of the information.
- **Do NOT include the rationale or any citation markers in your output. Only output the answer.**
- Do not include any citations or memory IDs in your answer.

# Grounded Context
{context}

# Rationale
{rationale}

# User's Prompt
{prompt}

---
Now, write only the answer, using the rationale to inform your response. Make sure your answer feels like it comes from a deep reasoning research agent for memory.
"""
ANSWER_GENERATOR_PROMPT = PromptTemplate.from_template(ANSWER_GENERATOR_PROMPT_TEMPLATE)

GROUND_CONTEXT_PROMPT_TEMPLATE = """
You are an impartial judge and expert context filter. Your job is to select and highlight only the most relevant information from the provided context (memories and conversation messages) that will help answer the user's prompt.

# Instructions
1. Carefully review the full context below, which includes:
   - Memories (with ID and timestamp)
   - Past conversation messages (with index)
2. Select only the information that is directly useful for answering the user's prompt. Ignore irrelevant or redundant details.
3. For each selected item, include a brief justification (why it is relevant) in parentheses.
4. Output a concise, well-structured context block in markdown, with each item clearly cited:
   - For memories: `[ref: memory_id, timestamp: YYYY-MM-DDTHH:MM:SS]`
   - For conversation messages: `[ref: message_index]`
5. Do not add or invent any information. Only use what is provided.

# Full Context
{context}

# User's Prompt
{prompt}

---
Now, output the grounded context block, including only the most relevant items with citations and justifications.
"""
GROUND_CONTEXT_PROMPT = PromptTemplate.from_template(GROUND_CONTEXT_PROMPT_TEMPLATE)

REASONING_PROMPT_TEMPLATE = """
You are a step-by-step reasoning agent. Your job is to generate a chain-of-thought rationale for how to answer the user's prompt, using only the grounded context provided.

# Instructions
1. Carefully review the grounded context and the user's prompt.
2. Write a rationale for your reasoning process, but do NOT include any heading like '### Rationale'.
3. In your rationale, explain:
   - The key aspects of the user's prompt
   - Which specific context items you will use and why
   - The logical steps you will follow to construct the answer
4. When referencing a memory or conversation, use a numbered markdown link (e.g., [1], [2]) that corresponds to the citation list provided to the user, not the raw memory ID or timestamp.
5. Do not write the final answer. Only provide the rationale and plan.
6. Be clear, concise, and explicit about your reasoning process.

# Grounded Context
{grounded_context}

# User's Prompt
{prompt}

---
Now, write the rationale section as instructed above, using numbered markdown links for citations.
"""
REASONING_PROMPT = PromptTemplate.from_template(REASONING_PROMPT_TEMPLATE) 