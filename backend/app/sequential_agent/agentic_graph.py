from langgraph.graph import StateGraph, START, END
from .agentic_state import ResearchState
from .agents import memory_agent, conversation_agent, context_agent, reasoning_agent, answer_agent, citation_agent

workflow = StateGraph(ResearchState)
workflow.add_node("memory_agent", memory_agent)
workflow.add_node("conversation_agent", conversation_agent)
workflow.add_node("context_agent", context_agent)
workflow.add_node("reasoning_agent", reasoning_agent)
workflow.add_node("answer_agent", answer_agent)
workflow.add_node("citation_agent", citation_agent)

workflow.add_edge(START, "memory_agent")
workflow.add_edge("memory_agent", "conversation_agent")
workflow.add_edge("conversation_agent", "context_agent")
workflow.add_edge("context_agent", "reasoning_agent")
workflow.add_edge("reasoning_agent", "answer_agent")
workflow.add_edge("answer_agent", "citation_agent")
workflow.add_edge("citation_agent", END)

graph = workflow.compile() 