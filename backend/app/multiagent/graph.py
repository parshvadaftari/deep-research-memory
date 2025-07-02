from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda, RunnableParallel
from .state import MultiAgentState
from .agents import supervisor_agent, memory_agent, conversation_agent, context_agent, reasoning_agent, answer_agent, citation_agent, memory_retrieval_agent, conversation_retrieval_agent

memory_retrieval_runnable = RunnableLambda(memory_retrieval_agent)
conversation_retrieval_runnable = RunnableLambda(conversation_retrieval_agent)

parallel_retrieval = RunnableParallel({
    "memories": memory_retrieval_runnable,
    "conversations": conversation_retrieval_runnable,
})

async def merge_retrievals(input_state):
    results = await parallel_retrieval.ainvoke(input_state)
    input_state.memories = results["memories"].memories
    input_state.conversations = results["conversations"].conversations
    input_state.history += results["memories"].history + results["conversations"].history
    return input_state

merge_retrievals_runnable = RunnableLambda(merge_retrievals)

workflow = StateGraph(MultiAgentState)
workflow.add_node("merge_retrievals", merge_retrievals_runnable)
workflow.add_edge(START, "merge_retrievals")
workflow.add_edge("merge_retrievals", "supervisor")

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("memory_agent", memory_agent)
workflow.add_node("conversation_agent", conversation_agent)
workflow.add_node("context_agent", context_agent)
workflow.add_node("reasoning_agent", reasoning_agent)
workflow.add_node("answer_agent", answer_agent)
workflow.add_node("citation_agent", citation_agent)

def supervisor_conditional(state: MultiAgentState):
    if state.clarifications:
        return END
    return "memory_agent"

workflow.add_conditional_edges("supervisor", supervisor_conditional, ["memory_agent", END])
workflow.add_edge("memory_agent", "conversation_agent")
workflow.add_edge("conversation_agent", "context_agent")
workflow.add_edge("context_agent", "reasoning_agent")
workflow.add_edge("reasoning_agent", "answer_agent")
workflow.add_edge("answer_agent", "citation_agent")
workflow.add_edge("citation_agent", END)

graph = workflow.compile() 