from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict

class MultiAgentState(BaseModel):
    user_id: str
    prompt: str
    memories: Optional[List[Dict]] = []
    conversations: Optional[List[Tuple]] = []
    context: Optional[str] = ""
    rationale: Optional[str] = ""
    answer: Optional[str] = ""
    answer_html: Optional[str] = ""
    citations: Optional[List[Dict]] = []
    history: Optional[List[str]] = []
    supervisor_plan: Optional[str] = ""
    subagent_tasks: Optional[List[str]] = []
    clarifications: Optional[List[str]] = [] 