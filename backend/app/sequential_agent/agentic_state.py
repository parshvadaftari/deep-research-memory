from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict

class ResearchState(BaseModel):
    user_id: str
    prompt: str
    memories: Optional[List[Dict]] = []
    conversations: Optional[List[Tuple]] = []
    context: Optional[str] = ""
    rationale: Optional[str] = ""
    answer: Optional[str] = ""
    citations: Optional[List[Dict]] = []
    history: Optional[List[str]] = [] 