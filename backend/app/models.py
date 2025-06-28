from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    user_id: str
    prompt: str

class Citation(BaseModel):
    memory_id: str
    timestamp: str
    content: str

class SearchResponse(BaseModel):
    rationale: str
    answer: str
    citations: List[Citation]
