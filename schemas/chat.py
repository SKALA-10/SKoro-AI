from pydantic import BaseModel
from typing import Literal, Optional

class ChatRequest(BaseModel):
    user_id: str
    chat_mode: Literal["default", "qna_response"]
    message: str
    appeal_complete: Optional[bool] = False

class ChatResponse(BaseModel):
    type: str
    response: Optional[str] = None
    summary: Optional[str] = None
    message: Optional[str] = None 
    user_id: str