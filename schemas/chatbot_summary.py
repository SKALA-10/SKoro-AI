from pydantic import BaseModel
from typing import Optional, List

class ChatbotSummaryResponse(BaseModel):
    code: int
    message: str