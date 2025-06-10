from pydantic import BaseModel
from typing import List

class EvaluationFeedbackRequest(BaseModel):
    emp_no: str
    chat_history: List[str]
    period_id: int

class EvaluationFeedbackResponse(BaseModel):
    emp_no: str
    content: str