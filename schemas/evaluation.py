from pydantic import BaseModel
from typing import Optional, List

class EvaluationRequest(BaseModel):
    period_id: int
    teams: Optional[List[int]] = None

class EvaluationResponse(BaseModel):
    period_id: int
    code: int
    message: str