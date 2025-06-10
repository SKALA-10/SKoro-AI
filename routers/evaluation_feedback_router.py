from fastapi import APIRouter
from schemas.evaluation_feedback import EvaluationFeedbackRequest
from services.evaluation_feedback_service import EvaluationFeedbackService

router = APIRouter()
evaluation_feedback_service = EvaluationFeedbackService()

# AI에 팀장에 대한 피드백 요약 요청(본인)
@router.post("/", response_model=EvaluationFeedbackResponse)
def generate_feedback_summary(request: EvaluationFeedbackRequest):
    return evaluation_feedback_service.generate_feedback_summary(request)