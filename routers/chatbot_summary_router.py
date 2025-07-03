from fastapi import APIRouter
from services.chatbot_summary_service import ChatbotSummaryService
from schemas.chatbot_summary import ChatbotSummaryRequest, ChatbotSummaryResponse

router = APIRouter()
chatbot_summary_service = ChatbotSummaryService()

# 평가 피드백 요약
@router.post("/quarterly", response_model=ChatbotSummaryResponse, summary="평가 피드백 요약")
def start_chatbot(request: ChatbotSummaryRequest):
    return chatbot_summary_service.start_quarterly_evaluation(request.period_id, request.teams)