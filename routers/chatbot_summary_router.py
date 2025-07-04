from fastapi import APIRouter
from services.chatbot_summary_service import ChatbotSummaryService
from schemas.chatbot_summary import ChatbotSummaryResponse

router = APIRouter()
chatbot_summary_service = ChatbotSummaryService()

router = APIRouter(tags=["평가 피드백 요약"])

# 평가 피드백 요약
@router.post("", response_model=ChatbotSummaryResponse, summary="평가 피드백 요약")
def run_summary_for_all_teams():
    return chatbot_summary_service.run_summary_for_all_teams()