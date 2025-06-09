from fastapi import APIRouter

router = APIRouter()

# AI에 피드백 요약 요청 및 저장
@router.post("/")
def generate_and_save_feedback_summary():