from fastapi import APIRouter
from services.team_evaluation_service import (
    save_team_evaluation_report,
    save_temp_evaluation_to_redis,
)

router = APIRouter()

# AI에 업무 요약 요청(중간 산출물)
@router.post("/task-summary")
def generate_task_summary_ai():
    # 팀 업무 요약 AI 호출

    # 응답 받으면

    # 팀 평가 레포트 저장(중간 산출물)
    save_team_evaluation_report()

    # 임시 평가 저장(Redis 데이터 생성)
    save_temp_evaluation_to_redis()

# 평가 완료 및 AI 메서드 호출
@router.post("/{team_evaluation_id}/complete")
def complete_team_evaluation():
    # 최종 평가 AI 호출

    # 응답 받으면
    return {"msg": "피드백 레포트 목록"}

# 팀 평가 레포트 다운로드
@router.get("/report/download")
def download_team_evaluation_report():
    # 레포트 생성 AI 호출

    # 저장 및 전송

