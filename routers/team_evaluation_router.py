from fastapi import APIRouter
from services.team_evaluation_service import TeamEvaluationService

router = APIRouter()
team_evaluation_service = TeamEvaluationService()

# AI에 업무 요약 요청(중간 산출물), url에 따라 다른데 옮겨야될수도있음
@router.post("/task-summary")
def generate_task_summary():
    return team_evaluation_service.generate_task_summary()

# 팀 평가 레포트 다운로드
@router.get("/report/{period_id}/download")
def download_team_evaluation_report():
    emp_no = "E001"
    return team_evaluation_service.download_team_evaluation_report(emp_no, period_id)
    
# 최종 평가
@router.post("/{team_evaluation_id}/complete")
def complete_team_evaluation():
    # 최종 평가 AI 호출

    # 필요한 데이터 AI에 전송

    # 데이터 받으면 다 저장
