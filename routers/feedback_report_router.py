from fastapi import APIRouter

router = APIRouter()

# 분기별 팀원 평가 레포트 다운로드
@router.get("/employees/{empNo}/feedback-report/{periodId}/download")
def download_member_feedback_report_by_period():

# 본인의 분기 평가 레포트 다운로드
@router.get("/download")
def download_my_feedback_report():