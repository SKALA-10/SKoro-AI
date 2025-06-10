from fastapi import APIRouter
from services.feedback_report_service import FeedbackReportService

router = APIRouter()
feedback_report_service = FeedbackReportService()

# 팀원의 분기 평가 레포트 다운로드
@router.get("/employees/{empNo}/feedback-report/{periodId}/download")
def download_member_feedback_report(emp_no: str, period_id: int):
    return feedback_report_service.download_feedback_report(emp_no, period_id)

# 본인의 분기 평가 레포트 다운로드
@router.get("/{period_id}/download")
def download_my_feedback_report(period_id: int):
    emp_no = "E001"  # 인증 정보에서 가져오는 구조로 교체
    return feedback_report_service.download_feedback_report(emp_no, period_id)