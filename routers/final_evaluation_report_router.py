from fastapi import APIRouter
from services.final_evaluation_report_service import FinalEvaluationReportService

router = APIRouter()
final_evaluation_report_service = FinalEvaluationReportService()

# 팀원의 최종 평가 레포트 다운로드
@router.get("/employees/{emp_no}/final-evaluation-report/{period_id}/download")
def download_member_final_evaluation_report(emp_no: str, period_id: int):
    return final_evaluation_report_service.download_final_evaluation_report(emp_no, period_id)

# 본인의 최종 평가 레포트 다운로드
@router.get("/final-evaluation-report/{period_id}/download")
def download_my_final_evaluation_report(period_id: int):
    emp_no = "E001"  # 인증 정보에서 가져오는 구조로 교체
    return final_evaluation_report_service.download_final_evaluation_report(emp_no, period_id)

