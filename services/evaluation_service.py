from typing import Optional, List
from agents.workflow.quarterly_evaluation_workflow import run_auto_workflow as run_quarterly_workflow
from agents.workflow.annual_phase1_workflow import run_auto_workflow as run_middle_workflow
from agents.workflow.annual_phase2_workflow import run_auto_workflow as run_final_workflow


class EvaluationService:
    def start_quarterly_evaluation(self, period_id: int, teams: Optional[List[int]] = None):
        run_quarterly_workflow(period_id=period_id, specific_teams=teams)
        return {"period_id": period_id, "code": 201, "message": "분기 평가가 완료되었습니다."}

    def start_middle_evaluation(self, period_id: int, teams: Optional[List[int]] = None):
        run_middle_workflow(period_id=period_id, specific_teams=teams)
        return {"period_id": period_id, "code": 201, "message": "중간 평가가 완료되었습니다."}

    def start_final_evaluation(self, period_id: int, teams: Optional[List[int]] = None):
        run_final_workflow(period_id=period_id, specific_teams=teams)
        return {"period_id": period_id, "code": 201, "message": "최종 평가가 완료되었습니다."}
