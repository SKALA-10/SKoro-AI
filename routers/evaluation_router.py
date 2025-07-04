from fastapi import APIRouter
from services.evaluation_service import EvaluationService
from schemas.evaluation import EvaluationRequest, EvaluationResponse

router = APIRouter()
evaluation_service = EvaluationService()

router = APIRouter(tags=["평가"])

# 분기 평가 시작
@router.post("/quarterly", response_model=EvaluationResponse, summary="분기 평가 시작")
def start_quarterly_evaluation(request: EvaluationRequest):
    return evaluation_service.start_quarterly_evaluation(request.period_id, request.teams)

# 중간 평가 시작
@router.post("/middle", response_model=EvaluationResponse, summary="중간 평가 시작")
def start_middle_evaluation(request: EvaluationRequest):
    return evaluation_service.start_middle_evaluation(request.period_id, request.teams)

# 최종 평가 시작
@router.post("/final", response_model=EvaluationResponse, summary="최종 평가 시작")
def start_final_evaluation(request: EvaluationRequest):
    return evaluation_service.start_final_evaluation(request.period_id, request.teams)