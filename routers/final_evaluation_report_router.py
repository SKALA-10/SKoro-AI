from fastapi import APIRouter

router = APIRouter()

# 본인의 최종 평가 레포트 다운로드
@router.get("/download")
def download_my_final_evaluation_report():

