from fastapi import APIRouter

router = APIRouter()

# 챗봇 SKoro와 대화
@router.post("/skoro")
def chat_with_skoro():