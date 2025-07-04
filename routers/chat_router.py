from fastapi import APIRouter
from schemas.chat import ChatRequest, ChatResponse
from pydantic import BaseModel
from services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService() 

router = APIRouter(tags=["챗봇"])

# 챗봇 SKoro와 대화
@router.post("/skoro", response_model=ChatResponse, summary="SKoro와 대화")
def chat_with_skoro(request: ChatRequest):

    response_dict = chat_service.chat_with_skoro(request)

    return ChatResponse(**response_dict)