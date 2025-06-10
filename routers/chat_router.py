from fastapi import APIRouter
from schemas.chat import ChatRequest, ChatResponse
from pydantic import BaseModel
from services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService() 

# 챗봇 SKoro와 대화
@router.post("/skoro", response_model=ChatResponse)
def chat_with_skoro(request: ChatRequest):

    response = chat_service.chat_with_skoro(request)

    return ChatResponse(response=response)