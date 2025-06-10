from schemas.chat import ChatRequest, ChatResponse

class ChatService:
    def chat_with_skoro(self, request: ChatRequest) -> str:
        
        # 챗봇한테 메시지 보내고 받음

        # 프론트에 메시지 반환
        return f"{request.message}에 대한 응답입니다."