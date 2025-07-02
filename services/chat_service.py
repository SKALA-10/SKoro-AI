from schemas.chat import ChatRequest, ChatResponse
from agents.chatbot.main_chatbot import SKChatbot

class ChatService:
    def __init__(self):
        self.chatbot = SKChatbot()

    def chat_with_skoro(self, request: ChatRequest) -> dict:
        return self.chatbot.chat(
            user_id=request.user_id,
            chat_mode=request.chat_mode,
            user_input=request.message,
            appeal_complete=request.appeal_complete or False
        )