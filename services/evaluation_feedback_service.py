from schemas.evaluation_feedback import EvaluationFeedbackRequest, EvaluationFeedbackResponse

class EvaluationFeedbackService:
    def generate_feedback_summary(self, request: EvaluationFeedbackRequest) -> EvaluationFeedbackResponse:
        summary = self.call_ai_summary(request.chat_history)
        return EvaluationFeedbackResponse(emp_no=request.emp_no, content=summary)

    def call_ai_generate_feedback_summary(self, chat_history: list[str]) -> str:
        # 실제 LLM 호출 자리
        return "AI 결과값 반환"