class EvaluationFeedbackSummary:
    def summarize_and_save_team_feedback(self, period_id: int) -> None:
        # period에 맞는 team_evaluation_id 단위로 조회해서 AI에 전달

        # AI에서 받은 데이터를 team_evaluation_id, period_id를 바탕으로 evaluation_feedback_summaries 테이블에 저장 