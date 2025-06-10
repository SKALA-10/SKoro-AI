class FeedbackReportService:
    def download_feedback_report(self, emp_no: str, period_id: int):

        # 분기 평가 레포트에서 report 가져옴(markdown)

        self.call_ai_generate_feedback_report()

        # 분기 평가 레포트 반환
        return "PDF 파일"

    # 분기평가 PDF 생성 AI 호출
    def call_ai_generate_feedback_report(self):
        # pdf 생성 AI 호출
        return "PDF 파일"
