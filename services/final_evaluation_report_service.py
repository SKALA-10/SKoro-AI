class FinalEvaluationReportService:
    def download_final_evaluation_report(self, emp_no: str, period_id: int):

        # 최종 평가 레포트에서 report 가져옴(markdown)

        self.call_ai_generate_final_evaluation_report()

        # 최종 평가 레포트 반환
        return "PDF 파일"

    # 최종평가 PDF 생성 AI 호출
    def call_ai_generate_final_evaluation_report(self):
        # pdf 생성 AI 호출
        return "PDF 파일"
