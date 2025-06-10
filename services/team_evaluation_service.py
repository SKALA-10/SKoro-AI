class TeamEvaluationService:
    def generate_task_summary(self):
        # 필요한 데이터 db에서 가져오기

        # 업무 요약 AI 호출
        call_ai_generate_task_summary()

        # 팀 평가 레포트 DB에 저장(중간 산출물)
        save_team_evaluation_report()

        # 임시 평가 Redis에 저장
        save_temp_evaluation_to_redis()

        # 팀 평가 레포트 반환
        return "팀 평가 레포트 반환"

    # 업무 요약 AI 호출
    def call_ai_generate_task_summary(self):
        return "업무 요약"

    # 팀 평가 레포트 DB에 저장(중간 산출물)
    def save_team_evaluation_report(self):

    # 임시 평가 Redis에 저장
    def save_temp_evaluation_to_redis(self):

    
    def download_team_evaluation_report(self, emp_no: str, period_id: int):

        # 팀 평가에서 report 가져옴(markdown)

        self.call_ai_generate_team_evaluation_report()

        # 팀 평가 레포트 반환
        return "PDF 파일"

    # 최종평가 PDF 생성 AI 호출
    def call_ai_generate_team_evaluation_report(self):
        # pdf 생성 AI 호출
        return "PDF 파일"