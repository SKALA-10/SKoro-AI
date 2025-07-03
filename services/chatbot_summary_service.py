from agents.chatbot_summary.run_module_chatbot_summary import FeedbackSummaryAgent, setup_environment

class ChatbotSummaryService:
    def run_summary_for_all_teams(self) -> dict:
        """모든 팀에 대해 피드백 요약을 실행"""
        print("📦 팀 피드백 요약 시스템 시작")
        print("=" * 50)

        database_url = setup_environment()
        if not database_url:
            return {"code": 500, "message": "❌ DB 설정 실패"}

        agent = FeedbackSummaryAgent(database_url)

        if not agent.initialize():
            return {"code": 500, "message": "❌ 에이전트 초기화 실패"}

        teams = agent.get_teams_status()
        if not teams:
            return {"code": 200, "message": "📭 처리할 팀이 없습니다."}

        success_count, total_count = agent.process_all_teams()
        agent.get_summary_results()

        return {
            "code": 201,
            "message": f"✅ 전체 {total_count}팀 중 {success_count}팀 생성 완료"
        }