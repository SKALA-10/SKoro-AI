"""
메인 에이전트 로직 모듈
팀 피드백 요약 시스템 - 핵심 처리 로직
"""

from typing import List, Dict, Optional, Tuple
from .db_utils import DatabaseManager
from .llm_utils import LLMSummarizer


class FeedbackSummaryAgent:
    """팀 피드백 요약 에이전트 클래스"""
    
    def __init__(self, database_url: str, model_name: str = "gpt-4o-mini"):
        """
        에이전트 초기화
        
        Args:
            database_url: 데이터베이스 연결 URL
            model_name: 사용할 LLM 모델명
        """
        self.db_manager = DatabaseManager(database_url)
        self.llm_summarizer = LLMSummarizer(model_name)
    
    def initialize(self) -> bool:
        """
        에이전트 초기화 및 연결 테스트
        
        Returns:
            bool: 초기화 성공 시 True
        """
        return self.db_manager.test_connection()
    
    def process_all_teams(self) -> Tuple[int, int]:
        """
        모든 팀의 피드백을 요약 처리
        
        Returns:
            Tuple[int, int]: (성공한 팀 수, 전체 팀 수)
        """
        print("🚀 팀별 피드백 요약 처리 시작")
        print("=" * 50)
        
        # 처리할 팀 목록 조회
        teams = self.db_manager.get_teams_to_summarize()
        
        if not teams:
            print("📭 처리할 팀이 없습니다.")
            return 0, 0
        
        success_count = 0
        total_count = len(teams)
        
        # 각 팀별로 처리
        for i, team in enumerate(teams, 1):
            team_name = team['team_name']
            team_evaluation_id = team['team_evaluation_id']
            period_id = team['period_id']
            
            print(f"\n📋 [{i}/{total_count}] {team_name} 처리 중...")
            
            try:
                if self._process_single_team(team_evaluation_id, period_id, team_name):
                    print(f"✅ {team_name}: 완료")
                    success_count += 1
                else:
                    print(f"❌ {team_name}: 처리 실패")
                    
            except Exception as e:
                print(f"❌ {team_name}: 처리 중 오류 - {e}")
                continue
        
        # 최종 결과
        print("\n" + "=" * 50)
        print(f"🎯 처리 완료: {success_count}/{total_count} 성공")
        if total_count > 0:
            print(f"📊 성공률: {success_count/total_count*100:.1f}%")
        
        return success_count, total_count
    
    def process_specific_team(self, team_evaluation_id: int, period_id: int) -> bool:
        """
        특정 팀의 피드백만 처리
        
        Args:
            team_evaluation_id: 팀 평가 ID
            period_id: 기간 ID
            
        Returns:
            bool: 처리 성공 시 True
        """
        print(f"🎯 특정 팀 처리 시작 - Team Evaluation ID: {team_evaluation_id}, Period ID: {period_id}")
        
        try:
            # 팀 이름 조회
            team_name = self.db_manager.get_team_name(team_evaluation_id)
            
            if not team_name:
                print("❌ 해당 팀을 찾을 수 없습니다.")
                return False
            
            return self._process_single_team(team_evaluation_id, period_id, team_name)
            
        except Exception as e:
            print(f"❌ 특정 팀 처리 중 오류: {e}")
            return False
    
    def _process_single_team(self, team_evaluation_id: int, period_id: int, team_name: str) -> bool:
        """
        단일 팀 처리 (내부 메서드)
        
        Args:
            team_evaluation_id: 팀 평가 ID
            period_id: 기간 ID
            team_name: 팀 이름
            
        Returns:
            bool: 처리 성공 시 True
        """
        # 1. 피드백 조회
        feedbacks = self.db_manager.get_team_feedbacks(team_evaluation_id, period_id)
        
        if not feedbacks:
            print(f"⚠️ {team_name}: 피드백 없음")
            return False
        
        # 2. LLM 요약 생성
        print(f"🤖 {team_name}: 요약 생성 중... (피드백 {len(feedbacks)}개)")
        summary = self.llm_summarizer.summarize_team_feedbacks(team_name, feedbacks)
        
        if not summary:
            print(f"❌ {team_name}: 요약 생성 실패")
            return False
        
        # 3. 요약 저장
        if self.db_manager.save_summary(team_evaluation_id, period_id, summary):
            return True
        else:
            print(f"❌ {team_name}: 저장 실패")
            return False
    
    def get_summary_results(self) -> None:
        """저장된 요약 결과 확인"""
        df = self.db_manager.get_summary_results()
        if not df.empty:
            print("📊 저장된 요약 결과:")
            print(df.to_string(index=False))
        else:
            print("📭 저장된 요약 결과가 없습니다.")
    
    def get_detailed_summary(self, team_evaluation_id: int, period_id: int) -> None:
        """
        특정 팀의 상세 요약 내용 출력
        
        Args:
            team_evaluation_id: 팀 평가 ID
            period_id: 기간 ID
        """
        result = self.db_manager.get_detailed_summary(team_evaluation_id, period_id)
        
        if result:
            print(f"📋 {result['team_name']} - {result['period_name']} 요약:")
            print("-" * 50)
            print(result['content'])
        else:
            print("해당 팀의 요약을 찾을 수 없습니다.")
    
    def get_teams_status(self) -> List[Dict]:
        """
        처리 대상 팀 목록 조회
        
        Returns:
            List[Dict]: 팀 상태 정보 리스트
        """
        return self.db_manager.get_teams_to_summarize()