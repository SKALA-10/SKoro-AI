"""
팀 피드백 요약 시스템 - 실행 및 테스트 모듈
"""

import sys
import os
from dotenv import load_dotenv
from .agent import FeedbackSummaryAgent

# 환경변수 로드
load_dotenv()

def setup_environment():
    """환경 설정"""
    # DB 설정 경로 추가
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
    
    try:
        from config.settings import DatabaseConfig
        db_config = DatabaseConfig()
        return db_config.DATABASE_URL
    except ImportError:
        print("❌ 데이터베이스 설정을 찾을 수 없습니다.")
        print("config/settings.py 파일과 DatabaseConfig 클래스를 확인해주세요.")
        return None

def main():
    """메인 실행 함수"""
    print("📦 팀 피드백 요약 시스템 시작")
    print("=" * 50)
    
    # 환경 설정
    database_url = setup_environment()
    if not database_url:
        return
    
    # 에이전트 초기화
    agent = FeedbackSummaryAgent(database_url)
    
    if not agent.initialize():
        print("❌ 에이전트 초기화 실패")
        return
    
    print("✅ 에이전트 초기화 완료")
    
    # 처리 대상 팀 확인
    teams = agent.get_teams_status()
    if not teams:
        print("📭 처리할 팀이 없습니다.")
        return
    
    print(f"\n📋 처리 대상 팀: {len(teams)}개")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team['team_name']} (피드백 {team['feedback_count']}개)")
    
    # 모든 팀 처리
    print("\n🚀 모든 팀 처리 시작...")
    success_count, total_count = agent.process_all_teams()
    
    # 결과 확인
    if success_count > 0:
        print("\n📊 처리 결과 확인:")
        agent.get_summary_results()

def run_specific_team(team_evaluation_id: int, period_id: int):
    """
    특정 팀만 처리하는 함수
    
    Args:
        team_evaluation_id: 팀 평가 ID
        period_id: 기간 ID
    """
    database_url = setup_environment()
    if not database_url:
        return
    
    agent = FeedbackSummaryAgent(database_url)
    
    if not agent.initialize():
        print("❌ 에이전트 초기화 실패")
        return
    
    # 특정 팀 처리
    success = agent.process_specific_team(team_evaluation_id, period_id)
    
    if success:
        print("✅ 처리 완료")
        agent.get_detailed_summary(team_evaluation_id, period_id)
    else:
        print("❌ 처리 실패")

def check_results():
    """저장된 결과 확인"""
    database_url = setup_environment()
    if not database_url:
        return
    
    agent = FeedbackSummaryAgent(database_url)
    
    if not agent.initialize():
        print("❌ 에이전트 초기화 실패")
        return
    
    agent.get_summary_results()

def view_detailed_summary(team_evaluation_id: int, period_id: int):
    """
    상세 요약 내용 보기
    
    Args:
        team_evaluation_id: 팀 평가 ID
        period_id: 기간 ID
    """
    database_url = setup_environment()
    if not database_url:
        return
    
    agent = FeedbackSummaryAgent(database_url)
    
    if not agent.initialize():
        print("❌ 에이전트 초기화 실패")
        return
    
    agent.get_detailed_summary(team_evaluation_id, period_id)

if __name__ == "__main__":
    """
    메인 실행부
    
    사용 방법:
    1. 모든 팀 처리: python run_module_chatbot_summary.py
    2. 특정 팀 처리: run_specific_team(team_evaluation_id, period_id)
    3. 결과 확인: check_results()
    4. 상세 요약 보기: view_detailed_summary(team_evaluation_id, period_id)
    """
    main()