# ================================================================
# run_module7.py - 모듈 7 실행 파일
# ================================================================

from typing import List, Optional
from langchain_core.messages import HumanMessage

from agents.evaluation.modules.module_07_final_evaluation.agent import TeamModule7AgentState, create_team_module7_graph
from agents.evaluation.modules.module_07_final_evaluation.db_utils import get_all_teams_with_data
from agents.evaluation.modules.module_07_final_evaluation.scoring_utils import preview_achievement_scoring
from agents.evaluation.modules.module_07_final_evaluation.llm_utils import *
from sqlalchemy import create_engine, text
import sys
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

load_dotenv()

# DB 설정
from config.settings import *

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ================================================================
# 실행 함수들
# ================================================================

def run_team_module7_evaluation(team_id: str, period_id: int = 4):
    """팀 단위 모듈 7 연말 종합평가 실행 (SK 등급 기반 절대평가 + CL 정규화 포함)"""
    
    print(f"🚀 팀 단위 모듈 7 + SK 등급 기반 절대평가 + CL 정규화 실행 시작: {team_id} (period_id: {period_id})")
    
    # State 정의
    state = TeamModule7AgentState(
        messages=[HumanMessage(content=f"팀 {team_id}: SK 등급 기반 종합평가 + CL 정규화 시작")],
        team_id=team_id,
        period_id=period_id,
        team_members=[],
        team_achievement_data=[],
        team_fourp_data=[],
        team_quarterly_data={},
        weights_by_cl={},
        individual_scores=[],
        evaluation_comments=[],
        processed_count=0,
        failed_members=[]
    )
    
    # 그래프 생성 및 실행
    team_module7_graph = create_team_module7_graph()
    
    try:
        result = team_module7_graph.invoke(state)
        
        print("✅ 팀 단위 모듈 7 + SK 등급 기반 절대평가 + CL 정규화 실행 완료!")
        print(f"📊 결과:")
        for message in result['messages']:
            print(f"  - {message.content}")
        
        if result.get('processed_count'):
            print(f"🎯 처리 완료: {result['processed_count']}명")
            if result.get('failed_members'):
                print(f"❌ 실패한 팀원: {result['failed_members']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 팀 단위 모듈 7 + SK 등급 기반 절대평가 + CL 정규화 실행 실패: {e}")
        return None

def run_multiple_teams_module7(team_ids: List[str], period_id: int = 4):
    """여러 팀 일괄 실행"""
    print(f"🚀 다중 팀 모듈 7 + SK 등급 기반 절대평가 + CL 정규화 실행: {len(team_ids)}개 팀")
    
    results = {}
    total_processed = 0
    total_failed = 0
    
    for team_id in team_ids:
        print(f"\n{'='*50}")
        print(f"팀 {team_id} 처리 중...")
        
        result = run_team_module7_evaluation(team_id, period_id)
        results[team_id] = result
        
        if result:
            total_processed += result.get('processed_count', 0)
            total_failed += len(result.get('failed_members', []))
    
    print(f"\n🎯 전체 결과:")
    print(f"   처리된 팀: {len([r for r in results.values() if r is not None])}/{len(team_ids)}")
    print(f"   처리된 인원: {total_processed}명")
    print(f"   실패한 인원: {total_failed}명")
    
    return results

# ================================================================
# 개별 실행 호환 함수 (기존 코드 호환성)
# ================================================================

def run_module7_evaluation(emp_no: str, period_id: int = 4):
    """개별 실행 (기존 호환성을 위한 래퍼 함수)"""
    # 직원의 팀 ID 조회
    with engine.connect() as connection:
        query = text("SELECT team_id FROM employees WHERE emp_no = :emp_no")
        result = connection.execute(query, {"emp_no": emp_no}).fetchone()
        
        if not result:
            print(f"❌ 직원 정보 없음: {emp_no}")
            return None
        
        team_id = result.team_id
    
    print(f"🔄 개별 실행을 팀 단위로 변환: {emp_no} → 팀 {team_id}")
    
    # 팀 단위로 실행
    return run_team_module7_evaluation(team_id, period_id)

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def test_team_module7(team_id: Optional[str] = None, period_id: int = 4):
    """팀 모듈 7 + SK 등급 기반 절대평가 + CL 정규화 테스트"""
    if not team_id:
        teams = get_all_teams_with_data(period_id)
        if teams:
            team_id = teams[0]
            print(f"🧪 테스트 팀 자동 선택: {team_id}")
        else:
            print("❌ 테스트할 팀이 없습니다")
            return
    
    return run_team_module7_evaluation(team_id, period_id)

# ================================================================
# 실행 예시
# ================================================================

if __name__ == "__main__":
    print("🚀 팀 단위 모듈 7 + SK 등급 기반 절대평가 + CL별 정규화 준비 완료!")
    print("\n🔥 주요 변경사항:")
    print("✅ SK 등급 체계 기반 절대평가 (S/A/B/C/D → 1-5점)")
    print("✅ 달성률 100% = 3.5점, 120% = 5.0점 명확한 기준")
    print("✅ CL별 정규화: 4명 이상일 때만 적용, 3명 이하는 원시점수 유지")
    print("✅ DB 저장: raw_score(원시점수) + score(정규화점수) 별도 저장")
    print("✅ 달성률:4P 가중치 유지 (CL3:6:4, CL2:5:5, CL1:4:6)")
    
    print("\n실행 명령어:")
    print("1. run_team_module7_evaluation('TEAM001', 4)     # 단일 팀 실행")
    print("2. run_multiple_teams_module7(['TEAM001', 'TEAM002'], 4)  # 다중 팀 실행")
    print("3. run_module7_evaluation('E002', 4)             # 개별 실행 (호환성)")
    print("4. test_team_module7()                           # 테스트 실행")
    print("5. preview_achievement_scoring()                 # 점수 체계 미리보기")
    
    # 점수 체계 미리보기 실행
    preview_achievement_scoring()
    
    # 자동 테스트 (필요시 주석 해제)
    test_team_module7()