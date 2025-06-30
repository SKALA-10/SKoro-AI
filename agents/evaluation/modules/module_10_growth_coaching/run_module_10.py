# ================================================================
# run_module10.py - 모듈 10 실행 파일
# ================================================================

from typing import Optional, List, Dict
from langchain_core.messages import HumanMessage

from agents.evaluation.modules.module_10_growth_coaching.agent import Module10AgentState, create_module10_graph
from agents.evaluation.modules.module_10_growth_coaching.db_utils import *

from sqlalchemy import create_engine, text
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# DB 설정
from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ================================================================
# 실행 함수들
# ================================================================

def run_module10_evaluation(emp_no: str, period_id: int, report_type: str = "quarterly"):
    """모듈 10 개인 성장 및 코칭 분석 실행"""
    
    print(f"🚀 모듈 10 개인 성장 및 코칭 분석 시작: {emp_no} ({report_type})")
    
    # State 정의
    state = Module10AgentState(
        messages=[HumanMessage(content=f"모듈 10 시작: {emp_no}")],
        emp_no=emp_no,
        period_id=period_id,
        report_type=report_type,
        basic_info={},
        performance_data={},
        peer_talk_data={},
        fourp_data={},
        collaboration_data={},
        module7_score_data={},
        module9_final_data={},
        growth_analysis={},
        focus_coaching_needed=False,
        focus_coaching_analysis={},
        individual_growth_result={},
        manager_coaching_result={},
        overall_comment="",
        storage_result={},
        processing_status="started",
        error_messages=[]
    )
    
    # 그래프 생성 및 실행
    module10_graph = create_module10_graph()
    
    try:
        result = module10_graph.invoke(state)
        
        print("✅ 모듈 10 개인 성장 및 코칭 분석 완료!")
        print(f"📊 처리 상태: {result.get('processing_status')}")
        
        if result.get('storage_result'):
            storage = result['storage_result']
            print(f"💾 저장 결과: {storage.get('updated_records', 0)}개 레코드 업데이트")
            
        if result.get('error_messages'):
            print(f"⚠️ 오류 메시지: {result['error_messages']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 모듈 10 실행 실패: {e}")
        return None

def run_team_module10_evaluation(team_id: str, period_id: int, report_type: str = "quarterly"):
    """팀 단위 모듈 10 실행"""
    
    print(f"🚀 팀 단위 모듈 10 실행: {team_id} ({report_type})")
    
    # 팀원 목록 조회
    with engine.connect() as connection:
        query = text("SELECT emp_no, emp_name FROM employees WHERE team_id = :team_id")
        results = connection.execute(query, {"team_id": team_id}).fetchall()
        team_members = [{"emp_no": row[0], "emp_name": row[1]} for row in results]
    
    if not team_members:
        print(f"❌ 팀원이 없습니다: {team_id}")
        return None
    
    print(f"📋 대상 팀원: {len(team_members)}명")
    
    results = {}
    success_count = 0
    
    for member in team_members:
        emp_no = member["emp_no"]
        emp_name = member["emp_name"]
        
        print(f"\n{'='*30}")
        print(f"처리 중: {emp_name}({emp_no})")
        
        result = run_module10_evaluation(emp_no, period_id, report_type)
        results[emp_no] = result
        
        if result and result.get('processing_status') == 'completed':
            success_count += 1
    
    print(f"\n🎯 팀 단위 실행 완료:")
    print(f"   성공: {success_count}/{len(team_members)}명")
    
    return results

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def test_module10(emp_no: Optional[str] = None, period_id: int = 4, report_type: str = "quarterly"):
    """모듈 10 테스트"""
    if not emp_no:
        # 테스트용 직원 자동 선택
        with engine.connect() as connection:
            query = text("""
                SELECT e.emp_no, e.emp_name 
                FROM employees e
                JOIN final_evaluation_reports fer ON e.emp_no = fer.emp_no
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE te.period_id = :period_id
                LIMIT 1
            """)
            result = connection.execute(query, {"period_id": period_id}).fetchone()
            
            if result:
                emp_no = result[0]
                print(f"🧪 테스트 직원 자동 선택: {result[1]}({emp_no})")
            else:
                print("❌ 테스트할 직원이 없습니다")
                return
    
    if emp_no is None:
        print("❌ emp_no가 None입니다")
        return
        
    return run_module10_evaluation(emp_no, period_id, report_type)

def display_result_summary(result: Dict):
    """결과 요약 출력"""
    if not result:
        print("❌ 결과가 없습니다.")
        return
    
    emp_no = result.get('emp_no', 'Unknown')
    status = result.get('processing_status', 'Unknown')
    
    print(f"\n📊 {emp_no} 결과 요약:")
    print(f"   상태: {status}")
    
    if status == 'completed':
        individual_result = result.get('individual_growth_result', {})
        manager_result = result.get('manager_coaching_result', {})
        overall_comment = result.get('overall_comment', '')
        
        print(f"   성장 포인트: {len(individual_result.get('growth_points', []))}개")
        print(f"   보완 영역: {len(individual_result.get('improvement_areas', []))}개")
        print(f"   추천 활동: {len(individual_result.get('recommended_activities', []))}개")
        print(f"   집중 코칭: {'필요' if result.get('focus_coaching_needed') else '불필요'}")
        print(f"   종합 총평: {len(overall_comment)}자")
        
        storage = result.get('storage_result', {})
        print(f"   저장 상태: {storage.get('updated_records', 0)}개 레코드")
    
    if result.get('error_messages'):
        print(f"   ⚠️ 오류: {len(result['error_messages'])}건")

# ================================================================
# 메인 실행 부분
# ================================================================

if __name__ == "__main__":
    print("🚀 모듈 10: 개인 성장 및 코칭 모듈 준비 완료!")
    print("\n🔥 주요 기능:")
    print("✅ 7개 데이터 소스 통합 분석 (기본 5개 + 연말 2개)")
    print("✅ LLM 기반 성장 포인트 및 보완 영역 추출")
    print("✅ 집중 코칭 대상 자동 선정")
    print("✅ 개인용/팀장용 차별화된 결과 생성")
    print("✅ 종합 총평 생성 (모든 모듈 결과 통합)")
    print("✅ JSON + TEXT 형태로 DB 저장")
    
    print("\n📋 실행 명령어:")
    print("1. run_module10_evaluation('E002', 4, 'quarterly')      # 개별 실행 (분기)")
    print("2. run_module10_evaluation('E002', 4, 'annual')        # 개별 실행 (연말)")
    print("3. run_team_module10_evaluation('1', 4, 'annual')      # 팀 단위 실행")
    print("4. test_module10()                                     # 테스트 실행")
    print("5. get_teams_with_data(4)                              # 데이터 있는 팀 조회")
    print("6. display_result_summary(result)                      # 결과 요약 출력")
    print("7. clean_ai_team_coaching_data('1', 4)                # 특정 팀 데이터 정리")
    print("8. clean_all_team_coaching_data(4)                    # 모든 팀 데이터 정리")
    
    print("\n📊 DB 저장 구조:")
    print("- ai_growth_coaching: 성장 제안 3개 항목 (JSON)")
    print("- overall_comment: 전체 레포트 종합 총평 (TEXT)")
    print("- ai_team_coaching: 팀장용 코칭 정보 (JSON)")
    
    print("\n🎯 필요한 DB 스키마:")
    print("ALTER TABLE feedback_reports ADD COLUMN overall_comment TEXT;")
    print("ALTER TABLE final_evaluation_reports ADD COLUMN overall_comment TEXT;")
    
    print("\n🔧 수정 사항:")
    print("✅ LLM 응답 후 emp_no/name 강제 설정")
    print("✅ JSON 구조 명확화")
    print("✅ 빈 emp_no 데이터 정리 함수 추가")
    
    # 자동 테스트 (필요시 주석 해제)
    # test_module10()
    
    # 실제 실행 예시
    run_module10_evaluation('SK0002', 2, 'quarterly')
    run_module10_evaluation('SK0003', 2, 'quarterly')
    run_module10_evaluation('SK0004', 2, 'quarterly')

    run_module10_evaluation('SK0002', 4, 'annual')
    run_module10_evaluation('SK0003', 4, 'annual')
    run_module10_evaluation('SK0004', 4, 'annual')