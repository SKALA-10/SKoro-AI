"""
run_module_04.py - 메인 실행 파일
동료평가 분석 워크플로우 실행을 위한 모듈입니다.
"""

from typing import Dict, List
from sqlalchemy import create_engine, text
from langgraph.graph import StateGraph, END

from agent import (
    complete_data_mapping_agent,
    simple_context_generation_agent,
    weighted_analysis_agent,
    improved_feedback_generation_agent,
    database_storage_agent
)
from db_utils import (
    get_all_employees_in_period,
    row_to_dict
)


def create_safe_langgraph_nodes(engine):
    """안전한 랭그래프 노드 함수들 생성 (상태 보존)"""
    
    def safe_data_mapping_node(state: Dict) -> Dict:
        """안전한 데이터 매핑 노드"""
        print("🔄 [안전] 데이터 매핑 노드 시작...")
        try:
            result = complete_data_mapping_agent(state, engine)
            print(f"   결과: 평가자 {len(result.get('평가하는사번_리스트', []))}명")
            
            # 상태 검증
            required_keys = ['평가하는사번_리스트', '비중', '키워드모음', '구체적업무내용']
            for key in required_keys:
                if key not in result:
                    print(f"   ⚠️ 누락된 키: {key}")
                    result[key] = []
                else:
                    print(f"   ✅ {key}: {len(result[key])}개")
            
            return result
        except Exception as e:
            print(f"   ❌ 데이터 매핑 실패: {str(e)}")
            # 실패 시 기본값으로 초기화
            state["평가하는사번_리스트"] = []
            state["비중"] = []
            state["키워드모음"] = []
            state["구체적업무내용"] = []
            return state
    
    def safe_context_generation_node(state: Dict) -> Dict:
        """안전한 맥락 생성 노드"""
        print("🔄 [안전] 맥락 생성 노드 시작...")
        
        # 입력 상태 검증
        print(f"   입력 검증:")
        print(f"     평가하는사번_리스트: {state.get('평가하는사번_리스트', 'MISSING')}")
        print(f"     비중: {state.get('비중', 'MISSING')}")
        print(f"     키워드모음: {len(state.get('키워드모음', []))}개")
        
        try:
            # 필수 키가 없으면 추가
            if '평가하는사번_리스트' not in state:
                state['평가하는사번_리스트'] = []
            if '비중' not in state:
                state['비중'] = []
            if '키워드모음' not in state:
                state['키워드모음'] = []
            if '구체적업무내용' not in state:
                state['구체적업무내용'] = []
            if '동료평가요약줄글들' not in state:
                state['동료평가요약줄글들'] = []
                
            result = simple_context_generation_agent(state)
            print(f"   결과: 문장 {len(result.get('동료평가요약줄글들', []))}개")
            return result
        except Exception as e:
            print(f"   ❌ 맥락 생성 실패: {str(e)}")
            # 실패 시 상태 유지
            state["동료평가요약줄글들"] = []
            return state
    
    def safe_weighted_analysis_node(state: Dict) -> Dict:
        """안전한 가중치 분석 노드"""
        print("🔄 [안전] 가중치 분석 노드 시작...")
        try:
            result = weighted_analysis_agent(state)
            analysis = result.get('_weighted_analysis', {})
            print(f"   결과: 키워드 {len(analysis.get('weighted_scores', {}))}개")
            return result
        except Exception as e:
            print(f"   ❌ 가중치 분석 실패: {str(e)}")
            # 실패 시 상태 유지
            state["_weighted_analysis"] = {}
            return state
    
    def safe_feedback_generation_node(state: Dict) -> Dict:
        """안전한 피드백 생성 노드"""
        print("🔄 [안전] 피드백 생성 노드 시작...")
        try:
            result = improved_feedback_generation_agent(state)
            print(f"   결과: 강점 {len(result.get('강점', []))}, 우려 {len(result.get('우려', []))}, 협업관찰 {len(result.get('협업관찰', []))}")
            return result
        except Exception as e:
            print(f"   ❌ 피드백 생성 실패: {str(e)}")
            # 실패 시 기본값 설정
            state["강점"] = ["분석 중 오류가 발생했습니다"]
            state["우려"] = ["추가 분석이 필요합니다"]
            state["협업관찰"] = ["데이터 부족으로 분석이 제한적입니다"]
            return state
    
    def safe_database_storage_node(state: Dict) -> Dict:
        """안전한 DB 저장 노드"""
        print("🔄 [안전] DB 저장 노드 시작...")
        try:
            result = database_storage_agent(state, engine)
            print("   결과: DB 저장 완료")
            return result
        except Exception as e:
            print(f"   ❌ DB 저장 실패: {str(e)}")
            # 실패해도 상태 반환
            return state
    
    return {
        "data_mapping": safe_data_mapping_node,
        "context_generation": safe_context_generation_node,
        "weighted_analysis": safe_weighted_analysis_node,
        "feedback_generation": safe_feedback_generation_node,
        "database_storage": safe_database_storage_node
    }


def create_safe_peer_evaluation_workflow(engine):
    """안전한 동료평가 분석 랭그래프 워크플로우 생성"""
    
    # 노드 함수들 생성
    nodes = create_safe_langgraph_nodes(engine)
    
    # StateGraph 생성
    workflow = StateGraph(Dict)
    
    # 노드들 추가
    workflow.add_node("data_mapping", nodes["data_mapping"])
    workflow.add_node("context_generation", nodes["context_generation"])
    workflow.add_node("weighted_analysis", nodes["weighted_analysis"])
    workflow.add_node("feedback_generation", nodes["feedback_generation"])
    workflow.add_node("database_storage", nodes["database_storage"])
    
    # 엣지 연결 (순차 실행)
    workflow.set_entry_point("data_mapping")
    workflow.add_edge("data_mapping", "context_generation")
    workflow.add_edge("context_generation", "weighted_analysis")
    workflow.add_edge("weighted_analysis", "feedback_generation")
    workflow.add_edge("feedback_generation", "database_storage")
    workflow.add_edge("database_storage", END)
    
    return workflow.compile()


def create_initial_state(period_id: int, emp_no: str) -> Dict:
    """초기 상태 생성"""
    return {
        "분기": str(period_id),
        "평가받는사번": emp_no,
        "평가하는사번_리스트": [],
        "성과지표ID_리스트": [],
        "비중": [],
        "키워드모음": [],
        "구체적업무내용": [],
        "동료평가요약줄글들": [],
        "강점": [],
        "우려": [],
        "협업관찰": [],
        "_weighted_analysis": {}
    }


def check_feedback_reports(engine, period_id: int, emp_no: str) -> bool:
    """해당 직원의 피드백 보고서가 이미 존재하는지 확인"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT fr.feedback_report_id
                FROM feedback_reports fr
                JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                WHERE te.period_id = :period_id
                  AND fr.emp_no = :emp_no
                  AND fr.peer_review_result IS NOT NULL
                LIMIT 1
            """)
            
            result = conn.execute(query, {
                "period_id": period_id,
                "emp_no": emp_no
            }).fetchone()
            
            return result is not None
    except Exception as e:
        print(f"❌ 피드백 보고서 확인 실패: {str(e)}")
        return False


def run_safe_period_analysis(engine, period_id: int):
    """안전한 랭그래프 워크플로우로 특정 분기의 모든 직원 분석"""
    print("=" * 80)
    print(f"🚀 안전한 {period_id}분기 전체 직원 동료평가 분석 시작")
    print("=" * 80)
    
    # 해당 분기의 모든 평가 대상자 조회
    employee_list = get_all_employees_in_period(engine, period_id)
    
    if not employee_list:
        print(f"❌ {period_id}분기에 분석할 직원이 없습니다.")
        return {}
    
    # 워크플로우 생성
    workflow = create_safe_peer_evaluation_workflow(engine)
    
    # 결과 저장
    results = {}
    success_count = 0
    
    # 각 직원 분석
    for i, emp_no in enumerate(employee_list, 1):
        print(f"\n[{i}/{len(employee_list)}] 📊 {emp_no} 분석 시작...")
        print("-" * 60)
        
        try:
            # 초기 상태 생성
            initial_state = create_initial_state(period_id, emp_no)
            
            # 워크플로우 실행
            result = workflow.invoke(initial_state)
            
            # 결과 확인
            if result and result.get("강점") and result.get("우려") and result.get("협업관찰"):
                print(f"✅ {emp_no} 분석 완료!")
                print(f"   강점: {result['강점'][0][:50]}...")
                print(f"   우려: {result['우려'][0][:50]}...")
                print(f"   협업관찰: {result['협업관찰'][0][:50]}...")
                success_count += 1
            else:
                print(f"⚠️ {emp_no} 분석 완료되었으나 결과가 불완전")
            
            results[emp_no] = result
            
        except Exception as e:
            print(f"❌ {emp_no} 분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            results[emp_no] = None
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 분석 결과 요약")
    print("=" * 80)
    print(f"총 대상자: {len(employee_list)}명")
    print(f"성공: {success_count}명")
    print(f"실패: {len(employee_list) - success_count}명")
    print(f"\n✅ {period_id}분기 전체 분석 완료!")
    
    return results


def run_single_employee_analysis(engine, period_id: int, emp_no: str):
    """안전한 랭그래프 워크플로우로 단일 직원 분석"""
    print(f"🎯 {emp_no} {period_id}분기 동료평가 분석 시작")
    
    # 워크플로우 생성
    workflow = create_safe_peer_evaluation_workflow(engine)
    
    # 초기 상태 생성
    initial_state = create_initial_state(period_id, emp_no)
    
    try:
        # 워크플로우 실행
        result = workflow.invoke(initial_state)
        
        # 결과 확인
        if result and result.get("강점") and result.get("우려") and result.get("협업관찰"):
            print(f"✅ {emp_no} 분석 완료!")
            print(f"   강점: {result['강점'][0]}")
            print(f"   우려: {result['우려'][0]}")
            print(f"   협업관찰: {result['협업관찰'][0]}")
        else:
            print(f"⚠️ {emp_no} 분석 완료되었으나 결과가 불완전")
        
        return result
        
    except Exception as e:
        print(f"❌ {emp_no} 분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_multiple_employees_analysis(engine, period_id: int, emp_list: List[str]):
    """안전한 랭그래프 워크플로우로 여러 직원 일괄 분석"""
    print(f"🎯 {len(emp_list)}명 {period_id}분기 일괄 분석 시작")
    
    # 워크플로우 생성
    workflow = create_safe_peer_evaluation_workflow(engine)
    
    # 결과 저장
    results = {}
    success_count = 0
    
    # 각 직원 분석
    for i, emp_no in enumerate(emp_list, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(emp_list)}] 처리 중: {emp_no}")
        print(f"{'='*50}")
        
        try:
            # 초기 상태 생성
            initial_state = create_initial_state(period_id, emp_no)
            
            # 워크플로우 실행
            result = workflow.invoke(initial_state)
            
            # 결과 확인
            if result and result.get("강점") and result.get("우려") and result.get("협업관찰"):
                print(f"✅ {emp_no} 분석 완료!")
                success_count += 1
            else:
                print(f"⚠️ {emp_no} 분석 완료되었으나 결과가 불완전")
            
            results[emp_no] = result
            
        except Exception as e:
            print(f"❌ {emp_no} 분석 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            results[emp_no] = None
    
    # 요약 출력
    print(f"\n📊 일괄 분석 결과 요약:")
    print(f"   - 성공: {success_count}명")
    print(f"   - 실패: {len(emp_list) - success_count}명")
    
    return results


# 메인 실행 부분
if __name__ == "__main__":
    # 데이터베이스 연결 설정
    DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/skoro_db"
    engine = create_engine(DATABASE_URL, echo=False)
    
    # 기본 실행: 3분기 모든 직원 분석
    print("🚀 3분기 동료평가 분석을 시작합니다.")
    results = run_safe_period_analysis(engine, 3)
    
    # 결과 확인
    print(f"\n총 {len(results)}명의 직원이 분석되었습니다.")