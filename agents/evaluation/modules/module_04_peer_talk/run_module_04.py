# agents/evaluation/modules/module_04_peer_talk/run_module_04.py

from agent import create_module4_graph, Module4AgentState
from db_utils import get_all_employees_in_period
from langchain_core.messages import HumanMessage

def create_initial_state(period_id: int, emp_no: str) -> Module4AgentState:
    """초기 상태 생성"""
    return Module4AgentState(
        messages=[HumanMessage(content=f"모듈 4: {emp_no} Peer Talk 분석 시작")],
        분기=str(period_id),
        평가받는사번=emp_no,
        평가하는사번_리스트=[],
        성과지표ID_리스트=[],
        비중=[],
        키워드모음=[],
        구체적업무내용=[],
        동료평가요약줄글들=[],
        강점=[],
        우려=[],
        협업관찰=[],
        _weighted_analysis={}
    )


def run_module4_single(period_id: int, emp_no: str):
    """모듈 4 단일 직원 실행"""
    print(f"🎯 {emp_no} {period_id}분기 Peer Talk 분석 시작")
    
    # 초기 상태 생성
    state = create_initial_state(period_id, emp_no)

    # 그래프 생성 및 실행
    print("모듈 4 실행 시작...")
    module4_graph = create_module4_graph()
    result = module4_graph.invoke(state)
    
    print("모듈 4 실행 완료!")
    print(f"최종 메시지: {result['messages'][-1].content}")
    
    # 결과 확인
    if result and result.get("강점") and result.get("우려") and result.get("협업관찰"):
        print(f"✅ {emp_no} 분석 완료!")
        print(f"   강점: {result['강점'][0]}")
        print(f"   우려: {result['우려'][0]}")
        print(f"   협업관찰: {result['협업관찰'][0]}")
    else:
        print(f"⚠️ {emp_no} 분석 완료되었으나 결과가 불완전")
    
    return result


def run_module4_period(period_id: int):
    """모듈 4 특정 분기 전체 직원 실행"""
    print("=" * 80)
    print(f"🚀 모듈 4: {period_id}분기 전체 직원 Peer Talk 분석 시작")
    print("=" * 80)
    
    from db_utils import engine
    
    # 해당 분기의 모든 평가 대상자 조회
    employee_list = get_all_employees_in_period(engine, period_id)
    
    if not employee_list:
        print(f"❌ {period_id}분기에 분석할 직원이 없습니다.")
        return {}
    
    # 그래프 생성
    module4_graph = create_module4_graph()
    
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
            result = module4_graph.invoke(initial_state)
            
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
    print(f"\n✅ {period_id}분기 모듈 4 전체 분석 완료!")
    
    return results


def run_module4_quarterly():
    """모듈 4 분기별 실행 (기본값: 3분기)"""
    return run_module4_period(3)


if __name__ == "__main__":
    # 기본 실행: 3분기 전체 직원 분석
    print("🚀 모듈 4: 3분기 Peer Talk 분석을 시작합니다.")
    results = run_module4_quarterly()
    
    # 결과 확인
    print(f"\n총 {len(results)}명의 직원이 분석되었습니다.")