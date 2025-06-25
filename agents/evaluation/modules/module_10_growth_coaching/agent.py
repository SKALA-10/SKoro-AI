# ================================================================
# agent_module10.py - 모듈 10 LangGraph 에이전트 및 상태 관리
# ================================================================

from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage
import operator
from langgraph.graph import StateGraph, START, END

from db_utils import (
    fetch_basic_info,
    fetch_performance_data,
    fetch_peer_talk_data,
    fetch_fourp_data,
    fetch_collaboration_data,
    fetch_module7_score_data,
    fetch_module9_final_score_data,
    calculate_ranking_by_achievement,
    save_individual_result,
    save_manager_result
)

from llm_utils import (
    call_llm_for_growth_analysis,
    call_llm_for_focus_coaching_analysis,
    call_llm_for_individual_result,
    call_llm_for_overall_comment,
    call_llm_for_manager_result
)

# ================================================================
# Module10AgentState 정의
# ================================================================

class Module10AgentState(TypedDict):
    """모듈 10 (개인 성장 및 코칭) 상태"""
    messages: Annotated[List[HumanMessage], operator.add]
    
    # 입력 정보
    emp_no: str
    period_id: int
    report_type: str  # "quarterly" or "annual"
    
    # 수집된 데이터 (기본 5개 + 연말 추가 2개)
    basic_info: Dict
    performance_data: Dict
    peer_talk_data: Dict
    fourp_data: Dict
    collaboration_data: Dict
    
    # 연말 추가 데이터
    module7_score_data: Dict  # 팀 내 정규화 점수
    module9_final_data: Dict  # 부문 정규화 최종 점수
    
    # 중간 처리 결과
    growth_analysis: Dict
    focus_coaching_needed: bool
    focus_coaching_analysis: Dict
    
    # 최종 결과
    individual_growth_result: Dict  # 개인용 JSON
    manager_coaching_result: Dict   # 팀장용 JSON
    overall_comment: str            # 종합 총평
    storage_result: Dict
    
    # 처리 상태
    processing_status: str
    error_messages: List[str]

# ================================================================
# 서브모듈 함수들
# ================================================================

def data_collection_submodule(state: Module10AgentState) -> Module10AgentState:
    """1. 데이터 수집 서브모듈 (종합 총평용 데이터 포함)"""
    
    emp_no = state["emp_no"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    
    try:
        print(f"🔍 모듈 10 데이터 수집 시작: {emp_no} ({report_type})")
        
        # 기본 5개 데이터 소스 수집
        basic_info = fetch_basic_info(emp_no)
        if not basic_info or not basic_info.get("team_id"):
            raise ValueError(f"{emp_no}의 기본 정보 또는 팀 정보를 찾을 수 없습니다.")

        team_id = basic_info["team_id"]

        performance_data = fetch_performance_data(emp_no, period_id, report_type)

        # 달성률 기반으로 실시간 순위 계산
        ranking = calculate_ranking_by_achievement(emp_no, team_id, period_id, report_type)
        performance_data['ranking'] = ranking
        print(f"   📊 달성률 기반 순위 계산 완료: {ranking}위")

        peer_talk_data = fetch_peer_talk_data(emp_no, period_id, report_type)
        fourp_data = fetch_fourp_data(emp_no, period_id, report_type)
        collaboration_data = fetch_collaboration_data(emp_no, period_id)
        
        # 연말 추가 데이터 수집
        module7_score_data = fetch_module7_score_data(emp_no, period_id, report_type)
        module9_final_data = fetch_module9_final_score_data(emp_no, period_id, report_type)
        
        total_sources = 5 + (2 if report_type == "annual" else 0)
        print(f"   ✅ {total_sources}개 데이터 소스 수집 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="데이터 수집 완료")],
            "basic_info": basic_info,
            "performance_data": performance_data,
            "peer_talk_data": peer_talk_data,
            "fourp_data": fourp_data,
            "collaboration_data": collaboration_data,
            "module7_score_data": module7_score_data,
            "module9_final_data": module9_final_data,
            "processing_status": "data_collected"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"데이터 수집 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": [str(e)]
        })
        return updated_state

def growth_analysis_submodule(state: Module10AgentState) -> Module10AgentState:
    """2. 성장 분석 서브모듈"""
    
    try:
        print(f"📊 성장 분석 시작")
        
        growth_analysis = call_llm_for_growth_analysis(
            state["basic_info"],
            state["performance_data"], 
            state["peer_talk_data"],
            state["fourp_data"],
            state["collaboration_data"]
        )
        
        print(f"   ✅ 성장 분석 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="성장 분석 완료")],
            "growth_analysis": growth_analysis,
            "processing_status": "growth_analyzed"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 성장 분석 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"성장 분석 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def focus_coaching_selection_submodule(state: Module10AgentState) -> Module10AgentState:
    """3. 집중 코칭 대상 선정 서브모듈"""
    
    try:
        print(f"🎯 집중 코칭 필요성 분석 시작")
        
        focus_analysis = call_llm_for_focus_coaching_analysis(
            state["peer_talk_data"],
            state["performance_data"],
            state["collaboration_data"],
            state["fourp_data"]
        )
        
        focus_needed = focus_analysis.get("focus_coaching_needed", False)
        print(f"   ✅ 집중 코칭 필요성: {focus_needed}")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"집중 코칭 분석 완료: {focus_needed}")],
            "focus_coaching_needed": focus_needed,
            "focus_coaching_analysis": focus_analysis,
            "processing_status": "focus_analyzed"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 집중 코칭 분석 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"집중 코칭 분석 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def individual_result_generation_submodule(state: Module10AgentState) -> Module10AgentState:
    """4. 개인용 결과 생성 서브모듈 (overall_comment 포함)"""
    
    try:
        print(f"👤 개인용 결과 생성 시작")
        
        # 개인용 성장 제안 결과 생성
        individual_result = call_llm_for_individual_result(
            state["basic_info"],
            state["growth_analysis"],
            state["performance_data"],
            state["report_type"]
        )
        
        # 종합 총평 생성 (모든 모듈 결과 포함)
        overall_comment = call_llm_for_overall_comment(
            state["basic_info"],
            state["performance_data"],
            state["peer_talk_data"],
            state["fourp_data"],
            state["collaboration_data"],
            state["growth_analysis"],
            state["module7_score_data"],
            state["module9_final_data"],
            state["report_type"]
        )
        
        print(f"   ✅ 개인용 결과 + 종합 총평 생성 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="개인용 결과 생성 완료")],
            "individual_growth_result": individual_result,
            "overall_comment": overall_comment,
            "processing_status": "individual_generated"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 개인용 결과 생성 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"개인용 결과 생성 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def manager_result_generation_submodule(state: Module10AgentState) -> Module10AgentState:
    """5. 팀장용 결과 생성 서브모듈"""
    
    try:
        print(f"👨‍💼 팀장용 결과 생성 시작")
        
        manager_result = call_llm_for_manager_result(
            state["basic_info"],
            state["growth_analysis"],
            state["performance_data"],
            state["collaboration_data"],
            state["focus_coaching_analysis"],
            state["focus_coaching_needed"]
        )
        
        print(f"   ✅ 팀장용 결과 생성 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="팀장용 결과 생성 완료")],
            "manager_coaching_result": manager_result,
            "processing_status": "manager_generated"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ 팀장용 결과 생성 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"팀장용 결과 생성 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)]
        })
        return updated_state

def storage_submodule(state: Module10AgentState) -> Module10AgentState:
    """6. DB 저장 서브모듈 (종합 총평 포함)"""
    
    try:
        print(f"💾 DB 저장 시작")
        
        emp_no = state["emp_no"]
        period_id = state["period_id"]
        report_type = state["report_type"]
        
        # 개인용 결과 + 종합 총평 저장
        individual_saved = save_individual_result(
            emp_no, period_id, report_type, 
            state["individual_growth_result"],
            state["overall_comment"]
        )
        
        # 팀장용 결과 저장
        manager_saved = save_manager_result(
            emp_no, period_id,
            state["manager_coaching_result"]
        )
        
        storage_result = {
            "individual_saved": individual_saved,
            "manager_saved": manager_saved,
            "updated_records": int(individual_saved) + int(manager_saved)
        }
        
        print(f"   ✅ 저장 완료: 개인용({individual_saved}), 팀장용({manager_saved})")
        print(f"   📝 종합 총평 저장: {len(state['overall_comment'])}자")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="DB 저장 완료")],
            "storage_result": storage_result,
            "processing_status": "completed"
        })
        return updated_state
        
    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"DB 저장 실패: {str(e)}")],
            "processing_status": "failed",
            "error_messages": state.get("error_messages", []) + [str(e)],
            "storage_result": {"individual_saved": False, "manager_saved": False, "updated_records": 0}
        })
        return updated_state

# ================================================================
# 워크플로우 생성
# ================================================================

def create_module10_graph():
    """모듈 10 그래프 생성 및 반환"""
    module10_workflow = StateGraph(Module10AgentState)
    
    # 노드 추가 (State 키와 겹치지 않도록 이름 변경)
    module10_workflow.add_node("collect_data", data_collection_submodule)
    module10_workflow.add_node("analyze_growth", growth_analysis_submodule)
    module10_workflow.add_node("select_focus_coaching", focus_coaching_selection_submodule)
    module10_workflow.add_node("generate_individual_result", individual_result_generation_submodule)
    module10_workflow.add_node("generate_manager_result", manager_result_generation_submodule)
    module10_workflow.add_node("store_results", storage_submodule)
    
    # 엣지 정의 (순차 실행)
    module10_workflow.add_edge(START, "collect_data")
    module10_workflow.add_edge("collect_data", "analyze_growth")
    module10_workflow.add_edge("analyze_growth", "select_focus_coaching")
    module10_workflow.add_edge("select_focus_coaching", "generate_individual_result")
    module10_workflow.add_edge("generate_individual_result", "generate_manager_result")
    module10_workflow.add_edge("generate_manager_result", "store_results")
    module10_workflow.add_edge("store_results", END)
    
    return module10_workflow.compile()