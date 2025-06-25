# ================================================================
# agent_module9.py - 모듈 9 LangGraph 에이전트 및 상태 관리
# ================================================================

from typing import Annotated, List, TypedDict, Dict
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END
import statistics

from db_utils import (
    fetch_headquarter_members,
    batch_update_final_evaluation_reports,
    batch_update_final_evaluation_ranking
)

from normalization_utils import (
    normalize_cl_group
)

# ================================================================
# HeadquarterModule9AgentState 정의 - 본부 단위 처리
# ================================================================

class HeadquarterModule9AgentState(TypedDict):
    """모듈 9 (본부 단위 CL별 정규화) 상태 - 본부 단위 처리"""
    messages: Annotated[List[HumanMessage], operator.add]
    
    # 본부 기본 정보
    headquarter_id: str
    period_id: int  # 연말: 4
    
    # 본부 전체 데이터
    headquarter_members: List[Dict]  # 본부 내 모든 직원 데이터
    cl_groups: Dict  # CL별 그룹화된 데이터
    
    # 정규화 결과
    normalized_scores: List[Dict]  # 정규화된 점수 및 코멘트
    
    # 처리 결과
    processed_count: int
    failed_members: List[str]

# ================================================================
# 본부 단위 서브모듈 함수들
# ================================================================

def headquarter_data_collection_submodule(state: HeadquarterModule9AgentState) -> HeadquarterModule9AgentState:
    """1. 본부 데이터 수집 서브모듈"""
    
    headquarter_id = state["headquarter_id"]
    period_id = state["period_id"]
    
    try:
        print(f"🔍 본부 데이터 수집 시작: {headquarter_id}")
        
        # 본부 내 모든 직원 데이터 조회
        headquarter_members = fetch_headquarter_members(headquarter_id, period_id)
        print(f"   본부 내 직원 수: {len(headquarter_members)}명")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"본부 데이터 수집 완료: {len(headquarter_members)}명")],
            "headquarter_members": headquarter_members
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"데이터 수집 실패: {str(e)}")]
        raise e

def headquarter_cl_grouping_submodule(state: HeadquarterModule9AgentState) -> HeadquarterModule9AgentState:
    """2. CL별 그룹화 서브모듈"""
    
    try:
        headquarter_members = state["headquarter_members"]
        
        print("📊 본부 내 CL별 그룹화 시작...")
        
        # CL별 그룹화 (숫자/문자열 모두 처리)
        cl_groups = {
            "CL1": [],
            "CL2": [], 
            "CL3": []
        }
        
        for member in headquarter_members:
            cl_raw = member.get("cl", 2)  # 기본값 2
            
            # CL 값 정규화
            if isinstance(cl_raw, (int, float)):
                cl = f"CL{int(cl_raw)}"
            else:
                cl = str(cl_raw).upper()
                if not cl.startswith("CL"):
                    cl = f"CL{cl}"
            
            # 유효한 CL인지 확인
            if cl in cl_groups:
                cl_groups[cl].append(member)
                member["cl"] = cl  # 정규화된 CL 값으로 업데이트
            else:
                print(f"⚠️ 알 수 없는 CL: {cl_raw} → CL2로 처리")
                cl_groups["CL2"].append(member)
                member["cl"] = "CL2"
        
        print(f"   CL별 분포: CL3({len(cl_groups['CL3'])}명), CL2({len(cl_groups['CL2'])}명), CL1({len(cl_groups['CL1'])}명)")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="CL별 그룹화 완료")],
            "cl_groups": cl_groups
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"CL별 그룹화 실패: {str(e)}")]
        raise e

def headquarter_cl_normalization_submodule(state: HeadquarterModule9AgentState) -> HeadquarterModule9AgentState:
    """3. 본부 내 CL별 정규화 서브모듈"""
    
    try:
        cl_groups = state["cl_groups"]
        
        print("🔄 본부 내 CL별 정규화 시작...")
        
        # CL별 정규화 실행 (무조건 정규화)
        normalized_scores = []
        
        for cl, members in cl_groups.items():
            if len(members) > 0:
                print(f"\n📊 {cl} 정규화 처리:")
                normalized_members = normalize_cl_group(members, cl)
                normalized_scores.extend(normalized_members)
        
        # 정규화 통계 출력
        raw_scores = [m["manager_score"] for m in normalized_scores]
        norm_scores = [m["final_score"] for m in normalized_scores]
        
        print(f"\n📈 정규화 결과:")
        print(f"   원시점수: 평균 {statistics.mean(raw_scores):.2f}, 표준편차 {statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0:.2f}")
        print(f"   정규화점수: 평균 {statistics.mean(norm_scores):.2f}, 표준편차 {statistics.stdev(norm_scores) if len(norm_scores) > 1 else 0:.2f}")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"CL별 정규화 완료: {len(normalized_scores)}명")],
            "normalized_scores": normalized_scores
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"정규화 실패: {str(e)}")]
        raise e

def headquarter_batch_storage_submodule(state: HeadquarterModule9AgentState) -> HeadquarterModule9AgentState:
    """4. 본부 배치 저장 서브모듈"""
    
    try:
        normalized_scores = state["normalized_scores"]
        
        print("💾 배치 저장 시작...")
        
        # 배치 업데이트 실행 (score, cl_reason)
        update_result = batch_update_final_evaluation_reports(normalized_scores)

        # === 팀별 ranking 산출 및 DB 업데이트 ===
        from collections import defaultdict
        teams = defaultdict(list)
        for member in normalized_scores:
            teams[member['team_id']].append(member)
        ranking_data = []
        for team_id, members in teams.items():
            sorted_members = sorted(members, key=lambda x: x['final_score'], reverse=True)
            for rank, member in enumerate(sorted_members, 1):
                member['ranking'] = rank
                ranking_data.append({
                    'final_evaluation_report_id': member['final_evaluation_report_id'],
                    'ranking': rank,
                    'emp_no': member['emp_no']
                })
        ranking_update_result = batch_update_final_evaluation_ranking(ranking_data)
        print(f"팀별 ranking 업데이트 완료: 성공 {ranking_update_result['success_count']}건, 실패 {len(ranking_update_result['failed_members'])}건")
        # ===

        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"배치 저장 완료: 성공 {update_result['success_count']}건, ranking {ranking_update_result['success_count']}건")],
            "processed_count": update_result["success_count"],
            "failed_members": update_result["failed_members"] + ranking_update_result["failed_members"]
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"배치 저장 실패: {str(e)}")],
            "processed_count": 0,
            "failed_members": []
        })
        raise e

# ================================================================
# 본부 단위 워크플로우 생성
# ================================================================

def create_headquarter_module9_graph():
    """본부 단위 모듈 9 그래프 생성 및 반환"""
    headquarter_module9_workflow = StateGraph(HeadquarterModule9AgentState)
    
    # 노드 추가
    headquarter_module9_workflow.add_node("headquarter_data_collection", headquarter_data_collection_submodule)
    headquarter_module9_workflow.add_node("headquarter_cl_grouping", headquarter_cl_grouping_submodule)
    headquarter_module9_workflow.add_node("headquarter_cl_normalization", headquarter_cl_normalization_submodule)
    headquarter_module9_workflow.add_node("headquarter_batch_storage", headquarter_batch_storage_submodule)
    
    # 엣지 정의 (순차 실행)
    headquarter_module9_workflow.add_edge(START, "headquarter_data_collection")
    headquarter_module9_workflow.add_edge("headquarter_data_collection", "headquarter_cl_grouping")
    headquarter_module9_workflow.add_edge("headquarter_cl_grouping", "headquarter_cl_normalization")
    headquarter_module9_workflow.add_edge("headquarter_cl_normalization", "headquarter_batch_storage")
    headquarter_module9_workflow.add_edge("headquarter_batch_storage", END)
    
    return headquarter_module9_workflow.compile()