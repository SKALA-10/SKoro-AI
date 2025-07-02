# ================================================================
# agent_module4.py - 모듈 4 LangGraph 에이전트 및 상태 관리
# ================================================================

from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage
import operator
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass

from agents.evaluation.modules.module_04_collaboration.db_utils import *
from agents.evaluation.modules.module_04_collaboration.llm_utils import *


class Module4State(TypedDict):
    """모듈 4 협업 분석 State - 모듈 2와 일관된 DB 기반 전달 방식"""
    # 기본 정보
    report_type: Literal["quarterly", "annual"]
    team_id: int
    period_id: int
    
    # 타겟 ID들
    target_team_kpi_ids: List[int]
    
    # 처리 결과 추적용 (DB ID만 저장)
    collaboration_relationships: Optional[List[Dict]]
    individual_collaboration_analysis: Optional[Dict[str, Dict]]
    team_collaboration_matrix: Optional[Dict]
    team_evaluation_id: Optional[int]
    
    # 메시지 추적용 (LangGraph 호환성)
    messages: Optional[List[HumanMessage]]

# ================================================================
# 서브모듈 함수들
# ================================================================

def collaboration_data_collection_submodule(state: Module4State) -> Module4State:
    """
    협업 분석을 위한 기초 데이터를 수집합니다.
    - 동일 KPI 내 Task 그룹핑
    - Task Summary 데이터 수집
    """
    print("=== 모듈 4: 협업 데이터 수집 시작 ===")
    
    # team_evaluation_id 확인/생성
    team_evaluation_id = fetch_team_evaluation_id(state["team_id"], state["period_id"])
    if not team_evaluation_id:
        raise Exception(f"team_evaluation_id not found for team {state['team_id']}, period {state['period_id']}")
    
    state["team_evaluation_id"] = team_evaluation_id
    
    collaboration_relationships = []
    
    # 각 KPI별로 협업 관계 1차 구성
    for team_kpi_id in state["target_team_kpi_ids"]:
        kpi_tasks = fetch_collaboration_tasks_by_kpi(team_kpi_id, state["period_id"])
        
        if len(kpi_tasks) > 1:  # 2개 이상 Task가 있어야 협업 가능
            for task in kpi_tasks:
                potential_collaborators = [t["emp_no"] for t in kpi_tasks if t["emp_no"] != task["emp_no"]]
                
                collaboration_relationships.append({
                    "task_id": task["task_id"],
                    "task_summary_id": task.get("task_summary_Id"),
                    "emp_no": task["emp_no"],
                    "emp_name": task["emp_name"],
                    "task_name": task["task_name"],
                    "task_summary": task.get("task_summary", ""),
                    "ai_contribution_score": task.get("ai_contribution_score", 0),
                    "team_kpi_id": team_kpi_id,
                    "potential_collaborators": potential_collaborators,
                    "start_date": task.get("start_date"),
                    "end_date": task.get("end_date"),
                    "collaboration_confirmed": False
                })
    
    print(f"총 {len(collaboration_relationships)}개 잠재적 협업 관계 수집 완료")
    
    state["collaboration_relationships"] = collaboration_relationships
    return state

def individual_collaboration_analysis_submodule(state: Module4State) -> Module4State:
    """
    Task Summary를 LLM으로 분석하여 실제 협업 관계를 확인하고,
    개인별 협업 패턴을 분석합니다.
    """
    print("=== 모듈 4: 개인 협업 분석 시작 ===")
    
    # 1. LLM으로 실제 협업 관계 확인
    confirmed_collaborations = []
    
    if state["collaboration_relationships"]:
        for relation in state["collaboration_relationships"]:
            task_summary = relation["task_summary"]
            potential_collaborators = relation["potential_collaborators"]
            
            if task_summary and potential_collaborators:
                llm_collaboration_result = call_llm_for_collaboration_detection(
                    task_summary=task_summary,
                    task_name=relation["task_name"],
                    potential_collaborators=potential_collaborators,
                    emp_name=relation["emp_name"]
                )
                
                if llm_collaboration_result.get("is_collaboration", False):
                    confirmed_collaborations.append({
                        **relation,
                        "confirmed_collaborators": llm_collaboration_result.get("collaborators", []),
                        "collaboration_description": llm_collaboration_result.get("description", ""),
                        "collaboration_confirmed": True
                    })
    
    # 2. 개인별 협업 패턴 분석
    team_members = fetch_team_members_with_tasks(state["team_id"], state["period_id"])
    individual_analysis = {}
    
    for member in team_members:
        emp_no = member["emp_no"]
        emp_name = member["emp_name"]
        role = member["role"]
        total_tasks = member["total_task_count"]
        avg_contribution_score = member.get("avg_contribution_score", 0)
        
        # 팀장(MANAGER) 제외
        if role == "MANAGER":
            print(f"팀장 {emp_name}({emp_no}) 분석에서 제외")
            continue
        
        # 해당 개인의 협업 Task들 필터링
        member_collaborations = [c for c in confirmed_collaborations if c["emp_no"] == emp_no]
        collaboration_task_count = len(member_collaborations)
        
        # 협업자별 카운트
        collaborator_counts = {}
        all_collaborators = []
        for collab in member_collaborations:
            for collaborator in collab["confirmed_collaborators"]:
                collaborator_counts[collaborator] = collaborator_counts.get(collaborator, 0) + 1
                all_collaborators.extend(collab["confirmed_collaborators"])
        
        # 핵심 협업자 (상위 2-3명)
        sorted_collaborators = sorted(collaborator_counts.items(), key=lambda x: x[1], reverse=True)
        key_collaborators = [collab[0] for collab in sorted_collaborators[:3]]
        
        # 협업 편중도 계산을 위한 의존도 지표
        dependency_metrics = {
            "collaboration_concentration": (max(collaborator_counts.values()) / sum(collaborator_counts.values()) * 100) if collaborator_counts else 0,
            "unique_collaborators": len(collaborator_counts),
            "total_collaborations": sum(collaborator_counts.values())
        }
        
        # Task Summary들 수집 (역할 분석용)
        member_task_summaries = [c["task_summary"] for c in member_collaborations if c["task_summary"]]
        
        individual_analysis[emp_no] = {
            "emp_name": emp_name,
            "total_tasks": total_tasks,
            "collaboration_tasks": collaboration_task_count,
            "collaboration_rate": (collaboration_task_count / total_tasks * 100) if total_tasks > 0 else 0,
            "avg_contribution_score": avg_contribution_score or 0,
            "collaborator_counts": collaborator_counts,
            "key_collaborators": key_collaborators,
            "dependency_metrics": dependency_metrics,
            "task_summaries": member_task_summaries
        }
    
    print(f"개인별 협업 분석 완료: {len(individual_analysis)}명")
    
    state["individual_collaboration_analysis"] = individual_analysis
    return state

def team_collaboration_network_submodule(state: Module4State) -> Module4State:
    """
    팀 전체 협업 네트워크 매트릭스를 생성합니다.
    """
    print("=== 모듈 4: 팀 협업 네트워크 분석 시작 ===")
    
    individual_analysis = state["individual_collaboration_analysis"] or {}
    period_id = state["period_id"]
    report_type = state["report_type"]
    
    collaboration_matrix = []
    
    for emp_no, analysis in individual_analysis.items():
        emp_name = analysis["emp_name"]
        
        print(f"처리 중: {emp_name}({emp_no})")
        
        # 1. 팀 내 역할 분석
        team_role_result = call_llm_for_team_role_analysis(
            task_summaries=analysis["task_summaries"],
            emp_name=emp_name,
            emp_no=emp_no
        )
        
        # 2. 핵심 협업자 이름 매핑
        key_collaborators_with_names = []
        for collaborator_emp_no in analysis["key_collaborators"]:
            if collaborator_emp_no in individual_analysis:
                collaborator_name = individual_analysis[collaborator_emp_no]["emp_name"]
                key_collaborators_with_names.append(f"{collaborator_name}({collaborator_emp_no})")
            else:
                key_collaborators_with_names.append(f"({collaborator_emp_no})")
        
        # 3. Peer Talk 요약
        peer_talk_content = fetch_peer_talk_summary(emp_no, period_id, report_type)
        peer_talk_summary = "동료평가 없음"
        if peer_talk_content:
            peer_talk_summary = call_llm_for_peer_talk_summary(peer_talk_content, emp_name)
        
        # 4. 협업 편중도 분석 (신뢰성 있는 방법)
        collaboration_data = {
            "total_tasks": analysis["total_tasks"],
            "collaboration_tasks": analysis["collaboration_tasks"],
            "collaborator_counts": analysis["collaborator_counts"],
            "dependency_metrics": analysis["dependency_metrics"]
        }
        
        bias_result = call_llm_for_collaboration_bias_analysis(collaboration_data, emp_name, emp_no)
        collaboration_bias = f"{bias_result['bias_level']}"
        if bias_result['bias_level'] == "높음":
            collaboration_bias += "(과의존 위험)"
        elif bias_result['bias_level'] == "낮음":
            collaboration_bias += "(협업 부족)"
        else:
            collaboration_bias += "(적절)"
        
        # 5. 종합 평가
        collaboration_analysis_summary = {
            "team_role": team_role_result["team_role"],
            "collaboration_rate": analysis["collaboration_rate"],
            "key_collaborators": key_collaborators_with_names,
            "collaboration_bias": collaboration_bias
        }
        
        overall_evaluation = call_llm_for_overall_evaluation(
            collaboration_analysis_summary, emp_name, emp_no
        )
        
        # 매트릭스 항목 구성
        matrix_item = {
            "emp_no": emp_no,
            "name": f"{emp_name}({emp_no})",
            "total_tasks": analysis["total_tasks"],
            "collaboration_rate": round(analysis["collaboration_rate"], 1),
            "avg_contribution_score": round(analysis["avg_contribution_score"], 1),
            "team_role": team_role_result["team_role"],
            "key_collaborators": key_collaborators_with_names,
            "peer_talk_summary": peer_talk_summary,
            "collaboration_bias": collaboration_bias,
            "overall_evaluation": overall_evaluation
        }
        
        collaboration_matrix.append(matrix_item)

    # 팀 전체 요약
    total_members = len(collaboration_matrix)
    avg_collaboration_rate = sum([item["collaboration_rate"] for item in collaboration_matrix]) / total_members if total_members > 0 else 0
    avg_contribution_score = sum([item["avg_contribution_score"] for item in collaboration_matrix]) / total_members if total_members > 0 else 0
    high_bias_members = [item["name"] for item in collaboration_matrix if "과의존 위험" in item["collaboration_bias"]]
    
    team_summary = call_llm_for_team_summary(collaboration_matrix)
    
    # 최종 결과 구성
    team_collaboration_matrix = {
        "collaboration_matrix": collaboration_matrix,
        "team_summary": team_summary,
        "analysis_period": period_id,
        "analysis_date": "2024-12-19",  # 실제로는 현재 날짜
        "total_members": total_members,
        "avg_collaboration_rate": round(avg_collaboration_rate, 1),
        "avg_contribution_score": round(avg_contribution_score, 1)
    }
    
    print(f"팀 협업 네트워크 매트릭스 생성 완료: {total_members}명")
    
    state["team_collaboration_matrix"] = team_collaboration_matrix
    return state

def collaboration_comprehensive_analysis_submodule(state: Module4State) -> Module4State:
    """
    협업 기여도 종합 분석 (현재는 팀장용만 처리하므로 패스)
    """
    print("=== 모듈 4: 협업 기여도 종합 분석 (스킵) ===")
    
    return state

def collaboration_formatter_submodule(state: Module4State) -> Module4State:
    """
    협업 매트릭스를 DB에 저장합니다.
    """
    print("=== 모듈 4: 협업 매트릭스 DB 저장 시작 ===")
    
    team_collaboration_matrix = state["team_collaboration_matrix"]
    team_evaluation_id = state["team_evaluation_id"]
    
    # DB에 저장
    if team_collaboration_matrix and team_evaluation_id:
        success = save_collaboration_matrix_to_db(team_evaluation_id, team_collaboration_matrix)
        status_message = "모듈 4: 협업 매트릭스 DB 저장 완료" if success else "모듈 4: 협업 매트릭스 DB 저장 실패"
    else:
        status_message = "모듈 4: 협업 매트릭스 DB 저장 실패 - 데이터 누락"
    
    # messages 필드 초기화 (LangGraph 호환성)
    if state["messages"] is None:
        state["messages"] = []
    state["messages"].append(HumanMessage(content=status_message))
    
    return state

# ================================================================
# 워크플로우 생성
# ================================================================

def create_module4_graph():
    """모듈 4 그래프 생성 및 반환"""
    module4_workflow = StateGraph(Module4State)
    
    # 노드 추가 (State 키와 겹치지 않도록 이름 수정)
    module4_workflow.add_node("data_collection", collaboration_data_collection_submodule)
    module4_workflow.add_node("individual_analysis", individual_collaboration_analysis_submodule)
    module4_workflow.add_node("team_network", team_collaboration_network_submodule)
    module4_workflow.add_node("comprehensive_analysis", collaboration_comprehensive_analysis_submodule)
    module4_workflow.add_node("formatter", collaboration_formatter_submodule)
    
    # 엣지 정의
    module4_workflow.add_edge(START, "data_collection")
    module4_workflow.add_edge("data_collection", "individual_analysis")
    module4_workflow.add_edge("individual_analysis", "team_network")
    module4_workflow.add_edge("team_network", "comprehensive_analysis")
    module4_workflow.add_edge("comprehensive_analysis", "formatter")
    module4_workflow.add_edge("formatter", END)
    
    return module4_workflow.compile()