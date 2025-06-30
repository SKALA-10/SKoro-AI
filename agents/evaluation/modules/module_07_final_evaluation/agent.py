import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

# ================================================================
# agent_module7.py - 모듈 7 LangGraph 에이전트 및 상태 관리
# ================================================================

from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END
import json
import statistics

from db_utils import (
    fetch_team_members,
    fetch_team_achievement_data,
    fetch_team_fourp_data,
    fetch_team_quarterly_data,
    batch_update_temp_evaluations
)

from scoring_utils import (
    get_evaluation_weights_by_cl,
    calculate_achievement_score_by_grade,
    normalize_cl_group
)

from llm_utils import (
    call_llm_for_normalized_evaluation_comments
)

# ================================================================
# TeamModule7AgentState 정의 - 팀 단위 처리
# ================================================================

class TeamModule7AgentState(TypedDict):
    """모듈 7 (종합평가 점수 산정) 상태 - 팀 단위 처리"""
    messages: Annotated[List[HumanMessage], operator.add]
    
    # 팀 기본 정보
    team_id: str
    period_id: int  # 연말: 4
    
    # 팀 전체 데이터 (한 번에 조회)
    team_members: List[Dict]  # 팀원 기본 정보
    team_achievement_data: List[Dict]  # 팀 전체 달성률 데이터
    team_fourp_data: List[Dict]  # 팀 전체 4P 데이터
    team_quarterly_data: Dict  # 팀원별 분기 데이터
    
    # 통계 계산 결과 (팀 공통)
    weights_by_cl: Dict  # CL별 가중치
    
    # 개별 계산 결과
    individual_scores: List[Dict]  # 각 팀원별 점수
    evaluation_comments: List[Dict]  # 각 팀원별 코멘트
    
    # 처리 결과
    processed_count: int
    failed_members: List[str]

# ================================================================
# 팀 단위 서브모듈 함수들
# ================================================================

def team_data_collection_submodule(state: TeamModule7AgentState) -> TeamModule7AgentState:
    """1. 팀 데이터 수집 서브모듈"""
    
    team_id = state["team_id"]
    period_id = state["period_id"]
    
    try:
        print(f"🔍 팀 데이터 수집 시작: {team_id}")
        
        # 1. 팀원 기본 정보 조회
        team_members = fetch_team_members(team_id)
        print(f"   팀원 수: {len(team_members)}명")
        
        # 2. 팀 전체 달성률 데이터 조회
        team_achievement_data = fetch_team_achievement_data(team_id, period_id)
        print(f"   달성률 데이터: {len(team_achievement_data)}건")
        
        # 3. 팀 전체 4P 데이터 조회
        team_fourp_data = fetch_team_fourp_data(team_id, period_id)
        print(f"   4P 데이터: {len(team_fourp_data)}건")
        
        # 4. 팀 전체 분기별 데이터 조회
        team_quarterly_data = fetch_team_quarterly_data(team_id, period_id)
        print(f"   분기별 데이터: {len(team_quarterly_data)}명")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"데이터 수집 완료")],
            "team_members": team_members,
            "team_achievement_data": team_achievement_data,
            "team_fourp_data": team_fourp_data,
            "team_quarterly_data": team_quarterly_data
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"데이터 수집 실패: {str(e)}")]
        raise e

def team_weights_calculation_submodule(state: TeamModule7AgentState) -> TeamModule7AgentState:
    """2. CL별 가중치 계산 서브모듈"""
    
    try:
        team_members = state["team_members"]
        
        # CL별 가중치 계산
        weights_by_cl = {}
        for member in team_members:
            cl = member.get("cl", "CL2")
            if cl not in weights_by_cl:
                weights_by_cl[cl] = get_evaluation_weights_by_cl(cl)
        
        print(f"📊 CL별 가중치 설정 완료: {list(weights_by_cl.keys())}")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content="가중치 계산 완료")],
            "weights_by_cl": weights_by_cl
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"가중치 계산 실패: {str(e)}")]
        raise e

def team_score_calculation_submodule(state: TeamModule7AgentState) -> TeamModule7AgentState:
    """3. 팀 전체 점수 계산 서브모듈 (SK 등급 기반 절대평가)"""
    
    try:
        team_achievement_data = state["team_achievement_data"]
        team_fourp_data = state["team_fourp_data"]
        weights_by_cl = state["weights_by_cl"]
        
        individual_scores = []
        
        print("🧮 SK 등급 기반 절대평가 점수 계산 시작...")
        
        for achievement_data in team_achievement_data:
            emp_no = achievement_data["emp_no"]
            cl = achievement_data.get("cl", "CL2")
            
            # 가중치 조회
            weights = weights_by_cl.get(cl, {"achievement": 0.5, "fourp": 0.5})
            
            # SK 등급 기반 절대평가 달성률 점수 계산
            achievement_score, achievement_reason = calculate_achievement_score_by_grade(
                achievement_data["ai_annual_achievement_rate"]
            )
            
            # 4P 점수 조회
            fourp_data = next((fp for fp in team_fourp_data if fp["emp_no"] == emp_no), {})
            fourp_results = fourp_data.get("fourp_results", {})
            fourp_score = fourp_results.get("overall", {}).get("average_score", 3.0)
            
            # 하이브리드 점수 계산 (정규화 전 원시점수)
            hybrid_score = (achievement_score * weights["achievement"]) + (fourp_score * weights["fourp"])
            final_score = round(hybrid_score, 2)
            
            individual_scores.append({
                "emp_no": emp_no,
                "emp_name": achievement_data.get("emp_name"),
                "cl": cl,
                "achievement_score": achievement_score,
                "achievement_reason": achievement_reason,
                "fourp_score": fourp_score,
                "hybrid_score": final_score,  # 정규화 전 원시점수
                "weights": weights,
                "emp_data": achievement_data,
                "fourp_results": fourp_results
            })
            
            print(f"   {emp_no}: {final_score}점 (달성률 {achievement_score}, 4P {fourp_score}) - {achievement_reason}")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"절대평가 점수 계산 완료: {len(individual_scores)}명")],
            "individual_scores": individual_scores
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"점수 계산 실패: {str(e)}")]
        raise e

def team_normalization_submodule(state: TeamModule7AgentState) -> TeamModule7AgentState:
    """4. 팀 내 CL별 정규화 서브모듈"""
    
    try:
        individual_scores = state["individual_scores"]
        
        print("🔄 팀 내 CL별 정규화 시작...")
        
        # 1. CL별 그룹화 (숫자/문자열 모두 처리)
        cl_groups = {
            "CL1": [],
            "CL2": [], 
            "CL3": []
        }
        
        for score in individual_scores:
            cl_raw = score.get("cl", 2)  # 기본값 2
            
            # CL 값 정규화
            if isinstance(cl_raw, (int, float)):
                cl = f"CL{int(cl_raw)}"
            else:
                cl = str(cl_raw).upper()
                if not cl.startswith("CL"):
                    cl = f"CL{cl}"
            
            # 유효한 CL인지 확인
            if cl in cl_groups:
                cl_groups[cl].append(score)
                score["cl"] = cl  # 정규화된 CL 값으로 업데이트
            else:
                print(f"⚠️ 알 수 없는 CL: {cl_raw} → CL2로 처리")
                cl_groups["CL2"].append(score)
                score["cl"] = "CL2"
        
        print(f"   CL별 분포: CL3({len(cl_groups['CL3'])}명), CL2({len(cl_groups['CL2'])}명), CL1({len(cl_groups['CL1'])}명)")
        
        # 2. CL별 정규화 실행 (4명 이상일 때만)
        normalized_scores = []
        
        for cl, members in cl_groups.items():
            if len(members) > 0:
                print(f"\n📊 {cl} 정규화 처리:")
                normalized_members = normalize_cl_group(members, cl)
                normalized_scores.extend(normalized_members)
        
        # 3. 정규화 통계 출력
        raw_scores = [s["hybrid_score"] for s in individual_scores]
        norm_scores = [s["normalized_score"] for s in normalized_scores]
        
        print(f"\n📈 정규화 결과:")
        print(f"   원시점수: 평균 {statistics.mean(raw_scores):.2f}, 표준편차 {statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0:.2f}")
        print(f"   정규화점수: 평균 {statistics.mean(norm_scores):.2f}, 표준편차 {statistics.stdev(norm_scores) if len(norm_scores) > 1 else 0:.2f}")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"CL별 정규화 완료: {len(normalized_scores)}명")],
            "individual_scores": normalized_scores
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"정규화 실패: {str(e)}")]
        raise e

def team_comment_generation_submodule(state: TeamModule7AgentState) -> TeamModule7AgentState:
    """5. 팀 전체 코멘트 생성 서브모듈 (정규화된 점수 기준)"""
    
    try:
        individual_scores = state["individual_scores"]
        team_quarterly_data = state["team_quarterly_data"]
        
        evaluation_comments = []
        
        print("💬 정규화 후 평가 코멘트 생성 시작...")
        
        for score_data in individual_scores:
            emp_no = score_data["emp_no"]
            quarterly_tasks = team_quarterly_data.get(emp_no, [])
            
            # LLM을 통한 정규화 후 코멘트 생성
            llm_result = call_llm_for_normalized_evaluation_comments(
                emp_data=score_data["emp_data"],
                normalized_score=score_data["normalized_score"],
                raw_hybrid_score=score_data["raw_hybrid_score"],
                achievement_score=score_data["achievement_score"],
                fourp_score=score_data["fourp_score"],
                quarterly_tasks=quarterly_tasks,
                fourp_results=score_data["fourp_results"],
                achievement_reason=score_data["achievement_reason"],
                normalization_reason=score_data["normalization_reason"]
            )
            
            # raw_score에 저장할 JSON 데이터 구성
            fourp_results = score_data.get("fourp_results", {})
            raw_score_details = {
                "achievement_score": score_data.get("achievement_score"),
                "passionate_score": fourp_results.get("passionate", {}).get("score", 3.0),
                "proactive_score": fourp_results.get("proactive", {}).get("score", 3.0),
                "professional_score": fourp_results.get("professional", {}).get("score", 3.0),
                "people_score": fourp_results.get("people", {}).get("score", 3.0),
                "raw_hybrid_score": score_data.get("raw_hybrid_score")
            }
            raw_score_json = json.dumps(raw_score_details, ensure_ascii=False)

            evaluation_comments.append({
                "emp_no": emp_no,
                "emp_name": score_data.get("emp_name"),
                "raw_score": raw_score_json,  # 원시점수를 JSON 문자열로 저장
                "score": score_data["normalized_score"],  # 정규화된 점수
                "ai_reason": llm_result["ai_reason"],
                "comment": llm_result["comment"]
            })
            
            print(f"   {emp_no}: 정규화 후 코멘트 생성 완료")
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"코멘트 생성 완료: {len(evaluation_comments)}명")],
            "evaluation_comments": evaluation_comments
        })
        return updated_state
        
    except Exception as e:
        updated_state = state.copy()
        updated_state["messages"] = [HumanMessage(content=f"코멘트 생성 실패: {str(e)}")]
        raise e

def team_batch_storage_submodule(state: TeamModule7AgentState) -> TeamModule7AgentState:
    """6. 팀 배치 저장 서브모듈"""
    
    try:
        evaluation_comments = state["evaluation_comments"]
        period_id = state["period_id"]
        
        print("💾 배치 저장 시작...")
        
        # 배치 업데이트 실행
        update_result = batch_update_temp_evaluations(evaluation_comments, period_id)
        
        updated_state = state.copy()
        updated_state.update({
            "messages": [HumanMessage(content=f"배치 저장 완료: 성공 {update_result['success_count']}건")],
            "processed_count": update_result["success_count"],
            "failed_members": update_result["failed_members"]
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
# 팀 단위 워크플로우 생성
# ================================================================

def create_team_module7_graph():
    """팀 단위 모듈 7 그래프 생성 및 반환 (SK 등급 기반 절대평가 + CL 정규화 포함)"""
    team_module7_workflow = StateGraph(TeamModule7AgentState)
    
    # 노드 추가
    team_module7_workflow.add_node("team_data_collection", team_data_collection_submodule)
    team_module7_workflow.add_node("team_weights_calculation", team_weights_calculation_submodule)
    team_module7_workflow.add_node("team_score_calculation", team_score_calculation_submodule)
    team_module7_workflow.add_node("team_normalization", team_normalization_submodule)
    team_module7_workflow.add_node("team_comment_generation", team_comment_generation_submodule)
    team_module7_workflow.add_node("team_batch_storage", team_batch_storage_submodule)
    
    # 엣지 정의 (순차 실행)
    team_module7_workflow.add_edge(START, "team_data_collection")
    team_module7_workflow.add_edge("team_data_collection", "team_weights_calculation")
    team_module7_workflow.add_edge("team_weights_calculation", "team_score_calculation")
    team_module7_workflow.add_edge("team_score_calculation", "team_normalization")
    team_module7_workflow.add_edge("team_normalization", "team_comment_generation")
    team_module7_workflow.add_edge("team_comment_generation", "team_batch_storage")
    team_module7_workflow.add_edge("team_batch_storage", END)
    
    return team_module7_workflow.compile()