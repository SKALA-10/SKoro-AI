# ================================================================
# agent_module8.py - 모듈 8 LangGraph 에이전트 및 상태 관리
# ================================================================

import os
import logging
from typing import Dict, List, Optional, Any, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from agents.evaluation.modules.module_08_team_comparision.db_utils import *
from agents.evaluation.modules.module_08_team_comparision.comparison_utils import *
from agents.evaluation.modules.module_08_team_comparision.llm_utils import *
from shared.team_performance_comparator import TeamPerformanceComparator

# 로깅 설정
logger = logging.getLogger(__name__)

# ================================================================
# 상태 정의
# ================================================================

class Module8AgentState(TypedDict):
    """
    모듈 8 (팀 성과 비교 모듈)의 내부 상태를 정의합니다.
    """
    messages: List[HumanMessage]
    
    # 기본 정보
    team_id: int
    period_id: int
    report_type: Literal["quarterly", "annual_manager"]
    
    # 클러스터링 결과
    our_team_cluster_id: int
    similar_teams: List[int]
    cluster_stats: Dict
    
    # 성과 데이터
    our_team_kpis: List[Dict]
    our_team_overall_rate: float
    similar_teams_performance: List[Dict]
    
    # 비교 분석 결과
    kpi_comparison_results: List[Dict]
    team_performance_summary: Dict
    
    # 최종 결과
    team_performance_comment: str
    final_comparison_json: Dict
    
    # 업데이트된 ID
    updated_team_evaluation_id: Optional[int]

# ================================================================
# 서브모듈 함수 정의
# ================================================================

def check_cluster_stats_submodule(state: Module8AgentState) -> Module8AgentState:
    """1. 클러스터 통계 존재 확인"""
    period_id = state["period_id"]
    
    # TeamPerformanceComparator 인스턴스 생성
    comparator = TeamPerformanceComparator()
    
    # 클러스터 통계 상태 확인
    status = comparator.get_cluster_status(period_id)
    
    if status["cache_file_exists"]:
        message = f"클러스터 통계 확인 완료: 기존 캐시 사용 (Q{period_id})"
        logger.info(f"✅ 클러스터 캐시 파일 존재 - Q{period_id}")
    else:
        message = f"클러스터 통계 없음: 새로 계산 예정 (Q{period_id})"
        logger.info(f"📊 클러스터 분석 필요 - Q{period_id}")
    
    return {
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=message)]
    }

def calculate_cluster_stats_submodule(state: Module8AgentState) -> Module8AgentState:
    """2. 필요시 전사 클러스터링 + 성과 통계 계산"""
    team_id = state["team_id"]
    period_id = state["period_id"]
    
    logger.info(f"🔄 클러스터 분석 시작 - 팀 {team_id}")
    
    # TeamPerformanceComparator로 클러스터 분석 실행
    comparator = TeamPerformanceComparator()
    result_data = comparator.analyze_team_cluster_performance(team_id, period_id)
    
    if not result_data["success"]:
        logger.error(f"❌ 클러스터 분석 실패: {result_data['error']}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"클러스터 분석 실패: {result_data['error']}")
            ]
        }
    
    team_cluster_info = result_data["team_cluster_info"]
    logger.info(f"✅ 클러스터 분석 완료 - 클러스터 {team_cluster_info['cluster_id']}, 유사팀 {len(team_cluster_info['similar_teams'])}개")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"클러스터 분석 완료: 클러스터 {team_cluster_info['cluster_id']}, 유사팀 {len(team_cluster_info['similar_teams'])}개")
        ],
        "our_team_cluster_id": team_cluster_info["cluster_id"],
        "similar_teams": team_cluster_info["similar_teams"],
        "cluster_stats": team_cluster_info["cluster_stats"]
    }

def team_performance_collection_submodule(state: Module8AgentState) -> Module8AgentState:
    """3. 우리팀 + 유사팀 성과 데이터 수집"""
    team_id = state["team_id"]
    period_id = state["period_id"]
    similar_teams = state["similar_teams"]
    
    logger.info(f"📋 성과 데이터 수집 중 - 팀 {team_id} + 유사팀 {len(similar_teams)}개")
    
    # 우리팀 데이터 수집
    our_team_data = fetch_team_kpis_data(team_id, period_id)
    if not our_team_data:
        logger.error(f"❌ 우리팀 데이터 조회 실패 - 팀 {team_id}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"우리팀 성과 데이터 조회 실패: 팀 {team_id}")
            ]
        }
    
    # 유사팀들 KPI 데이터 수집
    similar_teams_kpis = fetch_multiple_teams_kpis(similar_teams, period_id)
    
    logger.info(f"✅ 성과 데이터 수집 완료 - 우리팀 KPI {len(our_team_data['kpis'])}개, 유사팀 KPI {len(similar_teams_kpis)}개")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"팀 성과 데이터 수집 완료: 우리팀 KPI {len(our_team_data['kpis'])}개, 유사팀 KPI {len(similar_teams_kpis)}개")
        ],
        "our_team_kpis": our_team_data["kpis"],
        "our_team_overall_rate": our_team_data["overall_rate"],
        "similar_teams_performance": similar_teams_kpis
    }

def kpi_comparison_submodule(state: Module8AgentState) -> Module8AgentState:
    """4. KPI별 유사도 매칭 + 비교 분석"""
    our_team_kpis = state["our_team_kpis"]
    similar_teams_performance = state["similar_teams_performance"]
    
    logger.info(f"🔍 KPI 비교 분석 중 - {len(our_team_kpis)}개 KPI")
    
    # KPI별 비교 분석 실행
    kpi_comparison_results = compare_kpis_with_similar_teams(our_team_kpis, similar_teams_performance)
    
    # 비교 가능한 KPI 개수 계산
    comparable_kpis = len([kpi for kpi in kpi_comparison_results if kpi["comparison_result"] != "-"])
    
    logger.info(f"✅ KPI 비교 분석 완료 - {len(kpi_comparison_results)}개 중 {comparable_kpis}개 비교 가능")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"KPI 비교 분석 완료: {len(kpi_comparison_results)}개 KPI 중 {comparable_kpis}개 비교 가능")
        ],
        "kpi_comparison_results": kpi_comparison_results
    }

def generate_team_comment_submodule(state: Module8AgentState) -> Module8AgentState:
    """5. LLM 기반 팀 성과 코멘트 생성"""
    our_team_overall_rate = state["our_team_overall_rate"]
    cluster_stats = state["cluster_stats"]
    kpi_comparison_results = state["kpi_comparison_results"]
    similar_teams = state["similar_teams"]
    
    logger.info(f"🤖 LLM 코멘트 생성 중 - 종합 달성률 {our_team_overall_rate}%")
    
    # LLM으로 팀 성과 코멘트 생성
    team_comment = call_llm_for_team_performance_comment(
        our_team_overall_rate, cluster_stats, kpi_comparison_results, len(similar_teams)
    )
    
    # 최종 비교 JSON 구성
    final_comparison_json = {
        "overall": {
            "our_rate": our_team_overall_rate,
            "similar_avg_rate": cluster_stats["avg_rate"],
            "similar_teams_count": len(similar_teams),
            "comparison_result": get_comparison_result_detailed(our_team_overall_rate, cluster_stats),
            "comment": team_comment
        },
        "kpis": kpi_comparison_results
    }
    
    logger.info(f"✅ LLM 코멘트 생성 완료 - {len(team_comment)}자")
    
    return {
        **state,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"LLM 코멘트 생성 완료 ({len(team_comment)}자)")
        ],
        "team_performance_comment": team_comment,
        "final_comparison_json": final_comparison_json
    }

def save_results_submodule(state: Module8AgentState) -> Module8AgentState:
    """6. JSON 결과 DB 저장"""
    team_id = state["team_id"]
    period_id = state["period_id"]
    final_comparison_json = state["final_comparison_json"]
    
    logger.info(f"💾 DB 저장 중 - 팀 {team_id}")
    
    # team_evaluation_id 조회
    team_evaluation_id = fetch_team_evaluation_id(team_id, period_id)
    
    if not team_evaluation_id:
        logger.error(f"❌ team_evaluation_id 조회 실패 - 팀 {team_id}, Q{period_id}")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"team_evaluation_id 조회 실패: 팀 {team_id}, 분기 {period_id}")
            ]
        }
    
    # DB 저장
    success = save_team_comparison_results(team_evaluation_id, final_comparison_json)
    
    if success:
        logger.info(f"✅ DB 저장 완료 - team_evaluations[{team_evaluation_id}]")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"DB 저장 완료: team_evaluations[{team_evaluation_id}] 업데이트")
            ],
            "updated_team_evaluation_id": team_evaluation_id
        }
    else:
        logger.error(f"❌ DB 저장 실패 - team_evaluations[{team_evaluation_id}]")
        return {
            **state,
            "messages": state.get("messages", []) + [
                HumanMessage(content=f"DB 저장 실패: team_evaluations[{team_evaluation_id}]")
            ]
        }

# ================================================================
# LangGraph 워크플로우 구성
# ================================================================

def create_module8_graph():
    """모듈 8 그래프 생성 및 반환"""
    # 모듈 8 워크플로우 정의
    module8_workflow = StateGraph(Module8AgentState)

    # 노드 추가
    module8_workflow.add_node("check_cluster_stats", check_cluster_stats_submodule)
    module8_workflow.add_node("calculate_cluster_stats", calculate_cluster_stats_submodule)
    module8_workflow.add_node("team_performance_collection", team_performance_collection_submodule)
    module8_workflow.add_node("kpi_comparison", kpi_comparison_submodule)
    module8_workflow.add_node("generate_team_comment", generate_team_comment_submodule)
    module8_workflow.add_node("save_results", save_results_submodule)

    # 엣지 정의
    module8_workflow.add_edge(START, "check_cluster_stats")
    module8_workflow.add_edge("check_cluster_stats", "calculate_cluster_stats")
    module8_workflow.add_edge("calculate_cluster_stats", "team_performance_collection")
    module8_workflow.add_edge("team_performance_collection", "kpi_comparison")
    module8_workflow.add_edge("kpi_comparison", "generate_team_comment")
    module8_workflow.add_edge("generate_team_comment", "save_results")
    module8_workflow.add_edge("save_results", END)

    # 모듈 8 그래프 컴파일
    return module8_workflow.compile()