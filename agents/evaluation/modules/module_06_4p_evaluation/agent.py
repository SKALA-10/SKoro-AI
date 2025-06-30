# ================================================================
# agent.py - LangGraph 에이전트 및 상태 관리
# ================================================================

from typing import Annotated, List, Literal, TypedDict, Dict, Optional, Any
import operator
from langgraph.graph import StateGraph, START, END

from db_utils import (
    fetch_employee_basic_info,
    fetch_task_data_for_passionate,
    fetch_task_data_for_proactive,
    fetch_task_data_for_professional,
    fetch_peer_talk_data,
    fetch_collaboration_matrix_data,
    save_quarterly_4p_results,
    save_annual_4p_results
)

from llm_utils import (
    load_and_cache_evaluation_criteria,
    call_llm_for_passionate_evaluation,
    call_llm_for_proactive_evaluation,
    call_llm_for_professional_evaluation,
    call_llm_for_people_evaluation
)


# ================================================================
# State 정의
# ================================================================

class Module6AgentState(TypedDict, total=True):
    """모듈 6 (4P BARS 평가) 상태 - 병렬 처리 완전 지원"""

    # ✅ 병렬 누적 필드
    messages: Annotated[List[str], operator.add]
    
    # ✅ 읽기 전용 기본 정보 (초기 설정 후 변경 안함)
    report_type: Literal["quarterly", "annual"]
    team_id: int
    period_id: int
    emp_no: str
    feedback_report_id: Optional[int]
    final_evaluation_report_id: Optional[int]
    raw_evaluation_criteria: str
    
    # ✅ 병렬 업데이트 가능한 딕셔너리 필드들
    evaluation_criteria: Annotated[Dict[str, str], lambda x, y: {**x, **y}]
    evaluation_results: Annotated[Dict[str, Dict], lambda x, y: {**x, **y}]
    integrated_data: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]


# ================================================================
# Agent 함수들
# ================================================================

def initialize_evaluation_criteria_agent(state: Module6AgentState) -> Dict:
    """평가 기준 초기화 - 파일 캐시 활용하여 raw와 parsed 모두 설정"""
    
    try:
        # 파일 캐시 기반으로 평가 기준 로드
        parsed_criteria = load_and_cache_evaluation_criteria()
        
        # 파일 캐시에서 raw_text도 가져오기
        from llm_utils import load_cache_from_file
        cache_data = load_cache_from_file()
        raw_text = cache_data.get("raw_text", "")
        
        return {
            "raw_evaluation_criteria": raw_text,  # DB 원본 텍스트
            "evaluation_criteria": parsed_criteria,  # 파싱된 4P 딕셔너리
            "messages": ["✅ 평가 기준 초기화 완료 (파일 캐시 활용)"]
        }
        
    except Exception as e:
        print(f"❌ 평가 기준 초기화 실패: {e}")
        raise e


def passionate_evaluation_submodule(state: Module6AgentState) -> Dict:
    """Passionate 평가 서브모듈 - 수정됨"""
    
    emp_no = state["emp_no"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    evaluation_criteria = state.get("evaluation_criteria", {})

    print(f"Passionate 평가 시작: {emp_no}")

    basic_info = fetch_employee_basic_info(emp_no)
    task_data = fetch_task_data_for_passionate(emp_no, period_id, report_type)

    if not basic_info:
        passionate_result = {
            "score": 3.0,
            "evidence": ["직원 정보 없음"],
            "reasoning": "직원 정보를 찾을 수 없어 기본 평가",
            "bars_level": "기본 열정",
            "improvement_points": ["정보 확인 필요"],
        }
    else:
        passionate_result = call_llm_for_passionate_evaluation(
            task_data, basic_info, evaluation_criteria
        )

    # ✅ 특정 키만 반환
    return {
        "evaluation_results": {"passionate": passionate_result},
        "messages": [f"Passionate 평가 완료: {passionate_result['score']}점 ({passionate_result['bars_level']})"]
    }


def proactive_evaluation_submodule(state: Module6AgentState) -> Dict:
    """Proactive 평가 서브모듈 - 수정됨"""
    
    emp_no = state["emp_no"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    evaluation_criteria = state.get("evaluation_criteria", {})

    print(f"Proactive 평가 시작: {emp_no}")

    basic_info = fetch_employee_basic_info(emp_no)
    task_data = fetch_task_data_for_proactive(emp_no, period_id, report_type)

    if not basic_info:
        proactive_result = {
            "score": 3.0,
            "evidence": ["직원 정보 없음"],
            "reasoning": "직원 정보를 찾을 수 없어 기본 평가",
            "bars_level": "기본 주도성",
            "improvement_points": ["정보 확인 필요"],
        }
    else:
        proactive_result = call_llm_for_proactive_evaluation(task_data, basic_info, evaluation_criteria)

    # ✅ 특정 키만 반환
    return {
        "evaluation_results": {"proactive": proactive_result},
        "messages": [f"Proactive 평가 완료: {proactive_result['score']}점 ({proactive_result['bars_level']})"]
    }


def professional_evaluation_submodule(state: Module6AgentState) -> Dict:
    """Professional 평가 서브모듈 - 수정됨"""
    
    emp_no = state["emp_no"]
    period_id = state["period_id"]
    report_type = state["report_type"]
    evaluation_criteria = state.get("evaluation_criteria", {})

    print(f"Professional 평가 시작: {emp_no}")

    basic_info = fetch_employee_basic_info(emp_no)
    task_data = fetch_task_data_for_professional(emp_no, period_id, report_type)

    if not basic_info:
        professional_result = {
            "score": 3.0,
            "evidence": ["직원 정보 없음"],
            "reasoning": "직원 정보를 찾을 수 없어 기본 평가",
            "bars_level": "기본 전문성",
            "improvement_points": ["정보 확인 필요"],
        }
    else:
        professional_result = call_llm_for_professional_evaluation(
            task_data, basic_info, evaluation_criteria
        )

    # ✅ 특정 키만 반환
    return {
        "evaluation_results": {"professional": professional_result},
        "messages": [f"Professional 평가 완료: {professional_result['score']}점 ({professional_result['bars_level']})"]
    }


def people_evaluation_submodule(state: Module6AgentState) -> Dict:
    """People 평가 서브모듈 - 수정됨"""
    
    emp_no = state["emp_no"]
    period_id = state["period_id"]
    team_id = state["team_id"]
    report_type = state["report_type"]
    evaluation_criteria = state.get("evaluation_criteria", {})

    print(f"People 평가 시작: {emp_no}")

    basic_info = fetch_employee_basic_info(emp_no)
    task_data = fetch_task_data_for_professional(emp_no, period_id, report_type)
    collaboration_data = fetch_collaboration_matrix_data(emp_no, team_id, period_id)
    peer_talk_data = fetch_peer_talk_data(emp_no, period_id)

    if not basic_info:
        people_result = {
            "score": 3.0,
            "evidence": ["직원 정보 없음"],
            "reasoning": "직원 정보를 찾을 수 없어 기본 평가",
            "bars_level": "기본적 협력",
            "improvement_points": ["정보 확인 필요"],
        }
    else:
        people_result = call_llm_for_people_evaluation(
            task_data, collaboration_data, peer_talk_data, basic_info, evaluation_criteria
        )

    # ✅ 특정 키만 반환
    return {
        "evaluation_results": {"people": people_result},
        "messages": [f"People 평가 완료: {people_result['score']}점 ({people_result['bars_level']})"]
    }


def bars_integration_submodule(state: Module6AgentState) -> Dict:
    """4P 통합 평가 서브모듈 - 수정됨"""
    
    evaluation_results = state.get("evaluation_results", {})
    passionate = evaluation_results.get("passionate", {})
    proactive = evaluation_results.get("proactive", {})
    professional = evaluation_results.get("professional", {})
    people = evaluation_results.get("people", {})

    print("4P 통합 평가 시작")

    # 4P 평균 점수 계산
    scores = [
        passionate.get("score", 3.0),
        proactive.get("score", 3.0),
        professional.get("score", 3.0),
        people.get("score", 3.0),
    ]
    average_score = sum(scores) / len(scores)

    # 강점/약점 분석
    score_dict = {
        "passionate": passionate.get("score", 3.0),
        "proactive": proactive.get("score", 3.0),
        "professional": professional.get("score", 3.0),
        "people": people.get("score", 3.0),
    }

    # max/min 함수 오류 수정
    top_strength = max(score_dict.items(), key=lambda x: x[1])[0]
    improvement_area = min(score_dict.items(), key=lambda x: x[1])[0]

    # 4P 균형도 분석
    max_score = max(scores)
    min_score = min(scores)
    balance_gap = max_score - min_score

    if balance_gap <= 0.5:
        balance_analysis = "4P 영역이 매우 균형있게 발달"
    elif balance_gap <= 1.0:
        balance_analysis = f"{top_strength.capitalize()} 영역이 강하며, 전반적으로 균형 잡힌 발전"
    else:
        balance_analysis = f"{top_strength.capitalize()} 영역이 특히 강하며, {improvement_area.capitalize()} 영역에서 성장 여지"

    # 종합 평가
    if average_score >= 4.5:
        overall_level = "탁월"
    elif average_score >= 4.0:
        overall_level = "우수"
    elif average_score >= 3.5:
        overall_level = "양호"
    elif average_score >= 3.0:
        overall_level = "보통"
    else:
        overall_level = "개선 필요"

    integrated_result = {
        "scores": score_dict,
        "average_score": round(average_score, 2),
        "top_strength": top_strength,
        "improvement_area": improvement_area,
        "balance_analysis": balance_analysis,
        "overall_level": overall_level,
        "comprehensive_assessment": f"{overall_level} 수준의 4P 역량을 보유하고 있으며, {balance_analysis}",
        "passionate": passionate,
        "proactive": proactive,
        "professional": professional,
        "people": people,
    }

    # ✅ 특정 키만 반환
    return {
        "integrated_data": {"integrated_4p_result": integrated_result},
        "messages": [f"4P 통합 평가 완료: 평균 {average_score:.1f}점 ({overall_level})"]
    }


def quarterly_format_and_save_submodule(state: Module6AgentState) -> Dict:
    """분기 저장 서브모듈 - 수정됨"""
    
    feedback_report_id = state.get("feedback_report_id")
    integrated_result = state.get("integrated_data", {}).get("integrated_4p_result", {})

    print(f"분기 결과 저장 시작: feedback_report_id={feedback_report_id}")

    if not feedback_report_id:
        return {"messages": ["분기 저장 실패: feedback_report_id 없음"]}

    success = save_quarterly_4p_results(feedback_report_id, integrated_result)

    if success:
        message = f"분기 4P 평가 결과 저장 완료 (ID: {feedback_report_id})"
    else:
        message = "분기 4P 평가 결과 저장 실패"

    return {"messages": [message]}


def annual_format_and_save_submodule(state: Module6AgentState) -> Dict:
    """연말 저장 서브모듈 - 수정됨"""
    
    final_evaluation_report_id = state.get("final_evaluation_report_id")
    integrated_result = state.get("integrated_data", {}).get("integrated_4p_result", {})

    print(f"연말 결과 저장 시작: final_evaluation_report_id={final_evaluation_report_id}")

    if not final_evaluation_report_id:
        return {"messages": ["연말 저장 실패: final_evaluation_report_id 없음"]}
        
    if not integrated_result:
        return {"messages": ["연말 저장 실패: integrated_4p_result가 전달되지 않음"]}

    success = save_annual_4p_results(final_evaluation_report_id, integrated_result)

    if success:
        message = f"연말 4P 평가 결과 저장 완료 (ID: {final_evaluation_report_id})"
    else:
        message = "연말 4P 평가 결과 저장 실패"

    return {"messages": [message]}


# ================================================================
# 그래프 생성
# ================================================================

def create_module6_graph_efficient():
    """효율적인 모듈 6 그래프 - 파일 캐시 활용"""
    module6 = StateGraph(Module6AgentState)

    # 노드 정의 (initialize만 있음, parse 제거)
    module6.add_node("initialize_criteria", initialize_evaluation_criteria_agent)  # 👈 파일 캐시 기반
    module6.add_node("passionate_evaluation", passionate_evaluation_submodule)
    module6.add_node("proactive_evaluation", proactive_evaluation_submodule)
    module6.add_node("professional_evaluation", professional_evaluation_submodule)
    module6.add_node("people_evaluation", people_evaluation_submodule)
    module6.add_node("bars_integration", bars_integration_submodule)
    module6.add_node("quarterly_format_and_save", quarterly_format_and_save_submodule)
    module6.add_node("annual_format_and_save", annual_format_and_save_submodule)

    # 흐름 (initialize → 바로 평가들)
    module6.add_edge(START, "initialize_criteria")
    
    # initialize → 4개 평가 노드 직접 연결
    module6.add_edge("initialize_criteria", "passionate_evaluation")
    module6.add_edge("initialize_criteria", "proactive_evaluation") 
    module6.add_edge("initialize_criteria", "professional_evaluation")
    module6.add_edge("initialize_criteria", "people_evaluation")

    # 나머지는 동일
    module6.add_edge("passionate_evaluation", "bars_integration")
    module6.add_edge("proactive_evaluation", "bars_integration")
    module6.add_edge("professional_evaluation", "bars_integration")
    module6.add_edge("people_evaluation", "bars_integration")

    def decide_save_path(state):
        return (
            "quarterly_format_and_save"
            if state["report_type"] == "quarterly"
            else "annual_format_and_save"
        )

    module6.add_conditional_edges(
        "bars_integration",
        decide_save_path,
        {
            "quarterly_format_and_save": "quarterly_format_and_save",
            "annual_format_and_save": "annual_format_and_save",
        },
    )

    module6.add_edge("quarterly_format_and_save", END)
    module6.add_edge("annual_format_and_save", END)

    return module6.compile()