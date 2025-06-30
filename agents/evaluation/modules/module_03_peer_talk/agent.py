from agents.evaluation.modules.module_03_peer_talk.db_utils import *
from agents.evaluation.modules.module_03_peer_talk.llm_utils import *

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Literal, TypedDict, Dict, Optional
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END


# --- Module3AgentState 정의 ---
class Module3AgentState(TypedDict):
    """
    모듈 3 (동료평가 분석 모듈)의 내부 상태를 정의합니다.
    이 상태는 모듈 3 내의 모든 서브모듈이 공유하고 업데이트합니다.
    """
    messages: Annotated[List[HumanMessage], operator.add] 

    period_id: int 
    target_emp_no: str 
    
    # 동료평가 기본 데이터
    peer_evaluation_ids: List[int]
    evaluator_emp_nos: List[str]
    evaluation_weights: List[float]
    
    # 키워드 및 업무 데이터
    keyword_collections: List[str]
    task_summaries: List[str]
    
    # 분석 결과
    peer_evaluation_summary_sentences: List[str]
    strengths: List[str]
    concerns: List[str]
    collaboration_observations: List[str]
    
    # 내부 분석 데이터
    weighted_analysis_result: Dict
    
    # DB 저장 결과
    feedback_report_id: Optional[int]
    final_evaluation_report_id: Optional[int]


# --- 서브모듈 함수 정의 ---

# 1. 데이터 수집 서브모듈
def data_collection_submodule(state: Module3AgentState) -> Dict:
    """동료평가 데이터 수집 및 초기화"""
    messages = state.get("messages", []) + [HumanMessage(content="모듈 3: 동료평가 데이터 수집 시작")] 
    
    try:
        period_id = state["period_id"]
        target_emp_no = state["target_emp_no"]
        
        # 1. 동료평가 기본 데이터 조회
        peer_evaluations = fetch_peer_evaluations_for_target(period_id, target_emp_no)
        
        if not peer_evaluations:
            messages = messages + [HumanMessage(content=f"모듈 3: {target_emp_no} 동료평가 데이터 없음")]
            return {
                "messages": messages,
                "peer_evaluation_ids": [],
                "evaluator_emp_nos": [],
                "evaluation_weights": [],
                "keyword_collections": [],
                "task_summaries": []
            }
        
        # 2. 기본 데이터 추출
        peer_eval_ids = [pe["peer_evaluation_id"] for pe in peer_evaluations]
        evaluator_emp_nos = [pe["evaluator_emp_no"] for pe in peer_evaluations]
        evaluation_weights = [pe["weight"] for pe in peer_evaluations]
        
        # 3. 키워드 데이터 조회
        keyword_map = fetch_keywords_for_peer_evaluations(peer_eval_ids)
        keyword_collections = [
            ", ".join(keyword_map.get(pid, [])) if keyword_map.get(pid) else ""
            for pid in peer_eval_ids
        ]
        
        # 4. 업무 요약 데이터 조회
        task_map = fetch_tasks_for_peer_evaluations(peer_eval_ids)
        all_task_ids = [tid for tids in task_map.values() for tid in tids]
        summary_map = fetch_task_summaries(period_id, all_task_ids) if all_task_ids else {}
        
        task_summaries = []
        for pid in peer_eval_ids:
            if pid in task_map and task_map[pid]:
                first_task_id = task_map[pid][0]
                summary = summary_map.get(first_task_id, "")
                task_summaries.append(summary)
            else:
                task_summaries.append("")
        
        messages = messages + [HumanMessage(content=f"모듈 3: 동료평가 데이터 수집 완료 ({len(peer_evaluations)}건)")]
        
        return {
            "messages": messages,
            "peer_evaluation_ids": peer_eval_ids,
            "evaluator_emp_nos": evaluator_emp_nos,
            "evaluation_weights": evaluation_weights,
            "keyword_collections": keyword_collections,
            "task_summaries": task_summaries
        }
        
    except Exception as e:
        messages = messages + [HumanMessage(content=f"모듈 3: 데이터 수집 실패 - {str(e)}")]
        return {
            "messages": messages,
            "peer_evaluation_ids": [],
            "evaluator_emp_nos": [],
            "evaluation_weights": [],
            "keyword_collections": [],
            "task_summaries": []
        }


# 2. 맥락 생성 서브모듈
def context_generation_submodule(state: Module3AgentState) -> Dict:
    """동료평가 맥락 문장 생성"""
    messages = state.get("messages", []) + [HumanMessage(content="모듈 3: 동료평가 맥락 생성 시작")]
    
    try:
        target_emp_no = state["target_emp_no"]
        keyword_collections = state.get("keyword_collections", [])
        task_summaries = state.get("task_summaries", [])
        evaluation_weights = state.get("evaluation_weights", [])
        
        if not keyword_collections:
            messages = messages + [HumanMessage(content="모듈 3: 키워드 데이터 없음, 맥락 생성 스킵")]
            return {"messages": messages, "peer_evaluation_summary_sentences": []}
        
        summary_sentences = []
        
        for i in range(len(keyword_collections)):
            keywords = keyword_collections[i] if i < len(keyword_collections) else ""
            work_situation = task_summaries[i] if i < len(task_summaries) else ""
            weight = evaluation_weights[i] if i < len(evaluation_weights) else 1.0
            
            llm_result = call_llm_for_peer_evaluation_context(keywords, work_situation, weight)
            summary_sentence = llm_result.get("context_sentence", "업무 진행 과정에서 동료가 다양한 특성을 보임")
            summary_sentences.append(summary_sentence)
        
        messages = messages + [HumanMessage(content=f"모듈 3: 동료평가 맥락 생성 완료 ({len(summary_sentences)}개 문장)")]
        
        return {"messages": messages, "peer_evaluation_summary_sentences": summary_sentences}
        
    except Exception as e:
        messages = messages + [HumanMessage(content=f"모듈 3: 맥락 생성 실패 - {str(e)}")]
        return {"messages": messages, "peer_evaluation_summary_sentences": []}


# 3. 가중치 분석 서브모듈
def weighted_analysis_submodule(state: Module3AgentState) -> Dict:
    """키워드 가중치 분석"""
    messages = state.get("messages", []) + [HumanMessage(content="모듈 3: 가중치 분석 시작")]
    
    try:
        keyword_collections = state.get("keyword_collections", [])
        evaluation_weights = state.get("evaluation_weights", [])
        
        if not keyword_collections:
            messages = messages + [HumanMessage(content="모듈 3: 키워드 데이터 없음, 가중치 분석 스킵")]
            return {"messages": messages, "weighted_analysis_result": {}}
        
        analysis_result = call_llm_for_keyword_weighted_analysis(keyword_collections, evaluation_weights)
        
        messages = messages + [HumanMessage(content="모듈 3: 가중치 분석 완료")]
        
        return {"messages": messages, "weighted_analysis_result": analysis_result}
        
    except Exception as e:
        messages = messages + [HumanMessage(content=f"모듈 3: 가중치 분석 실패 - {str(e)}")]
        return {"messages": messages, "weighted_analysis_result": {}}


# 4. 피드백 생성 서브모듈
def feedback_generation_submodule(state: Module3AgentState) -> Dict:
    """최종 피드백 생성 (강점, 우려, 협업관찰)"""
    messages = state.get("messages", []) + [HumanMessage(content="모듈 3: 피드백 생성 시작")]
    
    try:
        target_emp_no = state["target_emp_no"]
        summary_sentences = state.get("peer_evaluation_summary_sentences", [])
        weighted_analysis = state.get("weighted_analysis_result", {})
        
        # 상위 비중 문장들 선별
        evaluation_weights = state.get("evaluation_weights", [])
        top_sentences = []
        
        if summary_sentences and evaluation_weights:
            sentences_with_weights = list(zip(summary_sentences, evaluation_weights))
            sorted_sentences = sorted(sentences_with_weights, key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in sorted_sentences[:3]]
        
        # LLM을 통한 피드백 생성
        feedback_result = call_llm_for_final_feedback_generation(
            weighted_analysis, top_sentences, target_emp_no
        )
        
        strengths = [feedback_result.get("strengths", "동료들로부터 긍정적인 평가를 받고 있습니다")]
        concerns = [feedback_result.get("concerns", "지속적인 성장을 위한 개선 영역이 있습니다")]
        collaboration_observations = [feedback_result.get("collaboration_observations", "팀 내에서 협업에 참여하고 있습니다")]
        
        messages = messages + [HumanMessage(content="모듈 3: 피드백 생성 완료")]
        
        return {
            "messages": messages,
            "strengths": strengths,
            "concerns": concerns,
            "collaboration_observations": collaboration_observations
        }
        
    except Exception as e:
        messages = messages + [HumanMessage(content=f"모듈 3: 피드백 생성 실패 - {str(e)}")]
        
        return {
            "messages": messages,
            "strengths": ["동료 평가 분석 중 오류가 발생했습니다"],
            "concerns": ["추가 분석이 필요합니다"],
            "collaboration_observations": ["데이터 부족으로 분석이 제한적입니다"]
        }


# 5. DB 저장 서브모듈
def database_storage_submodule(state: Module3AgentState) -> Dict:
    """
    분기별 DB 저장 (feedback_reports 또는 final_evaluation_reports)
    """
    messages = state.get("messages", []) + [HumanMessage(content="모듈 3: DB 저장 시작")]
    try:
        period_id = state["period_id"]
        target_emp_no = state["target_emp_no"]
        # feedback_report_id가 없으면 team_id, period_id로 조회
        feedback_report_id = state.get("feedback_report_id")
        if not feedback_report_id and "team_id" in state:
            from .db_utils import fetch_feedback_report_id
            feedback_report_id = fetch_feedback_report_id(state["team_id"], period_id, target_emp_no)
        # 피드백 결과 포맷팅
        peer_talk_summary = format_peer_evaluation_result(
            state.get("strengths", []),
            state.get("concerns", []),
            state.get("collaboration_observations", [])
        )
        if period_id == 4:
            report_id = save_final_evaluation_peer_summary(target_emp_no, period_id, peer_talk_summary)
            if report_id:
                messages = messages + [HumanMessage(content=f"모듈 3: final_evaluation_reports 저장 완료 (ID: {report_id})")]
                return {"messages": messages, "final_evaluation_report_id": report_id}
            else:
                messages = messages + [HumanMessage(content="모듈 3: final_evaluation_reports 저장 실패")]
                return {"messages": messages}
        else:
            report_id = save_feedback_peer_summary(target_emp_no, period_id, peer_talk_summary)
            if report_id:
                messages = messages + [HumanMessage(content=f"모듈 3: feedback_reports 저장 완료 (ID: {report_id})")]
                return {"messages": messages, "feedback_report_id": report_id}
            else:
                messages = messages + [HumanMessage(content="모듈 3: feedback_reports 저장 실패")]
                return {"messages": messages}
    except Exception as e:
        messages = messages + [HumanMessage(content=f"모듈 3: DB 저장 실패 - {str(e)}")]
        return {"messages": messages}


# 6. 포맷터 서브모듈
def formatter_submodule(state: Module3AgentState) -> Dict:
    """최종 결과 포맷팅"""
    messages = state.get("messages", []) + [HumanMessage(content="모듈 3: 동료평가 분석 완료")]
    return {"messages": messages}


# --- 워크플로우 생성 ---
def create_module3_graph():
    """모듈 3 그래프 생성 및 반환"""
    module4_workflow = StateGraph(Module3AgentState)
    
    # 노드 추가
    module4_workflow.add_node("data_collection", data_collection_submodule)
    module4_workflow.add_node("context_generation", context_generation_submodule)
    module4_workflow.add_node("weighted_analysis", weighted_analysis_submodule)
    module4_workflow.add_node("feedback_generation", feedback_generation_submodule)
    module4_workflow.add_node("database_storage", database_storage_submodule)
    module4_workflow.add_node("formatter", formatter_submodule)
    
    # 엣지 정의
    module4_workflow.add_edge(START, "data_collection")
    module4_workflow.add_edge("data_collection", "context_generation")
    module4_workflow.add_edge("context_generation", "weighted_analysis")
    module4_workflow.add_edge("weighted_analysis", "feedback_generation")
    module4_workflow.add_edge("feedback_generation", "database_storage")
    module4_workflow.add_edge("database_storage", "formatter")
    module4_workflow.add_edge("formatter", END)
    
    return module4_workflow.compile()