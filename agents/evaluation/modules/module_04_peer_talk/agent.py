# agents/evaluation/modules/module_04_peer_talk/agent.py

from db_utils import *
from llm_utils import (
    call_llm_for_context_generation, 
    call_llm_for_keyword_sentiment_analysis, 
    call_llm_for_feedback_generation,
    _extract_work_keywords, 
    _process_work_content_for_context, 
    _validate_and_enhance_sentence, 
    _create_enhanced_fallback_sentence
)


from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Literal, TypedDict, Dict
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END
from collections import defaultdict
import re
import json


# --- Module4AgentState 정의 ---
class Module4AgentState(TypedDict):
    """
    모듈 4 (Peer Talk 분석 모듈)의 내부 상태를 정의합니다.
    이 상태는 모듈 4 내의 모든 서브모듈이 공유하고 업데이트합니다.
    """
    messages: Annotated[List[HumanMessage], operator.add]
    
    # 기본 정보
    분기: str
    평가받는사번: str
    
    # 수집된 평가 데이터
    평가하는사번_리스트: List[str]
    성과지표ID_리스트: List[str]
    비중: List[int]
    키워드모음: List[str]
    
    # 매핑된 업무 정보
    구체적업무내용: List[str]
    
    # 생성된 분석 결과
    동료평가요약줄글들: List[str]
    
    # 최종 피드백
    강점: List[str]
    우려: List[str]
    협업관찰: List[str]
    
    # 임시 분석 데이터
    _weighted_analysis: Dict


# --- 서브모듈 함수 정의 ---

# 1. 데이터 수집 서브모듈
def data_collection_submodule(state: Module4AgentState) -> Module4AgentState:
    """
    DB에서 동료 평가 데이터를 조회하여 PeerTalkState로 매핑하는 에이전트
    """
    employee_id = state["평가받는사번"]
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 데이터 수집 시작")]
    
    try:
        # 입력 검증
        period_id = int(state["분기"])
        target_emp_no = state["평가받는사번"]
        
        if not target_emp_no:
            raise ValueError("평가받는사번이 필요합니다.")

        print(f"[CompleteDataMappingAgent] {target_emp_no}: 완전한 데이터 매핑 시작 (분기: {period_id})")

        # 1. 동료 평가 리스트 조회
        peer_evals = fetch_peer_evaluations_for_target(engine, period_id, target_emp_no)
        
        if not peer_evals:
            print(f"[CompleteDataMappingAgent] {target_emp_no}: 평가 데이터 없음")
            # 빈 데이터일 경우 기본값 설정
            for field in ["평가하는사번_리스트", "비중", "키워드모음", "구체적업무내용", "성과지표ID_리스트"]:
                state[field] = []
            for field in ["동료평가요약줄글들", "강점", "우려", "협업관찰"]:
                state[field] = []
            state["_weighted_analysis"] = {}
            messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 평가 데이터 없음")]
            return {"messages": messages, **state}

        peer_eval_ids = [pe["peer_evaluation_id"] for pe in peer_evals]
        print(f"[CompleteDataMappingAgent] {target_emp_no}: {len(peer_evals)}개 평가 발견")

        # 2. 기본 평가 정보 매핑
        state["평가하는사번_리스트"] = [pe["evaluator_emp_no"] for pe in peer_evals]
        state["비중"] = [pe["weight"] for pe in peer_evals]

        # 3. 키워드 모음 조회 및 매핑
        keyword_map = fetch_keywords_for_peer_evaluations(engine, peer_eval_ids)
        state["키워드모음"] = [
            ", ".join(keyword_map.get(pid, [])) if keyword_map.get(pid) else ""
            for pid in peer_eval_ids
        ]

        # 4. 업무 내용 조회 및 매핑 (수정된 함수 사용)
        task_map = fetch_tasks_for_peer_evaluations_fixed(engine, peer_eval_ids)
        all_task_ids = [tid for tids in task_map.values() for tid in tids]
        summary_map = fetch_task_summaries_fixed(engine, period_id, all_task_ids) if all_task_ids else {}
        
        # 각 평가별 첫 번째 task의 summary 사용
        state["구체적업무내용"] = []
        for pid in peer_eval_ids:
            if pid in task_map and task_map[pid]:
                first_task_id = task_map[pid][0]
                summary = summary_map.get(first_task_id, "")
                state["구체적업무내용"].append(summary)
            else:
                state["구체적업무내용"].append("")

        # 5. 성과지표ID_리스트 초기화
        state["성과지표ID_리스트"] = ["1"] * len(peer_evals)  # 기본값

        # 6. 기타 필드들 초기화
        state["동료평가요약줄글들"] = []
        state["강점"] = []
        state["우려"] = []
        state["협업관찰"] = []
        state["_weighted_analysis"] = {}
        
        print(f"[CompleteDataMappingAgent] {target_emp_no}: 완전한 매핑 완료")
        print(f"  - 평가자: {len(state['평가하는사번_리스트'])}명")
        print(f"  - 키워드: {len([k for k in state['키워드모음'] if k])}개 평가")
        print(f"  - 업무내용: {len([c for c in state['구체적업무내용'] if c])}개")
        print(f"  - 비중: {state['비중']}")
        
    except Exception as e:
        print(f"[CompleteDataMappingAgent] {target_emp_no}: 매핑 실패 - {str(e)}")
        import traceback
        traceback.print_exc()
        # 오류 발생시 기본값으로 초기화
        for field in ["평가하는사번_리스트", "비중", "키워드모음", "구체적업무내용", "성과지표ID_리스트"]:
            state[field] = []
        for field in ["동료평가요약줄글들", "강점", "우려", "협업관찰"]:
            state[field] = []
        state["_weighted_analysis"] = {}
    
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 데이터 수집 완료")]
    return {"messages": messages, **state}


# 2. 키워드 분석 엔진 서브모듈
def keyword_analysis_engine_submodule(state: Module4AgentState) -> Module4AgentState:
    """
    비중을 반영한 키워드 가중치 분석 (새로운 키워드 감정 분석 포함)
    """
    
    employee_id = state["평가받는사번"]
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 키워드 분석 시작")]
    print(f"[WeightedAnalysisAgent] {employee_id}: 가중치 분석 시작")
    
    # 기존 정의된 키워드들
    positive_keywords = {
        "배려", "긍정마인드", "열정", "책임감 있는", "성실한", "꼼꼼한",
        "추진력 있는", "문제해결력", "주도적인", "목표지향적", "의사결정력",
        "신뢰할 수 있는", "능동적인", "조율력", "유쾌한", "밝은", "리더십 있는",
        "열린 소통", "유연한", "빠른 실행력", "친절한", "협업", "현실적인",
        "아이디어", "고객중심", "창의적인", "분석적인", "체계적인", "논리적인",
        "침착한", "신중한", "적극적인", "전문적인", "세심한", "효율적인"
    }
    
    negative_keywords = {
        "소극적인", "실수가 잦은", "기한 미준수", "감정적인", "불쾌한 언행",
        "방어적인", "회피형", "개인주의", "무관심", "소통단절", "무의욕자",
        "부정적인", "부정적", "갑질", "근무태만", "다혈질", "리더십 없는",
        "이기주의", "수동적인", "비판적인", "의욕없는", "거짓말", "고집",
        "산만한", "느린 일처리"
    }
    
    # 평가 데이터가 없으면 스킵
    if not state["키워드모음"]:
        print(f"[WeightedAnalysisAgent] {employee_id}: 키워드 데이터 없음, 스킵")
        state["_weighted_analysis"] = _create_empty_analysis()
        messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 키워드 데이터 없음")]
        return {"messages": messages, **state}
    
    # 1. 모든 고유 키워드 수집
    all_keywords = set()
    for keyword_string in state["키워드모음"]:
        keywords = [k.strip() for k in keyword_string.split(',') if k.strip()]
        all_keywords.update(keywords)
    
    print(f"[WeightedAnalysisAgent] {employee_id}: 총 {len(all_keywords)}개 고유 키워드 발견")
    
    # 2. 새로운 키워드 감정 분석
    analyzed_keywords = {}
    new_keywords = [k for k in all_keywords 
                   if k not in positive_keywords and k not in negative_keywords]
    
    if new_keywords:
        print(f"[WeightedAnalysisAgent] {employee_id}: {len(new_keywords)}개 새로운 키워드 감정 분석 중...")
        
        for keyword in new_keywords:
            try:
                sentiment = call_llm_for_keyword_sentiment_analysis(keyword)
                
                if sentiment == "긍정":
                    score = 1.0
                elif sentiment == "부정":
                    score = -1.0
                else:
                    score = 0.0
                
                analyzed_keywords[keyword] = score
                print(f"  └ '{keyword}' → {sentiment} (점수: {score})")
                
            except Exception as e:
                analyzed_keywords[keyword] = 0.0
                print(f"  └ '{keyword}' → 분석 실패, 중립 처리 ({str(e)})")
    
    # 3. 가중치 분석 수행
    weighted_scores = defaultdict(float)
    keyword_frequency = defaultdict(int)
    total_weight = sum(state["비중"])
    
    def get_keyword_score(keyword: str) -> float:
        """키워드 감정 점수 반환"""
        if keyword in positive_keywords:
            return 1.0
        elif keyword in negative_keywords:
            return -1.0
        elif keyword in analyzed_keywords:
            return analyzed_keywords[keyword]
        else:
            return 0.0
    
    # 각 평가별로 키워드 가중치 계산
    for i in range(len(state["키워드모음"])):
        keywords = [k.strip() for k in state["키워드모음"][i].split(',') if k.strip()]
        weight = state["비중"][i]
        
        for keyword in keywords:
            keyword_frequency[keyword] += 1
            score = get_keyword_score(keyword)
            weighted_scores[keyword] += score * weight
    
    # 정규화 (총 비중으로 나누기)
    if total_weight > 0:
        for keyword in weighted_scores:
            weighted_scores[keyword] = weighted_scores[keyword] / total_weight
    
    # 상위 키워드 추출
    positive_keywords_result = {k: v for k, v in weighted_scores.items() if v > 0}
    negative_keywords_result = {k: v for k, v in weighted_scores.items() if v < 0}
    
    # 정렬 (점수 높은 순)
    top_positive = dict(sorted(positive_keywords_result.items(), key=lambda x: x[1], reverse=True)[:5])
    top_negative = dict(sorted(negative_keywords_result.items(), key=lambda x: x[1])[:3])
    
    # 분석 결과 저장
    analysis_result = {
        "weighted_scores": dict(weighted_scores),
        "keyword_frequency": dict(keyword_frequency),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "total_evaluations": len(state["평가하는사번_리스트"]),
        "average_weight": total_weight / len(state["비중"]) if state["비중"] else 0,
        "total_weight": total_weight,
        "new_keywords_analyzed": len(new_keywords)
    }
    
    state["_weighted_analysis"] = analysis_result
    
    print(f"[WeightedAnalysisAgent] {employee_id}: 분석 완료")
    print(f"  - 총 키워드: {len(weighted_scores)}")
    print(f"  - 긍정 키워드: {len(top_positive)}")
    print(f"  - 부정 키워드: {len(top_negative)}")
    print(f"  - 새로 분석된 키워드: {len(new_keywords)}")
    print(f"  - 평균 비중: {analysis_result['average_weight']:.2f}")
    
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 키워드 분석 완료")]
    return {"messages": messages, **state}


# 3. 개인용 Peer Talk 서브모듈
def individual_peer_talk_submodule(state: Module4AgentState) -> Module4AgentState:
    """
    완전히 새로운 간단한 맥락 문장 생성 에이전트
    """
    
    employee_id = state["평가받는사번"]
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 개인 맥락 생성 시작")]
    print(f"[SimpleContextAgent] {employee_id}: 간단한 문장 생성 시작")
    
    # 평가 데이터가 없으면 스킵
    if not state["평가하는사번_리스트"]:
        print(f"[SimpleContextAgent] {employee_id}: 평가 데이터 없음")
        state["동료평가요약줄글들"] = []
        messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 평가 데이터 없음")]
        return {"messages": messages, **state}
    
    요약문장들 = []
    
    # 각 평가별로 문장 생성
    for i in range(len(state["평가하는사번_리스트"])):
        키워드 = state["키워드모음"][i]
        업무내용 = state["구체적업무내용"][i]
        비중 = state["비중"][i]
        
        try:
            # 업무내용에서 핵심 키워드 추출
            work_keywords = _extract_work_keywords(업무내용)
            
            # 업무 내용을 요약하되 핵심 기술/프로세스 유지
            work_content_processed = _process_work_content_for_context(업무내용, work_keywords)
            
            # 업무 상황 간단히 추출
            work_situation = 업무내용[:100] + "..." if len(업무내용) > 100 else 업무내용
            
            요약문장 = call_llm_for_context_generation(키워드, work_situation, 비중)
            
            # 문장 품질 검증 및 보정
            요약문장 = _validate_and_enhance_sentence(요약문장, 키워드, work_keywords, 비중)
            
            요약문장들.append(요약문장)
            print(f"[SimpleContextAgent] {i+1}번째 문장 생성 완료")
            
        except Exception as e:
            # 실패 시 고도화된 기본 템플릿 사용
            fallback = _create_enhanced_fallback_sentence(employee_id, 키워드, 업무내용, 비중)
            요약문장들.append(fallback)
            print(f"[SimpleContextAgent] {i+1}번째 생성 실패, 고도화된 기본 템플릿 사용: {str(e)}")
    
    state["동료평가요약줄글들"] = 요약문장들
    print(f"[SimpleContextAgent] {employee_id}: 총 {len(요약문장들)}개 문장 생성 완료")
    
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 개인 맥락 생성 완료")]
    return {"messages": messages, **state}


# 4. 팀장용 Peer Talk 서브모듈 (모듈 7에서 처리하므로 여기서는 패스)
def manager_peer_talk_submodule(state: Module4AgentState) -> Module4AgentState:
    """팀장용 Peer Talk 서브모듈 (모듈 7에서 처리)"""
    employee_id = state["평가받는사번"]
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 팀장용 Peer Talk (모듈 7에서 처리)")]
    return {"messages": messages, **state}


# 5. 포맷터 서브모듈
def formatter_submodule(state: Module4AgentState) -> Module4AgentState:
    """
    최종 구조화된 피드백 생성 - LLM이 반드시 생성하도록 강화
    """
    
    employee_id = state["평가받는사번"]
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 최종 피드백 생성 시작")]
    print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 강화된 피드백 생성 시작")
    
    # 최소한의 데이터만 확인
    if not state["평가하는사번_리스트"]:
        print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 평가 데이터 없음, 강제 생성 시도")
        # 그래도 LLM으로 생성 시도
    
    # 가중치 분석 결과가 없어도 진행
    analysis = state.get("_weighted_analysis", {})
    
    # 최대 3번 재시도
    for attempt in range(3):
        try:
            print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 생성 시도 {attempt + 1}/3")
            
            # 상위 요약 문장들 선별
            top_sentences = _get_top_sentences(state)
            
            # LLM 호출
            feedback = call_llm_for_feedback_generation(
                employee_id,
                len(state["평가하는사번_리스트"]),
                ", ".join(analysis.get("top_positive", {}).keys()) if analysis.get("top_positive") else "협업, 책임감",
                ", ".join(analysis.get("top_negative", {}).keys()) if analysis.get("top_negative") else "일부 개선 영역",
                top_sentences
            )
            
            # 필수 키 확인
            required_keys = ["강점", "우려", "협업관찰"]
            if all(key in feedback for key in required_keys):
                # 성공적으로 파싱됨
                state["강점"] = [feedback["강점"]]
                state["우려"] = [feedback["우려"]]
                state["협업관찰"] = [feedback["협업관찰"]]
                
                print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 피드백 생성 성공!")
                print(f"  - 강점: {state['강점'][0]}")
                print(f"  - 우려: {state['우려'][0]}")
                print(f"  - 협업관찰: {state['협업관찰'][0]}")
                
                # 임시 분석 데이터 정리
                if "_weighted_analysis" in state:
                    del state["_weighted_analysis"]
                
                messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 최종 피드백 생성 성공")]
                return {"messages": messages, **state}
            else:
                print(f"[ImprovedFeedbackGenerationAgent] 필수 키 누락, 재시도...")
                continue
                
        except Exception as e:
            print(f"[ImprovedFeedbackGenerationAgent] 생성 오류: {str(e)}, 재시도...")
            continue
    
    # 3번 시도 후에도 실패하면 강제로 최소한의 응답 생성
    print(f"[ImprovedFeedbackGenerationAgent] {employee_id}: 3번 시도 실패, 최소한의 응답 강제 생성")
    
    # 키워드나 문장에서 추출하여 강제 생성
    keywords = " ".join(state.get("키워드모음", []))
    
    state["강점"] = [f"동료 평가에서 나타난 {keywords.split(',')[0] if keywords else '협업'} 등의 긍정적 특성을 바탕으로 팀에 기여하고 있습니다"]
    state["우려"] = ["지속적인 성장을 위해 일부 영역에서 추가적인 개발과 개선이 필요한 상황으로 보입니다"]
    state["협업관찰"] = ["팀 내에서의 역할 수행과 동료들과의 관계에서 전반적으로 안정적인 모습을 보여주고 있습니다"]
    
    # 임시 분석 데이터 정리
    if "_weighted_analysis" in state:
        del state["_weighted_analysis"]
    
    # DB에 저장
    peer_review_result = _format_peer_review_result(state)
    period_id = int(state["분기"])
    save_success = save_peer_review_result_to_db(engine, period_id, employee_id, peer_review_result)
    
    if save_success:
        print(f"[DatabaseStorageAgent] {employee_id}: DB 저장 성공!")
    else:
        print(f"[DatabaseStorageAgent] {employee_id}: DB 저장 실패")
    
    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 4: {employee_id} 최종 피드백 생성 및 DB 저장 완료")]
    return {"messages": messages, **state}


# --- 헬퍼 함수들 ---

def _create_empty_analysis():
    """빈 분석 결과 생성"""
    return {
        "weighted_scores": {},
        "keyword_frequency": {},
        "top_positive": {},
        "top_negative": {},
        "total_evaluations": 0,
        "average_weight": 0,
        "total_weight": 0,
        "new_keywords_analyzed": 0
    }


def _get_top_sentences(state):
    """비중 높은 순으로 요약 문장 정렬"""
    if not state["동료평가요약줄글들"] or not state["비중"]:
        return []
    
    sentences_with_weights = list(zip(state["동료평가요약줄글들"], state["비중"]))
    sorted_sentences = sorted(sentences_with_weights, key=lambda x: x[1], reverse=True)
    
    # 상위 3개 문장만 선별 (핵심만)
    return [s[0] for s in sorted_sentences[:3]]


def _format_peer_review_result(state: Module4AgentState) -> str:
    """
    강점, 우려, 협업관찰을 줄바꿈 포함한 텍스트로 포맷팅
    """
    # 각 항목에서 첫 번째 요소 추출 (리스트 형태이므로)
    강점 = state["강점"][0] if state["강점"] else "동료들로부터 긍정적인 평가를 받고 있습니다."
    우려 = state["우려"][0] if state["우려"] else "지속적인 성장을 위한 개선 영역이 있습니다."
    협업관찰 = state["협업관찰"][0] if state["협업관찰"] else "팀 내에서 협업에 참여하고 있습니다."
    
    # 줄바꿈 포함하여 텍스트 생성
    peer_review_result = f"""강점: {강점}
우려: {우려}
협업관찰: {협업관찰}"""
    
    return peer_review_result


# 워크플로우 생성
def create_module4_graph():
    """모듈 4 그래프 생성 및 반환"""
    module4_workflow = StateGraph(Module4AgentState)
    
    # 노드 추가
    module4_workflow.add_node("data_collection", data_collection_submodule)
    module4_workflow.add_node("keyword_analysis_engine", keyword_analysis_engine_submodule)
    module4_workflow.add_node("individual_peer_talk", individual_peer_talk_submodule)
    module4_workflow.add_node("manager_peer_talk", manager_peer_talk_submodule)
    module4_workflow.add_node("formatter", formatter_submodule)
    
    # 엣지 정의
    module4_workflow.add_edge(START, "data_collection")
    module4_workflow.add_edge("data_collection", "keyword_analysis_engine")
    module4_workflow.add_edge("keyword_analysis_engine", "individual_peer_talk")
    module4_workflow.add_edge("individual_peer_talk", "manager_peer_talk")
    module4_workflow.add_edge("manager_peer_talk", "formatter")
    module4_workflow.add_edge("formatter", END)
    
    return module4_workflow.compile()