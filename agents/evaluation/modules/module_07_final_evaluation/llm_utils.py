# ================================================================
# llm_utils_module7.py - 모듈 7 LLM 관련 유틸리티
# ================================================================

import re
import json
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

from agents.evaluation.modules.module_07_final_evaluation.db_utils import *

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(f"LLM Client initialized: {llm_client.model_name}")

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ================================================================
# LLM 호출 함수
# ================================================================

def call_llm_for_normalized_evaluation_comments(
    emp_data: Dict,
    normalized_score: float,
    raw_hybrid_score: float,
    achievement_score: float,
    fourp_score: float,
    quarterly_tasks: List[Dict],
    fourp_results: Dict,
    achievement_reason: str,
    normalization_reason: str
) -> Dict:
    """정규화된 점수로 ai_reason과 comment 생성"""
    
    emp_no = emp_data["emp_no"]
    emp_name = emp_data.get("emp_name", emp_no)
    position = emp_data.get("position", "직책 정보 없음")
    cl = emp_data.get("cl", "CL 정보 없음")
    
    print(f"LLM Call: {emp_no} 정규화 후 종합평가 근거 생성")
    
    # 분기별 Task 요약
    quarterly_summary = ""
    for i, task in enumerate(quarterly_tasks[:8]):  # 최대 8개만 표시
        quarterly_summary += f"Q{task.get('period_id')}: {task.get('task_name')} - 달성률 {task.get('ai_contribution_score', 0)}점\n"
        if task.get('ai_analysis_comment_task'):
            quarterly_summary += f"  → {task.get('ai_analysis_comment_task')}\n"
    
    if not quarterly_summary:
        quarterly_summary = "분기별 Task 데이터 없음"
    
    # 4P 요약
    fourp_summary = f"""
    - Passionate: {fourp_results.get('passionate', {}).get('score', 3.0)}점
    - Proactive: {fourp_results.get('proactive', {}).get('score', 3.0)}점  
    - Professional: {fourp_results.get('professional', {}).get('score', 3.0)}점
    - People: {fourp_results.get('people', {}).get('score', 3.0)}점
    - 평균: {fourp_results.get('overall', {}).get('average_score', 3.0)}점
    """
    
    system_prompt = """
    당신은 SK 조직의 종합 평가 전문가입니다.
    SK 등급 체계 기반 절대평가(달성률)와 정성평가(4P BARS)를 종합하고 CL별 정규화를 거친 최종 점수를 바탕으로 
    두 가지 관점의 평가 근거를 생성해주세요.

    SK 등급 체계 (절대평가):
    - S+등급(120% 이상): 5.0점 (탁월한 성과)
    - S등급(110-120%): 4.0-5.0점 (매우 우수한 성과)
    - A등급(100-110%): 3.5-4.0점 (목표 달성)
    - B등급(80-100%): 2.5-3.5점 (목표 근접)
    - C등급(60-80%): 1.5-2.5점 (목표 미달)
    - D등급(60% 미만): 1.0-1.5점 (크게 미달)

    두 가지 관점의 근거를 생성하세요:
    1. ai_reason: 팀장이 보는 AI 평가 근거 (객관적, 분석적, 절대평가+CL별 정규화 과정 포함)
    2. comment: 팀원이 보는 평가 근거 초안 (개인 친화적, 발전지향적, 격려 포함)

    결과는 JSON 형식으로만 응답하세요.
    """
    
    human_prompt = f"""
    <직원 정보>
    이름: {emp_name}
    사번: {emp_no}
    직책: {position}
    CL: {cl}
    </직원 정보>

    <점수 산출 과정>
    최종 정규화 점수: {normalized_score}점
    ├── 원시 하이브리드 점수: {raw_hybrid_score:.2f}점
    │   ├── 달성률 점수: {achievement_score}점 ({achievement_reason})
    │   └── 4P 점수: {fourp_score}점
    └── CL별 정규화: {normalization_reason}
    </점수 산출 과정>

    <상세 근거 데이터>
    연간 달성률: {emp_data.get('ai_annual_achievement_rate', 0)}%
    성과 요약: {emp_data.get('ai_annual_performance_summary_comment', '성과 요약 없음')}
    동료평가: {emp_data.get('ai_peer_talk_summary', '동료평가 없음')}
    
    4P 평가 결과:
    {fourp_summary}
    
    분기별 Task 성과:
    {quarterly_summary}
    </상세 근거 데이터>

    JSON 응답:
    {{
        "ai_reason": "[팀장용: {emp_name}({emp_no})님에 대한 객관적이고 구체적인 AI 평가 근거. SK 등급 체계 기반 절대평가 결과와 CL별 정규화 과정을 포함하여 분석적 관점으로 설명]",
        "comment": "[팀원용: {emp_name}님께 드리는 개인 친화적이고 발전지향적인 평가 근거 초안. 최종 점수 {normalized_score}점에 대한 격려와 성장 방향 포함]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response = chain.invoke({})
        json_output_raw = response.content
        json_output = _extract_json_from_llm_response(str(json_output_raw))
        llm_parsed_data = json.loads(json_output)
        
        ai_reason = llm_parsed_data.get("ai_reason", "")
        comment = llm_parsed_data.get("comment", "")

        if not ai_reason or not comment:
            raise ValueError("LLM 응답에서 ai_reason 또는 comment가 누락됨")

        return {
            "ai_reason": ai_reason,
            "comment": comment
        }

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}")
        return {
            "ai_reason": f"{emp_name}님의 정규화 후 종합 평가 근거 생성 중 오류 발생",
            "comment": f"{emp_name}님께 드리는 정규화 후 평가 근거 생성 중 오류 발생"
        }
    except Exception as e:
        print(f"LLM 호출 중 오류: {e}")
        return {
            "ai_reason": f"{emp_name}님의 정규화 후 종합 평가 근거 생성 중 오류 발생",
            "comment": f"{emp_name}님께 드리는 정규화 후 평가 근거 생성 중 오류 발생"
        }