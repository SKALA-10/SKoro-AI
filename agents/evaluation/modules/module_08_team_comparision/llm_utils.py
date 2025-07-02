# ================================================================
# llm_utils_module8.py - 모듈 8 LLM 관련 유틸리티
# ================================================================

import re
import json
import time
import logging
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from agents.evaluation.modules.module_08_team_comparision.comparison_utils import *

load_dotenv()

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
logger = logging.getLogger(__name__)

def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ================================================================
# 에러 처리 클래스
# ================================================================

class Module8ValidationError(Exception):
    pass

# ================================================================
# LLM 함수
# ================================================================

def robust_llm_call(prompt: str, validation_func, max_retries: int = 3, context: str = ""):
    """견고한 LLM 호출"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = llm_client.invoke(prompt)
            content = str(response.content)
            validated_result = validation_func(content)
            return validated_result
            
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call attempt {attempt + 1} failed for {context}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 지수 백오프
    
    logger.error(f"All LLM attempts failed for {context}: {last_error}")
    raise Module8ValidationError(f"Failed after {max_retries} attempts: {last_error}")

def call_llm_for_team_performance_comment(our_overall_rate: float, cluster_stats: Dict, 
                                        kpi_comparison_results: List[Dict], 
                                        similar_teams_count: int) -> str:
    """팀 성과 비교 코멘트 생성"""
    
    # KPI별 상세 분석 문자열 생성
    kpi_details = ""
    for kpi in kpi_comparison_results:
        if kpi["similar_avg_rate"] is not None:
            kpi_details += f"- {kpi['kpi_name']}: {kpi['our_rate']}% vs {kpi['similar_avg_rate']}% (유사팀 평균) → {kpi['comparison_result']}\n"
        else:
            kpi_details += f"- {kpi['kpi_name']}: {kpi['our_rate']}% (유사 KPI 없음) → -\n"
    
    system_prompt = """
    당신은 SK의 팀 성과 분석 전문가입니다.
    다음 정보를 바탕으로 팀장에게 제공할 객관적이고 건설적인 성과 분석 코멘트를 작성해주세요.
    
    다음 구성으로 250-300자 분량의 코멘트를 작성해주세요:
    1. 종합 달성률 평가 (유사팀 대비)
    2. KPI별 강점/특징 분석
    3. 팀 성과의 전반적 특성
    4. 간단한 개선 방향 (필요시)
    
    결과는 다음 JSON 형식으로만 응답해주세요:
    {
      "comment": "[250-300자 분량의 팀 성과 분석 코멘트]"
    }
    """
    
    human_prompt = f"""
    <팀 성과 정보>
    종합 달성률: {our_overall_rate}% (유사팀 평균: {cluster_stats['avg_rate']}%)
    클러스터 내 위치: {similar_teams_count}개 유사팀과 비교
    </팀 성과 정보>
    
    <KPI별 상세 분석>
    {kpi_details}
    </KPI별 상세 분석>
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    def validate_comment_response(response: str) -> str:
        try:
            json_output_raw = response
            json_output = _extract_json_from_llm_response(json_output_raw)
            llm_parsed_data = json.loads(json_output)
            
            comment = llm_parsed_data.get("comment", "")
            if not comment:
                raise ValueError("LLM이 빈 코멘트를 반환했습니다.")
            
            return comment
            
        except Exception as e:
            logger.error(f"Comment validation failed: {e}")
            # 폴백 코멘트
            overall_comparison = get_comparison_result_detailed(
                our_overall_rate, cluster_stats
            )
            return f"귀하의 팀은 종합 달성률 {our_overall_rate}%로 유사팀 평균({cluster_stats['avg_rate']}%) 대비 {overall_comparison} 성과를 보이고 있습니다. 유사팀 {similar_teams_count}개와의 비교 결과를 바탕으로 지속적인 성장을 위한 전략 수립이 필요합니다."
    
    chain = prompt | llm_client
    
    try:
        response = chain.invoke({})
        content = str(response.content)
        return validate_comment_response(content)
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        # 폴백 코멘트
        overall_comparison = get_comparison_result_detailed(
            our_overall_rate, cluster_stats
        )
        return f"귀하의 팀은 종합 달성률 {our_overall_rate}%로 유사팀 평균({cluster_stats['avg_rate']}%) 대비 {overall_comparison} 성과를 보이고 있습니다. 유사팀 {similar_teams_count}개와의 비교 결과를 바탕으로 지속적인 성장을 위한 전략 수립이 필요합니다."