# ================================================================
# llm_utils_module2.py - 모듈 2 LLM 관련 유틸리티
# ================================================================

import re
import json
import time
import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from agents.evaluation.modules.module_02_goal_achievement.calculation_utils import *
from agents.evaluation.modules.module_02_goal_achievement.db_utils import *

load_dotenv()

# LLM 클라이언트 설정
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
logger = logging.getLogger(__name__)

def extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# ================================================================
# 에러 처리 클래스들
# ================================================================

class LLMValidationError(Exception):
    pass

# ================================================================
# LLM 호출 및 검증 함수들
# ================================================================

def robust_llm_call(prompt: str, validation_func, max_retries: int = 3, context: str = ""):
    """견고한 LLM 호출 - 우리가 상의한 에러 처리"""
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
    raise LLMValidationError(f"Failed after {max_retries} attempts: {last_error}")

def validate_achievement_rate(response: str) -> Dict:
    """달성률 응답 검증"""
    try:
        json_output = extract_json_from_llm_response(response)
        data = json.loads(json_output)
        
        rate = data.get("achievement_rate")
        grade = data.get("grade", "")
        
        if not isinstance(rate, (int, float)) or not (0 <= rate <= 200):
            raise ValueError(f"Invalid achievement rate: {rate}")
        
        if grade and grade not in ["S", "A", "B", "C", "D"]:
            raise ValueError(f"Invalid grade: {grade}")
            
        return {
            "achievement_rate": round(float(rate), 2),
            "grade": grade if grade else None
        }
        
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMValidationError(f"Achievement rate validation failed: {e}")

def validate_contribution_analysis(response: str) -> Dict:
    """기여도 분석 응답 검증"""
    try:
        json_output = extract_json_from_llm_response(response)
        data = json.loads(json_output)
        
        kpi_rate = data.get("kpi_overall_rate")
        contributions = data.get("individual_contributions", {})
        
        if not isinstance(kpi_rate, (int, float)) or not (0 <= kpi_rate <= 200):
            raise ValueError(f"Invalid KPI rate: {kpi_rate}")
            
        # 기여도 합계 검증 (100% ± 5% 허용)
        total_contribution = sum(float(v) for v in contributions.values())
        if abs(total_contribution - 100.0) > 5.0:
            logger.warning(f"Contribution sum: {total_contribution}%, normalizing to 100%")
            # 정규화
            if total_contribution > 0:
                contributions = {k: round((float(v) / total_contribution) * 100, 2) 
                               for k, v in contributions.items()}
        
        return {
            "kpi_overall_rate": round(float(kpi_rate), 2),
            "individual_contributions": contributions,
            "kpi_analysis_comment": data.get("kpi_analysis_comment", "")
        }
        
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMValidationError(f"Contribution analysis validation failed: {e}")

def classify_kpi_type_by_llm(kpi_data: Dict) -> str:
    """KPI 평가 방식 자동 분류"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        다음 KPI의 평가 방식을 분류해주세요:
        
        - quantitative: 개인별 수치 성과를 합산하여 기여도 계산 가능
        - qualitative: 수치적 비교 불가능한 정성 평가 필요
        
        답변: quantitative 또는 qualitative (한 단어만)
        """),
        HumanMessage(content=f"""
        KPI명: {kpi_data.get('kpi_name', '')}
        KPI 설명: {kpi_data.get('kpi_description', '')}
        """)
    ])
    
    def validate_type(response: str) -> str:
        response = response.strip().lower()
        if response in ["quantitative", "qualitative"]:
            return response
        raise ValueError(f"Invalid type: {response}")
    
    return robust_llm_call(str(prompt.format()), validate_type, context="KPI classification")

# ================================================================
# 배치 처리 함수들
# ================================================================

def batch_calculate_achievement_and_grades(batch_data: List[Dict], include_grades: bool) -> List[Dict]:
    """배치 달성률+등급 계산"""
    
    # 프롬프트 구성
    tasks_text = ""
    for i, data in enumerate(batch_data):
        tasks_text += f"\n{i+1}. Task ID: {data['task_id']}\n"
        tasks_text += f"   목표: {data['target_level']}\n"
        tasks_text += f"   성과: {data['cumulative_performance']}\n"
        tasks_text += f"   상세: {data['cumulative_summary'][:200]}...\n"
    
    if include_grades:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            다음 Task들의 달성률(0-200%)과 등급(S,A,B,C,D)을 계산해주세요.
            
            평가 기준:
            - 달성률: 목표 대비 실제 성과 (100% = 목표 달성, 100% 초과 = 목표 초과)
            - 등급: S(초과달성), A(완전달성), B(양호), C(미흡), D(불량)
            
            현재 성과를 객관적으로 평가해주세요.
            JSON 형식으로 응답해주세요.
            """),
            HumanMessage(content=f"""
            {tasks_text}
            
            답변 형식:
            {{
              "results": [
                {{"task_id": 1, "achievement_rate": 85, "grade": "B"}},
                {{"task_id": 2, "achievement_rate": 120, "grade": "S"}}
              ]
            }}
            """)
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            다음 Task들의 달성률(0-200%)을 계산해주세요.
            
            달성률 기준: 목표 대비 실제 성과 (100% = 목표 달성)
            현재 성과를 객관적으로 평가해주세요.
            """),
            HumanMessage(content=f"""
            {tasks_text}
            
            답변 형식:
            {{
              "results": [
                {{"task_id": 1, "achievement_rate": 85}},
                {{"task_id": 2, "achievement_rate": 120}}
              ]
            }}
            """)
        ])
    
    def validate_batch_response(response: str) -> List[Dict]:
        try:
            json_output = extract_json_from_llm_response(response)
            data = json.loads(json_output)
            results = data.get("results", [])
            
            validated_results = []
            for result in results:
                rate = result.get("achievement_rate")
                if not isinstance(rate, (int, float)) or not (0 <= rate <= 200):
                    rate = 80.0  # 기본값
                
                validated_result: Dict[str, Any] = {"achievement_rate": round(float(rate), 2)}
                
                if include_grades:
                    grade = result.get("grade", "C")
                    if grade not in ["S", "A", "B", "C", "D"]:
                        grade = "C"
                    validated_result["grade"] = str(grade)
                
                validated_results.append(validated_result)
            
            return validated_results
            
        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            # 기본값으로 폴백
            fallback_results = []
            for _ in batch_data:
                result: Dict[str, Any] = {"achievement_rate": 80.0}
                if include_grades:
                    result["grade"] = "C"
                fallback_results.append(result)
            return fallback_results
    
    return robust_llm_call(str(prompt.format()), validate_batch_response, context="batch achievement calculation")

def calculate_qualitative_contributions(kpi_id: int, period_id: int, kpi_data: Dict) -> Dict[str, float]:
    """정성 평가 기여도 계산 - 우리가 상의한 grade_rule 기반"""
    from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_kpi_tasks
    
    tasks = fetch_kpi_tasks(kpi_id, period_id)
    evaluation_criteria = get_evaluation_criteria(kpi_id)
    
    # LLM 프롬프트 구성
    criteria_text = "\n".join([f"- {criterion}" for criterion in evaluation_criteria])
    
    tasks_text = ""
    emp_nos = []
    for task in tasks:
        emp_nos.append(task['emp_no'])
        tasks_text += f"\n- {task['emp_name']}({task['emp_no']}): {task['task_name']}\n"
        tasks_text += f"  내용: {task['task_summary']}\n"
        tasks_text += f"  성과: {task['task_performance']}\n"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        다음 KPI에 대해 팀원들의 상대적 기여도를 분석해주세요.
        
        평가 기준:
        {criteria_text}
        
        모든 팀원의 기여도 합계는 정확히 100%가 되어야 합니다.
        현재 성과와 기여도를 객관적으로 분석해주세요.
        """),
        HumanMessage(content=f"""
        KPI: {kpi_data.get('kpi_name', '')}
        설명: {kpi_data.get('kpi_description', '')}
        
        팀원별 업무:
        {tasks_text}
        
        JSON 답변:
        {{
          "kpi_overall_rate": [KPI 전체 진행률 0-200%],
          "individual_contributions": {{
            "{emp_nos[0] if emp_nos else 'EMP001'}": [기여도 0-100%],
            "{emp_nos[1] if len(emp_nos) > 1 else 'EMP002'}": [기여도 0-100%]
          }},
          "kpi_analysis_comment": "[현재 상태 분석 코멘트]"
        }}
        """)
    ])
    
    result = robust_llm_call(str(prompt.format()), validate_contribution_analysis, context=f"KPI {kpi_id} qualitative analysis")
    
    # KPI 레벨 결과 저장
    update_team_kpi(kpi_id, {
        "ai_kpi_progress_rate": int(result['kpi_overall_rate']),
        "ai_kpi_analysis_comment": result['kpi_analysis_comment']
    })
    
    return result['individual_contributions']

def calculate_team_kpi_achievement_rate(kpi_id: int, period_id: int, kpi_data: Dict) -> Dict:
    """팀 KPI 달성률 LLM 계산"""
    from agents.evaluation.modules.module_02_goal_achievement.db_utils import fetch_kpi_tasks
    
    tasks = fetch_kpi_tasks(kpi_id, period_id)
    
    tasks_text = ""
    for task in tasks:
        tasks_text += f"\n- {task['emp_name']}: {task['task_name']}\n"
        tasks_text += f"  목표: {task['target_level']}\n"
        tasks_text += f"  성과: {task['task_performance']}\n"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        팀 KPI의 전체 달성률을 0-200% 범위로 계산해주세요.
        개별 팀원들의 성과를 종합하여 팀 전체의 현재 목표 달성 수준을 객관적으로 판단해주세요.
        """),
        HumanMessage(content=f"""
        KPI: {kpi_data.get('kpi_name', '')}
        KPI 목표: {kpi_data.get('kpi_description', '')}
        
        팀원별 개별 성과:
        {tasks_text}
        
        JSON 답변:
        {{
            "kpi_overall_rate": [팀 KPI 달성률 0-200%],
            "kpi_analysis_comment": "[현재 KPI 달성 상태 분석]"
        }}
        """)
    ])
    
    def validate_kpi_rate(response: str) -> Dict:
        try:
            json_output = extract_json_from_llm_response(response)
            data = json.loads(json_output)
            
            rate = data.get("kpi_overall_rate")
            if not isinstance(rate, (int, float)) or not (0 <= rate <= 200):
                rate = 80.0
                
            return {
                "rate": round(float(rate), 2),
                "comment": data.get("kpi_analysis_comment", "현재 상태 분석 실패")
            }
        except Exception as e:
            return {"rate": 80.0, "comment": f"KPI 분석 실패: {str(e)[:100]}"}
    
    return robust_llm_call(str(prompt.format()), validate_kpi_rate, context=f"Team KPI {kpi_id}")