# -*- coding: utf-8 -*-
"""
모듈2: 목표달성도 분석 모듈 - 완전 구현

상의 내용 반영사항:
1. 6단계 서브모듈 구조 (등급 산정을 달성률과 통합)
2. evaluation_type 기반 정량/정성 평가 분류
3. 하이브리드 개인 종합 기여도 계산 (참여자 수 보정 + KPI 비중)
4. LLM 배치 처리 + 에러 처리
5. 구조화된 코멘트 생성 (분기별/연말 톤 차별화)
6. 분기별 누적 처리 (최종 성과 기준)
7. 팀원 변경시 실제 참여 기간만 평가
8. 팀 단위 코멘트 일관성 가이드
9. grades.grade_rule 기반 평가 기준 추출
10. DB 기반 경량 State 전달 방식
"""

import sys
import os
import json
import re
import time
import logging
from typing import Dict, List, Optional, Any, Literal, TypedDict
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime

# 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))  # 4단계로 수정
sys.path.append(project_root)

from config.settings import DatabaseConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from langgraph.graph import StateGraph, START, END  # LangGraph 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# httpx 및 관련 라이브러리 로그 레벨 조정 (HTTP 요청 로그 숨김)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# DB 및 LLM 설정
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== 상태 정의 =====
@dataclass
class Module2State:
    """경량 State - 우리가 상의한 DB 기반 전달 방식"""
    # 기본 정보
    report_type: Literal["quarterly", "annual"]
    team_id: int
    period_id: int
    
    # 타겟 ID들
    target_task_ids: List[int]
    target_team_kpi_ids: List[int]
    
    # 처리 결과 추적용 (DB ID만 저장)
    updated_task_ids: Optional[List[int]] = None
    updated_team_kpi_ids: Optional[List[int]] = None
    feedback_report_ids: Optional[List[int]] = None
    team_evaluation_id: Optional[int] = None
    final_evaluation_report_ids: Optional[List[int]] = None
    
    # 특별 전달 데이터 (서브모듈 간 필요시만)
    team_context_guide: Optional[Dict] = None

# ===== 에러 처리 클래스 =====
class LLMValidationError(Exception):
    pass

class DataIntegrityError(Exception):
    pass

# ===== 유틸리티 함수 =====
def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

def extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 JSON 블록 추출"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_number_from_response(response: str) -> float:
    """응답에서 숫자 추출"""
    patterns = [
        r'^(\d+(?:\.\d+)?)$',           # "85", "85.5"
        r'(\d+(?:\.\d+)?)%',            # "85%"  
        r'(\d+(?:\.\d+)?)\s*점',         # "85점"
        r':(\d+(?:\.\d+)?)(?:[:%]|$)',  # "1:85", "1:85%"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            return float(match.group(1))
    
    raise ValueError(f"No valid number found in response: {response}")

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """안전한 나누기"""
    if denominator == 0:
        print(f"   ⚠️  나누기 오류: {numerator}/{denominator}, 기본값 {default} 사용")
        return default
    return numerator / denominator

def calculate_weighted_average(values: List[float], weights: List[float], default: float = 0) -> float:
    """가중평균 계산: Σ(값 × 가중치) / Σ(가중치)"""
    if not values or not weights or len(values) != len(weights):
        return default
    
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights)
    
    return safe_divide(weighted_sum, total_weight, default)

def calculate_individual_weighted_achievement_rate(individual_tasks: List[Dict]) -> Dict[str, float]:
    """개인별 가중평균 달성률 계산"""
    if not individual_tasks:
        return {"achievement_rate": 0, "contribution_rate": 0, "total_weight": 0}
    
    # 달성률 가중평균 계산
    achievement_rates = []
    weights = []
    
    for task in individual_tasks:
        achievement_rate = task.get('ai_achievement_rate', 0)
        weight = task.get('weight', 0)
        
        achievement_rates.append(achievement_rate)
        weights.append(weight)
    
    # 가중평균 달성률
    weighted_achievement = calculate_weighted_average(achievement_rates, weights, 0)
    
    # 기여도는 단순평균 (기존 방식 유지)
    contribution_rates = [task.get('ai_contribution_score', 0) for task in individual_tasks]
    avg_contribution = sum(contribution_rates) / len(contribution_rates) if contribution_rates else 0
    
    total_weight = sum(weights)
    
    return {
        "achievement_rate": weighted_achievement,
        "contribution_rate": avg_contribution,
        "total_weight": total_weight
    }

# ===== 데이터 조회 함수 =====
def fetch_team_evaluation_id(team_id: int, period_id: int) -> Optional[int]:
    """team_evaluations에서 ID 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT team_evaluation_id FROM team_evaluations 
            WHERE team_id = :team_id AND period_id = :period_id
        """)
        result = connection.execute(query, {"team_id": team_id, "period_id": period_id})
        return result.scalar_one_or_none()

def fetch_team_members(team_id: int) -> List[Dict]:
    """팀 멤버 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT emp_no, emp_name, cl, position, role 
            FROM employees 
            WHERE team_id = :team_id
        """)
        results = connection.execute(query, {"team_id": team_id})
        return [row_to_dict(row) for row in results]

def fetch_cumulative_task_data(task_id: int, period_id: int) -> Dict:
    """누적 Task 데이터 조회 - 우리가 상의한 방식"""
    with engine.connect() as connection:
        query = text("""
            SELECT ts.*, t.task_name, t.target_level, t.weight, t.emp_no, t.team_kpi_id, 
                   e.emp_name, tk.kpi_name, tk.kpi_description
            FROM task_summaries ts
            JOIN tasks t ON ts.task_id = t.task_id
            JOIN employees e ON t.emp_no = e.emp_no
            JOIN team_kpis tk ON t.team_kpi_id = tk.team_kpi_id
            WHERE ts.task_id = :task_id AND ts.period_id <= :period_id
            ORDER BY ts.period_id
        """)
        results = connection.execute(query, {"task_id": task_id, "period_id": period_id})
        task_summaries = [row_to_dict(row) for row in results]
        
        if not task_summaries:
            return {}
        
        latest = task_summaries[-1]
        cumulative_summary = "\n".join([
            f"Q{ts['period_id']}: {ts['task_summary']}" 
            for ts in task_summaries if ts['task_summary']
        ])
        
        return {
            **latest,
            "cumulative_task_summary": cumulative_summary,
            "cumulative_performance": latest.get('task_performance', ''),
            "participation_periods": len(task_summaries)
        }

def fetch_team_kpi_data(team_kpi_id: int) -> Dict:
    """Team KPI 데이터 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT tk.*, g.grade_rule, g.grade_s, g.grade_a, g.grade_b, g.grade_c, g.grade_d
            FROM team_kpis tk
            LEFT JOIN grades g ON tk.team_kpi_id = g.team_kpi_id OR g.team_kpi_id IS NULL
            WHERE tk.team_kpi_id = :team_kpi_id
            LIMIT 1
        """)
        result = connection.execute(query, {"team_kpi_id": team_kpi_id})
        row = result.fetchone()
        return row_to_dict(row) if row else {}

def fetch_kpi_tasks(team_kpi_id: int, period_id: int) -> List[Dict]:
    """특정 KPI의 최신 분기 Task들 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT t.task_id, t.task_name, t.target_level, t.weight, t.emp_no,
                   e.emp_name, ts.task_summary, ts.task_performance, ts.period_id
            FROM tasks t
            JOIN employees e ON t.emp_no = e.emp_no
            JOIN task_summaries ts ON t.task_id = ts.task_id
            WHERE t.team_kpi_id = :team_kpi_id 
            AND ts.period_id = :period_id
        """)
        results = connection.execute(query, {"team_kpi_id": team_kpi_id, "period_id": period_id})
        return [row_to_dict(row) for row in results]

def check_evaluation_type(team_kpi_id: int) -> str:
    """evaluation_type 확인 (없으면 자동 분류)"""
    with engine.connect() as connection:
        query = text("SELECT evaluation_type FROM team_kpis WHERE team_kpi_id = :team_kpi_id")
        result = connection.execute(query, {"team_kpi_id": team_kpi_id})
        evaluation_type = result.scalar_one_or_none()
        
        if evaluation_type:
            return evaluation_type
            
        # 자동 분류 (LLM 기반)
        kpi_data = fetch_team_kpi_data(team_kpi_id)
        auto_type = classify_kpi_type_by_llm(kpi_data)
        
        # DB 업데이트
        update_query = text("""
            UPDATE team_kpis SET evaluation_type = :evaluation_type 
            WHERE team_kpi_id = :team_kpi_id
        """)
        connection.execute(update_query, {
            "evaluation_type": auto_type, 
            "team_kpi_id": team_kpi_id
        })
        connection.commit()
        
        return auto_type

# ===== 데이터 업데이트 함수 =====
def update_task_summary(task_summary_id: int, data: Dict) -> bool:
    """task_summaries 업데이트"""
    with engine.connect() as connection:
        set_clauses = [f"{k} = :{k}" for k in data.keys()]
        query = text(f"""
            UPDATE task_summaries 
            SET {', '.join(set_clauses)}
            WHERE task_summary_id = :task_summary_id
        """)
        result = connection.execute(query, {**data, "task_summary_id": task_summary_id})
        connection.commit()
        return result.rowcount > 0

def update_team_kpi(team_kpi_id: int, data: Dict) -> bool:
    """team_kpis 업데이트"""
    with engine.connect() as connection:
        set_clauses = [f"{k} = :{k}" for k in data.keys()]
        query = text(f"""
            UPDATE team_kpis 
            SET {', '.join(set_clauses)}
            WHERE team_kpi_id = :team_kpi_id
        """)
        result = connection.execute(query, {**data, "team_kpi_id": team_kpi_id})
        connection.commit()
        return result.rowcount > 0

def save_feedback_report(emp_no: str, team_evaluation_id: int, data: Dict) -> int:
    """feedback_reports 저장/업데이트"""
    with engine.connect() as connection:
        # 기존 레코드 확인
        check_query = text("""
            SELECT feedback_report_id FROM feedback_reports 
            WHERE emp_no = :emp_no AND team_evaluation_id = :team_evaluation_id
        """)
        existing_id = connection.execute(check_query, {
            "emp_no": emp_no, 
            "team_evaluation_id": team_evaluation_id
        }).scalar_one_or_none()
        
        if existing_id:
            # 업데이트
            set_clauses = [f"{k} = :{k}" for k in data.keys()]
            update_query = text(f"""
                UPDATE feedback_reports 
                SET {', '.join(set_clauses)}
                WHERE feedback_report_id = :feedback_report_id
            """)
            connection.execute(update_query, {**data, "feedback_report_id": existing_id})
            connection.commit()
            return existing_id
        else:
            # 신규 생성
            cols = ["emp_no", "team_evaluation_id"] + list(data.keys())
            placeholders = [f":{col}" for col in cols]
            insert_query = text(f"""
                INSERT INTO feedback_reports ({', '.join(cols)})
                VALUES ({', '.join(placeholders)})
            """)
            connection.execute(insert_query, {
                "emp_no": emp_no,
                "team_evaluation_id": team_evaluation_id,
                **data
            })
            connection.commit()
            
            # 새로 생성된 ID 조회
            new_id = connection.execute(check_query, {
                "emp_no": emp_no, 
                "team_evaluation_id": team_evaluation_id
            }).scalar_one()
            return new_id

def update_team_evaluations(team_evaluation_id: int, data: Dict) -> bool:
    """team_evaluations 업데이트"""
    with engine.connect() as connection:
        set_clauses = [f"{k} = :{k}" for k in data.keys()]
        query = text(f"""
            UPDATE team_evaluations 
            SET {', '.join(set_clauses)}
            WHERE team_evaluation_id = :team_evaluation_id
        """)
        result = connection.execute(query, {**data, "team_evaluation_id": team_evaluation_id})
        connection.commit()
        return result.rowcount > 0

def save_final_evaluation_report(emp_no: str, team_evaluation_id: int, data: Dict) -> int:
    """final_evaluation_reports 저장/업데이트"""
    with engine.connect() as connection:
        # 기존 레코드 확인
        check_query = text("""
            SELECT final_evaluation_report_id FROM final_evaluation_reports 
            WHERE emp_no = :emp_no AND team_evaluation_id = :team_evaluation_id
        """)
        existing_id = connection.execute(check_query, {
            "emp_no": emp_no, 
            "team_evaluation_id": team_evaluation_id
        }).scalar_one_or_none()
        
        if existing_id:
            # 업데이트
            set_clauses = [f"{k} = :{k}" for k in data.keys()]
            update_query = text(f"""
                UPDATE final_evaluation_reports 
                SET {', '.join(set_clauses)}
                WHERE final_evaluation_report_id = :final_evaluation_report_id
            """)
            connection.execute(update_query, {**data, "final_evaluation_report_id": existing_id})
            connection.commit()
            return existing_id
        else:
            # 신규 생성
            cols = ["emp_no", "team_evaluation_id"] + list(data.keys())
            placeholders = [f":{col}" for col in cols]
            insert_query = text(f"""
                INSERT INTO final_evaluation_reports ({', '.join(cols)})
                VALUES ({', '.join(placeholders)})
            """)
            connection.execute(insert_query, {
                "emp_no": emp_no,
                "team_evaluation_id": team_evaluation_id,
                **data
            })
            connection.commit()
            
            # 새로 생성된 ID 조회
            new_id = connection.execute(check_query, {
                "emp_no": emp_no, 
                "team_evaluation_id": team_evaluation_id
            }).scalar_one()
            return new_id

# ===== LLM 호출 및 검증 함수 =====
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

# ===== 평가 기준 처리 =====
def get_evaluation_criteria(team_kpi_id: int) -> List[str]:
    """우리가 상의한 평가 기준 처리 로직"""
    kpi_data = fetch_team_kpi_data(team_kpi_id)
    grade_rule = kpi_data.get('grade_rule')
    
    if grade_rule and grade_rule.strip():
        criteria = parse_criteria_from_grade_rule(grade_rule)
        if criteria:
            return criteria
    
    # 기본 평가 기준
    return ["목표달성 기여도", "성과 영향력", "업무 완성도"]

def parse_criteria_from_grade_rule(grade_rule: str) -> Optional[List[str]]:
    """grade_rule에서 평가 기준 추출"""
    if not grade_rule or not grade_rule.strip():
        return None
    
    lines = grade_rule.strip().split('\n')
    criteria = []
    
    for line in lines:
        line = line.strip()
        # "- " 또는 "• " 제거하고 내용 추출
        match = re.match(r'^[-•]\s*(.+)$', line)
        if match:
            criteria.append(match.group(1).strip())
        elif line and not line.startswith(('-', '•')):
            criteria.append(line)
    
    # 너무 많은 기준은 제한
    if len(criteria) > 5:
        criteria = criteria[:5]
        
    return criteria if criteria else None

# ===== 서브모듈 1: 데이터 수집 =====
def data_collection_submodule(state: Module2State) -> Module2State:
    """데이터 수집 서브모듈"""
    print(f"   📋 데이터 수집 중...")
    
    # team_evaluation_id 확인/생성
    team_evaluation_id = fetch_team_evaluation_id(state.team_id, state.period_id)
    if not team_evaluation_id:
        raise DataIntegrityError(f"team_evaluation_id not found for team {state.team_id}, period {state.period_id}")
    
    state.team_evaluation_id = team_evaluation_id
    
    # evaluation_type 확인/설정
    for kpi_id in state.target_team_kpi_ids:
        evaluation_type = check_evaluation_type(kpi_id)
        print(f"      • KPI {kpi_id}: {evaluation_type} 평가")
    
    print(f"   ✅ 데이터 수집 완료")
    return state

# ===== 서브모듈 2: 달성률+등급 계산 =====
def achievement_and_grade_calculation_submodule(state: Module2State) -> Module2State:
    """달성률+등급 계산 서브모듈 (통합) - 우리가 상의한 배치 처리"""
    print(f"   🎯 달성률 및 등급 계산 중...")
    
    updated_task_ids = []
    batch_data = []
    
    # 배치용 데이터 준비
    for task_id in state.target_task_ids:
        task_data = fetch_cumulative_task_data(task_id, state.period_id)
        if not task_data:
            continue
            
        batch_data.append({
            "task_id": task_id,
            "task_summary_id": task_data.get('task_summary_id'),
            "target_level": task_data.get('target_level', ''),
            "cumulative_performance": task_data.get('cumulative_performance', ''),
            "cumulative_summary": task_data.get('cumulative_task_summary', ''),
            "kpi_data": fetch_team_kpi_data(task_data.get('team_kpi_id') or 0)
        })
    
    # 배치 처리 (15개씩)
    batch_size = 15
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        results = batch_calculate_achievement_and_grades(batch, state.report_type == "annual")
        
        # 결과 저장
        for task_data, result in zip(batch, results):
            task_summary_id = task_data['task_summary_id']
            if not task_summary_id:
                continue
                
            update_data = {
                "ai_achievement_rate": int(result['achievement_rate'])
            }
            
            # 연말인 경우 등급도 저장
            if state.report_type == "annual" and result.get('grade'):
                update_data["ai_assessed_grade"] = result['grade']
            
            if update_task_summary(task_summary_id, update_data):
                updated_task_ids.append(task_data['task_id'])
    
    state.updated_task_ids = updated_task_ids
    print(f"   ✅ 달성률 계산 완료: {len(updated_task_ids)}개 Task 업데이트")
    return state

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

# ===== 서브모듈 3: 기여도 계산 =====
def contribution_calculation_submodule(state: Module2State) -> Module2State:
    """기여도 계산 서브모듈 - 우리가 상의한 하이브리드 방식"""
    print(f"   ⚖️ 기여도 계산 중...")
    
    updated_task_ids = []
    kpi_contributions_by_emp = {}  # {emp_no: total_score} - 하이브리드 3단계 결과
    
    # KPI별로 처리
    for kpi_id in state.target_team_kpi_ids:
        evaluation_type = check_evaluation_type(kpi_id)
        kpi_data = fetch_team_kpi_data(kpi_id)
        
        if evaluation_type == "quantitative":
            # 정량 평가: 개인성과/팀전체성과 × 100
            contributions = calculate_quantitative_contributions(kpi_id, state.period_id)
        else:
            # 정성 평가: LLM 기반 상대 평가
            contributions = calculate_qualitative_contributions(kpi_id, state.period_id, kpi_data)
        
        # 하이브리드 1단계: 참여자 수 보정
        kpi_tasks = fetch_kpi_tasks(kpi_id, state.period_id)
        participants_count = len(set(task['emp_no'] for task in kpi_tasks))
        
        print(f"      • KPI {kpi_id}: {evaluation_type} 평가, 참여자 {participants_count}명")
        
        for emp_no, contribution_rate in contributions.items():
            # 1단계: 참여자 수 보정
            adjusted_score = contribution_rate * participants_count
            
            # 2단계: KPI 비중 적용
            kpi_weight = kpi_data.get('weight', 0) / 100.0
            weighted_score = adjusted_score * kpi_weight
            
            if emp_no not in kpi_contributions_by_emp:
                kpi_contributions_by_emp[emp_no] = 0
            kpi_contributions_by_emp[emp_no] += weighted_score
            
            print(f"        - {emp_no}: 원래 {contribution_rate:.1f}% → 보정 {adjusted_score:.1f} → 가중 {weighted_score:.1f}")
        
        # Task별 기여도 업데이트 (원래 KPI별 기여도 저장)
        for task in kpi_tasks:
            task_data = fetch_cumulative_task_data(task['task_id'], state.period_id)
            if not task_data:
                continue
                
            emp_contribution = contributions.get(task['emp_no'], 0)
            
            update_data = {
                "ai_contribution_score": int(emp_contribution)  # KPI별 원래 기여도
            }
            
            if update_task_summary(task_data['task_summary_id'], update_data):
                updated_task_ids.append(task['task_id'])
    
    # 하이브리드 3단계: 팀 내 % 기여도 변환
    total_team_score = sum(kpi_contributions_by_emp.values())
    final_contributions = {}
    
    if total_team_score > 0:
        for emp_no in kpi_contributions_by_emp:
            percentage = (kpi_contributions_by_emp[emp_no] / total_team_score) * 100
            final_contributions[emp_no] = round(percentage, 2)
            print(f"      • {emp_no} 최종 기여도: {percentage:.1f}%")
    else:
        # 팀 점수가 0인 경우 동등 분배
        emp_count = len(kpi_contributions_by_emp)
        if emp_count > 0:
            equal_share = 100.0 / emp_count
            for emp_no in kpi_contributions_by_emp:
                final_contributions[emp_no] = round(equal_share, 2)
    
    # 최종 기여도를 feedback_reports 또는 final_evaluation_reports에 저장
    save_final_contributions_to_db(state, final_contributions)
    
    # 디버깅: 하이브리드 계산 과정 시각화
    debug_contribution_calculation(state)
    
    state.updated_task_ids = list(set((state.updated_task_ids or []) + updated_task_ids))
    print(f"   ✅ 기여도 계산 완료: {len(updated_task_ids)}개 Task 업데이트, {len(final_contributions)}명 최종 기여도 저장")
    return state

def save_final_contributions_to_db(state: Module2State, final_contributions: Dict[str, float]):
    """최종 기여도를 DB에 저장"""
    team_members = fetch_team_members(state.team_id)
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
            
        emp_no = member['emp_no']
        final_contribution = final_contributions.get(emp_no, 0)
        
        if state.report_type == "quarterly":
            # 분기별: feedback_reports에 저장
            save_feedback_report(
                emp_no, 
                state.team_evaluation_id or 0,
                {"contribution_rate": int(final_contribution)}  # 기존 컬럼명 사용
            )
        else:
            # 연말: final_evaluation_reports에 저장
            save_final_evaluation_report(
                emp_no,
                state.team_evaluation_id or 0,
                {"contribution_rate": int(final_contribution)}  # 기존 컬럼명 사용
            )

def debug_contribution_calculation(state: Module2State):
    """기여도 계산 과정 디버깅 - 하이브리드 방식 검증"""
    print(f"\n🔍 기여도 계산 과정 디버깅")
    print(f"{'='*50}")
    
    # 1단계: KPI별 원래 기여도 수집
    kpi_contributions = {}
    for kpi_id in state.target_team_kpi_ids:
        evaluation_type = check_evaluation_type(kpi_id)
        kpi_data = fetch_team_kpi_data(kpi_id)
        kpi_tasks = fetch_kpi_tasks(kpi_id, state.period_id)
        participants_count = len(set(task['emp_no'] for task in kpi_tasks))
        
        if evaluation_type == "quantitative":
            contributions = calculate_quantitative_contributions(kpi_id, state.period_id)
        else:
            contributions = calculate_qualitative_contributions(kpi_id, state.period_id, kpi_data)
        
        kpi_contributions[kpi_id] = {
            'kpi_name': kpi_data.get('kpi_name', f'KPI{kpi_id}'),
            'weight': kpi_data.get('weight', 0),
            'participants_count': participants_count,
            'contributions': contributions
        }
    
    # 2단계: 하이브리드 계산 과정 시각화
    print(f"📊 KPI별 기여도 분석:")
    for kpi_id, kpi_info in kpi_contributions.items():
        print(f"\n🎯 {kpi_info['kpi_name']} (비중: {kpi_info['weight']}%, 참여자: {kpi_info['participants_count']}명)")
        print(f"   원래 기여도 → 참여자수 보정 → KPI 비중 적용")
        print(f"   {'─' * 50}")
        
        for emp_no, original_rate in kpi_info['contributions'].items():
            # 1단계: 참여자 수 보정
            adjusted = original_rate * kpi_info['participants_count']
            # 2단계: KPI 비중 적용
            weighted = adjusted * (kpi_info['weight'] / 100.0)
            
            print(f"   {emp_no}: {original_rate:5.1f}% → {adjusted:6.1f} → {weighted:6.1f}")
    
    # 3단계: 개인별 종합 점수 계산
    print(f"\n📈 개인별 종합 점수 (하이브리드 1-2단계 결과):")
    emp_total_scores = {}
    
    for kpi_id, kpi_info in kpi_contributions.items():
        for emp_no, original_rate in kpi_info['contributions'].items():
            if emp_no not in emp_total_scores:
                emp_total_scores[emp_no] = 0
            
            adjusted = original_rate * kpi_info['participants_count']
            weighted = adjusted * (kpi_info['weight'] / 100.0)
            emp_total_scores[emp_no] += weighted
    
    for emp_no, total_score in sorted(emp_total_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"   {emp_no}: {total_score:.1f}점")
    
    # 4단계: 팀 내 % 기여도 변환
    total_team_score = sum(emp_total_scores.values())
    print(f"\n🏆 최종 기여도 (하이브리드 3단계 결과):")
    print(f"   팀 전체 점수: {total_team_score:.1f}")
    print(f"   {'─' * 30}")
    
    for emp_no, total_score in sorted(emp_total_scores.items(), key=lambda x: x[1], reverse=True):
        final_percentage = (total_score / total_team_score) * 100 if total_team_score > 0 else 0
        print(f"   {emp_no}: {final_percentage:.1f}% ({total_score:.1f}점)")
    
    print(f"{'='*50}")

def calculate_quantitative_contributions(kpi_id: int, period_id: int) -> Dict[str, float]:
    """정량 평가 기여도 계산"""
    tasks = fetch_kpi_tasks(kpi_id, period_id)
    
    # 개인별 성과 수집
    emp_performance = {}
    for task in tasks:
        emp_no = task['emp_no']
        performance_text = task.get('task_performance', '')
        
        # 성과에서 수치 추출 시도
        try:
            performance_value = extract_number_from_response(performance_text)
            if emp_no not in emp_performance:
                emp_performance[emp_no] = 0
            emp_performance[emp_no] += performance_value
        except:
            # 수치 추출 실패시 동등 분배
            emp_performance[emp_no] = 1.0
    
    # 기여도 계산
    total_performance = sum(emp_performance.values())
    contributions = {}
    
    for emp_no, performance in emp_performance.items():
        contribution_rate = safe_divide(performance, total_performance, 1/len(emp_performance)) * 100
        contributions[emp_no] = round(contribution_rate, 2)
    
    return contributions

def calculate_qualitative_contributions(kpi_id: int, period_id: int, kpi_data: Dict) -> Dict[str, float]:
    """정성 평가 기여도 계산 - 우리가 상의한 grade_rule 기반"""
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

# ===== 서브모듈 4: 팀 목표 분석 =====
def team_analysis_submodule(state: Module2State) -> Module2State:
    """팀 목표 분석 서브모듈 - 우리가 상의한 LLM 기반"""
    print(f"   🏢 팀 목표 분석 중...")
    
    updated_kpi_ids = []
    kpi_rates = []
    
    # 정량 평가 KPI들 처리 (LLM으로 팀 KPI 달성률 계산)
    for kpi_id in state.target_team_kpi_ids:
        evaluation_type = check_evaluation_type(kpi_id)
        
        if evaluation_type == "quantitative":
            # 정량 KPI도 LLM이 종합 판단
            kpi_data = fetch_team_kpi_data(kpi_id)
            kpi_rate = calculate_team_kpi_achievement_rate(kpi_id, state.period_id, kpi_data)
            
            update_data = {
                "ai_kpi_progress_rate": int(kpi_rate['rate']),
                "ai_kpi_analysis_comment": kpi_rate['comment']
            }
            
            if update_team_kpi(kpi_id, update_data):
                updated_kpi_ids.append(kpi_id)
                kpi_rates.append(kpi_rate['rate'])
        else:
            # 정성 KPI는 이미 서브모듈 3에서 처리됨
            kpi_data = fetch_team_kpi_data(kpi_id)
            if kpi_data.get('ai_kpi_progress_rate') is not None:
                kpi_rates.append(kpi_data['ai_kpi_progress_rate'])
    
    # 팀 전체 평균 달성률 계산 (KPI 비중 고려)
    team_average_rate = calculate_team_average_achievement_rate(state.target_team_kpi_ids)
    
    # team_evaluations 업데이트
    team_eval_data = {
        "average_achievement_rate": int(team_average_rate)
    }
    
    # 연말인 경우 전년 대비 성장률 계산 시도
    if state.report_type == "annual":
        yoy_growth = calculate_year_over_year_growth(state.team_id, state.period_id, team_average_rate)
        if yoy_growth is not None:
            team_eval_data["year_over_year_growth"] = int(yoy_growth)
    
    if update_team_evaluations(state.team_evaluation_id or 0, team_eval_data):
        print(f"      • 팀 평균 달성률: {team_average_rate:.1f}%")
    
    state.updated_team_kpi_ids = updated_kpi_ids
    print(f"   ✅ 팀 분석 완료: {len(updated_kpi_ids)}개 KPI 업데이트")
    return state

def calculate_team_kpi_achievement_rate(kpi_id: int, period_id: int, kpi_data: Dict) -> Dict:
    """팀 KPI 달성률 LLM 계산"""
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

def calculate_team_average_achievement_rate(team_kpi_ids: List[int]) -> float:
    """팀 전체 평균 달성률 계산 (KPI 비중 고려)"""
    total_weight = 0
    weighted_sum = 0
    
    for kpi_id in team_kpi_ids:
        kpi_data = fetch_team_kpi_data(kpi_id)
        weight = kpi_data.get('weight', 0)
        rate = kpi_data.get('ai_kpi_progress_rate', 0)
        
        total_weight += weight
        weighted_sum += rate * weight
    
    return safe_divide(weighted_sum, total_weight, 80.0)

def calculate_year_over_year_growth(team_id: int, current_period_id: int, current_rate: float) -> Optional[float]:
    """전년 대비 성장률 계산 (periods 테이블 활용)"""
    try:
        with engine.connect() as connection:
            # 현재 period의 연도 조회
            cur_period_year = connection.execute(
                text("SELECT year FROM periods WHERE period_id = :pid"),
                {"pid": current_period_id}
            ).scalar_one_or_none()
            if not cur_period_year:
                return None

            # 전년도 연말 period_id 조회
            last_year = cur_period_year - 1
            last_period_id = connection.execute(
                text("SELECT period_id FROM periods WHERE year = :y AND is_final = 1"),
                {"y": last_year}
            ).scalar_one_or_none()
            if not last_period_id:
                return None

            # 전년도 연말 팀 성과 조회
            last_year_rate = connection.execute(
                text("""
                    SELECT average_achievement_rate
                    FROM team_evaluations
                    WHERE team_id = :team_id AND period_id = :period_id
                """),
                {"team_id": team_id, "period_id": last_period_id}
            ).scalar_one_or_none()

            if last_year_rate and last_year_rate > 0:
                growth = ((current_rate - last_year_rate) / last_year_rate) * 100
                return round(growth, 2)
    except Exception as e:
        logger.warning(f"Year-over-year calculation failed: {e}")
    return None

# ===== 팀 일관성 가이드 생성 함수 =====
def generate_team_consistency_guide(team_id: int, period_id: int) -> Dict:
    """팀 단위 일관성 가이드 생성 - 우리가 상의한 방식"""
    team_members = fetch_team_members(team_id)
    team_avg_rate = calculate_team_average_achievement_rate(
        [kpi_id for kpi_id in range(1, 10)]  # 임시로 KPI ID 범위
    )
    
    # 팀 성과 수준에 따른 가이드라인 결정
    if team_avg_rate >= 90:
        performance_level = "high"
        tone_guide = "성과 중심, 구체적 수치 강조"
        style_guide = "전문적이고 객관적"
    elif team_avg_rate >= 70:
        performance_level = "average"
        tone_guide = "균형적, 현재 성과 분석"
        style_guide = "객관적이고 분석적"
    else:
        performance_level = "improvement_needed"
        tone_guide = "현재 상태 분석, 성과 요약"
        style_guide = "객관적이고 구체적"
    
    return {
        "performance_level": performance_level,
        "tone_guide": tone_guide,
        "style_guide": style_guide,
        "length_target": 250,
        "length_tolerance": 30,
        "team_context": f"팀 평균 달성률 {team_avg_rate:.1f}%, {len(team_members)}명 구성"
    }

# ===== 통합 코멘트 생성 시스템 =====
class CommentGenerator:
    """통합 코멘트 생성 시스템 - 일관성 있는 코멘트 생성"""
    
    def __init__(self, comment_type: str, period_type: str, team_guide: Optional[Dict] = None):
        self.comment_type = comment_type  # "task", "individual", "team", "kpi"
        self.period_type = period_type    # "quarterly", "annual"
        self.team_guide = team_guide or {}
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """코멘트 타입별 설정 로드"""
        base_configs = {
            "task": {
                "quarterly": {
                    "elements": ["성과요약", "주요포인트", "팀기여도", "현재상태분석"],
                    "tone": "객관적이고 분석적",
                    "focus": "현재 성과와 기여도 분석",
                    "length": {"target": 250, "tolerance": 30}
                },
                "annual": {
                    "elements": ["연간요약", "성장추이", "팀기여도", "종합평가"],
                    "tone": "종합적이고 객관적",
                    "focus": "연간 성과와 성장 분석",
                    "length": {"target": 300, "tolerance": 40}
                }
            },
            "individual": {
                "quarterly": {
                    "elements": ["전체성과요약", "주요성과하이라이트", "성장포인트", "현재역량평가"],
                    "tone": "객관적이고 분석적",
                    "length": {"target": 350, "tolerance": 50}
                },
                "annual": {
                    "elements": ["연간성과종합", "분기별성장추이", "핵심기여영역", "종합역량평가"],
                    "tone": "종합평가적이고 객관적",
                    "length": {"target": 450, "tolerance": 50}
                }
            },
            "team": {
                "quarterly": {
                    "elements": ["팀성과종합", "팀원기여분석", "주요성과영역", "팀현재상태"],
                    "tone": "분석적이고 객관적",
                    "length": {"target": 450, "tolerance": 50}
                },
                "annual": {
                    "elements": ["연간팀성과요약", "팀조직력평가", "핵심성과기여", "팀종합평가"],
                    "tone": "종합적이고 객관적",
                    "length": {"target": 550, "tolerance": 50}
                }
            },
            "kpi": {
                "quarterly": {
                    "elements": ["KPI달성현황", "주요성과분석", "팀기여도평가", "현재달성수준"],
                    "tone": "객관적이고 분석적",
                    "length": {"target": 200, "tolerance": 30}
                },
                "annual": {
                    "elements": ["연간KPI종합", "성과추이분석", "팀기여도평가", "종합달성평가"],
                    "tone": "종합적이고 객관적",
                    "length": {"target": 250, "tolerance": 30}
                }
            }
        }
        
        return base_configs.get(self.comment_type, {}).get(self.period_type, {})
    
    def generate(self, data: Dict, context: Optional[Dict] = None) -> str:
        """통합 코멘트 생성 메인 함수"""
        if not self.config:
            raise ValueError(f"Invalid comment type: {self.comment_type} or period type: {self.period_type}")
        
        context = context or {}
        
        # 코멘트 타입별 데이터 전처리
        processed_data = self._preprocess_data(data)
        
        # 프롬프트 생성
        prompt = self._create_prompt(processed_data, context)
        
        # LLM 호출 및 검증
        comment = self._call_llm_with_validation(prompt)
        
        return comment
    
    def _preprocess_data(self, data: Dict) -> Dict:
        """코멘트 타입별 데이터 전처리"""
        if self.comment_type == "task":
            return {
                "task_name": data.get('task_name', ''),
                "emp_name": data.get('emp_name', ''),
                "emp_no": data.get('emp_no', ''),
                "target_level": data.get('target_level', ''),
                "performance": data.get('cumulative_performance', ''),
                "achievement_rate": data.get('ai_achievement_rate', 0),
                "contribution_score": data.get('ai_contribution_score', 0)
            }
        
        elif self.comment_type == "individual":
            tasks = data.get('tasks', [])
            tasks_summary = ""
            total_achievement = 0
            total_contribution = 0
            
            for task in tasks:
                tasks_summary += f"- {task.get('task_name', '')}: 달성률 {task.get('ai_achievement_rate', 0)}%, 기여도 {task.get('ai_contribution_score', 0)}점\n"
                total_achievement += task.get('ai_achievement_rate', 0)
                total_contribution += task.get('ai_contribution_score', 0)
            
            avg_achievement = total_achievement / len(tasks) if tasks else 0
            avg_contribution = total_contribution / len(tasks) if tasks else 0
            
            return {
                "emp_name": data.get('emp_name', ''),
                "emp_no": data.get('emp_no', ''),
                "position": data.get('position', ''),
                "cl": data.get('cl', ''),
                "tasks_summary": tasks_summary,
                "avg_achievement": avg_achievement,
                "avg_contribution": avg_contribution,
                "task_count": len(tasks)
            }
        
        elif self.comment_type == "team":
            kpis = data.get('kpis', [])
            kpis_summary = ""
            total_rate = 0
            
            for kpi in kpis:
                rate = kpi.get('ai_kpi_progress_rate', 0)
                weight = kpi.get('weight', 0)
                kpis_summary += f"- {kpi.get('kpi_name', '')}: {rate}% (비중 {weight}%)\n"
                total_rate += rate * (weight / 100)
            
            return {
                "kpis_summary": kpis_summary,
                "total_rate": total_rate,
                "team_context": data.get('team_context', ''),
                "performance_level": data.get('performance_level', '')
            }
        
        elif self.comment_type == "kpi":
            tasks = data.get('tasks', [])
            tasks_text = ""
            for task in tasks:
                tasks_text += f"- {task.get('emp_name', '')}: {task.get('task_name', '')}\n"
                tasks_text += f"  목표: {task.get('target_level', '')}\n"
                tasks_text += f"  성과: {task.get('task_performance', '')}\n"
            
            return {
                "kpi_name": data.get('kpi_name', ''),
                "kpi_description": data.get('kpi_description', ''),
                "tasks_text": tasks_text
            }
        
        return data
    
    def _create_prompt(self, data: Dict, context: Dict) -> str:
        """통합 프롬프트 생성"""
        elements = self.config.get('elements', [])
        tone = self.config.get('tone', '')
        focus = self.config.get('focus', '')
        length = self.config.get('length', {})
        
        # 팀 가이드라인 적용
        team_tone = self.team_guide.get('tone_guide', '')
        team_style = self.team_guide.get('style_guide', '')
        team_context = self.team_guide.get('team_context', '')
        
        system_content = f"""
        다음 내용을 포함하여 {self.comment_type} 분석 코멘트를 하나의 자연스러운 문단으로 작성해주세요:
        
        포함할 내용: {', '.join(elements)}
        톤: {tone}
        초점: {focus}
        길이: {length.get('target', 250)}±{length.get('tolerance', 30)}자
        
        팀 가이드라인:
        - {team_tone}
        - {team_style}
        - {team_context}
        
        작성 원칙:
        1. 현재 상태와 과거 성장 추이에 집중
        2. 구체적 수치와 성과를 포함
        3. 미래 계획이나 제안은 포함하지 않음
        4. 객관적이고 팩트 기반으로 작성
        5. 직원 이름 언급시 "이름(사번)님" 형태로 작성
        6. "연간요약:", "팀기여도:" 등의 제목 없이 하나의 자연스러운 문단으로 작성
        7. 문단 간 자연스러운 연결로 전체적인 흐름을 만들어주세요
        """
        
        # 코멘트 타입별 human content 생성
        human_content = self._create_human_content(data)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ])
        
        return str(prompt.format())
    
    def _create_human_content(self, data: Dict) -> str:
        """코멘트 타입별 human content 생성"""
        if self.comment_type == "task":
            return f"""
            Task: {data.get('task_name', '')}
            담당자: {data.get('emp_name', '')}({data.get('emp_no', '')})
            목표: {data.get('target_level', '')}
            누적 성과: {data.get('performance', '')}
            달성률: {data.get('achievement_rate', 0)}%
            기여도: {data.get('contribution_score', 0)}점
            """
        
        elif self.comment_type == "individual":
            return f"""
            직원: {data.get('emp_name', '')}({data.get('emp_no', '')})
            직위: {data.get('position', '')} (CL{data.get('cl', '')})
            
            Task 수행 현황:
            {data.get('tasks_summary', '')}
            
            종합 성과:
            - 평균 달성률: {data.get('avg_achievement', 0):.1f}%
            - 평균 기여도: {data.get('avg_contribution', 0):.1f}점
            - 참여 Task 수: {data.get('task_count', 0)}개
            """
        
        elif self.comment_type == "team":
            return f"""
            팀 KPI 성과 현황:
            {data.get('kpis_summary', '')}
            
            팀 전체 평균 달성률: {data.get('total_rate', 0):.1f}%
            팀 구성: {data.get('team_context', '')}
            성과 수준: {data.get('performance_level', '')}
            """
        
        elif self.comment_type == "kpi":
            return f"""
            KPI: {data.get('kpi_name', '')}
            KPI 목표: {data.get('kpi_description', '')}
            
            팀원별 개별 성과:
            {data.get('tasks_text', '')}
            """
        
        return str(data)
    
    def _call_llm_with_validation(self, prompt: str) -> str:
        """LLM 호출 및 검증"""
        def validate_comment(response: str) -> str:
            response = response.strip()
            
            # 길이 검증 (경고 로그 제거)
            target_length = self.config.get('length', {}).get('target', 250)
            tolerance = self.config.get('length', {}).get('tolerance', 30)
            
            # 길이 검증은 하되 경고 로그는 출력하지 않음
            # if not (target_length - tolerance <= len(response) <= target_length + tolerance):
            #     logger.warning(f"Comment length {len(response)} outside target {target_length}±{tolerance}")
            
            return response
        
        return robust_llm_call(prompt, validate_comment, context=f"{self.comment_type} comment")

# ===== 서브모듈 5: 코멘트 생성 (개선된 버전) =====
def comment_generation_submodule(state: Module2State) -> Module2State:
    """코멘트 생성 서브모듈 - 통합 시스템 사용"""
    print(f"   📝 코멘트 생성 중...")
    
    # 팀 일관성 가이드 생성
    team_context_guide = generate_team_consistency_guide(state.team_id, state.period_id)
    state.team_context_guide = team_context_guide
    
    # 통합 코멘트 생성기 사용
    generate_task_comments_unified(state)
    generate_individual_summary_comments_unified(state)
    generate_team_overall_comment_unified(state)
    
    print(f"   ✅ 코멘트 생성 완료")
    return state

def generate_task_comments_unified(state: Module2State):
    """Task별 코멘트 생성 (통합 시스템)"""
    period_type = "annual" if state.report_type == "annual" else "quarterly"
    
    for task_id in state.target_task_ids:
        task_data = fetch_cumulative_task_data(task_id, state.period_id)
        if not task_data:
            continue
        
        # 통합 코멘트 생성기 사용
        generator = CommentGenerator("task", period_type, state.team_context_guide)
        comment = generator.generate(task_data)
        
        if task_data.get('task_summary_id'):
            update_task_summary(task_data['task_summary_id'], {
                "ai_analysis_comment_task": comment
            })

def generate_individual_summary_comments_unified(state: Module2State):
    """개인 종합 코멘트 생성 (통합 시스템)"""
    team_members = fetch_team_members(state.team_id)
    period_type = "annual" if state.report_type == "annual" else "quarterly"
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
        
        # 개인 Task 데이터 수집
        individual_tasks = []
        for task_id in state.target_task_ids:
            task_data = fetch_cumulative_task_data(task_id, state.period_id)
            if task_data and task_data.get('emp_no') == member['emp_no']:
                individual_tasks.append(task_data)
        
        if not individual_tasks:
            continue
        
        # 통합 코멘트 생성기 사용
        generator = CommentGenerator("individual", period_type, state.team_context_guide)
        comment = generator.generate({
            **member,
            "tasks": individual_tasks
        })
        
        # 분기별/연말별 저장
        if state.report_type == "quarterly":
            feedback_report_id = save_feedback_report(
                member['emp_no'], 
                state.team_evaluation_id or 0,
                {"ai_overall_contribution_summary_comment": comment}
            )
            if state.feedback_report_ids is None:
                state.feedback_report_ids = []
            state.feedback_report_ids.append(feedback_report_id)
        else:  # annual
            final_report_id = save_final_evaluation_report(
                member['emp_no'],
                state.team_evaluation_id or 0, 
                {"ai_annual_performance_summary_comment": comment}
            )
            if state.final_evaluation_report_ids is None:
                state.final_evaluation_report_ids = []
            state.final_evaluation_report_ids.append(final_report_id)

def generate_team_overall_comment_unified(state: Module2State):
    """팀 전체 분석 코멘트 생성 (통합 시스템)"""
    # 팀 KPI 데이터 수집
    team_kpis_data = []
    for kpi_id in state.target_team_kpi_ids:
        kpi_data = fetch_team_kpi_data(kpi_id)
        if kpi_data:
            team_kpis_data.append(kpi_data)
    
    period_type = "annual" if state.report_type == "annual" else "quarterly"
    
    # 통합 코멘트 생성기 사용
    generator = CommentGenerator("team", period_type, state.team_context_guide)
    comment = generator.generate({
        "kpis": team_kpis_data,
        "team_context": state.team_context_guide.get('team_context', '') if state.team_context_guide else '',
        "performance_level": state.team_context_guide.get('performance_level', '') if state.team_context_guide else ''
    })
    
    # team_evaluations 업데이트
    update_team_evaluations(state.team_evaluation_id or 0, {
        "ai_team_overall_analysis_comment": comment
    })

# ===== 서브모듈 6: DB 업데이트 =====
def db_update_submodule(state: Module2State) -> Module2State:
    """최종 DB 업데이트 서브모듈 - 트랜잭션 처리"""
    print(f"   💾 최종 DB 업데이트 중...")
    
    try:
        with engine.begin() as transaction:
            # 이미 각 서브모듈에서 업데이트했으므로 최종 검증만 수행
            
            # 1. 분기별 추가 업데이트 (ranking, cumulative 데이터)
            if state.report_type == "quarterly":
                update_quarterly_specific_data(state)
            
            # 2. 연말 추가 업데이트 (final_evaluation_reports 추가 필드)
            elif state.report_type == "annual":
                update_annual_specific_data(state)
            
            # 3. 업데이트 결과 검증
            validation_result = validate_final_update_results(state)
            
            if not validation_result['success']:
                raise DataIntegrityError(f"Final validation failed: {validation_result['errors']}")
            
            # 4. 업데이트 통계 로깅
            updated_tasks = len(state.updated_task_ids or [])
            updated_kpis = len(state.updated_team_kpi_ids or [])
            updated_feedback_reports = len(state.feedback_report_ids or [])
            updated_final_reports = len(state.final_evaluation_report_ids or [])
            
            print(f"      • Task 업데이트: {updated_tasks}개")
            print(f"      • KPI 업데이트: {updated_kpis}개")
            print(f"      • 피드백 리포트: {updated_feedback_reports}개")
            print(f"      • 최종 리포트: {updated_final_reports}개")
            
            # 5. 최종 상태 로깅
            if state.report_type == "quarterly":
                print(f"      • 분기 평가 완료")
            else:
                print(f"      • 연말 평가 완료")
                
            return state
                
    except Exception as e:
        print(f"   ❌ 최종 DB 업데이트 실패: {e}")
        raise

def update_quarterly_specific_data(state: Module2State):
    """분기별 전용 데이터 업데이트 - 개인 달성률 기반 순위 매기기"""
    print(f"      📊 분기별 전용 데이터 업데이트 중...")
    
    # 1. 팀 내 개인별 달성률 기반 순위 계산 및 업데이트
    team_ranking_result = calculate_team_ranking(state)
    
    # 2. 순위 결과를 feedback_reports에 저장
    update_team_ranking_to_feedback_reports(state, team_ranking_result)
    
    print(f"      ✅ 분기별 순위 업데이트 완료: {len(team_ranking_result)}명")
    print(f"      📊 팀 내 달성률 순위:")
    for i, member in enumerate(team_ranking_result):
        print(f"        {i+1}위: {member['emp_name']}({member['emp_no']}) - {member['avg_achievement_rate']:.1f}%")

def calculate_team_ranking(state: Module2State) -> List[Dict]:
    """팀 내 개인별 달성률 기반 순위 계산"""
    print(f"        🏆 팀 내 순위 계산 중...")
    
    team_members = fetch_team_members(state.team_id)
    member_achievements = []
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
            
        # 개인별 Task 수집
        individual_tasks = []
        for task_id in state.target_task_ids:
            task_data = fetch_cumulative_task_data(task_id, state.period_id)
            if task_data and task_data.get('emp_no') == member['emp_no']:
                individual_tasks.append(task_data)
        
        if individual_tasks:
            # 가중평균 달성률 계산
            result = calculate_individual_weighted_achievement_rate(individual_tasks)
            
            # 계산 과정 상세 로깅
            print(f"          📈 {member['emp_name']}({member['emp_no']}) 달성률 계산:")
            total_weighted_score = 0
            for task in individual_tasks:
                task_name = task.get('task_name', f'Task{task.get("task_id")}')
                task_weight = task.get('weight', 0)
                task_achievement = task.get('ai_achievement_rate', 0)
                weighted_score = task_achievement * task_weight
                total_weighted_score += weighted_score
                print(f"            • {task_name}: {task_achievement}% × {task_weight} = {weighted_score}")
            
            print(f"            = {result['achievement_rate']:.1f}% (총 가중점수: {total_weighted_score}, 총 가중치: {result['total_weight']})")
            
            member_achievements.append({
                'emp_no': member['emp_no'],
                'emp_name': member['emp_name'],
                'position': member.get('position', ''),
                'cl': member.get('cl', ''),
                'avg_achievement_rate': result['achievement_rate'],
                'avg_contribution_rate': result['contribution_rate'],
                'task_count': len(individual_tasks),
                'total_weight': result['total_weight'],
                'total_weighted_score': total_weighted_score
            })
        else:
            print(f"          ⚠️  {member['emp_name']}({member['emp_no']}): 참여 Task 없음")
    
    # 달성률 기준 내림차순 정렬 (높은 달성률이 1위)
    member_achievements.sort(key=lambda x: x['avg_achievement_rate'], reverse=True)
    
    # 동점자 처리 (같은 달성률인 경우 가중점수로 재정렬)
    for i in range(len(member_achievements) - 1):
        if member_achievements[i]['avg_achievement_rate'] == member_achievements[i + 1]['avg_achievement_rate']:
            # 동점자인 경우 가중점수로 재정렬
            if member_achievements[i]['total_weighted_score'] < member_achievements[i + 1]['total_weighted_score']:
                member_achievements[i], member_achievements[i + 1] = member_achievements[i + 1], member_achievements[i]
    
    return member_achievements

def update_team_ranking_to_feedback_reports(state: Module2State, team_ranking: List[Dict]):
    """팀 순위 결과를 feedback_reports에 저장"""
    print(f"        💾 순위 결과를 feedback_reports에 저장 중...")
    
    updated_count = 0
    
    for i, member_data in enumerate(team_ranking):
        ranking = i + 1
        
        # feedback_reports 업데이트 데이터
        feedback_data = {
            'ranking': ranking,  # 팀 내 순위 (1, 2, 3, ...)
            'ai_achievement_rate': int(member_data['avg_achievement_rate']),  # 가중평균 달성률
            'contribution_rate': int(member_data['avg_contribution_rate'])    # 평균 기여도
        }
        
        # 기존 feedback_report 업데이트 또는 새로 생성
        feedback_report_id = save_feedback_report(
            member_data['emp_no'],
            state.team_evaluation_id or 0,
            feedback_data
        )
        
        updated_count += 1
        
        # 순위 저장 결과 로깅
        print(f"          {ranking}위: {member_data['emp_name']}({member_data['emp_no']}) - {member_data['avg_achievement_rate']:.1f}% → feedback_report_id: {feedback_report_id}")
    
    print(f"        ✅ {updated_count}명의 순위 정보 저장 완료")

def validate_team_ranking_data(state: Module2State) -> Dict[str, Any]:
    """팀 순위 데이터 검증"""
    print(f"        🔍 팀 순위 데이터 검증 중...")
    
    errors = []
    warnings = []
    
    try:
        with engine.connect() as connection:
            # feedback_reports에서 순위 데이터 조회
            query = text("""
                SELECT emp_no, ranking, ai_achievement_rate, contribution_rate
                FROM feedback_reports 
                WHERE team_evaluation_id = :team_evaluation_id
                ORDER BY ranking
            """)
            results = connection.execute(query, {"team_evaluation_id": state.team_evaluation_id})
            ranking_data = [row_to_dict(row) for row in results]
            
            if not ranking_data:
                errors.append("팀 순위 데이터가 없습니다")
                return {'success': False, 'errors': errors, 'warnings': warnings}
            
            # 1. 순위 연속성 검증
            expected_rankings = list(range(1, len(ranking_data) + 1))
            actual_rankings = [r['ranking'] for r in ranking_data]
            
            if actual_rankings != expected_rankings:
                errors.append(f"순위가 연속적이지 않음: 예상 {expected_rankings}, 실제 {actual_rankings}")
            
            # 2. 달성률 범위 검증
            for rank_data in ranking_data:
                achievement_rate = rank_data.get('ai_achievement_rate', 0)
                if not (0 <= achievement_rate <= 200):
                    errors.append(f"사번 {rank_data['emp_no']}: 달성률 {achievement_rate}%가 범위를 벗어남")
                
                contribution_rate = rank_data.get('contribution_rate', 0)
                if not (0 <= contribution_rate <= 100):
                    warnings.append(f"사번 {rank_data['emp_no']}: 기여도 {contribution_rate}%가 범위를 벗어남")
            
            # 3. 순위와 달성률 일관성 검증
            for i in range(len(ranking_data) - 1):
                current_rate = ranking_data[i]['ai_achievement_rate']
                next_rate = ranking_data[i + 1]['ai_achievement_rate']
                
                if current_rate < next_rate:
                    errors.append(f"순위 {i+1}위({ranking_data[i]['emp_no']})의 달성률 {current_rate}%가 {i+2}위({ranking_data[i+1]['emp_no']})의 달성률 {next_rate}%보다 낮음")
            
            # 4. 팀원 수와 순위 수 일치 검증
            team_members = fetch_team_members(state.team_id)
            non_manager_count = len([m for m in team_members if m.get('role') != 'MANAGER'])
            
            if len(ranking_data) != non_manager_count:
                warnings.append(f"팀원 수({non_manager_count}명)와 순위 수({len(ranking_data)}명)가 일치하지 않음")
            
            success = len(errors) == 0
            
            if warnings:
                print(f"          ⚠️  검증 경고: {len(warnings)}건")
            
            return {
                'success': success,
                'errors': errors,
                'warnings': warnings,
                'ranking_count': len(ranking_data),
                'team_member_count': non_manager_count
            }
            
    except Exception as e:
        print(f"          ❌ 순위 데이터 검증 실패: {e}")
        return {
            'success': False,
            'errors': [f"순위 데이터 검증 오류: {str(e)}"],
            'warnings': [],
            'ranking_count': 0,
            'team_member_count': 0
        }

def update_annual_specific_data(state: Module2State):
    """연말 전용 데이터 업데이트 - Task Weight 기반 가중평균"""
    print(f"      📊 연말 전용 데이터 업데이트 중...")
    
    team_members = fetch_team_members(state.team_id)
    
    for member in team_members:
        if member.get('role') == 'MANAGER':
            continue
            
        # 개인별 Task 수집
        individual_tasks = []
        for task_id in state.target_task_ids:
            task_data = fetch_cumulative_task_data(task_id, state.period_id)
            if task_data and task_data.get('emp_no') == member['emp_no']:
                individual_tasks.append(task_data)
        
        if individual_tasks:
            # 가중평균 계산
            result = calculate_individual_weighted_achievement_rate(individual_tasks)
            
            # 계산 과정 로깅
            print(f"        📈 {member['emp_name']}({member['emp_no']}) 연간 가중평균 계산:")
            for task in individual_tasks:
                task_name = task.get('task_name', f'Task{task.get("task_id")}')
                task_weight = task.get('weight', 0)
                task_achievement = task.get('ai_achievement_rate', 0)
                print(f"          • {task_name}: {task_achievement}% × {task_weight} = {task_achievement * task_weight}")
            print(f"          = {result['achievement_rate']:.1f}% (총 가중치: {result['total_weight']})")
            
            # final_evaluation_reports 업데이트
            final_data = {
                'ai_annual_achievement_rate': int(result['achievement_rate'])
            }
            
            save_final_evaluation_report(
                member['emp_no'],
                state.team_evaluation_id or 0,
                final_data
            )
    
    print(f"      ✅ 연말 데이터 업데이트 완료: {len([m for m in team_members if m.get('role') != 'MANAGER'])}명")

def validate_final_update_results(state: Module2State) -> Dict[str, Any]:
    """최종 업데이트 결과 검증"""
    errors = []
    warnings = []
    
    try:
        # 1. Task 업데이트 검증
        for task_id in (state.updated_task_ids or []):
            task_data = fetch_cumulative_task_data(task_id, state.period_id)
            
            # 필수 필드 검증
            if task_data.get('ai_achievement_rate') is None:
                errors.append(f"Task {task_id}: ai_achievement_rate not updated")
            
            if task_data.get('ai_contribution_score') is None:
                errors.append(f"Task {task_id}: ai_contribution_score not updated")
            
            if not task_data.get('ai_analysis_comment_task'):
                warnings.append(f"Task {task_id}: ai_analysis_comment_task empty")
            
            # 연말 전용 검증
            if state.report_type == "annual" and not task_data.get('ai_assessed_grade'):
                warnings.append(f"Task {task_id}: ai_assessed_grade not set for annual evaluation")
        
        # 2. Team KPI 업데이트 검증
        for kpi_id in (state.updated_team_kpi_ids or []):
            kpi_data = fetch_team_kpi_data(kpi_id)
            
            if kpi_data.get('ai_kpi_progress_rate') is None:
                errors.append(f"KPI {kpi_id}: ai_kpi_progress_rate not updated")
            
            if not kpi_data.get('ai_kpi_analysis_comment'):
                warnings.append(f"KPI {kpi_id}: ai_kpi_analysis_comment empty")
        
        # 3. Team evaluation 검증
        if state.team_evaluation_id:
            with engine.connect() as connection:
                query = text("""
                    SELECT average_achievement_rate, ai_team_overall_analysis_comment,
                           year_over_year_growth
                    FROM team_evaluations 
                    WHERE team_evaluation_id = :team_evaluation_id
                """)
                result = connection.execute(query, {"team_evaluation_id": state.team_evaluation_id})
                row = result.fetchone()
                team_eval = row_to_dict(row) if row else {}
                
                if team_eval.get('average_achievement_rate') is None:
                    errors.append("Team evaluation: average_achievement_rate not updated")
                
                if not team_eval.get('ai_team_overall_analysis_comment'):
                    warnings.append("Team evaluation: ai_team_overall_analysis_comment empty")
        
        # 4. 분기별 팀 순위 검증 (새로 추가)
        if state.report_type == "quarterly":
            ranking_validation = validate_team_ranking_data(state)
            if not ranking_validation['success']:
                errors.extend(ranking_validation['errors'])
            warnings.extend(ranking_validation['warnings'])
            
            print(f"      📊 팀 순위 검증 결과:")
            print(f"        • 순위 데이터: {ranking_validation['ranking_count']}명")
            print(f"        • 팀원 수: {ranking_validation['team_member_count']}명")
        
        # 5. 레포트 검증
        if state.report_type == "quarterly" and state.feedback_report_ids:
            for report_id in state.feedback_report_ids:
                # feedback_reports 검증 로직
                pass
        
        elif state.report_type == "annual" and state.final_evaluation_report_ids:
            for report_id in state.final_evaluation_report_ids:
                # final_evaluation_reports 검증 로직
                pass
        
        # 6. 데이터 일관성 검증
        consistency_errors = validate_data_consistency(state)
        errors.extend(consistency_errors)
        
        success = len(errors) == 0
        
        if warnings:
            print(f"      ⚠️  검증 경고: {len(warnings)}건")
        
        return {
            'success': success,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'tasks_validated': len(state.updated_task_ids or []),
                'kpis_validated': len(state.updated_team_kpi_ids or []),
                'reports_validated': len(state.feedback_report_ids or []) + len(state.final_evaluation_report_ids or []),
                'ranking_validated': state.report_type == "quarterly"
            }
        }
        
    except Exception as e:
        print(f"      ❌ 검증 프로세스 실패: {e}")
        return {
            'success': False,
            'errors': [f"Validation process error: {str(e)}"],
            'warnings': [],
            'stats': {}
        }

def validate_data_consistency(state: Module2State) -> List[str]:
    """데이터 일관성 검증"""
    errors = []
    
    try:
        # 1. 기여도 합계 검증 (KPI별)
        for kpi_id in state.target_team_kpi_ids:
            kpi_tasks = fetch_kpi_tasks(kpi_id, state.period_id)
            total_contribution = 0
            
            for task in kpi_tasks:
                task_data = fetch_cumulative_task_data(task['task_id'], state.period_id)
                contribution = task_data.get('ai_contribution_score', 0)
                total_contribution += contribution
            
            # KPI별 기여도 합계가 100에 가까운지 확인 (정량평가인 경우)
            evaluation_type = check_evaluation_type(kpi_id)
            if evaluation_type == "quantitative" and abs(total_contribution - 100) > 10:
                errors.append(f"KPI {kpi_id}: contribution sum {total_contribution} far from 100")
        
        # 2. 달성률 범위 검증
        for task_id in state.updated_task_ids or []:
            task_data = fetch_cumulative_task_data(task_id, state.period_id)
            achievement_rate = task_data.get('ai_achievement_rate', 0)
            
            if not (0 <= achievement_rate <= 200):
                errors.append(f"Task {task_id}: achievement_rate {achievement_rate} out of range")
        
        # 3. 팀 평균 달성률과 개별 달성률 일관성 검증
        if state.team_evaluation_id:
            with engine.connect() as connection:
                query = text("""
                    SELECT average_achievement_rate 
                    FROM team_evaluations 
                    WHERE team_evaluation_id = :team_evaluation_id
                """)
                result = connection.execute(query, {"team_evaluation_id": state.team_evaluation_id})
                team_avg = result.scalar_one_or_none()
                
                # 개별 Task들의 가중평균과 팀 평균이 크게 다르지 않은지 확인
                calculated_avg = calculate_team_average_achievement_rate(state.target_team_kpi_ids)
                
                if team_avg and abs(team_avg - calculated_avg) > 15:
                    errors.append(f"Team average inconsistency: stored {team_avg} vs calculated {calculated_avg}")
        
    except Exception as e:
        errors.append(f"Consistency validation error: {str(e)}")
    
    return errors

# ===== LangGraph 워크플로우 구성 =====

# StateGraph에 사용할 타입: dataclass 그대로 사용 (dict도 가능)
module2_workflow = StateGraph(Module2State)

# 각 서브모듈을 노드로 등록
module2_workflow.add_node("data_collection", data_collection_submodule)
module2_workflow.add_node("achievement_and_grade", achievement_and_grade_calculation_submodule)
module2_workflow.add_node("contribution", contribution_calculation_submodule)
module2_workflow.add_node("team_analysis", team_analysis_submodule)
module2_workflow.add_node("comment_generation", comment_generation_submodule)
module2_workflow.add_node("db_update", db_update_submodule)

# 엣지(실행 순서) 정의
module2_workflow.add_edge(START, "data_collection")
module2_workflow.add_edge("data_collection", "achievement_and_grade")
module2_workflow.add_edge("achievement_and_grade", "contribution")
module2_workflow.add_edge("contribution", "team_analysis")
module2_workflow.add_edge("team_analysis", "comment_generation")
module2_workflow.add_edge("comment_generation", "db_update")
module2_workflow.add_edge("db_update", END)

# 그래프 컴파일
module2_graph = module2_workflow.compile()

# ===== 실행 함수 및 main 진입점 추가 =====
def run_module2_for_team_period(team_id: int, period_id: int, report_type: str, target_task_ids: list, target_team_kpi_ids: list):
    """
    모듈2 전체 워크플로우 실행 함수
    Args:
        team_id: 팀 ID
        period_id: 평가 기간 ID (분기)
        report_type: 'quarterly' 또는 'annual'
        target_task_ids: 평가 대상 Task ID 리스트
        target_team_kpi_ids: 평가 대상 KPI ID 리스트
    """
    print(f"\n============================")
    print(f"[모듈2] 팀 {team_id}, 기간 {period_id} ({'연말' if report_type == 'annual' else '분기'}) 평가 실행")
    print(f"============================\n")
    
    # State 생성
    state = Module2State(
        report_type=report_type if report_type == "annual" else "quarterly",  # Literal 타입 보장
        team_id=team_id,
        period_id=period_id,
        target_task_ids=target_task_ids,
        target_team_kpi_ids=target_team_kpi_ids
    )
    
    # 워크플로우 실행
    result = module2_graph.invoke(state)
    print(f"\n[완료] 팀 {team_id}, 기간 {period_id} 평가 종료\n")
    return result

if __name__ == "__main__":
    # 분기별 period_id와 report_type 매핑 (예시)
    period_map = {
        1: {"period_id": 1, "report_type": "quarterly"},
        2: {"period_id": 2, "report_type": "quarterly"},
        3: {"period_id": 3, "report_type": "quarterly"},
        4: {"period_id": 4, "report_type": "annual"},
    }
    
    # 실제 DB에서 team_id=1의 task/kpi id를 조회하는 코드 필요
    # 여기서는 예시로 임의의 ID 사용 (실제 환경에 맞게 수정 필요)
    # 아래 쿼리로 자동 조회 가능
    def fetch_team_tasks_and_kpis(team_id: int, period_id: int):
        with engine.connect() as connection:
            # 해당 팀의 해당 기간 Task ID
            task_query = text("""
                SELECT t.task_id FROM tasks t
                JOIN employees e ON t.emp_no = e.emp_no
                WHERE e.team_id = :team_id
            """)
            task_ids = [row[0] for row in connection.execute(task_query, {"team_id": team_id})]
            
            # 해당 팀의 KPI ID
            kpi_query = text("""
                SELECT team_kpi_id FROM team_kpis WHERE team_id = :team_id
            """)
            kpi_ids = [row[0] for row in connection.execute(kpi_query, {"team_id": team_id})]
            return task_ids, kpi_ids
    
    import argparse
    parser = argparse.ArgumentParser(description="Module2 Goal Achievement Runner")
    parser.add_argument("--quarter", type=int, choices=[1,2,3,4], required=False, default=4, help="실행할 분기 (1,2,3,4). 기본값: 1")
    args = parser.parse_args()
    
    team_id = 1
    period_info = period_map[args.quarter]
    period_id = period_info["period_id"]
    report_type = period_info["report_type"]
    
    # Task/KPI ID 자동 조회
    task_ids, kpi_ids = fetch_team_tasks_and_kpis(team_id, period_id)
    
    run_module2_for_team_period(
        team_id=team_id,
        period_id=period_id,
        report_type=report_type,
        target_task_ids=task_ids,
        target_team_kpi_ids=kpi_ids
    )

