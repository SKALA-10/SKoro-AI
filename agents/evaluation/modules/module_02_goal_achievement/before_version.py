# ai-performance-management-system/shared/tools/py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from typing import Optional, List, Dict, Any

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

from config.settings import DatabaseConfig

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# --- 도우미 함수: SQLAlchemy Row 객체를 딕셔너리로 변환 ---
def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환합니다."""
    if row is None:
        return {}
    return row._asdict() # ._asdict() 사용


# --- 데이터 조회 함수 (`SELECT` 쿼리 구현) ---
def fetch_task_summary_by_id(task_summary_id: int) -> Optional[Dict]:
    """
    `task_summary_Id`로 `task_summaries` 및 관련 `tasks`, `employees` 테이블에서 상세 Task Summary 데이터를 조회합니다.
    """
    with engine.connect() as connection:
        query = text(f"""
            SELECT ts.*, t.task_name, t.target_level, t.task_performance, 
                    t.emp_no, t.team_kpi_id, e.emp_name, -- 수정: e.emp_name 추가
                    t.ai_contribution_score, t.ai_achievement_rate, t.ai_assessed_grade, t.ai_analysis_comment_task
            FROM task_summaries ts
            JOIN tasks t ON ts.task_id = t.task_id
            JOIN employees e ON t.emp_no = e.emp_no 
            WHERE ts.task_summary_Id = :task_summary_id
        """)
        result = connection.execute(query, {"task_summary_id": task_summary_id}).fetchone()
        return row_to_dict(result) if result else None


def fetch_kpi_data_by_id(team_kpi_id: int) -> Optional[Dict]:
    """
    `team_kpi_id`로 `team_kpis` 테이블에서 상세 KPI 데이터를 조회합니다.
    """
    with engine.connect() as connection:
        query = text("SELECT * FROM team_kpis WHERE team_kpi_id = :team_kpi_id")
        result = connection.execute(query, {"team_kpi_id": team_kpi_id}).fetchone()
        return row_to_dict(result) if result else None
    

def fetch_tasks_for_kpi(team_kpi_id: int, period_id: int) -> List[Dict]:
    """
    특정 KPI에 속한 Task들을 조회합니다.
    """
    with engine.connect() as connection:
        # 먼저 디버깅용으로 각 테이블 데이터 확인
        
        query = text("""
            SELECT t.task_id, t.task_name, t.emp_no, ts.task_summary, ts.task_summary_Id, 
                    e.emp_name, t.ai_contribution_score, t.ai_achievement_rate, 
                    t.ai_assessed_grade, t.ai_analysis_comment_task
            FROM tasks t
            JOIN task_summaries ts ON t.task_id = ts.task_id
            JOIN employees e ON t.emp_no = e.emp_no
            WHERE t.team_kpi_id = :team_kpi_id AND ts.period_id = :period_id
        """)
        
        results = connection.execute(query, {"team_kpi_id": team_kpi_id, "period_id": period_id}).fetchall()
        result_dicts = [row_to_dict(row) for row in results]
        
        return result_dicts


def fetch_grade_definitions_from_db() -> Dict:
    """
    `grades` 테이블에서 LLM이 참고할 등급 정의 (`grade_s`, `grade_a` 등 컬럼의 텍스트)를 조회합니다.
    """
    with engine.connect() as connection:
        query = text("SELECT grade_id, grade_s, grade_a, grade_b, grade_c, grade_d, grade_rule FROM grades")
        results = connection.execute(query).fetchall()
        
        if results:
            first_row = row_to_dict(results[0])
            return {
                "S": first_row.get("grade_s", "목표를 초과 달성"),
                "A": first_row.get("grade_a", "목표를 완벽하게 달성하며 높은 품질의 결과물 제공"),
                "B": first_row.get("grade_b", "목표 수준을 정확히 달성"),
                "C": first_row.get("grade_c", "목표에 미달했으나 일부 성과 달성"),
                "D": first_row.get("grade_d", "목표 달성 미흡")
            }
        return {}


def fetch_team_evaluation_id_by_team_and_period(team_id: int, period_id: int) -> Optional[int]:
    """
    `team_evaluations` 테이블에서 `team_id`와 `period_id`로 `team_evaluation_id`를 조회합니다.
    Spring에서 이 레코드를 미리 생성하고 ID를 관리한다고 가정합니다.
    """
    with engine.connect() as connection:
        query = text("SELECT team_evaluation_id FROM team_evaluations WHERE team_id = :team_id AND period_id = :period_id")
        result = connection.execute(query, {"team_id": team_id, "period_id": period_id}).scalar_one_or_none()
        return result

def fetch_temp_evaluation_id_by_emp_and_period(emp_no: str, period_id: int) -> Optional[int]:
    """
    `temp_evaluations` 테이블에서 `TempEvaluation_empNo`와 `period_id`로 `TempEvaluation_id`를 조회합니다.
    (ERD상 `temp_evaluations`에 `period_id`가 직접 없고 `team_evaluation_id`를 통해 간접 연결되므로, Spring의 테이블 구조에 맞춰 쿼리 수정)
    """
    with engine.connect() as connection:
        query = text("""
            SELECT te.TempEvaluation_id
            FROM temp_evaluations te
            JOIN team_evaluations t_eval ON te.team_evaluation_id = t_eval.team_evaluation_id
            WHERE te.TempEvaluation_empNo = :emp_no AND t_eval.period_id = :period_id
        """)
        result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).scalar_one_or_none()
        return result

def fetch_employees_by_team_id(team_id: int) -> List[Dict]:
    """
    특정 팀에 속한 모든 직원의 emp_no, emp_name, role을 조회합니다.
    """
    with engine.connect() as connection:
        query = text("SELECT emp_no, emp_name, role FROM employees WHERE team_id = :team_id")
        results = connection.execute(query, {"team_id": team_id}).fetchall()
        return [row_to_dict(row) for row in results]


def update_task_ai_results_in_db(task_id: int, update_data: Dict) -> bool:
    """
    `task_id`에 해당하는 `tasks` 테이블 레코드의 AI 컬럼들을 업데이트합니다.
    """
    with engine.connect() as connection:
        set_clauses = [f"`{k}` = :{k}" for k in update_data.keys()]
        query = text(f"UPDATE `tasks` SET {', '.join(set_clauses)} WHERE `task_id` = :task_id")
        
        params = {**update_data, "task_id": task_id}
        result = connection.execute(query, params)
        connection.commit()
        return result.rowcount > 0

def update_team_kpi_ai_results_in_db(team_kpi_id: int, update_data: Dict) -> bool:
    """
    `team_kpi_id`에 해당하는 `team_kpis` 테이블 레코드의 AI 컬럼들을 업데이트합니다.
    """
    with engine.connect() as connection:
        set_clauses = [f"`{k}` = :{k}" for k in update_data.keys()]
        query = text(f"UPDATE `team_kpis` SET {', '.join(set_clauses)} WHERE `team_kpi_id` = :team_kpi_id")
        
        params = {**update_data, "team_kpi_id": team_kpi_id}
        result = connection.execute(query, params)
        connection.commit()
        return result.rowcount > 0
    

def save_feedback_report_module2_results_to_db(emp_no: str, team_evaluation_id: int, results: Dict) -> int: 
    """
    `feedback_reports` 테이블에 모듈 2 관련 AI 결과를 삽입하거나 업데이트합니다.
    `emp_no`와 `team_evaluation_id`가 중복되면 업데이트를 수행합니다.
    """
    with engine.connect() as connection:
        # INSERT ... ON DUPLICATE KEY UPDATE 사용
        # emp_no와 team_evaluation_id는 UNIQUE 키(또는 복합 PK)로 설정되어 있어야 합니다.
        cols_for_insert = ["emp_no", "team_evaluation_id"] + list(results.keys())
        values_placeholder = ", ".join([f":{col}" for col in cols_for_insert])
        cols_str = ", ".join([f"`{col}`" for col in cols_for_insert])

        # ON DUPLICATE KEY UPDATE 절에 사용할 컬럼들 (AI 결과 컬럼만 업데이트)
        on_duplicate_set_clauses = [f"`{k}` = VALUES(`{k}`)" for k in results.keys()]
        
        query = text(f"""
            INSERT INTO `feedback_reports` ({cols_str}) VALUES ({values_placeholder})
            ON DUPLICATE KEY UPDATE {", ".join(on_duplicate_set_clauses)}
        """)
        
        params = {"emp_no": emp_no, "team_evaluation_id": team_evaluation_id, **results} 
        
        connection.execute(query, params)
        connection.commit()
        
        # 삽입 또는 업데이트된 레코드의 ID를 다시 조회 (ON DUPLICATE KEY UPDATE의 LAST_INSERT_ID()는 복잡)
        inserted_or_updated_id_query = text("""
            SELECT feedback_report_id FROM `feedback_reports`
            WHERE `emp_no` = :emp_no AND `team_evaluation_id` = :team_evaluation_id
        """)
        ret_id = connection.execute(inserted_or_updated_id_query, {"emp_no": emp_no, "team_evaluation_id": team_evaluation_id}).scalar_one()
        
        print(f"DB: feedback_reports[{ret_id}] for emp_no={emp_no}, team_evaluation_id={team_evaluation_id} inserted/updated.")
        return ret_id
    

def update_team_evaluations_module2_results_in_db(team_evaluation_id: int, update_data: Dict) -> bool:
    """
    `team_evaluations` 테이블에 모듈 2 관련 AI 결과를 업데이트합니다.
    """
    with engine.connect() as connection:
        set_clauses = [f"`{k}` = :{k}" for k in update_data.keys()]
        query = text(f"UPDATE `team_evaluations` SET {', '.join(set_clauses)} WHERE `team_evaluation_id` = :team_evaluation_id")
        
        params = {**update_data, "team_evaluation_id": team_evaluation_id}
        result = connection.execute(query, params)
        connection.commit()
        return result.rowcount > 0

def save_final_evaluation_report_module2_results_to_db(emp_no: str, team_evaluation_id: int, results: Dict) -> int:
    """
    `final_evaluation_reports` 테이블에 모듈 2 관련 AI 결과를 삽입하거나 업데이트합니다.
    `emp_no`와 `team_evaluation_id`가 중복되면 업데이트를 수행합니다.
    """
    with engine.connect() as connection:
        # INSERT ... ON DUPLICATE KEY UPDATE 사용
        # emp_no와 team_evaluation_id는 UNIQUE 키(또는 복합 PK)로 설정되어 있어야 합니다.
        cols_for_insert = ["emp_no", "team_evaluation_id"] + list(results.keys())
        values_placeholder = ", ".join([f":{col}" for col in cols_for_insert])
        cols_str = ", ".join([f"`{col}`" for col in cols_for_insert])
        
        on_duplicate_set_clauses = [f"`{k}` = VALUES(`{k}`)" for k in results.keys()]
        
        query = text(f"""
            INSERT INTO `final_evaluation_reports` ({cols_str}) VALUES ({values_placeholder})
            ON DUPLICATE KEY UPDATE {", ".join(on_duplicate_set_clauses)}
        """)
        
        params = {"emp_no": emp_no, "team_evaluation_id": team_evaluation_id, **results}
        
        connection.execute(query, params)
        connection.commit()
        
        inserted_or_updated_id_query = text("""
            SELECT final_evaluation_report_id FROM `final_evaluation_reports`
            WHERE `emp_no` = :emp_no AND `team_evaluation_id` = :team_evaluation_id
        """)
        ret_id = connection.execute(inserted_or_updated_id_query, {"emp_no": emp_no, "team_evaluation_id": team_evaluation_id}).scalar_one()
        
        print(f"DB: final_evaluation_reports[{ret_id}] created/updated for emp_no={emp_no}.")
        return ret_id


def update_temp_evaluations_module2_results_in_db(temp_evaluation_id: int, update_data: Dict) -> bool:
    """
    `temp_evaluations` 테이블에 모듈 2 관련 AI 결과를 업데이트합니다.
    """
    with engine.connect() as connection:
        set_clauses = [f"`{k}` = :{k}" for k in update_data.keys()]
        query = text(f"UPDATE `temp_evaluations` SET {', '.join(set_clauses)} WHERE `TempEvaluation_id` = :temp_evaluation_id")
        
        params = {**update_data, "temp_evaluation_id": temp_evaluation_id}
        result = connection.execute(query, params)
        connection.commit()
        return result.rowcount > 0



from db_utils import *

import re
import json 
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv() 

# LangChain LLM 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# --- LLM 클라이언트 인스턴스 (전역 설정) ---
llm_client = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
print(f"LLM Client initialized with model: {llm_client.model_name}, temperature: {llm_client.temperature}")

# --- LLM 응답에서 JSON 코드 블록 추출 도우미 함수 ---
def _extract_json_from_llm_response(text: str) -> str:
    """LLM 응답 텍스트에서 ```json ... ``` 블록만 추출합니다."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip() # JSON 내용만 반환하고 양쪽 공백 제거
    return text.strip()


def call_llm_for_task_contribution(task_summary_text: str) -> Dict:
    print(f"LLM Call (Task Contribution): '{task_summary_text[:30]}...'")

    system_prompt = """
    당신은 SK 조직의 성과 평가 전문가입니다.
    아래 Task 요약 내용을 보고, 해당 Task가 전체 프로젝트/팀 목표에 얼마나 기여했는지, 
    그리고 업무의 난이도, 완성도, 중요도를 종합적으로 고려하여 100점 만점으로 기여도 점수를 산정하고, 
    간략한 분석 코멘트를 생성해주세요.

    평가 시 다음을 고려합니다:
    - Task의 복잡성과 달성 난이도
    - Task 결과물의 품질과 완성도
    - Task가 다음 단계 또는 다른 팀원에게 미친 긍정적 영향 (선행 조건 해결 등)
    - Task가 팀 목표 달성에 기여한 정도

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """
    
    human_prompt = f"""
    <Task 요약>
    {task_summary_text}
    </Task 요약>

    JSON 응답:
    {{
        "기여도 점수": [기여도 점수 (0-100점, 소수점 첫째 자리까지)],
        "분석 코멘트": "[Task에 대한 분석 코멘트]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"task_summary_text": task_summary_text})
        json_output_raw = response.content

        json_output = _extract_json_from_llm_response(json_output_raw)

        llm_parsed_data = json.loads(json_output)

        score = llm_parsed_data.get("기여도 점수") 
        comment = llm_parsed_data.get("분석 코멘트") 


        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            raise ValueError(f"LLM 반환 점수 {score}가 유효하지 않습니다.")
        if not isinstance(comment, str) or not comment:
            raise ValueError(f"LLM 반환 코멘트 {comment}가 유효하지 않습니다.")

        return {"score": round(float(score), 2), "comment": comment}
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"score": 0.0, "comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"score": 0.0, "comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"score": 0.0, "comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


def call_llm_for_task_achievement(target_level_text: str, task_performance_text: str, grade_definitions: Dict) -> Dict:
    
    system_prompt = """
    당신은 SK 조직의 성과 평가 전문가입니다.
    아래 Task 목표와 실제 성과를 비교하여, Task의 달성률(0-100점 이상)과 적절한 등급(S, A, B, C, D)을 판단하고,
    상세 분석 코멘트를 생성해주세요.

    평가 기준은 다음과 같습니다:
    - 달성률은 0점부터 시작하며, 100점을 초과하여 목표 초과 달성을 나타낼 수 있습니다. (예: 100.1% 이상)
    - 등급은 S, A, B, C, D 중 하나여야 합니다.

    <등급 정의 (LLM 참고용)>
    """
    for grade, desc in grade_definitions.items():
        system_prompt += f"- {grade} 등급: {desc}\n"
    system_prompt += "</등급 정의>\n"
    system_prompt += "결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다."

    human_prompt = f"""
    <Task 목표>
    {target_level_text}
    </Task 목표>

    <실제 성과>
    {task_performance_text}
    </실제 성과>

    JSON 응답:
    {{
      "달성률": [달성률 (0-100점 이상)],
      "등급": "[S, A, B, C, D 중 하나]",
      "상세 분석 코멘트": "[Task에 대한 상세 분석 코멘트]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"target_level_text": target_level_text, "task_performance_text": task_performance_text, "grade_definitions": grade_definitions})
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        rate = llm_parsed_data.get("달성률") 
        grade = llm_parsed_data.get("등급") 
        analysis = llm_parsed_data.get("상세 분석 코멘트") 

        # 수정된 부분: 달성률 유효성 검사 상한 제거
        if not isinstance(rate, (int, float)) or not (0 <= rate): 
            raise ValueError(f"LLM 반환 달성률 {rate}가 유효하지 않습니다 (0 이상이어야 합니다).")
        if grade not in ["S", "A", "B", "C", "D"]:
            raise ValueError(f"LLM 반환 등급 {grade}가 유효하지 않습니다.")
        if not isinstance(analysis, str) or not analysis:
            raise ValueError(f"LLM 반환 분석 코멘트 {analysis}가 유효하지 않습니다.")

        return {"grade": grade, "rate": round(float(rate), 2), "analysis": analysis}

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"grade": "D", "rate": 0.0, "analysis": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"grade": "D", "rate": 0.0, "analysis": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"grade": "D", "rate": 0.0, "analysis": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


def call_llm_for_overall_contribution_summary(all_individual_task_results: List[Dict], emp_name: str, emp_no: str) -> Dict: 
    print(f"LLM Call (Overall Contribution Summary): '{emp_name} ({emp_no})' Task {len(all_individual_task_results)}개 기반 요약 요청.") 

    task_details_str = ""
    for task in all_individual_task_results:
        task_details_str += f"- Task: {task.get('task_name')} (ID: {task.get('task_id')})\n"
        task_details_str += f"  Summary: {task.get('task_summary', task.get('task_performance', ''))}\n"
        if task.get('ai_contribution_score') is not None:
            task_details_str += f"  AI 기여도: {task.get('ai_contribution_score')}점\n"
        if task.get('ai_achievement_rate') is not None:
            task_details_str += f"  AI 달성률: {task.get('ai_achievement_rate')}%\n"
        if task.get('ai_assessed_grade'):
            task_details_str += f"  AI 등급: {task.get('ai_assessed_grade')}\n"
        task_details_str += "\n"

    system_prompt = """
    당신은 SK 조직의 HR 성과 전문가입니다.
    아래 제공된 개인의 모든 Task 정보, Task Summary, 그리고 AI가 분석한 개별 Task 기여도/달성률 점수를 종합적으로 고려하여,
    이 개인의 총체적인 기여도 점수 (팀 내 상대 비율, 0-100%)를 추정하고,
    이름과 사번을 명시하며 개인의 전체적인 성과와 기여에 대한 간략한 종합 코멘트를 생성해주세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    직원 이름을 언급할 때는 반드시 "이름(사번)님" 형태로 작성해주세요.
    """

    human_prompt = f"""
    <개인 Task 종합 정보>
    {task_details_str}
    </개인 Task 종합 정보>
    <평가 대상 개인 정보>
    이름: {emp_name}
    사번: {emp_no}
    </평가 대상 개인 정보>

    JSON 응답:
    {{
      "total_contribution": [개인의 총체적인 기여도 점수 (0-100점)],
      "comment": "[{emp_name}({emp_no})님의 전체 성과와 기여에 대한 종합 코멘트]",
      "average_rate": [Task 달성률들의 평균 또는 종합적인 달성률 추정 (0-100점 이상)]
    }}
    """


    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"all_individual_task_results": all_individual_task_results, "emp_name": emp_name, "emp_no": emp_no})
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        total_contribution = llm_parsed_data.get("total_contribution")
        comment = llm_parsed_data.get("comment")
        average_rate = llm_parsed_data.get("average_rate")

        if not isinstance(total_contribution, (int, float)) or not (0 <= total_contribution <= 100):
            raise ValueError(f"LLM 반환 총 기여도 {total_contribution}가 유효하지 않습니다.")
        if not isinstance(comment, str) or not comment:
            raise ValueError(f"LLM 반환 코멘트 {comment}가 유효하지 않습니다.")
        if not isinstance(average_rate, (int, float)) or not (0 <= average_rate): # 0-120점 -> 0점 이상으로 수정
            raise ValueError(f"LLM 반환 평균 달성률 {average_rate}가 유효하지 않습니다 (0 이상이어야 합니다).")

        return {"total_contribution": round(float(total_contribution), 2), "comment": comment, "average_rate": round(float(average_rate), 2)}

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"total_contribution": 0.0, "comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}...", "average_rate": 0.0}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"total_contribution": 0.0, "comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}...", "average_rate": 0.0}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"total_contribution": 0.0, "comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}...", "average_rate": 0.0}


def call_llm_for_team_overall_analysis(all_team_kpis_results: List[Dict]) -> Dict:
    print(f"LLM Call (Team Overall Analysis): KPI {len(all_team_kpis_results)}개 기반 분석 요청.")

    kpi_details_str = ""
    for kpi in all_team_kpis_results:
        kpi_details_str += f"- KPI: {kpi.get('kpi_name')} (ID: {kpi.get('team_kpi_id')})\n"
        kpi_details_str += f"  Description: {kpi.get('kpi_description')}\n"
        if kpi.get('ai_kpi_overall_progress_rate') is not None:
            kpi_details_str += f"  AI 진행률: {kpi.get('ai_kpi_overall_progress_rate')}%\n"
        if kpi.get('ai_kpi_analysis_comment'):
            kpi_details_str += f"  AI 코멘트: {kpi.get('ai_kpi_analysis_comment')}\n"
        kpi_details_str += "\n"

    system_prompt = """
    당신은 SK 조직의 고위 경영진을 위한 팀 성과 분석 전문가입니다.
    아래 제공된 팀의 KPI 정보, 설명, 그리고 AI가 분석한 각 KPI의 진행률 및 코멘트를 종합적으로 검토하여,
    이 팀의 전반적인 목표 달성률을 추정하고, 팀 성과의 주요 특징과 개선점에 대한 분석 코멘트를 생성해주세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """

    human_prompt = f"""
    <팀 KPI 종합 정보>
    {kpi_details_str}
    </팀 KPI 종합 정보>

    JSON 응답:
    {{
      "overall_rate": [팀 전체의 목표 달성률 추정 (0-100점)],
      "comment": "[팀 성과에 대한 전반적인 분석 코멘트]"
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"all_team_kpis_results": all_team_kpis_results})
        json_output_raw = response.content
        
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        overall_rate = llm_parsed_data.get("overall_rate")
        comment = llm_parsed_data.get("comment")

        if not isinstance(overall_rate, (int, float)) or not (0 <= overall_rate <= 100):
            raise ValueError(f"LLM 반환 전체 달성률 {overall_rate}가 유효하지 않습니다.")
        if not isinstance(comment, str) or not comment:
            raise ValueError(f"LLM 반환 코멘트 {comment}가 유효하지 않습니다.")

        return {"overall_rate": round(float(overall_rate), 2), "comment": comment}

    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"overall_rate": 0.0, "comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"overall_rate": 0.0, "comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"overall_rate": 0.0, "comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


def call_llm_for_kpi_relative_contribution(kpi_analysis_input: Dict) -> Dict:
    kpi_goal = kpi_analysis_input.get("kpi_goal", "알 수 없는 목표")
    kpi_description = kpi_analysis_input.get("kpi_description", "")
    team_tasks = kpi_analysis_input.get("team_members_tasks", [])
    
    print(f"LLM Call (KPI Relative Contribution): '{kpi_goal[:30]}...' KPI 내 개인별 상대 기여도 분석 요청.")
    
    actual_emp_nos_in_kpi = sorted(list(set(task.get('emp_no') for task in team_tasks if task.get('emp_no'))))

    system_prompt = """
    당신은 팀 KPI 성과에 대한 개인별 기여도를 평가하는 전문가입니다.
    아래는 특정 팀 KPI의 목표, 설명, 그리고 이 KPI에 기여한 팀원들의 Task 상세 내용 및 AI가 분석한 개별 Task 기여도 점수입니다.
    
    이 정보를 종합적으로 검토하여 다음을 수행하세요:
    1. 이 KPI에 대한 각 개인의 **상대적인 기여도 점수 (총합 100%)**를 판단하세요.
       - 반환하는 JSON의 `individual_relative_contributions_in_kpi` 딕셔너리에는 아래 <실제 팀원 사번 목록>에 있는 모든 사번에 대해 기여도를 포함해야 합니다.
       - 각 개인의 기여도 점수(0-100점)는 소수점 두 자리까지 허용합니다.
       - 어떤 팀원의 기여도가 0%이더라도 해당 사번과 0점을 명시적으로 포함해야 합니다.
       - 모든 팀원의 기여도 합계가 100%가 되도록 조정해야 합니다.
    2. KPI 전체의 진행 상황에 대한 간략한 분석 코멘트를 생성하세요.

    평가 시 다음을 고려해야 합니다:
    - 각 Task의 내용이 KPI 목표 달성에 얼마나 중요한가?
    - 각 Task의 AI 기여도 점수는 어떤 의미인가? (개별 Task의 품질 및 중요도)
    - 팀원 간 Task의 상호 의존성, 선행/후행 관계, 협업 기여도
    - 특정 팀원이 여러 Task를 수행했거나, 더 중요한 Task를 수행했는가?
    - 결과물 JSON에 불필요한 텍스트를 포함하지 마세요.
    - 직원 이름을 언급할 때는 반드시 "이름(사번)님" 형태로 작성해주세요.


    결과는 다음 JSON 형식으로만 응답해주세요:
    """

    team_tasks_str = ""
    for task in team_tasks:
        emp_name = task.get('emp_name', '이름없음') 
        emp_no = task.get('emp_no', '사번없음')
        team_tasks_str += f"- 팀원: {emp_name}({emp_no})님, Task: {task.get('task_name')}\n" 
        team_tasks_str += f"  요약: {task.get('task_summary')}\n"
        if task.get('ai_contribution_score_from_individual_analysis') is not None:
            team_tasks_str += f"  개별 AI 기여도 점수 (참고용): {task.get('ai_contribution_score_from_individual_analysis')}점\n"
        team_tasks_str += "\n"


    individual_contributions_json_example = ",\n".join([f'"{emp_no}": [상대 기여도 (0-100점)]' for emp_no in actual_emp_nos_in_kpi])
    if not individual_contributions_json_example:
        individual_contributions_json_example = '"EMP_NO_X": [상대 기여도 (0-100점)]'

    human_prompt = f"""
    <팀 KPI 목표>
    {kpi_goal}
    </팀 KPI 목표>
    <팀 KPI 설명>
    {kpi_description}
    </팀 KPI 설명>
    <팀원 Task 정보>
    {team_tasks_str}
    </팀원 Task 정보>
    <실제 팀원 사번 목록>
    {', '.join(actual_emp_nos_in_kpi) if actual_emp_nos_in_kpi else '없음'}
    </실제 팀원 사번 목록>

    JSON 응답:
    {{
      "kpi_overall_rate": [KPI 전체의 진행 상황에 대한 점수 (0-100점)],
      "kpi_analysis_comment": "[KPI 전체 진행 상황에 대한 분석 코멘트]",
      "individual_relative_contributions_in_kpi": {{
        {individual_contributions_json_example}
      }}
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({"kpi_analysis_input": kpi_analysis_input})
        json_output_raw = response.content
        json_output = _extract_json_from_llm_response(json_output_raw)
        
        llm_parsed_data = json.loads(json_output)
        
        kpi_overall_rate = llm_parsed_data.get("kpi_overall_rate")
        kpi_analysis_comment = llm_parsed_data.get("kpi_analysis_comment")
        individual_relative_contributions_raw_from_llm = llm_parsed_data.get("individual_relative_contributions_in_kpi")

        if not isinstance(kpi_overall_rate, (int, float)) or not (0 <= kpi_overall_rate <= 100):
            raise ValueError(f"LLM 반환 KPI 전체 진행률 {kpi_overall_rate}가 유효하지 않습니다.")
        if not isinstance(kpi_analysis_comment, str) or not kpi_analysis_comment:
            raise ValueError(f"LLM 반환 KPI 분석 코멘트 {kpi_analysis_comment}가 유효하지 않습니다.")
        if not isinstance(individual_relative_contributions_raw_from_llm, dict):
            raise ValueError(f"LLM 반환 개인 상대 기여도 형식 {individual_relative_contributions_raw_from_llm}가 유효하지 않습니다.")
        
        # --- 파싱 로직 보강: LLM이 반환한 사번 외의 사번 처리 및 합계 검증 ---
        final_relative_contributions = {}
        for emp_no in actual_emp_nos_in_kpi:
            final_relative_contributions[emp_no] = 0.0
        
        for emp_no_from_llm, score in individual_relative_contributions_raw_from_llm.items():
            if emp_no_from_llm in final_relative_contributions and isinstance(score, (int, float)):
                final_relative_contributions[emp_no_from_llm] = round(float(score), 2)
            else:
                print(f"Warning: LLM이 예상치 못한 사번 '{emp_no_from_llm}'를 반환했거나 점수가 유효하지 않아 무시됩니다. 점수: {score}")

        total_relative_sum = sum(final_relative_contributions.values())
        if total_relative_sum > 0 and not (99.9 <= total_relative_sum <= 100.1):
            print(f"Warning: 개인 상대 기여도 합계가 100%와 다릅니다: {total_relative_sum}%. 재조정 시도.")
            adjustment_factor = 100.0 / total_relative_sum if total_relative_sum > 0 else 1.0
            adjusted_contributions = {k: round(v * adjustment_factor, 2) for k, v in final_relative_contributions.items()}
            final_relative_contributions = adjusted_contributions
            print(f"재조정된 기여도: {final_relative_contributions}")
            
        return {
            "kpi_overall_rate": round(float(kpi_overall_rate), 2),
            "kpi_analysis_comment": kpi_analysis_comment,
            "individual_relative_contributions_in_kpi": final_relative_contributions
        }
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {
            "kpi_overall_rate": 0.0,
            "kpi_analysis_comment": f"AI 분석 실패: JSON 파싱 오류 - {json_output_raw[:100]}...",
            "individual_relative_contributions_in_kpi": {}
        }
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {
            "kpi_overall_rate": 0.0,
            "kpi_analysis_comment": f"AI 분석 실패: 유효성 오류 - {json_output_raw[:100]}...",
            "individual_relative_contributions_in_kpi": {}
        }
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {
            "kpi_overall_rate": 0.0,
            "kpi_analysis_comment": f"AI 분석 실패: 예기치 않은 오류 - {str(e)[:100]}...",
            "individual_relative_contributions_in_kpi": {}
        }
    


def call_llm_for_individual_contribution_reason_comment(
    task_info: Dict, 
    adjusted_contribution_score: float, 
    kpi_goal: str, 
    kpi_overall_comment: str) -> Dict:
    """
    개인의 Task 상세 내역, 조정된 기여도 점수, KPI 맥락을 종합하여
    Task에 대한 기여도 근거 코멘트를 생성합니다.
    """
    emp_name = task_info.get("emp_name", "이름 없음") # Task info에 emp_name이 없다면 db에서 조회해야 함
    emp_no = task_info.get("emp_no", "사번 없음")
    task_name = task_info.get("task_name", "알 수 없는 Task")
    task_summary_text = task_info.get("task_summary", task_info.get("task_performance", "상세 내용 없음"))

    print(f"LLM Call (Individual Contribution Reason): '{emp_name} ({emp_no})'의 '{task_name[:30]}...' Task 근거 요청.")

    system_prompt = """
    당신은 SK 조직의 성과 평가 전문가이자 명확한 근거를 제시하는 분석가입니다.
    아래 제공된 개인의 특정 Task 상세 내용, 이 Task가 속한 KPI의 목표, 그리고 팀 전체에 대한 KPI 분석 코멘트를 종합적으로 고려하여,
    이 Task의 최종 조정된 기여도 점수(KPI 내 상대적 기여도)가 왜 그렇게 산정되었는지에 대한 구체적이고 복합적인 근거 코멘트를 작성해주세요.

    코멘트는 다음 요소를 포함해야 합니다:
    - Task 자체의 내용과 중요도 (Task Summary 기반)
    - LLM이 판단한 KPI 내 상대적 기여도 점수 (제시된 점수 활용)
    - 이 Task가 팀 KPI 목표 달성에 어떻게 기여했는지 (KPI 목표, 전체 KPI 코멘트 기반)
    - Task 간의 상호 관계나 협업 등의 맥락이 기여도에 미친 영향 (제공된 정보 내에서 추론)
    - 직원 이름을 언급할 때는 반드시 "이름(사번)님" 형태로 작성해주세요.

    결과는 다음 JSON 형식으로만 응답해주세요. 불필요한 서문이나 추가 설명 없이 JSON만 반환해야 합니다.
    """

    human_prompt = f"""
    <Task 상세 정보>
    이름: {emp_name}
    사번: {emp_no}
    Task 이름: {task_name}
    Task 요약/성과: {task_summary_text}
    조정된 기여도 점수: {adjusted_contribution_score}점
    </Task 상세 정보>

    <KPI 정보>
    KPI 목표: {kpi_goal}
    KPI 전체 분석 코멘트: {kpi_overall_comment}
    </KPI 정보>

    JSON 응답:
    {{
      "comment_reason": "[{emp_name}({emp_no})님의 해당 Task에 대한 구체적 근거 코멘트]"
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = prompt | llm_client

    try:
        response: AIMessage = chain.invoke({
            "task_info": task_info, 
            "adjusted_contribution_score": adjusted_contribution_score, 
            "kpi_goal": kpi_goal, 
            "kpi_overall_comment": kpi_overall_comment
        })
        json_output_raw = response.content
        json_output = _extract_json_from_llm_response(json_output_raw)
        llm_parsed_data = json.loads(json_output)
        
        comment_reason = llm_parsed_data.get("comment_reason")

        if not isinstance(comment_reason, str) or not comment_reason:
            raise ValueError(f"LLM 반환 근거 코멘트 {comment_reason}가 유효하지 않습니다.")

        return {"comment": comment_reason}
        
    except json.JSONDecodeError as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}. 원본 응답: '{json_output_raw}'. 파싱 시도 텍스트: '{json_output[:100]}...'")
        return {"comment": f"AI 근거 생성 실패: JSON 파싱 오류 - {json_output_raw[:100]}..."}
    except ValueError as e:
        print(f"LLM 응답 데이터 유효성 오류: {e}. 응답: {json_output}")
        return {"comment": f"AI 근거 생성 실패: 유효성 오류 - {json_output[:100]}..."}
    except Exception as e:
        print(f"LLM 호출 중 예기치 않은 오류 발생: {e}. 원본 응답: '{json_output_raw}'")
        return {"comment": f"AI 근거 생성 실패: 예기치 않은 오류 - {str(e)[:100]}..."}


from db_utils import *
from llm_utils import *

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List, Literal, TypedDict, Dict
from langchain_core.messages import HumanMessage 
import operator
from langgraph.graph import StateGraph, START, END


# --- Module2AgentState 정의 ---
class Module2AgentState(TypedDict):
    """
    모듈 2 (목표달성도 분석 모듈)의 내부 상태를 정의합니다.
    이 상태는 모듈 2 내의 모든 서브모듈이 공유하고 업데이트합니다.
    """
    messages: Annotated[List[HumanMessage], operator.add] 

    report_type: Literal["quarterly", "annual"] 
    team_id: int 
    period_id: int 
    
    target_task_summary_ids: List[int] 
    target_team_kpi_ids: List[int] 

    updated_task_ids: List[int]
    updated_team_kpi_ids: List[int]
    
    kpi_individual_relative_contributions: List[Dict] = [] 

    feedback_report_id: int = None 
    team_evaluation_id: int = None 
    final_evaluation_report_id: int = None 
    updated_temp_evaluation_ids_list: List[int] = [] 

# --- 서브모듈 함수 정의 ---

# 1. 데이터 수집 서브모듈
def data_collection_submodule(state: Module2AgentState) -> Module2AgentState:
    messages = state.get("messages", []) + [HumanMessage(content="모듈 2: 데이터 수집 ID 초기화 완료")] 
    return {"messages": messages}


# 2. 개인 기여도 계산 서브모듈
def calculate_individual_contribution_submodule(state: Module2AgentState) -> Module2AgentState:
    report_type = state["report_type"] 
    target_task_summary_ids = state["target_task_summary_ids"] 
    
    updated_task_ids_list = [] 

    for task_summary_id in target_task_summary_ids: 
        task_data = fetch_task_summary_by_id(task_summary_id) 
        if not task_data: 
            print(f"Warning: Task data not found for task_summary_id {task_summary_id}.") 
            continue 
        
        task_id = task_data["task_id"] 
        
        llm_results = {} 
        update_data = {} 

        # --- 분기별 로직: 기여도만 계산 ---
        if report_type == "quarterly": 
            task_summary_text = task_data.get("task_summary", "") 
            if task_summary_text: 
                llm_results = call_llm_for_task_contribution(task_summary_text) 
                update_data = {
                    "ai_contribution_score": llm_results.get("score"), 
                    "ai_analysis_comment_task": llm_results.get("comment") 
                }
            else: 
                print(f"Warning: No task_summary found for task_id {task_id} in {report_type} report.") 
                continue 
        # --- 연말 로직: 달성률/등급 및 기여도 계산 ---
        elif report_type == "annual": 
            target_level = task_data.get("target_level", "") 
            task_performance = task_data.get("task_performance", "") 
            task_summary_text_q4 = task_data.get("task_summary", "") 

            grade_definitions = fetch_grade_definitions_from_db() 
            
            if target_level and task_performance: 
                llm_achievement_results = call_llm_for_task_achievement(target_level, task_performance, grade_definitions) 
                
                llm_contribution_results = call_llm_for_task_contribution(task_summary_text_q4) 
                
                update_data = {
                    "ai_contribution_score": llm_contribution_results.get("score"), 
                    "ai_achievement_rate": llm_achievement_results.get("rate"), 
                    "ai_assessed_grade": llm_achievement_results.get("grade"), 
                    "ai_analysis_comment_task": llm_achievement_results.get("analysis") 
                }
            else: 
                print(f"Warning: target_level or task_performance missing for task_id {task_id} in {report_type} report.") 
                continue 

        if update_task_ai_results_in_db(task_id, update_data): 
            updated_task_ids_list.append(task_id) 
        else: 
            print(f"Failed to update AI results for task_id: {task_id}") 

    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 개인 Task 기여도/달성률 계산 및 DB 업데이트 완료 ({len(updated_task_ids_list)}건)")] 
    return {"messages": messages, "updated_task_ids": updated_task_ids_list}



# 3. 팀 목표 분석 서브모듈 (수정: KPI 내 개인 상대 기여도 계산 및 LLM 요청 후 tasks 업데이트)
def analyze_team_goals_submodule(state: Module2AgentState) -> Module2AgentState:
    report_type = state["report_type"] 
    target_team_kpi_ids = state["target_team_kpi_ids"] 
    period_id = state["period_id"] 

    updated_team_kpi_ids_list = [] 
    kpi_individual_relative_contributions_for_state = [] 

    for team_kpi_id in target_team_kpi_ids: 
        kpi_data = fetch_kpi_data_by_id(team_kpi_id) 
        if not kpi_data: 
            print(f"Warning: Team KPI data not found for team_kpi_id {team_kpi_id}.") 
            continue 

        tasks_in_this_kpi = fetch_tasks_for_kpi(team_kpi_id, period_id) 
        
        llm_input_for_kpi_analysis = {
            "kpi_goal": kpi_data.get("kpi_name"), 
            "kpi_description": kpi_data.get("kpi_description"), 
            "team_members_tasks": [
                {
                    "emp_no": task.get("emp_no"), 
                    "task_id": task.get("task_id"), 
                    "task_name": task.get("task_name"), 
                    "task_summary": task.get("task_summary"), 
                    "ai_contribution_score_from_individual_analysis": task.get("ai_contribution_score") 
                } for task in tasks_in_this_kpi
            ]
        }

        llm_kpi_analysis_results = call_llm_for_kpi_relative_contribution(llm_input_for_kpi_analysis) 

        update_data_kpi = { 
            "ai_kpi_overall_progress_rate": llm_kpi_analysis_results.get("kpi_overall_rate"), 
            "ai_kpi_analysis_comment": llm_kpi_analysis_results.get("kpi_analysis_comment") 
        }

        if update_team_kpi_ai_results_in_db(team_kpi_id, update_data_kpi): 
            updated_team_kpi_ids_list.append(team_kpi_id) 
            
            if "individual_relative_contributions_in_kpi" in llm_kpi_analysis_results: 
                relative_contributions_by_emp = llm_kpi_analysis_results["individual_relative_contributions_in_kpi"]
                kpi_individual_relative_contributions_for_state.append({ 
                    "team_kpi_id": team_kpi_id,
                    "relative_contributions": relative_contributions_by_emp
                }) 

                # --- 수정된 부분: tasks 테이블 ai_contribution_score 및 ai_analysis_comment_task 업데이트 ---
                for task in tasks_in_this_kpi: # 현재 KPI에 속한 Task들을 다시 순회
                    emp_no_task = task.get("emp_no")
                    task_id_current = task.get("task_id")
                    
                    if emp_no_task in relative_contributions_by_emp:
                        new_contribution_score = relative_contributions_by_emp[emp_no_task]
                        
                        # LLM 호출을 위한 Task 상세 정보 준비
                        # task_data에는 emp_name도 포함될 수 있도록 fetch_task_summary_by_id 쿼리 확인
                        # (py의 fetch_employees_by_team_id도 emp_name을 가져옴)
                        task_data_for_comment = fetch_task_summary_by_id(task.get("task_summary_Id")) # task_summary_Id 필요
                        if not task_data_for_comment:
                            print(f"Warning: Task data for comment generation not found for task_summary_Id {task.get('task_summary_Id')}.")
                            continue

                        # 새로운 LLM 호출: Task별 상세 기여도 근거 코멘트 생성
                        reason_llm_results = call_llm_for_individual_contribution_reason_comment(
                            task_data_for_comment, # Task 상세 정보
                            float(new_contribution_score), # 조정된 점수 (LLM에 전달 시 float으로 변환)
                            kpi_data.get('kpi_name', ''), # KPI 목표
                            llm_kpi_analysis_results.get('kpi_analysis_comment', '') # KPI 전체 분석 코멘트
                        )
                        adjusted_comment = reason_llm_results.get("comment", f"AI 근거 생성 실패: {emp_no_task}의 Task {task_id_current}에 대한 근거를 생성할 수 없습니다.")
                        
                        update_data_task = {
                            "ai_contribution_score": new_contribution_score,
                            "ai_analysis_comment_task": adjusted_comment
                        }
                        
                        if not update_task_ai_results_in_db(task_id_current, update_data_task):
                            print(f"Warning: Failed to update ai_contribution_score for task_id {task_id_current} (emp_no: {emp_no_task}) with new relative contribution.")
                    else:
                        print(f"Warning: Emp_no {emp_no_task} from task_id {task_id_current} not found in LLM's relative contributions for KPI {team_kpi_id}. ai_contribution_score not updated for this task.")

        else: 
            print(f"Failed to update AI results for team_kpi_id: {team_kpi_id}") 

    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 팀 목표 분석 및 DB 업데이트 완료 ({len(updated_team_kpi_ids_list)}건)")] 
    
    return {"messages": messages, "updated_team_kpi_ids": updated_team_kpi_ids_list,
            "kpi_individual_relative_contributions": kpi_individual_relative_contributions_for_state}



# 4. 모듈 2 관련 레포트 테이블 데이터 생성/업데이트 서브모듈
def generate_module2_report_data_submodule(state: Module2AgentState) -> Module2AgentState:
    report_type = state["report_type"] 
    team_id = state["team_id"] 
    period_id = state["period_id"] 
    
    kpi_individual_relative_contributions = state.get("kpi_individual_relative_contributions", []) 
    
    updated_ids_for_state = {} 

    # 개인의 팀 전체 기여도 계산 (KPI별 상대 기여도 기반)
    emp_overall_relative_contributions = {} 
    
    for kpi_result in kpi_individual_relative_contributions: 
        for emp_no, relative_score in kpi_result["relative_contributions"].items(): 
            if emp_no not in emp_overall_relative_contributions: 
                emp_overall_relative_contributions[emp_no] = 0 
            emp_overall_relative_contributions[emp_no] += relative_score 

    # --- 팀 전체 기여도 합계 100%로 정규화 ---
    total_sum_of_relative_contributions = sum(emp_overall_relative_contributions.values())
    if total_sum_of_relative_contributions > 0:
        adjustment_factor = 100.0 / total_sum_of_relative_contributions
        for emp_no, score in emp_overall_relative_contributions.items():
            emp_overall_relative_contributions[emp_no] = round(score * adjustment_factor, 2)
    # ----------------------------------------

    # 모든 개인 Task 결과는 여전히 필요 
    all_individual_task_results_raw = [] 
    for task_summary_id in state["target_task_summary_ids"]: 
        task_data = fetch_task_summary_by_id(task_summary_id) 
        if task_data: 
            all_individual_task_results_raw.append(task_data) 


    # 개인용 분기별 피드백 레포트 (feedback_reports)
    if report_type == "quarterly": 
        # 1. 해당 팀의 모든 emp_no 조회 (피드백 레포트는 팀원용)
        all_team_members_in_db = fetch_employees_by_team_id(team_id)

        for member_info in all_team_members_in_db:
            emp_no_current_member = member_info["emp_no"]
            emp_name_current_member = member_info["emp_name"] # 직원 이름 추가

            # 팀장(MANAGER) 역할은 피드백 레포트를 직접 생성하지 않으므로 건너뜁니다.
            if member_info.get("role") == "MANAGER": 
                print(f"Info: Skipping feedback_reports for manager {emp_no_current_member}.")
                continue

            # 해당 팀원에게 해당하는 Task Summaries 필터링
            individual_tasks_for_report = [
                task for task in all_individual_task_results_raw 
                if task.get("emp_no") == emp_no_current_member and task.get("period_id") <= period_id 
            ]

            if not individual_tasks_for_report: 
                print(f"Warning: No individual tasks found for emp_no {emp_no_current_member} in period {period_id}. Skipping feedback_reports save for this member.") 
                continue 

            # LLM 호출 시 emp_name, emp_no 전달
            individual_overall_results = call_llm_for_overall_contribution_summary(
                individual_tasks_for_report, emp_name_current_member, emp_no_current_member
            ) 
            calculated_individual_quarterly_contribution = emp_overall_relative_contributions.get(emp_no_current_member, 0) 

            team_evaluation_id_for_report = state.get("team_evaluation_id") 
            if team_evaluation_id_for_report is None: 
                print(f"Warning: team_evaluation_id for team_id={team_id}, period_id={period_id} is missing in state. Cannot save feedback_reports for {emp_no_current_member}. (앞단 Agent에서 생성 필요)") 
            else: 
                actual_team_eval_id_in_db = fetch_team_evaluation_id_by_team_and_period(team_id, period_id) 
                if actual_team_eval_id_in_db != team_evaluation_id_for_report: 
                     print(f"Warning: team_evaluation_id {team_evaluation_id_for_report} from state does not match existing ID in DB for team={team_id}, period={period_id}. Skipping feedback_reports save for {emp_no_current_member}.") 
                else: 
                    # --- INSERT 또는 UPDATE 로직 (ON DUPLICATE KEY UPDATE 사용) ---
                    feedback_report_id = save_feedback_report_module2_results_to_db(
                        emp_no_current_member, team_evaluation_id_for_report, 
                        {
                            "ai_individual_total_contribution_quarterly": calculated_individual_quarterly_contribution, 
                            "ai_overall_contribution_summary_comment": individual_overall_results.get("comment") 
                        }
                    )
                    updated_ids_for_state["feedback_report_id"] = feedback_report_id 
                    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 개인 {emp_no_current_member} 분기별 레포트 내용 생성/업데이트 및 feedback_reports 저장 완료 (ID: {feedback_report_id})")] 

    # 팀장용 분기별/연말 팀 전체 평가 레포트 (team_evaluations)
    team_evaluation_id = state.get("team_evaluation_id") 
    if team_evaluation_id is None: 
        print(f"Warning: team_evaluation_id for team_id={team_id}, period_id={period_id} is missing in state. Cannot update team_evaluations. (앞단 Agent에서 생성 필요)") 
    else: 
        actual_team_eval_id_in_db = fetch_team_evaluation_id_by_team_and_period(team_id, period_id) 
        if actual_team_eval_id_in_db != team_evaluation_id: 
             print(f"Warning: team_evaluation_id {team_evaluation_id} from state does not match existing ID in DB for team={team_id}, period={period_id}. Skipping team_evaluations update.") 
        else: 
            all_team_kpis_results = [fetch_kpi_data_by_id(kpi_id) for kpi_id in state["target_team_kpi_ids"] if fetch_kpi_data_by_id(kpi_id)] 
            team_overall_results = call_llm_for_team_overall_analysis(all_team_kpis_results) 
            
            update_data = {
                "ai_team_overall_achievement_rate": team_overall_results.get("overall_rate"), 
                "ai_team_overall_analysis_comment": team_overall_results.get("comment") 
            }
            update_team_evaluations_module2_results_in_db(team_evaluation_id, update_data) 
            updated_ids_for_state["team_evaluation_id"] = team_evaluation_id 
            messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 팀 전체 분석 코멘트 생성 및 team_evaluations 업데이트 완료 (ID: {team_evaluation_id})")] 


    # 개인용 연말 최종 평가 레포트 (final_evaluation_reports)
    if report_type == "annual": 
        all_team_members_in_db = fetch_employees_by_team_id(team_id)

        for member_info in all_team_members_in_db:
            emp_no_current_member = member_info["emp_no"]
            emp_name_current_member = member_info["emp_name"] # 직원 이름 추가

            # 팀장(MANAGER) 역할은 최종 평가 레포트의 직접 대상이 아니므로 건너뜁니다.
            if member_info.get("role") == "MANAGER": 
                print(f"Info: Skipping final_evaluation_reports for manager {emp_no_current_member}.")
                continue

            # 해당 팀원에게 해당하는 Task Summaries 필터링
            individual_tasks_for_annual_report = [
                task for task in all_individual_task_results_raw 
                if task.get("emp_no") == emp_no_current_member and task.get("period_id") <= period_id
            ]
            if not individual_tasks_for_annual_report: 
                print(f"Warning: No individual tasks found for emp_no {emp_no_current_member} in period {period_id}. Skipping final_evaluation_reports save for this member.") 
                continue 

            # LLM 호출 시 emp_name, emp_no 전달
            annual_individual_summary_results = call_llm_for_overall_contribution_summary(
                individual_tasks_for_annual_report, emp_name_current_member, emp_no_current_member
            ) 
            
            calculated_annual_individual_total_contribution = emp_overall_relative_contributions.get(emp_no_current_member, 0) 
            
            final_team_evaluation_id_example = state.get("team_evaluation_id") 
            if final_team_evaluation_id_example is None: 
                print(f"Warning: team_evaluation_id for team_id={team_id}, period_id={period_id} is missing in state. Cannot save final_evaluation_reports for {emp_no_current_member}. (앞단 Agent에서 생성 필요)") 
            else: 
                actual_team_eval_id_in_db = fetch_team_evaluation_id_by_team_and_period(team_id, period_id) 
                if actual_team_eval_id_in_db != final_team_evaluation_id_example: 
                     print(f"Warning: team_evaluation_id {final_team_evaluation_id_example} from state does not match existing ID in DB for team={team_id}, period={period_id}. Skipping final_evaluation_reports save for {emp_no_current_member}.") 
                else: 
                    final_report_id = save_final_evaluation_report_module2_results_to_db(
                        emp_no_current_member, final_team_evaluation_id_example, 
                        {
                            "ai_annual_individual_total_contribution": calculated_annual_individual_total_contribution, 
                            "ai_annual_achievement_rate": annual_individual_summary_results.get("average_rate"), 
                            "ai_annual_performance_summary_comment": annual_individual_summary_results.get("comment") 
                        }
                    )
                    updated_ids_for_state["final_report_id"] = final_report_id 
                    messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 개인 {emp_no_current_member} 연말 최종 평가 레포트 내용 생성 및 final_evaluation_reports 저장 완료 (ID: {final_report_id})")] 
    
    # 최종 전 중간 평가 자료 (temp_evaluations)
    if report_type == "annual": 
        all_team_members = fetch_employees_by_team_id(team_id) 

        for member in all_team_members:
            emp_no_current_member = member["emp_no"]
            emp_name_current_member = member["emp_name"] # 직원 이름 추가

            # 팀장(MANAGER) 역할도 temp_evaluations에는 포함될 수 있으므로 (참고 자료)
            # 여기서는 MANAGER 역할도 포함하여 처리합니다.
            
            # 해당 팀원에게 해당하는 Task Summaries 필터링
            individual_tasks_for_temp_eval = [
                task for task in all_individual_task_results_raw
                if task.get("emp_no") == emp_no_current_member and task.get("period_id") <= period_id
            ]
            
            if not individual_tasks_for_temp_eval:
                print(f"Warning: No individual tasks found for emp_no {emp_no_current_member} in period {period_id}. Skipping temp_evaluations update for this member.") 
                continue 

            # LLM 호출 시 emp_name, emp_no 전달
            key_performance_summary_results = call_llm_for_overall_contribution_summary(
                individual_tasks_for_temp_eval, emp_name_current_member, emp_no_current_member
            )
            
            temp_eval_id_for_member = fetch_temp_evaluation_id_by_emp_and_period(emp_no_current_member, period_id) 

            if temp_eval_id_for_member is None: 
                print(f"Warning: temp_evaluation_id for emp_no={emp_no_current_member}, period_id={period_id} is missing in DB. Cannot update temp_evaluations. (앞단 Agent에서 생성 필요)") 
            else: 
                update_temp_evaluations_module2_results_in_db(
                    temp_eval_id_for_member,
                    {
                        "ai_annual_key_performance_contribution_summary": key_performance_summary_results.get("comment")
                    }
                )
                if "updated_temp_evaluation_ids_list" not in updated_ids_for_state: 
                     updated_ids_for_state["updated_temp_evaluation_ids_list"] = [] 
                updated_ids_for_state["updated_temp_evaluation_ids_list"].append(temp_eval_id_for_member) 
                
                messages = state.get("messages", []) + [HumanMessage(content=f"모듈 2: 팀원 {emp_no_current_member} 연간 핵심 성과 기여도 요약 생성 및 temp_evaluations 업데이트 완료 (ID: {temp_eval_id_for_member})")] 

    return {"messages": messages, **updated_ids_for_state}


# 5. 포맷터 서브모듈
def formatter_submodule(state: Module2AgentState) -> Module2AgentState:
    messages = state.get("messages", []) + [HumanMessage(content="모듈 2: 포맷팅 완료")]
    return {"messages": messages}


# 워크플로우 생성
def create_module2_graph():
    """모듈 2 그래프 생성 및 반환"""
    module2_workflow = StateGraph(Module2AgentState)
    
    # 노드 추가
    module2_workflow.add_node("data_collection", data_collection_submodule)
    module2_workflow.add_node("calculate_individual_contribution", calculate_individual_contribution_submodule)
    module2_workflow.add_node("analyze_team_goals", analyze_team_goals_submodule)
    module2_workflow.add_node("generate_module2_report_data", generate_module2_report_data_submodule)
    module2_workflow.add_node("formatter", formatter_submodule)
    
    # 엣지 정의
    module2_workflow.add_edge(START, "data_collection")
    module2_workflow.add_edge("data_collection", "calculate_individual_contribution")
    module2_workflow.add_edge("calculate_individual_contribution", "analyze_team_goals")
    module2_workflow.add_edge("analyze_team_goals", "generate_module2_report_data")
    module2_workflow.add_edge("generate_module2_report_data", "formatter")
    module2_workflow.add_edge("formatter", END)
    
    return module2_workflow.compile()

# 실행 스크립트 (간단함)
from agent import create_module2_graph, Module2AgentState
from langchain_core.messages import HumanMessage

def run_module2_quarterly():
    """모듈 2 분기별 실행"""
    # State 정의
    state = Module2AgentState(
        messages=[HumanMessage(content="모듈 2 분기별 평가 시작")],
        report_type="quarterly",
        team_id=1,
        period_id=2,
        target_task_summary_ids=[1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30],
        target_team_kpi_ids=[1, 2, 3],
        team_evaluation_id=101,
        updated_task_ids=None,
        updated_team_kpi_ids=None,
        kpi_individual_relative_contributions=None
    )

    # 그래프 생성 및 실행
    print("모듈 2 실행 시작...")
    module2_graph = create_module2_graph()
    result = module2_graph.invoke(state)
    print("모듈 2 실행 완료!")
    print(f"최종 메시지: {result['messages'][-1].content}")
    return result

if __name__ == "__main__":
    run_module2_quarterly()