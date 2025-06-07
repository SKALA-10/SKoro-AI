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