# ================================================================
# db_utils_module2.py - 모듈 2 데이터베이스 관련 유틸리티
# ================================================================

import sys
import os
import json
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from dotenv import load_dotenv

from config.settings import *

load_dotenv()

# DB 설정
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

# ================================================================
# 데이터 조회 함수들
# ================================================================

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
            
        # 자동 분류가 필요한 경우 기본값 반환
        return "quantitative"

def fetch_team_tasks_and_kpis(team_id: int, period_id: int):
    """팀의 Task와 KPI ID 조회"""
    with engine.connect() as connection:
        # 해당 팀의 해당 기간 Task ID
        task_query = text("""
            SELECT t.task_id FROM tasks t
            JOIN employees e ON t.emp_no = e.emp_no
            WHERE e.team_id = :team_id
        """)
        task_ids = [row[0] for row in connection.execute(task_query, {"team_id": team_id})]
        
        # period_id로 연도 조회
        year_query = text("SELECT year FROM periods WHERE period_id = :period_id")
        year = connection.execute(year_query, {"period_id": period_id}).scalar_one_or_none()
        
        # 해당 팀의 해당 연도 KPI ID만 조회
        kpi_query = text("""
            SELECT team_kpi_id FROM team_kpis 
            WHERE team_id = :team_id AND year = :year
        """)
        kpi_ids = [row[0] for row in connection.execute(kpi_query, {
            "team_id": team_id, 
            "year": year
        })]
        return task_ids, kpi_ids

# ================================================================
# 데이터 업데이트 함수들
# ================================================================

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
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Year-over-year calculation failed: {e}")
    return None