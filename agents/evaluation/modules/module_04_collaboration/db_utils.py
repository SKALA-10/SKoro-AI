# ================================================================
# db_utils_module4.py - 모듈 4 데이터베이스 관련 유틸리티
# ================================================================

import json
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from dotenv import load_dotenv

load_dotenv()

from config.settings import *

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict[str, Any]:
    """SQLAlchemy Row 객체를 딕셔너리로 변환합니다."""
    if row is None:
        return {}
    return row._asdict()

# ================================================================
# 모듈 4 전용 DB 함수들
# ================================================================

def fetch_collaboration_tasks_by_kpi(team_kpi_id: int, period_id: int) -> List[Dict]:
    """KPI별 협업 가능한 Task들을 조회합니다."""
    with engine.connect() as connection:
        query = text("""
            SELECT t.task_id, t.task_name, t.emp_no, t.start_date, t.end_date,
                   ts.task_summary, ts.task_summary_Id, 
                   ts.ai_contribution_score, ts.ai_analysis_comment_task,
                   e.emp_name
            FROM tasks t
            JOIN task_summaries ts ON t.task_id = ts.task_id
            JOIN employees e ON t.emp_no = e.emp_no
            WHERE t.team_kpi_id = :team_kpi_id AND ts.period_id = :period_id
            ORDER BY t.start_date
        """)
        results = connection.execute(query, {"team_kpi_id": team_kpi_id, "period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]

def fetch_peer_talk_summary(emp_no: str, period_id: int, report_type: str) -> Optional[str]:
    """개인의 Peer Talk 요약을 조회합니다."""
    with engine.connect() as connection:
        if report_type == "quarterly":
            # feedback_reports에서 조회 (분기별)
            query = text("""
                SELECT fr.ai_peer_talk_summary
                FROM feedback_reports fr
                JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
                WHERE fr.emp_no = :emp_no AND te.period_id = :period_id
                LIMIT 1
            """)
        else:  # annual
            # final_evaluation_reports에서 조회 (연말)
            query = text("""
                SELECT fer.ai_peer_talk_summary
                FROM final_evaluation_reports fer
                JOIN team_evaluations te ON fer.team_evaluation_id = te.team_evaluation_id
                WHERE fer.emp_no = :emp_no AND te.period_id = :period_id
                LIMIT 1
            """)
        
        result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).scalar()
        return result

def fetch_team_members_with_tasks(team_id: int, period_id: int) -> List[Dict]:
    """팀원들과 그들의 Task 통계를 조회합니다. (팀장 제외)"""
    with engine.connect() as connection:
        query = text("""
            SELECT e.emp_no, e.emp_name, e.role,
                   COUNT(DISTINCT t.task_id) as total_task_count,
                   AVG(ts.ai_contribution_score) as avg_contribution_score
            FROM employees e
            LEFT JOIN tasks t ON e.emp_no = t.emp_no
            LEFT JOIN task_summaries ts ON t.task_id = ts.task_id AND ts.period_id = :period_id
            WHERE e.team_id = :team_id AND e.role != 'MANAGER'
            GROUP BY e.emp_no, e.emp_name, e.role
        """)
        results = connection.execute(query, {"team_id": team_id, "period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]

def save_collaboration_matrix_to_db(team_evaluation_id: int, collaboration_matrix: Dict) -> bool:
    """협업 매트릭스를 team_evaluations 테이블의 ai_collaboration_matrix 컬럼에 저장합니다."""
    
    # Decimal을 float로 변환하는 함수
    def convert_decimal(obj):
        if isinstance(obj, dict):
            return {k: convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimal(v) for v in obj]
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'Decimal':
            return float(obj)
        else:
            return obj
    
    # Decimal 변환 후 JSON으로 직렬화
    converted_matrix = convert_decimal(collaboration_matrix)
    collaboration_matrix_json = json.dumps(converted_matrix, ensure_ascii=False)
    
    with engine.connect() as connection:
        try:
            query = text("""
                UPDATE team_evaluations 
                SET ai_collaboration_matrix = :collaboration_matrix
                WHERE team_evaluation_id = :team_evaluation_id
            """)
            
            connection.execute(query, {
                "collaboration_matrix": collaboration_matrix_json,
                "team_evaluation_id": team_evaluation_id
            })
            connection.commit()
            
            print(f"협업 매트릭스 저장 성공: team_evaluation_id={team_evaluation_id}")
            print(f"저장된 JSON 데이터:")
            print(json.dumps(converted_matrix, ensure_ascii=False, indent=2))
            
            return True
            
        except Exception as e:
            print(f"협업 매트릭스 저장 실패: {e}")
            connection.rollback()
            return False

def fetch_team_kpi_progress(team_kpi_ids: List[int], period_id: int) -> List[Dict]:
    """팀 KPI 진행률 정보를 조회합니다."""
    if not team_kpi_ids:
        return []
    
    kpi_ids_str = ','.join(map(str, team_kpi_ids))
    
    with engine.connect() as connection:
        query = text(f"""
            SELECT tk.team_kpi_id, tk.kpi_name, tk.target_value, tk.current_value,
                   tk.ai_kpi_progress_rate, tk.ai_kpi_analysis_comment
            FROM team_kpis tk
            WHERE tk.team_kpi_id IN ({kpi_ids_str}) AND tk.period_id = :period_id
        """)
        results = connection.execute(query, {"period_id": period_id}).fetchall()
        return [row_to_dict(row) for row in results]

def fetch_feedback_report_data(emp_no: str, period_id: int) -> Optional[Dict]:
    """개인의 분기별 피드백 리포트 데이터를 조회합니다."""
    with engine.connect() as connection:
        query = text("""
            SELECT fr.contribution_rate, fr.ai_overall_contribution_summary_comment,
                   fr.ai_peer_talk_summary
            FROM feedback_reports fr
            JOIN team_evaluations te ON fr.team_evaluation_id = te.team_evaluation_id
            WHERE fr.emp_no = :emp_no AND te.period_id = :period_id
            LIMIT 1
        """)
        result = connection.execute(query, {"emp_no": emp_no, "period_id": period_id}).fetchone()
        return row_to_dict(result) if result else None

def fetch_team_evaluation_id(team_id: int, period_id: int) -> Optional[int]:
    """team_evaluations에서 ID 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT team_evaluation_id FROM team_evaluations 
            WHERE team_id = :team_id AND period_id = :period_id
        """)
        result = connection.execute(query, {"team_id": team_id, "period_id": period_id})
        return result.scalar_one_or_none()

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