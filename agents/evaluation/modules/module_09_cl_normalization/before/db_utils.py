# ================================================================
# db_utils_module9.py - 모듈 9 데이터베이스 관련 유틸리티
# ================================================================

from config.settings import *
from typing import Dict, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Row
from dotenv import load_dotenv

load_dotenv()

# DB 설정

db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def row_to_dict(row: Row) -> Dict:
    """SQLAlchemy Row 객체를 딕셔너리로 변환"""
    if row is None:
        return {}
    return row._asdict()

# ================================================================
# 본부 단위 DB 조회 함수들
# ================================================================

def fetch_headquarter_members(headquarter_id: str, period_id: int) -> List[Dict]:
    """본부 내 모든 직원의 manager_score + 기본정보 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT 
                e.emp_no, e.emp_name, e.cl, e.position, e.team_id,
                te.manager_score,
                fer.final_evaluation_report_id
            FROM employees e
            JOIN teams t ON e.team_id = t.team_id
            JOIN headquarters h ON t.headquarter_id = h.headquarter_id
            JOIN temp_evaluations te ON e.emp_no = te.emp_no
            JOIN team_evaluations tea ON t.team_id = tea.team_id
            JOIN final_evaluation_reports fer ON (e.emp_no = fer.emp_no AND tea.team_evaluation_id = fer.team_evaluation_id)
            WHERE h.headquarter_id = :headquarter_id 
              AND tea.period_id = :period_id
              AND te.manager_score IS NOT NULL
            ORDER BY e.cl DESC, e.position DESC
        """)
        results = connection.execute(query, {
            "headquarter_id": headquarter_id, 
            "period_id": period_id
        }).fetchall()
        return [row_to_dict(row) for row in results]

def batch_update_final_evaluation_reports(score_data: List[Dict]) -> Dict:
    """본부 전체 final_evaluation_reports 배치 업데이트 (score, cl_reason 저장)"""
    success_count = 0
    failed_members = []
    
    with engine.connect() as connection:
        try:
            for data in score_data:
                try:
                    query = text("""
                        UPDATE final_evaluation_reports 
                        SET score = :score,
                            cl_reason = :cl_reason
                        WHERE final_evaluation_report_id = :report_id
                    """)
                    
                    result = connection.execute(query, {
                        "report_id": data["final_evaluation_report_id"],
                        "score": data["final_score"],
                        "cl_reason": data["cl_reason"]
                    })
                    
                    if result.rowcount > 0:
                        success_count += 1
                        print(f"DB 업데이트 성공: {data['emp_no']} (정규화: {data['final_score']})")
                    else:
                        failed_members.append(data["emp_no"])
                        print(f"DB 업데이트 실패: {data['emp_no']} (행 없음)")
                        
                except Exception as e:
                    failed_members.append(data["emp_no"])
                    print(f"DB 업데이트 실패: {data['emp_no']} - {e}")
            
            connection.commit()
            print(f"배치 업데이트 완료: 성공 {success_count}건, 실패 {len(failed_members)}건")
            
            return {
                "success_count": success_count,
                "failed_members": failed_members
            }
            
        except Exception as e:
            print(f"배치 업데이트 실패: {e}")
            connection.rollback()
            return {
                "success_count": 0,
                "failed_members": [data["emp_no"] for data in score_data]
            }

def batch_update_final_evaluation_ranking(ranking_data: List[Dict]) -> Dict:
    """final_evaluation_reports 테이블에 ranking 일괄 업데이트"""
    success_count = 0
    failed_members = []
    
    with engine.connect() as connection:
        try:
            for data in ranking_data:
                try:
                    query = text("""
                        UPDATE final_evaluation_reports 
                        SET ranking = :ranking
                        WHERE final_evaluation_report_id = :report_id
                    """)
                    result = connection.execute(query, {
                        "report_id": data["final_evaluation_report_id"],
                        "ranking": data["ranking"]
                    })
                    if result.rowcount > 0:
                        success_count += 1
                    else:
                        failed_members.append(data["emp_no"])
                except Exception as e:
                    failed_members.append(data["emp_no"])
            connection.commit()
            return {
                "success_count": success_count,
                "failed_members": failed_members
            }
        except Exception as e:
            connection.rollback()
            return {
                "success_count": 0,
                "failed_members": [data["emp_no"] for data in ranking_data]
            }

# ================================================================
# 테스트 및 디버깅 함수들
# ================================================================

def get_all_headquarters_with_data(period_id: int = 4) -> List[str]:
    """평가 데이터가 있는 모든 본부 ID 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT h.headquarter_id
            FROM headquarters h
            JOIN teams t ON h.headquarter_id = t.headquarter_id
            JOIN employees e ON t.team_id = e.team_id
            JOIN temp_evaluations te ON e.emp_no = te.emp_no
            WHERE te.manager_score IS NOT NULL
            ORDER BY h.headquarter_id
        """)
        results = connection.execute(query).fetchall()
        return [row.headquarter_id for row in results]

def get_all_headquarters_info() -> List[Dict]:
    """모든 본부 정보 조회"""
    with engine.connect() as connection:
        query = text("""
            SELECT DISTINCT h.headquarter_id, h.headquarter_name
            FROM headquarters h
            JOIN teams t ON h.headquarter_id = t.headquarter_id
            JOIN employees e ON t.team_id = e.team_id
            JOIN temp_evaluations te ON e.emp_no = te.emp_no
            WHERE te.manager_score IS NOT NULL
            ORDER BY h.headquarter_id
        """)
        results = connection.execute(query).fetchall()
        return [row_to_dict(row) for row in results]